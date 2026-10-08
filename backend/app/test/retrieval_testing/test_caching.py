"""The cache in front of retrieval.

A cache is only correct if it is wrong about nothing: it must not hand back an
answer older than its TTL, must not grow without bound, and must not evict a
live entry to make room while holding entries nobody can use. Each of those is
a way of returning the wrong thing quickly, which is worse than slowly.
"""

import sys
import threading

import pytest

from retrieval_layer.caching import CacheStats, ResultCache, cache_key
from retrieval_layer.models import RetrievalRequest
from retrieval_layer.settings import RetrievalSettings


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def cache(max_size: int = 4, ttl: float = 10.0) -> tuple:
    clock = FakeClock()
    return ResultCache(max_size=max_size, ttl_seconds=ttl, clock=clock), clock


def ask(query: str, top_k: int = 8, rerank: bool = True) -> RetrievalRequest:
    return RetrievalRequest(query=query, top_k=top_k, rerank=rerank)


class TestWhatMakesTwoQuestionsTheSame:
    def test_the_same_question_twice(self):
        assert cache_key(ask("how does WAL work")) == cache_key(
            ask("how does WAL work")
        )

    def test_a_different_question(self):
        assert cache_key(ask("a")) != cache_key(ask("b"))

    def test_asking_for_more_results_is_a_different_question(self):
        """Top-8 of a ranking is not the first 8 of top-32 once MMR has run."""
        assert cache_key(ask("q", top_k=8)) != cache_key(ask("q", top_k=32))

    def test_asking_without_reranking_is_a_different_question(self):
        assert cache_key(ask("q", rerank=True)) != cache_key(
            ask("q", rerank=False)
        )

    def test_case_does_not_split_an_entry(self):
        """The embedder is uncased and FTS5 folds case, so these do match."""
        assert cache_key(ask("Python GIL")) == cache_key(ask("python gil"))

    def test_spacing_does_not_split_an_entry(self):
        assert cache_key(ask("  how   does WAL work ")) == cache_key(
            ask("how does WAL work")
        )

    def test_resolved_defaults_agree_with_the_same_values_spelled_out(self):
        settings = RetrievalSettings()
        resolved = RetrievalRequest(query="q").with_defaults(settings)
        explicit = ask("q", top_k=settings.top_k, rerank=settings.rerank)
        assert cache_key(resolved) == cache_key(explicit)

    def test_an_unresolved_request_keys_differently(self):
        """Why the orchestrator must resolve defaults before it keys."""
        assert cache_key(RetrievalRequest(query="q")) != cache_key(ask("q"))


class TestHitsAndMisses:
    def test_a_question_never_asked_is_a_miss(self):
        store, _ = cache()
        assert store.get("k") is None
        assert store.stats.misses == 1
        assert store.stats.hits == 0

    def test_a_stored_answer_comes_back(self):
        store, _ = cache()
        store.put("k", ["a passage"])
        assert store.get("k") == ["a passage"]
        assert store.stats.hits == 1

    def test_an_empty_answer_is_still_an_answer(self):
        """A query that found nothing will find nothing again; that is the
        expensive search worth not repeating."""
        store, _ = cache()
        store.put("k", [])
        assert store.get("k") == []
        assert store.stats.hits == 1

    def test_re_putting_replaces_rather_than_duplicates(self):
        store, _ = cache()
        store.put("k", "first")
        store.put("k", "second")
        assert store.get("k") == "second"
        assert store.stats.size == 1

    def test_the_hit_rate(self):
        store, _ = cache()
        store.put("k", 1)
        store.get("k")
        store.get("k")
        store.get("missing")
        assert store.stats.hit_rate == pytest.approx(2 / 3)

    def test_the_hit_rate_of_a_cache_nobody_has_asked(self):
        assert CacheStats().hit_rate == 0.0


class TestExpiry:
    def test_an_answer_past_its_ttl_is_not_returned(self):
        store, clock = cache(ttl=10.0)
        store.put("k", "stale")
        clock.advance(11.0)
        assert store.get("k") is None

    def test_an_expired_answer_is_dropped_not_merely_withheld(self):
        """Withholding it would leave it occupying a slot for ever."""
        store, clock = cache(ttl=10.0)
        store.put("k", "stale")
        clock.advance(11.0)
        store.get("k")
        assert store.stats.size == 0
        assert store.stats.expirations == 1

    def test_an_expiry_counts_as_a_miss(self):
        store, clock = cache(ttl=10.0)
        store.put("k", "stale")
        clock.advance(11.0)
        store.get("k")
        assert store.stats.misses == 1

    def test_an_answer_within_its_ttl_is_returned(self):
        store, clock = cache(ttl=10.0)
        store.put("k", "fresh")
        clock.advance(9.99)
        assert store.get("k") == "fresh"

    def test_exactly_at_the_ttl_has_expired(self):
        store, clock = cache(ttl=10.0)
        store.put("k", "edge")
        clock.advance(10.0)
        assert store.get("k") is None

    def test_the_ttl_runs_from_the_write_not_the_read(self):
        """A hit must not extend an entry's life, or a popular stale answer
        would never be refreshed."""
        store, clock = cache(ttl=10.0)
        store.put("k", "v")
        clock.advance(6.0)
        assert store.get("k") == "v"
        clock.advance(5.0)
        assert store.get("k") is None

    def test_the_default_clock_is_monotonic(self):
        """A wall-clock step backwards would resurrect an expired entry."""
        import time
        assert ResultCache().clock is time.monotonic


class TestStayingBounded:
    def test_it_never_holds_more_than_its_size(self):
        store, _ = cache(max_size=3)
        for i in range(10):
            store.put(f"k{i}", i)
        assert store.stats.size == 3

    def test_the_least_recently_used_goes_first(self):
        store, _ = cache(max_size=2)
        store.put("a", 1)
        store.put("b", 2)
        store.put("c", 3)
        assert store.get("a") is None
        assert store.get("b") == 2
        assert store.get("c") == 3
        assert store.stats.evictions == 1

    def test_a_hit_protects_an_entry_from_the_next_eviction(self):
        """The U in LRU: reading 'a' makes 'b' the oldest."""
        store, _ = cache(max_size=2)
        store.put("a", 1)
        store.put("b", 2)
        store.get("a")
        store.put("c", 3)
        assert store.get("a") == 1
        assert store.get("b") is None

    def test_re_putting_an_entry_refreshes_its_place_in_the_queue(self):
        """Assigning an existing OrderedDict key leaves it where it was, so a
        rewrite has to remove the entry before storing it again."""
        store, _ = cache(max_size=3)
        store.put("a", 1)
        store.put("b", 2)
        store.put("a", 11)
        store.put("c", 3)
        store.put("d", 4)

        assert store.get("a") == 11
        assert store.get("b") is None

    def test_re_putting_into_a_full_cache_evicts_nothing(self):
        """It takes the slot it already had."""
        store, _ = cache(max_size=2)
        store.put("a", 1)
        store.put("b", 2)
        store.put("a", 11)
        assert store.stats.evictions == 0
        assert store.stats.size == 2

    def test_expired_entries_are_reclaimed_before_a_live_one_is_evicted(self):
        store, clock = cache(max_size=2, ttl=10.0)
        store.put("old1", 1)
        store.put("old2", 2)
        clock.advance(11.0)
        store.put("new", 3)

        assert store.stats.expirations == 2
        assert store.stats.evictions == 0
        assert store.get("new") == 3

    def test_a_cache_of_size_zero_stores_nothing(self):
        """How caching is turned off, so it must not raise on the way."""
        store = ResultCache(max_size=0)
        store.put("k", "v")
        assert store.get("k") is None
        assert store.stats.size == 0

    def test_a_negative_size_is_read_as_off(self):
        store = ResultCache(max_size=-5)
        store.put("k", "v")
        assert store.stats.size == 0

    def test_a_cache_of_size_one_holds_the_newest(self):
        store, _ = cache(max_size=1)
        store.put("a", 1)
        store.put("b", 2)
        assert store.get("a") is None
        assert store.get("b") == 2


class TestClearing:
    def test_clearing_forgets_every_answer(self):
        store, _ = cache()
        store.put("a", 1)
        store.put("b", 2)
        store.clear()
        assert store.stats.size == 0
        assert store.get("a") is None

    def test_clearing_keeps_the_lifetime_counters(self):
        """They describe the process, not the contents."""
        store, _ = cache()
        store.put("a", 1)
        store.get("a")
        store.get("missing")
        store.clear()
        assert store.stats.hits == 1
        assert store.stats.misses == 1

    def test_clearing_an_empty_cache_is_quiet(self, caplog):
        store, _ = cache()
        with caplog.at_level("DEBUG", logger="retrieval_layer.caching"):
            store.clear()
        assert caplog.records == []


class TestUnderConcurrentUse:
    """One cache serves every request, so every operation has to hold up when
    several arrive at once.

    The races are check-then-act pairs: `get` finds an entry and then moves it,
    `put` sees the cache full and then evicts, and the expiry sweep lists stale
    keys and then deletes them. Each is a `KeyError` if another thread removes
    the key in between. The switch interval is dropped so the interpreter
    actually swaps threads inside those windows instead of leaving the race to
    chance.
    """

    @staticmethod
    def hammer(store, keys, rounds, errors):
        try:
            for i in range(rounds):
                key = keys[i % len(keys)]
                store.put(key, i)
                store.get(key)
                store.get(keys[(i * 3) % len(keys)])
                store.get("never stored")
        except BaseException as error:
            errors.append(error)

    def run_hammer(self, store, workers=16, rounds=3000):
        errors = []
        keys = [f"k{i}" for i in range(4)]
        threads = [
            threading.Thread(
                target=self.hammer, args=(store, keys, rounds, errors)
            )
            for _ in range(workers)
        ]
        original = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            sys.setswitchinterval(original)
        return errors

    def test_constant_eviction_races_nobody(self):
        """A cache smaller than the key set evicts on nearly every write."""
        store = ResultCache(max_size=2, ttl_seconds=60.0)
        assert self.run_hammer(store) == []
        assert store.stats.size <= 2

    def test_constant_expiry_races_nobody(self):
        """A clock that moves every time it is read expires everything at once,
        so the sweep and the readers collide on the same keys."""
        ticking = iter(range(10**9))
        store = ResultCache(
            max_size=4, ttl_seconds=2.0, clock=lambda: float(next(ticking))
        )
        assert self.run_hammer(store) == []
        assert store.stats.size <= 4

    def test_the_bound_holds_under_load(self):
        store = ResultCache(max_size=16, ttl_seconds=60.0)
        assert self.run_hammer(store) == []
        assert store.stats.size <= 16
