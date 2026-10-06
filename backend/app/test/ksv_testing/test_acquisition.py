"""The fetch -> chunk -> embed -> verify loop.

No network. The fetcher is injected, so these run the real loop — real chunking,
real cosine scoring, real banding — over documents the test supplies.
"""

import numpy as np
import pytest

from knowledge_sufficiency.acquisition import (
    Acquisition,
    KnowledgeAcquisition,
    split_into_chunks,
)
from knowledge_sufficiency.fetching import Document
from knowledge_sufficiency.ksv_exceptions import FetchFailed, UntrustedSource
from knowledge_sufficiency.ksv_manager import KSVManager
from knowledge_sufficiency.target import AcquisitionTarget
from knowledge_sufficiency.verdict import Sufficiency

DIMENSIONS = 128
ANSWER = "the answer"


def unit(seed: int) -> np.ndarray:
    """A random unit vector that is genuinely unrelated to the others.

    standard_normal, not random(): uniform [0, 1) values put every vector in the
    positive orthant, where two unrelated ones score about 0.75 against each
    other — above the sufficiency floor. Tests built on those would show
    knowledge arriving that never did.
    """
    v = np.random.default_rng(seed).standard_normal(DIMENSIONS).astype(np.float32)
    return v / np.linalg.norm(v)


QUERY = unit(1)
ELSEWHERE = unit(2)


def target_for(query_vector=QUERY, topic_vector=None, subtopics=()):
    return AcquisitionTarget(
        topic="anything", query="a question",
        topic_vector=topic_vector if topic_vector is not None else unit(5),
        query_vector=query_vector,
        subtopics=tuple(name for name, _ in subtopics),
        subtopic_vectors=tuple(v for _, v in subtopics),
    )


def embed(texts):
    """A chunk containing ANSWER embeds onto the query; anything else does not."""
    return [
        (f"c{i}", QUERY.copy() if ANSWER in text else ELSEWHERE.copy())
        for i, text in enumerate(texts)
    ]


def page(text: str):
    return lambda url: Document(url=url, text=text, content_type="text/plain")


def pages(mapping):
    def fetch(url):
        if url not in mapping:
            raise FetchFailed(url, "not in this test's fixture")
        value = mapping[url]
        if isinstance(value, Exception):
            raise value
        return Document(url=url, text=value, content_type="text/plain")
    return fetch


@pytest.fixture
def verify():
    return KSVManager().sufficiency_verification


def build(verify, **kwargs):
    """The loop with the relevance gate open.

    `topic_floor=-1.0` admits every chunk, because these tests are about the
    loop — rounds, accumulation, failures, storage. The gate is covered in
    test_relevance.py, where a floor that admits nothing would hide the point.
    """
    kwargs.setdefault("topic_floor", -1.0)
    return KnowledgeAcquisition(verify=verify, embed=embed, **kwargs)


class TestItStopsWhenTheAnswerIsFound:
    def test_one_round_is_enough(self, verify):
        acq = build(verify, fetch=page(f"something {ANSWER} here"))
        result = acq.acquire_until_sufficient(target_for(), ["https://a/1", "https://a/2"])

        assert result.verdict.sufficiency is Sufficiency.FULL
        assert result.rounds == 1

    def test_it_keeps_going_while_the_answer_is_missing(self, verify):
        acq = build(verify, fetch=pages({
            "https://a/1": "nothing useful",
            "https://a/2": "still nothing",
            "https://a/3": f"here it is: {ANSWER}",
        }))
        result = acq.acquire_until_sufficient(
            target_for(), ["https://a/1", "https://a/2", "https://a/3"]
        )

        assert result.verdict.sufficiency is Sufficiency.FULL
        assert result.rounds == 3

    def test_it_gives_up_after_max_rounds(self, verify):
        acq = build(verify, fetch=page("unrelated material"), max_rounds=2)
        result = acq.acquire_until_sufficient(
            target_for(), ["https://a/1", "https://a/2", "https://a/3", "https://a/4"]
        )

        assert result.rounds == 2
        assert not result.verdict.is_sufficient

    def test_max_rounds_below_one_is_refused(self, verify):
        with pytest.raises(ValueError, match="at least 1"):
            build(verify, max_rounds=0)


class TestKnowledgeAccumulates:
    def test_what_earlier_rounds_found_is_kept(self, verify):
        """Two sources that each fall short may answer the query together."""
        acq = build(verify, fetch=pages({
            "https://a/1": "partial material",
            "https://a/2": f"the rest, including {ANSWER}",
        }))
        result = acq.acquire_until_sufficient(target_for(), ["https://a/1", "https://a/2"])

        assert len(result.acquired) > 1
        assert result.verdict.is_sufficient

    def test_the_verdict_is_over_everything_gathered(self, verify):
        acq = build(verify, fetch=pages({
            "https://a/1": "nothing",
            "https://a/2": f"{ANSWER}",
        }))
        result = acq.acquire_until_sufficient(target_for(), ["https://a/1", "https://a/2"])

        assert len(result.verdict.scores) == len(result.acquired)


class TestABadSourceDoesNotEndTheAttempt:
    def test_an_untrusted_url_is_skipped(self, verify):
        acq = build(verify, fetch=pages({
            "https://bad/1": UntrustedSource("https://bad/1", "not listed"),
            "https://a/2": f"{ANSWER}",
        }))
        result = acq.acquire_until_sufficient(target_for(), ["https://bad/1", "https://a/2"])

        assert result.verdict.is_sufficient
        assert result.failures[0][0] == "https://bad/1"

    def test_a_failed_fetch_is_skipped(self, verify):
        acq = build(verify, fetch=pages({
            "https://a/1": FetchFailed("https://a/1", "HTTP 503"),
            "https://a/2": f"{ANSWER}",
        }))
        result = acq.acquire_until_sufficient(target_for(), ["https://a/1", "https://a/2"])

        assert result.verdict.is_sufficient
        assert len(result.failures) == 1

    def test_every_source_failing_leaves_nothing(self, verify):
        acq = build(verify, fetch=pages({}))
        result = acq.acquire_until_sufficient(target_for(), ["https://a/1", "https://a/2"])

        assert result.acquired == []
        assert result.verdict.sufficiency is Sufficiency.NONE
        assert len(result.failures) == 2

    def test_an_empty_document_is_not_an_error(self, verify):
        acq = build(verify, fetch=pages({
            "https://a/1": "   ",
            "https://a/2": f"{ANSWER}",
        }))
        result = acq.acquire_until_sufficient(target_for(), ["https://a/1", "https://a/2"])

        assert result.verdict.is_sufficient
        assert result.failures == []


class TestWhatGetsStored:
    def test_knowledge_that_answered_the_query_is_stored(self, verify):
        stored = []
        acq = build(verify, fetch=page(f"{ANSWER}"), store=stored.extend)
        result = acq.acquire_until_sufficient(target_for(), ["https://a/1"])

        assert result.stored
        assert len(stored) == len(result.acquired)

    def test_knowledge_that_did_not_is_discarded(self, verify):
        """Otherwise every miss grows the index with material that answered nothing."""
        stored = []
        acq = build(verify, fetch=page("unrelated material"), store=stored.extend)
        result = acq.acquire_until_sufficient(target_for(), ["https://a/1"])

        assert not result.stored
        assert stored == []

    def test_partial_can_be_kept_deliberately(self, verify):
        stored = []
        middling = unit(1) * 0.5 + unit(3) * 0.5

        def half(texts):
            return [("c0", (middling / np.linalg.norm(middling)).astype(np.float32))]

        acq = KnowledgeAcquisition(
            verify=verify, embed=half, fetch=page("some material"),
            store=stored.extend, store_on_partial=True, max_rounds=1,
            topic_floor=-1.0,
        )
        result = acq.acquire_until_sufficient(target_for(), ["https://a/1"])

        if result.verdict.sufficiency is Sufficiency.PARTIAL:
            assert result.stored

    def test_nothing_is_stored_without_a_store(self, verify):
        acq = build(verify, fetch=page(f"{ANSWER}"))
        assert not acq.acquire_until_sufficient(target_for(), ["https://a/1"]).stored


class TestChunking:
    def test_it_reuses_the_project_windowing(self):
        chunks = split_into_chunks("word " * 400, size=256, overlap=20)
        assert all(len(c) <= 256 for c in chunks)

    def test_blank_chunks_are_dropped(self):
        assert split_into_chunks("   \n\n   ") == []

    def test_a_short_document_is_one_chunk(self):
        assert len(split_into_chunks("a short line")) == 1


class TestTheManagerApi:
    def test_acquire_runs_the_loop(self):
        ksv = KSVManager()
        result = ksv.acquire(
            target_for(), embed=embed, urls=["https://a/1"],
            fetch=page(f"{ANSWER}"), topic_floor=-1.0,
        )

        assert isinstance(result, Acquisition)
        assert result.verdict.is_sufficient

    def test_it_verifies_with_this_manager_s_floors(self):
        """A stricter manager should refuse what a default one accepts."""
        strict = KSVManager(sufficient_floor=0.999, partial_floor=0.998)
        result = strict.acquire(
            target_for(), embed=lambda t: [("c0", unit(7))],
            urls=["https://a/1"], fetch=page("material"), topic_floor=-1.0,
        )
        assert not result.verdict.is_sufficient


class TestTheLoopAppliesTheGate:
    """The rest of this file opens the gate to test the loop. These close it.

    Removing the gate from the loop entirely passed every other test in this
    file, because they all set topic_floor=-1.0, and test_relevance.py exercises
    the filter directly rather than through an acquisition.
    """

    def on_topic_embedder(self, topic_vector):
        """Chunks mentioning the topic embed onto it; the rest land elsewhere."""
        def embed_(texts):
            return [
                (f"c{i}", topic_vector.copy() if "ontopic" in t else unit(50 + i))
                for i, t in enumerate(texts)
            ]
        return embed_

    def test_off_topic_chunks_never_reach_the_store(self, verify):
        topic = unit(11)
        stored = []
        acq = KnowledgeAcquisition(
            verify=verify, embed=self.on_topic_embedder(topic),
            fetch=page("a page about something else entirely"),
            store=stored.extend, max_rounds=1,
        )
        result = acq.acquire_until_sufficient(
            target_for(query_vector=topic, topic_vector=topic), ["https://a/1"]
        )

        assert result.acquired == []
        assert result.discarded > 0
        assert stored == []

    def test_a_mixed_page_contributes_only_its_relevant_part(self, verify):
        topic = unit(12)
        acq = KnowledgeAcquisition(
            verify=verify, embed=self.on_topic_embedder(topic),
            fetch=page("ontopic " + ("filler " * 200)), max_rounds=1,
        )
        result = acq.acquire_until_sufficient(
            target_for(query_vector=topic, topic_vector=topic), ["https://a/1"]
        )

        assert result.acquired
        assert result.discarded > 0

    def test_the_discarded_count_is_reported(self, verify):
        topic = unit(13)
        acq = KnowledgeAcquisition(
            verify=verify, embed=self.on_topic_embedder(topic),
            fetch=page("nothing relevant " * 100), max_rounds=1,
        )
        result = acq.acquire_until_sufficient(
            target_for(query_vector=topic, topic_vector=topic), ["https://a/1"]
        )
        assert result.discarded == len(split_into_chunks("nothing relevant " * 100))

    def test_the_cap_bounds_what_one_acquisition_keeps(self, verify):
        topic = unit(14)
        acq = KnowledgeAcquisition(
            verify=verify, embed=lambda texts: [(f"c{i}", topic.copy())
                                                for i, _ in enumerate(texts)],
            fetch=page("ontopic " * 500), max_rounds=1, max_kept=5,
        )
        result = acq.acquire_until_sufficient(
            target_for(query_vector=topic, topic_vector=topic), ["https://a/1"]
        )
        assert len(result.acquired) <= 5

    def test_max_kept_below_one_is_refused(self, verify):
        with pytest.raises(ValueError, match="max_kept"):
            KnowledgeAcquisition(verify=verify, embed=embed, max_kept=0)


class TestTheLoopUsesSourceSelection:
    """Discovery must go through the index when one is supplied.

    Skipping selection entirely passed every other test here, because they all
    pass `urls` and never reach discovery, and test_selection.py exercises the
    index directly. This is the second gap of exactly that shape in this
    subsystem — the unit is covered, the seam is not.
    """

    def index_over(self, *pairs):
        from knowledge_sufficiency.selection import SourceIndex
        from knowledge_sufficiency.sources import TrustedSource

        vectors = {name: vec for name, vec in pairs}
        sources = tuple(
            TrustedSource(name, f"{name}.example", name,
                          f"https://{name}.example/s?q={{terms}}")
            for name, _ in pairs
        )
        return SourceIndex(lambda text: vectors[text], sources), sources

    def test_only_the_selected_source_is_searched(self, verify):
        wanted, other = unit(21), unit(22)
        index, _ = self.index_over(("wanted", wanted), ("other", other))
        asked = []

        def fetch(url):
            asked.append(url)
            return Document(url=url, text="", content_type="text/html")

        acq = KnowledgeAcquisition(
            verify=verify, embed=embed, fetch=fetch, source_index=index,
            topic_floor=-1.0,
        )
        acq.acquire_until_sufficient(
            target_for(query_vector=wanted, topic_vector=wanted)
        )

        assert asked, "discovery never ran"
        assert all("wanted.example" in url for url in asked), asked

    def test_without_an_index_every_searchable_source_is_asked(self, verify):
        _, sources = self.index_over(("a", unit(31)), ("b", unit(32)))
        asked = []

        def fetch(url):
            asked.append(url)
            return Document(url=url, text="", content_type="text/html")

        acq = KnowledgeAcquisition(
            verify=verify, embed=embed, fetch=fetch, sources=sources,
            topic_floor=-1.0,
        )
        acq.acquire_until_sufficient(target_for())

        assert len(asked) == 2

    def test_supplying_urls_skips_discovery_altogether(self, verify):
        index, _ = self.index_over(("wanted", unit(41)))
        asked = []

        def fetch(url):
            asked.append(url)
            return Document(url=url, text=f"{ANSWER}", content_type="text/plain")

        acq = KnowledgeAcquisition(
            verify=verify, embed=embed, fetch=fetch, source_index=index,
            topic_floor=-1.0,
        )
        acq.acquire_until_sufficient(target_for(), ["https://given/1"])

        assert asked == ["https://given/1"]


class TestTheLoopRecordsWhatItAcquired:
    """Recording is tied to storage, not to fetching.

    The table says what the system took knowledge from. A document that was
    fetched and then discarded as off-subject, or kept but never stored because
    it did not answer the query, was never taken from.
    """

    def store_for(self, tmp_path):
        from knowledge_sufficiency.acquisition_store import AcquisitionStore
        return AcquisitionStore(tmp_path / "acquired.sql")

    def test_a_successful_acquisition_is_recorded(self, verify, tmp_path):
        store = self.store_for(tmp_path)
        acq = build(verify, fetch=page(f"{ANSWER}"), store=lambda c: None,
                    acquisition_store=store)
        result = acq.acquire_until_sufficient(target_for(), ["https://a/1"])

        assert result.recorded
        assert [r.url for r in store.all_records()] == ["https://a/1"]
        store.close()

    def test_the_topic_and_query_are_recorded_with_it(self, verify, tmp_path):
        store = self.store_for(tmp_path)
        acq = build(verify, fetch=page(f"{ANSWER}"), store=lambda c: None,
                    acquisition_store=store)
        acq.acquire_until_sufficient(target_for(), ["https://a/1"])

        record = store.all_records()[0]
        assert record.topic == "anything"
        assert record.query == "a question"
        store.close()

    def test_an_unsuccessful_acquisition_records_nothing(self, verify, tmp_path):
        """Nothing was stored, so nothing was taken from anywhere."""
        store = self.store_for(tmp_path)
        acq = build(verify, fetch=page("unrelated material"),
                    store=lambda c: None, acquisition_store=store)
        result = acq.acquire_until_sufficient(target_for(), ["https://a/1"])

        assert not result.stored
        assert result.recorded == []
        assert store.all_records() == []
        store.close()

    def test_a_document_that_contributed_nothing_is_not_recorded(
        self, verify, tmp_path
    ):
        """It was fetched, but the gate discarded all of it."""
        topic = unit(61)
        store = self.store_for(tmp_path)

        def embed_(texts):
            return [
                (f"c{i}", topic.copy() if "ontopic" in t else unit(70 + i))
                for i, t in enumerate(texts)
            ]

        acq = KnowledgeAcquisition(
            verify=verify, embed=embed_, store=lambda c: None,
            acquisition_store=store, max_rounds=2,
            fetch=pages({
                "https://off/1": "entirely about something else",
                "https://on/2": "ontopic",
            }),
        )
        acq.acquire_until_sufficient(
            target_for(query_vector=topic, topic_vector=topic),
            ["https://off/1", "https://on/2"],
        )

        assert [r.url for r in store.all_records()] == ["https://on/2"]
        store.close()

    def test_every_contributing_document_is_recorded(self, verify, tmp_path):
        store = self.store_for(tmp_path)
        acq = build(verify, store=lambda c: None, acquisition_store=store,
                    max_rounds=2, fetch=pages({
                        "https://a/1": "material",
                        "https://a/2": f"{ANSWER}",
                    }))
        acq.acquire_until_sufficient(target_for(), ["https://a/1", "https://a/2"])

        assert len(store.all_records()) == 2
        store.close()

    def test_no_store_means_no_recording_and_no_error(self, verify):
        acq = build(verify, fetch=page(f"{ANSWER}"), store=lambda c: None)
        assert acq.acquire_until_sufficient(target_for(), ["https://a/1"]).recorded == []
