"""The types every retrieval stage passes along, and the knobs it reads.

Small, but load-bearing: fusion depends on `rank` meaning position, the token
budget depends on `tokens()` agreeing with the memory layer's estimate, and the
settings guard the values that would otherwise fail deep inside a stage.
"""

import numpy as np
import pytest

from retrieval_layer.models import (
    FUSED,
    KEYWORD,
    Passage,
    QueryPlan,
    RetrievalRequest,
    RetrievalResult,
    ScoredId,
    StageTiming,
    VECTOR,
    as_ranked,
    rank_of,
)
from retrieval_layer.retrieval_exceptions import (
    EmptyQuery,
    IndexUnavailable,
    InvalidRetrievalSetting,
    RerankerUnavailable,
)
from retrieval_layer.settings import (
    CHARS_PER_TOKEN,
    RetrievalSettings,
)


def passage(text="some text", score=1.0, vector_id=1):
    return Passage(vector_id=vector_id, chunk_id=f"c{vector_id}", text=text,
                   score=score, source=VECTOR)


class TestTheRequest:
    def test_unset_fields_take_the_settings(self):
        settings = RetrievalSettings(top_k=5, rerank=False)
        filled = RetrievalRequest("a query").with_defaults(settings)
        assert filled.top_k == 5
        assert filled.rerank is False

    def test_what_the_caller_set_is_kept(self):
        settings = RetrievalSettings(top_k=5, rerank=False)
        filled = RetrievalRequest("q", top_k=2, rerank=True).with_defaults(settings)
        assert (filled.top_k, filled.rerank) == (2, True)

    def test_rerank_false_is_not_mistaken_for_unset(self):
        """`if self.rerank` would read False as absent and turn it back on."""
        settings = RetrievalSettings(rerank=True)
        assert RetrievalRequest("q", rerank=False).with_defaults(settings).rerank is False

    def test_top_k_zero_is_not_mistaken_for_unset(self):
        settings = RetrievalSettings(top_k=8)
        assert RetrievalRequest("q", top_k=0).with_defaults(settings).top_k == 0


class TestRanking:
    def test_scores_become_positions(self):
        ranked = as_ranked([(1, 0.2), (2, 0.9), (3, 0.5)], VECTOR)
        assert [r.vector_id for r in ranked] == [2, 3, 1]
        assert [r.rank for r in ranked] == [1, 2, 3]

    def test_ranks_start_at_one(self):
        """RRF divides by (k + rank); a rank of 0 would weight differently."""
        assert as_ranked([(1, 0.5)], VECTOR)[0].rank == 1

    def test_a_distance_ranks_the_other_way(self):
        """DiskANN returns distance, where smaller is closer."""
        ranked = as_ranked([(1, 0.1), (2, 0.9)], VECTOR, descending=False)
        assert [r.vector_id for r in ranked] == [1, 2]

    def test_the_source_is_carried(self):
        assert as_ranked([(1, 0.5)], KEYWORD)[0].source == KEYWORD

    def test_nothing_ranks_to_nothing(self):
        assert as_ranked([], VECTOR) == []

    def test_rank_of_maps_id_to_position(self):
        ranked = as_ranked([(1, 0.2), (2, 0.9)], VECTOR)
        assert rank_of(ranked) == {2: 1, 1: 2}

    def test_ids_come_back_as_ints(self):
        """They arrive from numpy as uint32 and index dictionaries later."""
        ranked = as_ranked([(np.uint32(7), np.float32(0.5))], VECTOR)
        assert isinstance(ranked[0].vector_id, int)
        assert isinstance(ranked[0].score, float)


class TestThePassage:
    def test_tokens_use_the_shared_estimate(self):
        """Not a second estimate: one corpus measured two ways budgets wrongly."""
        assert passage("a" * 40).tokens(CHARS_PER_TOKEN) == 10

    def test_rescoring_keeps_everything_else(self):
        original = passage(text="kept", vector_id=3)
        again = original.rescored(0.1, FUSED)
        assert (again.text, again.vector_id) == ("kept", 3)
        assert (again.score, again.source) == (0.1, FUSED)

    def test_rescoring_does_not_mutate_the_original(self):
        original = passage(score=1.0)
        original.rescored(0.5)
        assert original.score == 1.0

    def test_rescoring_may_keep_the_source(self):
        assert passage().rescored(0.5).source == VECTOR


class TestTheResult:
    def test_the_text_joins_the_passages(self):
        result = RetrievalResult([passage(text="one"), passage(text="two")], [])
        assert result.text == "one\n\ntwo"

    def test_an_empty_result_has_empty_text(self):
        assert RetrievalResult([], []).text == ""

    def test_the_total_is_the_sum_of_the_stages(self):
        result = RetrievalResult([], [StageTiming("a", 1.5, 0),
                                      StageTiming("b", 2.5, 0)])
        assert result.total_milliseconds == 4.0

    def test_the_slowest_stage_is_named(self):
        """So a slow retrieval says which stage, instead of needing a profiler."""
        result = RetrievalResult([], [StageTiming("fast", 1.0, 0),
                                      StageTiming("slow", 9.0, 0)])
        assert result.slowest_stage().stage == "slow"

    def test_no_timings_has_no_slowest(self):
        assert RetrievalResult([], []).slowest_stage() is None


class TestTheSettings:
    def test_candidates_over_fetch(self):
        """Reranking can only reorder what it is given."""
        assert RetrievalSettings(top_k=8, candidate_multiplier=4).candidates == 32

    def test_they_are_frozen(self):
        with pytest.raises(Exception):
            RetrievalSettings().top_k = 99

    @pytest.mark.parametrize("field,value,expected", [
        ("top_k", 0, "at least 1"),
        ("candidate_multiplier", 0, "at least 1"),
        ("rrf_k", 0, "greater than 0"),
        ("rerank_batch", 0, "at least 1"),
        ("mmr_lambda", 1.5, r"\[0, 1\]"),
        ("mmr_lambda", -0.1, r"\[0, 1\]"),
        ("chars_per_token", 0, "at least 1"),
        ("token_budget", 0, "at least 1"),
        ("cache_size", -1, "0 or more"),
        ("cache_ttl_seconds", 0, "greater than 0"),
    ])
    def test_a_value_that_would_fail_later_is_refused_now(self, field, value, expected):
        with pytest.raises(InvalidRetrievalSetting, match=expected):
            RetrievalSettings(**{field: value})

    def test_a_cache_size_of_zero_is_allowed(self):
        """Zero means caching off, which is different from a nonsense value."""
        assert RetrievalSettings(cache_size=0).cache_size == 0

    def test_the_token_estimate_matches_the_memory_layer(self):
        """Two estimates over one corpus would budget it two different ways."""
        from memory.topic_pool.project_pool.conversation_pool.conversation_summary_pipeline import (
            conversation_summary,
        )
        assert CHARS_PER_TOKEN == conversation_summary._CHARS_PER_TOKEN


class TestTheExceptions:
    def test_an_empty_query_says_what_it_got(self):
        assert "''" in str(EmptyQuery(""))

    def test_a_missing_index_says_how_to_make_one(self):
        message = str(IndexUnavailable("data/chunks", "no chunk store"))
        assert "Ingest a corpus" in message

    def test_a_failed_reranker_says_retrieval_still_works(self):
        message = str(RerankerUnavailable("some/model", "offline"))
        assert "without it" in message
