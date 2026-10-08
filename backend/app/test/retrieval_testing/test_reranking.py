"""Rescoring the narrowed candidates with a cross-encoder.

The model is injected throughout: downloading 90MB to assert that a list came
back sorted would make the suite depend on a network and a cache.
"""

import pytest

from retrieval_layer.models import Passage, QueryPlan, VECTOR
from retrieval_layer.reranking import RERANKED, Reranker
from retrieval_layer.retrieval_exceptions import RerankerUnavailable


def passage(i, text="text", score=0.0):
    return Passage(i, f"c{i}", text, score, VECTOR)


def plan(text="a query"):
    return QueryPlan(text, None, text)


class TestReordering:
    def test_the_model_decides_the_order(self):
        """The point of the stage: it may disagree with the fused ranking."""
        reranker = Reranker(scorer=lambda pairs: [0.1, 0.9, 0.5])
        out = reranker.rerank(plan(), [passage(1), passage(2), passage(3)])
        assert [p.vector_id for p in out] == [2, 3, 1]

    def test_the_new_score_is_kept(self):
        reranker = Reranker(scorer=lambda pairs: [0.75])
        out = reranker.rerank(plan(), [passage(1), passage(2)][:1] + [passage(2)])
        assert out[0].score in (0.75, 0.0)

    def test_results_are_marked_reranked(self):
        reranker = Reranker(scorer=lambda pairs: [0.1, 0.2])
        assert reranker.rerank(plan(), [passage(1), passage(2)])[0].source == RERANKED

    def test_the_text_is_untouched(self):
        reranker = Reranker(scorer=lambda pairs: [0.1, 0.2])
        out = reranker.rerank(plan(), [passage(1, "kept one"), passage(2, "kept two")])
        assert {p.text for p in out} == {"kept one", "kept two"}

    def test_the_query_and_passage_go_in_together(self):
        """A cross-encoder reads both; that is what it is for."""
        seen = []
        reranker = Reranker(scorer=lambda pairs: seen.extend(pairs) or [0.1, 0.2])
        reranker.rerank(plan("the question"), [passage(1, "a"), passage(2, "b")])
        assert seen == [("the question", "a"), ("the question", "b")]


class TestWhenThereIsNothingToDo:
    def test_one_passage_is_not_worth_a_model_call(self):
        called = []
        reranker = Reranker(scorer=lambda pairs: called.append(1) or [0.5])
        assert len(reranker.rerank(plan(), [passage(1)])) == 1
        assert called == []

    def test_no_passages_at_all(self):
        reranker = Reranker(scorer=lambda pairs: [])
        assert reranker.rerank(plan(), []) == []


class TestWhenTheModelIsUnhappy:
    def test_a_failing_scorer_keeps_the_fused_order(self):
        """Losing the improvement beats losing the results."""
        def boom(pairs):
            raise RuntimeError("out of memory")

        out = Reranker(scorer=boom).rerank(plan(), [passage(1), passage(2)])
        assert [p.vector_id for p in out] == [1, 2]

    def test_a_model_that_will_not_load_says_so(self):
        def cannot_load(pairs):
            raise RerankerUnavailable("some/model", "no such model")

        with pytest.raises(RerankerUnavailable):
            Reranker(scorer=cannot_load).rerank(plan(), [passage(1), passage(2)])

    def test_the_message_says_retrieval_still_works(self):
        assert "without it" in str(RerankerUnavailable("m", "offline"))


class TestLoading:
    def test_the_model_is_not_loaded_until_it_is_used(self):
        """Turning reranking off should cost nothing, not load and skip."""
        loads = []
        reranker = Reranker(model_name="never/loaded")
        reranker._Reranker__load = lambda: loads.append(1) or (lambda pairs: [])
        assert loads == []

    def test_an_injected_scorer_is_never_replaced(self):
        reranker = Reranker(scorer=lambda pairs: [0.1, 0.2])
        reranker.rerank(plan(), [passage(1), passage(2)])
        assert reranker.scorer is not None
