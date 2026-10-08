"""Choosing a spread of passages instead of the same one several times."""

import numpy as np
import pytest

from retrieval_layer.diversity import maximal_marginal_relevance
from retrieval_layer.models import Passage, QueryPlan, VECTOR

DIMENSIONS = 128


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(DIMENSIONS).astype(np.float32)
    return v / np.linalg.norm(v)


QUERY = unit(1)


def _basis():
    """Two directions orthogonal to the query, to build vectors at known angles."""
    e1 = unit(2) - QUERY * float(unit(2) @ QUERY)
    e1 /= np.linalg.norm(e1)
    e2 = unit(3) - QUERY * float(unit(3) @ QUERY) - e1 * float(unit(3) @ e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


def at(to_query: float, along_e1: float, along_e2: float = 0.0) -> np.ndarray:
    """A unit vector with the given components, so angles are exact not hoped for."""
    e1, e2 = _basis()
    v = QUERY * to_query + e1 * along_e1 + e2 * along_e2
    return (v / np.linalg.norm(v)).astype(np.float32)


def passage(i, vector=None, score=1.0, text=None):
    return Passage(i, f"c{i}", text or f"text {i}", score, VECTOR,
                   embedding=vector)


def plan():
    return QueryPlan("a query", QUERY, "a query")


class TestItAvoidsRepetition:
    def test_a_near_duplicate_is_passed_over(self):
        """Overlapping chunks put the same sentence in the ranking twice.

        The alternative has to be genuinely relevant for the trade-off to mean
        anything: MMR weighs relevance against redundancy, so a near-duplicate
        that answers the query beats a diverse passage that does not, and
        should. Here the top passage is relevant without being the query
        itself, the second nearly repeats it, and the third is a little less
        relevant but says something else.
        """
        best = at(0.80, 0.60)
        near_duplicate = at(0.78, 0.62)
        elsewhere = at(0.70, -0.60, 0.39)

        assert float(near_duplicate @ best) > 0.98, "should nearly repeat it"
        assert float(elsewhere @ best) < 0.3, "should say something else"
        assert float(elsewhere @ QUERY) > 0.6, "should still be relevant"

        chosen = maximal_marginal_relevance(
            plan(),
            [passage(1, best), passage(2, near_duplicate), passage(3, elsewhere)],
            k=2,
        )
        assert [p.vector_id for p in chosen] == [1, 3]

    def test_the_most_relevant_is_always_first(self):
        """Diversity is a tie-breaker, not a reason to drop the best answer."""
        chosen = maximal_marginal_relevance(
            plan(), [passage(1, unit(9)), passage(2, QUERY)], k=1
        )
        assert chosen[0].vector_id == 2

    def test_lambda_one_is_pure_relevance(self):
        near = (QUERY * 0.98 + unit(2) * 0.02)
        near /= np.linalg.norm(near)
        chosen = maximal_marginal_relevance(
            plan(),
            [passage(1, QUERY), passage(2, near.astype(np.float32)),
             passage(3, unit(5))],
            k=2, lambda_=1.0,
        )
        assert [p.vector_id for p in chosen] == [1, 2]

    def test_lambda_zero_is_pure_diversity(self):
        near = (QUERY * 0.98 + unit(2) * 0.02)
        near /= np.linalg.norm(near)
        chosen = maximal_marginal_relevance(
            plan(),
            [passage(1, QUERY), passage(2, near.astype(np.float32)),
             passage(3, unit(5))],
            k=2, lambda_=0.0,
        )
        assert [p.vector_id for p in chosen] == [1, 3]


class TestHowMany:
    def test_it_returns_exactly_k(self):
        passages = [passage(i, unit(i)) for i in range(1, 6)]
        assert len(maximal_marginal_relevance(plan(), passages, k=3)) == 3

    def test_fewer_passages_than_k_are_all_returned(self):
        passages = [passage(1, unit(1)), passage(2, unit(2))]
        assert len(maximal_marginal_relevance(plan(), passages, k=10)) == 2

    @pytest.mark.parametrize("k", [0, -1])
    def test_asking_for_none_returns_none(self, k):
        assert maximal_marginal_relevance(plan(), [passage(1, unit(1))], k) == []

    def test_no_passages_at_all(self):
        assert maximal_marginal_relevance(plan(), [], k=3) == []


class TestWhereTheEmbeddingsComeFrom:
    def test_passage_embeddings_are_used_when_present(self):
        passages = [passage(i, unit(i)) for i in range(1, 5)]
        assert len(maximal_marginal_relevance(plan(), passages, k=2)) == 2

    def test_an_embedder_is_called_when_they_are_absent(self):
        asked = []

        def embed(texts):
            asked.extend(texts)
            return [unit(i) for i in range(len(texts))]

        passages = [passage(i) for i in range(1, 5)]
        maximal_marginal_relevance(plan(), passages, k=2, embed=embed)
        assert len(asked) == 4

    def test_without_either_it_keeps_the_ranking_it_was_given(self):
        """Repetition cannot be told from coverage without something to compare."""
        passages = [passage(i, score=1.0 / i) for i in range(1, 6)]
        chosen = maximal_marginal_relevance(plan(), passages, k=3)
        assert [p.vector_id for p in chosen] == [1, 2, 3]


class TestRelevanceFromTheStageBefore:
    """MMR runs after the reranker. If it re-derived relevance from cosine it
    would replace the cross-encoder's judgement with the bi-encoder's, and the
    most expensive stage in the pipeline would change nothing."""

    def scenario(self):
        answer = passage(1, at(0.55, 0.835))
        lexical = passage(2, at(0.90, -0.436))
        filler = passage(3, at(0.40, 0.0, 0.917))
        return [answer, lexical, filler]

    def test_without_it_the_closest_embedding_leads(self):
        picked = maximal_marginal_relevance(plan(), self.scenario(), k=1)
        assert picked[0].vector_id == 2

    def test_with_it_the_previous_judgement_leads(self):
        """The cross-encoder said passage 1; MMR must not overrule it."""
        picked = maximal_marginal_relevance(
            plan(), self.scenario(), k=1, relevance=[9.0, -2.0, -5.0]
        )
        assert picked[0].vector_id == 1

    def test_scores_on_any_scale_are_usable(self):
        """Logits, bm25 and RRF all differ in range; only their order matters."""
        small = maximal_marginal_relevance(
            plan(), self.scenario(), k=3, relevance=[0.033, 0.016, 0.015]
        )
        large = maximal_marginal_relevance(
            plan(), self.scenario(), k=3, relevance=[330.0, 160.0, 150.0]
        )
        assert [p.vector_id for p in small] == [p.vector_id for p in large]

    def test_relevance_still_trades_against_repetition(self):
        """A near-copy of the leader loses to a less relevant but new passage."""
        best = passage(1, at(0.80, 0.60))
        copy = passage(2, at(0.80, 0.599, 0.035))
        other = passage(3, at(0.80, -0.60))
        weak = passage(4, at(0.10, 0.0, 0.995))
        picked = maximal_marginal_relevance(
            plan(), [best, copy, other, weak], k=2,
            relevance=[1.0, 0.95, 0.80, 0.0],
        )
        assert [p.vector_id for p in picked] == [1, 3]

    def test_relevance_is_relative_to_the_pool(self):
        """Min-max: the weakest candidate present scores 0 whatever its raw
        score, which is why MMR is given the whole candidate pool and not a
        shortlist — in a pool of three, a good passage can be the floor."""
        best = passage(1, at(0.80, 0.60))
        copy = passage(2, at(0.80, 0.599, 0.035))
        other = passage(3, at(0.80, -0.60))
        picked = maximal_marginal_relevance(
            plan(), [best, copy, other], k=2, relevance=[1.0, 0.95, 0.80]
        )
        assert [p.vector_id for p in picked] == [1, 2]

    def test_equal_scores_do_not_divide_by_zero(self):
        picked = maximal_marginal_relevance(
            plan(), self.scenario(), k=2, relevance=[0.5, 0.5, 0.5]
        )
        assert len(picked) == 2
