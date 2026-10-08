"""Combining two ranked lists whose scores are not comparable."""

import pytest

from retrieval_layer.fusion import reciprocal_rank_fusion, sources_of
from retrieval_layer.models import FUSED, KEYWORD, ScoredId, VECTOR


def ranked(ids, source):
    return [ScoredId(i, 0.0, source, n + 1) for n, i in enumerate(ids)]


class TestFusing:
    def test_agreement_between_searchers_wins(self):
        """The argument for consulting two: what both found is likelier right."""
        dense = ranked([1, 2, 3], VECTOR)
        lexical = ranked([3, 1], KEYWORD)
        fused = [f.vector_id for f in reciprocal_rank_fusion([dense, lexical])]
        assert fused[:2] == [1, 3]
        assert fused[-1] == 2

    def test_a_candidate_in_one_list_still_appears(self):
        dense = ranked([1], VECTOR)
        lexical = ranked([2], KEYWORD)
        assert len(reciprocal_rank_fusion([dense, lexical])) == 2

    def test_the_order_within_a_single_list_is_kept(self):
        fused = reciprocal_rank_fusion([ranked([5, 6, 7], VECTOR)])
        assert [f.vector_id for f in fused] == [5, 6, 7]

    def test_results_are_marked_fused(self):
        fused = reciprocal_rank_fusion([ranked([1], VECTOR)])
        assert fused[0].source == FUSED

    def test_ranks_are_reassigned_from_one(self):
        fused = reciprocal_rank_fusion([ranked([9, 8], VECTOR)])
        assert [f.rank for f in fused] == [1, 2]

    def test_nothing_fuses_to_nothing(self):
        assert reciprocal_rank_fusion([[], []]) == []

    def test_no_lists_at_all(self):
        assert reciprocal_rank_fusion([]) == []


class TestItUsesRankNotScore:
    def test_the_incoming_scores_are_ignored(self):
        """A bm25 value and a cosine distance are not on one scale.

        The lexical list here carries enormous scores and the dense list tiny
        ones; if either leaked into the fusion the result would be decided by
        whichever searcher happened to use bigger numbers.
        """
        dense = [ScoredId(1, 0.0001, VECTOR, 1), ScoredId(2, 0.0002, VECTOR, 2)]
        lexical = [ScoredId(2, -9999.0, KEYWORD, 1), ScoredId(1, -1.0, KEYWORD, 2)]
        fused = reciprocal_rank_fusion([dense, lexical])
        assert {f.vector_id for f in fused} == {1, 2}
        assert fused[0].score == pytest.approx(fused[1].score)

    def test_a_tie_in_rank_is_a_tie_in_score(self):
        dense = ranked([1, 2], VECTOR)
        lexical = ranked([2, 1], KEYWORD)
        fused = reciprocal_rank_fusion([dense, lexical])
        assert fused[0].score == pytest.approx(fused[1].score)


class TestTheDampingConstant:
    def test_a_small_k_lets_one_first_place_dominate(self):
        """With k tiny, 1/(k+1) dwarfs everything below it."""
        dense = ranked([1], VECTOR)
        lexical = ranked([2, 2, 2], KEYWORD)
        fused = reciprocal_rank_fusion([dense, ranked([9, 8, 2], KEYWORD)], rrf_k=1)
        assert fused[0].vector_id in (1, 9)

    def test_a_larger_k_flattens_the_contributions(self):
        dense = ranked([1], VECTOR)
        lexical = ranked([2], KEYWORD)
        close = reciprocal_rank_fusion([dense, lexical], rrf_k=1000)
        assert close[0].score == pytest.approx(close[1].score, rel=1e-3)


class TestExplaining:
    def test_it_reports_which_searchers_found_each(self):
        found = sources_of([ranked([1, 2], VECTOR), ranked([2], KEYWORD)])
        assert found[2] == {VECTOR, KEYWORD}
        assert found[1] == {VECTOR}
