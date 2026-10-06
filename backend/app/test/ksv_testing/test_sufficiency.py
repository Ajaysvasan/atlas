"""Knowledge Sufficiency Verification.

Nothing is mocked. The subsystem takes vectors and returns a band, so every test
here runs the real scoring over real arrays — which is the point of it taking
candidates rather than fetching them.
"""

import numpy as np
import pytest

from knowledge_sufficiency.ksv_manager import KSVManager
from knowledge_sufficiency.similarity import cosine_scores
from knowledge_sufficiency.thresholds import (
    PARTIAL_FLOOR,
    SUFFICIENT_FLOOR,
    classify,
)
from knowledge_sufficiency.verdict import Sufficiency, Verdict

DIMENSIONS = 128


def unit(seed: int) -> np.ndarray:
    """A random unit vector that is genuinely unrelated to the others.

    standard_normal, not random(): uniform [0, 1) values put every vector in the
    positive orthant, where two unrelated ones score about 0.75 against each
    other — above the sufficiency floor. Tests built on those would show
    knowledge arriving that never did.
    """
    v = np.random.default_rng(seed).standard_normal(DIMENSIONS).astype(np.float32)
    return v / np.linalg.norm(v)


def at_cosine(reference: np.ndarray, target: float) -> np.ndarray:
    """A unit vector whose cosine with `reference` is `target`.

    Built from the reference plus an orthogonal component, so the tests can sit
    a candidate exactly on a band edge instead of hoping a random vector lands
    near one.
    """
    orthogonal = unit(999)
    orthogonal = orthogonal - reference * float(orthogonal @ reference)
    orthogonal = orthogonal / np.linalg.norm(orthogonal)
    built = reference * target + orthogonal * np.sqrt(max(0.0, 1.0 - target**2))
    return (built / np.linalg.norm(built)).astype(np.float32)


@pytest.fixture
def ksv():
    return KSVManager()


@pytest.fixture
def query():
    return unit(1)


class TestTheBands:
    def test_a_close_match_is_full(self, ksv, query):
        verdict = ksv.sufficiency_verification(query, [("a", query.copy())])
        assert verdict.sufficiency is Sufficiency.FULL

    def test_a_middling_match_is_partial(self, ksv, query):
        middle = (SUFFICIENT_FLOOR + PARTIAL_FLOOR) / 2
        verdict = ksv.sufficiency_verification(
            query, [("a", at_cosine(query, middle))]
        )
        assert verdict.sufficiency is Sufficiency.PARTIAL

    def test_a_distant_match_is_none(self, ksv, query):
        verdict = ksv.sufficiency_verification(
            query, [("a", at_cosine(query, PARTIAL_FLOOR - 0.15))]
        )
        assert verdict.sufficiency is Sufficiency.NONE

    def test_the_best_candidate_decides_the_band(self, ksv, query):
        """One good answer is enough; a crowd of poor ones is not."""
        verdict = ksv.sufficiency_verification(query, [
            ("poor", at_cosine(query, 0.1)),
            ("good", query.copy()),
            ("poor2", at_cosine(query, 0.05)),
        ])
        assert verdict.sufficiency is Sufficiency.FULL

    def test_many_mediocre_candidates_do_not_add_up(self, ksv, query):
        middle = (SUFFICIENT_FLOOR + PARTIAL_FLOOR) / 2
        verdict = ksv.sufficiency_verification(
            query, [(f"c{i}", at_cosine(query, middle)) for i in range(20)]
        )
        assert verdict.sufficiency is Sufficiency.PARTIAL


class TestTheBandEdges:
    """The floors are inclusive, so a score sitting exactly on one is inside."""

    def test_exactly_on_the_sufficient_floor_is_full(self):
        assert classify(SUFFICIENT_FLOOR) is Sufficiency.FULL

    def test_just_below_the_sufficient_floor_is_partial(self):
        assert classify(SUFFICIENT_FLOOR - 1e-6) is Sufficiency.PARTIAL

    def test_exactly_on_the_partial_floor_is_partial(self):
        assert classify(PARTIAL_FLOOR) is Sufficiency.PARTIAL

    def test_just_below_the_partial_floor_is_none(self):
        assert classify(PARTIAL_FLOOR - 1e-6) is Sufficiency.NONE

    def test_a_negative_score_is_none(self):
        assert classify(-1.0) is Sufficiency.NONE


class TestNothingToScore:
    def test_no_candidates_is_none(self, ksv, query):
        assert ksv.sufficiency_verification(query, []).sufficiency is Sufficiency.NONE

    def test_no_candidates_supports_nothing(self, ksv, query):
        assert ksv.sufficiency_verification(query, []).supporting == []

    def test_no_candidates_scores_below_every_floor(self, ksv, query):
        """-1.0, not 0.0: zero is a real cosine and would sit inside a band."""
        assert ksv.sufficiency_verification(query, []).best == -1.0

    def test_an_iterator_is_accepted(self, ksv, query):
        """It is consumed twice internally, so it has to be realised first."""
        verdict = ksv.sufficiency_verification(
            query, iter([("a", query.copy())])
        )
        assert verdict.sufficiency is Sufficiency.FULL


class TestWhatItHandsOn:
    def test_supporting_holds_everything_above_the_partial_floor(self, ksv, query):
        """Not just the best one: a PARTIAL caller combines all of these."""
        verdict = ksv.sufficiency_verification(query, [
            ("keep_1", query.copy()),
            ("keep_2", at_cosine(query, PARTIAL_FLOOR + 0.1)),
            ("drop", at_cosine(query, PARTIAL_FLOOR - 0.1)),
        ])
        assert set(verdict.supporting) == {"keep_1", "keep_2"}

    def test_supporting_is_ordered_best_first(self, ksv, query):
        verdict = ksv.sufficiency_verification(query, [
            ("middling", at_cosine(query, PARTIAL_FLOOR + 0.05)),
            ("best", query.copy()),
            ("good", at_cosine(query, SUFFICIENT_FLOOR + 0.05)),
        ])
        assert verdict.supporting == ["best", "good", "middling"]

    def test_a_none_verdict_supports_nothing(self, ksv, query):
        verdict = ksv.sufficiency_verification(
            query, [("far", at_cosine(query, 0.05))]
        )
        assert verdict.sufficiency is Sufficiency.NONE
        assert verdict.supporting == []

    def test_the_scores_come_back_too(self, ksv, query):
        verdict = ksv.sufficiency_verification(query, [
            ("a", query.copy()), ("b", at_cosine(query, 0.2)),
        ])
        assert len(verdict.scores) == 2

    def test_ids_are_not_required_to_be_strings(self, ksv, query):
        """Callers key vectors by whatever their store uses — ints, here."""
        verdict = ksv.sufficiency_verification(query, [(4242, query.copy())])
        assert verdict.supporting == [4242]


class TestDegenerateVectors:
    def test_a_zero_candidate_scores_minus_one(self, ksv, query):
        verdict = ksv.sufficiency_verification(
            query, [("zero", np.zeros(DIMENSIONS, dtype=np.float32))]
        )
        assert verdict.best == -1.0
        assert verdict.sufficiency is Sufficiency.NONE

    def test_a_zero_candidate_does_not_poison_a_good_one(self, ksv, query):
        """A nan from the zero row would propagate through max()."""
        verdict = ksv.sufficiency_verification(query, [
            ("zero", np.zeros(DIMENSIONS, dtype=np.float32)),
            ("good", query.copy()),
        ])
        assert verdict.sufficiency is Sufficiency.FULL
        assert verdict.supporting == ["good"]

    def test_a_mismatched_dimension_is_refused(self, ksv, query):
        """Matched against our own wording, not just ValueError.

        numpy's matmul raises a ValueError containing the word "dimension" on
        its own, so a looser assertion passes whether or not the guard exists —
        and the point of the guard is the message, which names both sizes
        instead of talking about gufunc signatures.
        """
        with pytest.raises(ValueError, match=r"candidates have"):
            ksv.sufficiency_verification(
                query, [("short", np.ones(8, dtype=np.float32))]
            )

    def test_the_dimension_message_names_both_sizes(self, ksv, query):
        with pytest.raises(ValueError) as caught:
            ksv.sufficiency_verification(
                query, [("short", np.ones(8, dtype=np.float32))]
            )
        assert "128" in str(caught.value) and "8" in str(caught.value)


class TestTheFloors:
    def test_they_can_be_overridden(self, query):
        strict = KSVManager(sufficient_floor=0.99, partial_floor=0.98)
        verdict = strict.sufficiency_verification(
            query, [("a", at_cosine(query, 0.8))]
        )
        assert verdict.sufficiency is Sufficiency.NONE

    def test_a_sufficient_floor_below_the_partial_floor_is_refused(self):
        """It would leave no range in which anything could be PARTIAL."""
        with pytest.raises(ValueError, match="PARTIAL"):
            KSVManager(sufficient_floor=0.2, partial_floor=0.8)

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_a_floor_outside_the_cosine_range_is_refused(self, bad):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            KSVManager(sufficient_floor=bad)

    def test_the_defaults_are_ordered(self):
        assert PARTIAL_FLOOR <= SUFFICIENT_FLOOR

    def test_the_documented_defaults_are_the_actual_defaults(self):
        """Pinned so recalibration cannot drift away from the docs silently.

        Unlike most pinned constants this one is *expected* to change: the
        values are placeholders until measured. Changing them should fail this
        test, which is the reminder to update README.md and the API reference
        in the same commit — not a reason to leave them as they are.
        """
        assert (SUFFICIENT_FLOOR, PARTIAL_FLOOR) == (0.60, 0.35)


class TestTheVerdict:
    def test_is_sufficient_only_for_full(self, ksv, query):
        assert ksv.sufficiency_verification(
            query, [("a", query.copy())]
        ).is_sufficient

    def test_partial_still_needs_retrieval(self, ksv, query):
        middle = (SUFFICIENT_FLOOR + PARTIAL_FLOOR) / 2
        verdict = ksv.sufficiency_verification(
            query, [("a", at_cosine(query, middle))]
        )
        assert not verdict.is_sufficient
        assert verdict.needs_retrieval

    def test_none_needs_retrieval(self, ksv, query):
        assert ksv.sufficiency_verification(query, []).needs_retrieval

    def test_it_unpacks_like_a_tuple(self, ksv, query):
        sufficiency, best, supporting, scores = ksv.sufficiency_verification(
            query, [("a", query.copy())]
        )
        assert sufficiency is Sufficiency.FULL


class TestTheSharedCosine:
    """It moved here from ProjectManager, which now imports it."""

    def test_an_empty_matrix_scores_nothing(self, query):
        assert cosine_scores(query, np.empty((0, DIMENSIONS), dtype=np.float32)).size == 0

    def test_an_identical_vector_scores_one(self, query):
        scores = cosine_scores(query, np.array([query]))
        assert scores[0] == pytest.approx(1.0, abs=1e-5)

    def test_project_manager_uses_this_one(self):
        from memory.topic_pool.project_pool import project_manager
        assert project_manager.cosine_scores is cosine_scores
