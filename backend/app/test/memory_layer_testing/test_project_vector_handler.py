"""Tests for ProjectVectorHandler — a project's summary vector, by project id.

The vector store is a stateful fake rather than a MagicMock, for the reason
test_project_meta_data.py gives: what matters here is which row a call lands on,
and a mock records the call without holding the row.
"""

import numpy as np
import pytest

from config import Config
from data_layer.datalayer_exceptions.datalayer_exceptions import (
    DuplicateVectorException,
    VectorNotFoundEror,
)
from memory.memory_pool_exceptions import InvalidVectorDimension
from memory.memory_pool_exceptions import MisMatchCount
from memory.topic_pool.project_pool.project_data_repo.project_vector_handler import (
    ProjectVectorHandler,
    _derive,
    description_vector_id,
    summary_vector_id,
)

DIMENSIONS = Config.EMBEDDING_DIMENSIONS


class FakeVectorRepository:
    """Same contract and failure modes as VectorRepository, scoped to a project."""

    instances: list = []

    def __init__(self, project_id):
        self.project_id = project_id
        self.store = {}
        self.closed = False
        FakeVectorRepository.instances.append(self)

    def insert(self, vector_id, vector):
        if int(vector_id) in self.store:
            raise DuplicateVectorException(vector_id)
        self.store[int(vector_id)] = np.asarray(vector, dtype=np.float32)

    def batch_insert(self, vector_ids, vectors):
        for vector_id, vector in zip(vector_ids, vectors):
            self.store.setdefault(int(vector_id), np.asarray(vector, dtype=np.float32))

    def batch_search(self, vector_ids):
        return np.array([self.search(v) for v in vector_ids])

    def update(self, vector_id, vector):
        if int(vector_id) not in self.store:
            raise VectorNotFoundEror(vector_id)
        self.store[int(vector_id)] = np.asarray(vector, dtype=np.float32)

    def search(self, vector_id):
        if int(vector_id) not in self.store:
            raise VectorNotFoundEror(vector_id)
        return self.store[int(vector_id)]

    def delete(self, vector_id):
        self.store.pop(int(vector_id), None)

    def close(self):
        self.closed = True


def vec(fill: float) -> np.ndarray:
    return np.full(DIMENSIONS, float(fill), dtype=np.float32)


@pytest.fixture
def handler():
    FakeVectorRepository.instances = []
    h = ProjectVectorHandler(repository_factory=FakeVectorRepository)
    yield h
    h.close()


class TestSummaryVectorId:
    def test_is_deterministic(self):
        assert summary_vector_id("project_a") == summary_vector_id("project_a")

    def test_differs_between_projects(self):
        assert summary_vector_id("project_a") != summary_vector_id("project_b")

    def test_fits_the_signed_64_bit_range(self):
        """Hash-derived ids are unsigned and overflow a signed column for about
        half of all inputs unless they are masked."""
        for project in [f"project_{index}" for index in range(500)]:
            identifier = summary_vector_id(project)
            assert 0 <= identifier <= Config.VECTOR_ID_MASK

    def test_is_an_int_sqlite_and_postgres_both_accept(self):
        assert isinstance(summary_vector_id("project_a"), int)


class TestAdd:
    def test_stores_the_vector_under_the_derived_id(self, handler):
        handler.add_project_summary_vector("project_a", vec(1))
        repository = FakeVectorRepository.instances[0]
        assert list(repository.store) == [summary_vector_id("project_a")]

    def test_round_trips(self, handler):
        handler.add_project_summary_vector("project_a", vec(1))
        assert handler.get_project_summary_vector("project_a").tolist() == vec(1).tolist()

    def test_accepts_a_plain_list(self, handler):
        handler.add_project_summary_vector("project_a", [1.0] * DIMENSIONS)
        assert handler.get_project_summary_vector("project_a").tolist() == vec(1).tolist()

    def test_stores_float32(self, handler):
        handler.add_project_summary_vector("project_a", [1.0] * DIMENSIONS)
        assert handler.get_project_summary_vector("project_a").dtype == np.float32

    def test_a_second_add_is_refused(self, handler):
        """The slot is taken; replacing it is update's job, not a silent add."""
        handler.add_project_summary_vector("project_a", vec(1))
        with pytest.raises(DuplicateVectorException):
            handler.add_project_summary_vector("project_a", vec(2))

    def test_a_refused_add_leaves_the_stored_vector_alone(self, handler):
        handler.add_project_summary_vector("project_a", vec(1))
        with pytest.raises(DuplicateVectorException):
            handler.add_project_summary_vector("project_a", vec(2))
        assert handler.get_project_summary_vector("project_a").tolist() == vec(1).tolist()


class TestUpdate:
    def test_replaces_in_place(self, handler):
        handler.add_project_summary_vector("project_a", vec(1))
        handler.update_project_summary_vector("project_a", vec(2))
        assert handler.get_project_summary_vector("project_a").tolist() == vec(2).tolist()

    def test_returns_the_project_id(self, handler):
        handler.add_project_summary_vector("project_a", vec(1))
        assert handler.update_project_summary_vector("project_a", vec(2)) == "project_a"

    def test_does_not_add_a_second_row(self, handler):
        handler.add_project_summary_vector("project_a", vec(1))
        handler.update_project_summary_vector("project_a", vec(2))
        assert len(FakeVectorRepository.instances[0].store) == 1

    def test_a_project_with_no_vector_raises(self, handler):
        with pytest.raises(VectorNotFoundEror):
            handler.update_project_summary_vector("project_a", vec(1))


class TestDelete:
    def test_removes_the_vector(self, handler):
        handler.add_project_summary_vector("project_a", vec(1))
        handler.delete_project_summary_vector("project_a")
        with pytest.raises(VectorNotFoundEror):
            handler.get_project_summary_vector("project_a")

    def test_is_idempotent(self, handler):
        """The caller that wants it gone does not care whether it was there."""
        handler.delete_project_summary_vector("project_a")
        handler.delete_project_summary_vector("project_a")

    def test_add_works_again_afterwards(self, handler):
        handler.add_project_summary_vector("project_a", vec(1))
        handler.delete_project_summary_vector("project_a")
        handler.add_project_summary_vector("project_a", vec(2))
        assert handler.get_project_summary_vector("project_a").tolist() == vec(2).tolist()


class TestGet:
    def test_a_project_with_no_vector_raises(self, handler):
        with pytest.raises(VectorNotFoundEror):
            handler.get_project_summary_vector("project_a")

    def test_reads_do_not_consume(self, handler):
        handler.add_project_summary_vector("project_a", vec(1))
        first = handler.get_project_summary_vector("project_a")
        second = handler.get_project_summary_vector("project_a")
        assert first.tolist() == second.tolist()


class TestProjectScoping:
    def test_each_project_gets_its_own_repository(self, handler):
        handler.add_project_summary_vector("project_a", vec(1))
        handler.add_project_summary_vector("project_b", vec(2))
        assert [r.project_id for r in FakeVectorRepository.instances] == [
            "project_a",
            "project_b",
        ]

    def test_one_project_does_not_see_another_s_vector(self, handler):
        handler.add_project_summary_vector("project_a", vec(1))
        with pytest.raises(VectorNotFoundEror):
            handler.get_project_summary_vector("project_b")

    def test_updating_one_leaves_the_other_alone(self, handler):
        handler.add_project_summary_vector("project_a", vec(1))
        handler.add_project_summary_vector("project_b", vec(2))
        handler.update_project_summary_vector("project_a", vec(9))
        assert handler.get_project_summary_vector("project_b").tolist() == vec(2).tolist()

    def test_deleting_one_leaves_the_other_alone(self, handler):
        handler.add_project_summary_vector("project_a", vec(1))
        handler.add_project_summary_vector("project_b", vec(2))
        handler.delete_project_summary_vector("project_a")
        assert handler.get_project_summary_vector("project_b").tolist() == vec(2).tolist()


class TestConnectionReuse:
    def test_repeated_calls_reuse_one_connection(self, handler):
        """VectorRepository opens PostgreSQL in its constructor, so one per call
        would open and drop a connection on every read."""
        handler.add_project_summary_vector("project_a", vec(1))
        for _ in range(10):
            handler.get_project_summary_vector("project_a")
        handler.update_project_summary_vector("project_a", vec(2))
        handler.delete_project_summary_vector("project_a")
        assert len(FakeVectorRepository.instances) == 1

    def test_close_releases_every_connection(self, handler):
        handler.add_project_summary_vector("project_a", vec(1))
        handler.add_project_summary_vector("project_b", vec(2))
        handler.close()
        assert all(r.closed for r in FakeVectorRepository.instances)

    def test_close_is_repeatable(self, handler):
        handler.add_project_summary_vector("project_a", vec(1))
        handler.close()
        handler.close()

    def test_a_call_after_close_opens_a_fresh_connection(self, handler):
        handler.add_project_summary_vector("project_a", vec(1))
        handler.close()
        handler.add_project_summary_vector("project_a", vec(1))
        assert len(FakeVectorRepository.instances) == 2

    def test_works_as_a_context_manager(self):
        FakeVectorRepository.instances = []
        with ProjectVectorHandler(repository_factory=FakeVectorRepository) as handler:
            handler.add_project_summary_vector("project_a", vec(1))
        assert FakeVectorRepository.instances[0].closed


class TestValidation:
    @pytest.mark.parametrize("bad", [None, "", "   ", 7, b"project_a"])
    def test_an_unusable_project_id_is_refused(self, handler, bad):
        with pytest.raises(ValueError):
            handler.add_project_summary_vector(bad, vec(1))

    def test_a_bad_project_id_opens_no_connection(self, handler):
        with pytest.raises(ValueError):
            handler.add_project_summary_vector("", vec(1))
        assert FakeVectorRepository.instances == []

    @pytest.mark.parametrize("size", [1, DIMENSIONS - 1, DIMENSIONS + 1, 0])
    def test_the_wrong_width_is_refused(self, handler, size):
        with pytest.raises(InvalidVectorDimension):
            handler.add_project_summary_vector("project_a", [1.0] * size)

    def test_a_two_dimensional_array_is_refused(self, handler):
        with pytest.raises(InvalidVectorDimension):
            handler.add_project_summary_vector("project_a", np.vstack([vec(1), vec(2)]))

    def test_a_bad_vector_opens_no_connection(self, handler):
        """Validated before the store is touched, so a malformed vector costs
        nothing and cannot half-write."""
        with pytest.raises(InvalidVectorDimension):
            handler.add_project_summary_vector("project_a", [1.0])
        assert FakeVectorRepository.instances == []

    def test_update_validates_too(self, handler):
        handler.add_project_summary_vector("project_a", vec(1))
        with pytest.raises(InvalidVectorDimension):
            handler.update_project_summary_vector("project_a", [1.0])
        assert handler.get_project_summary_vector("project_a").tolist() == vec(1).tolist()


# ---------------------------------------------------------------------------
# Bug 4.46 — a project holds one summary vector and N description vectors
# ---------------------------------------------------------------------------


class TestManyVectorsPerProject:
    def test_summary_and_descriptions_coexist(self, handler):
        """The defect: a second vector for one project had nowhere to live."""
        handler.add_project_summary_vector("project_a", vec(1))
        handler.add_project_description_vector("project_a", "goal", vec(2))
        handler.add_project_description_vector("project_a", "scope", vec(3))
        assert len(FakeVectorRepository.instances[0].store) == 3

    def test_each_reads_back_as_itself(self, handler):
        handler.add_project_summary_vector("project_a", vec(1))
        handler.add_project_description_vector("project_a", "goal", vec(2))
        assert handler.get_project_summary_vector("project_a").tolist() == vec(1).tolist()
        assert handler.get_project_description_vector(
            "project_a", "goal"
        ).tolist() == vec(2).tolist()

    def test_a_description_called_summary_does_not_collide(self, handler):
        """The source kind is part of the derivation, so the names cannot meet."""
        assert summary_vector_id("project_a") != description_vector_id(
            "project_a", "summary"
        )
        handler.add_project_summary_vector("project_a", vec(1))
        handler.add_project_description_vector("project_a", "summary", vec(2))
        assert handler.get_project_summary_vector("project_a").tolist() == vec(1).tolist()

    def test_the_source_kind_participates_in_the_id(self):
        """Today the summary is told apart from a description by its empty
        source id alone. The kind is what keeps a *third* source — a title, a
        tag — from colliding with a description that shares its id."""
        assert _derive("description", "p", "goal") != _derive("title", "p", "goal")

    def test_ids_separate_their_fields(self, handler):
        """NUL between the fields: ("a","bc") and ("ab","c") are different
        inputs, the way chunk_id in the conversation layer already works."""
        assert description_vector_id("a", "bc") != description_vector_id("ab", "c")

    def test_description_ids_are_deterministic_and_distinct(self, handler):
        assert description_vector_id("p", "goal") == description_vector_id("p", "goal")
        assert description_vector_id("p", "goal") != description_vector_id("p", "scope")
        assert description_vector_id("p", "goal") != description_vector_id("q", "goal")

    def test_updating_one_description_leaves_the_others(self, handler):
        handler.add_project_summary_vector("project_a", vec(1))
        handler.add_project_description_vector("project_a", "goal", vec(2))
        handler.add_project_description_vector("project_a", "scope", vec(3))
        handler.update_project_description_vector("project_a", "goal", vec(9))
        assert handler.get_project_description_vector("project_a", "goal").tolist() == vec(9).tolist()
        assert handler.get_project_description_vector("project_a", "scope").tolist() == vec(3).tolist()
        assert handler.get_project_summary_vector("project_a").tolist() == vec(1).tolist()

    def test_deleting_one_description_leaves_the_others(self, handler):
        handler.add_project_description_vector("project_a", "goal", vec(2))
        handler.add_project_description_vector("project_a", "scope", vec(3))
        handler.delete_project_description_vector("project_a", "goal")
        with pytest.raises(VectorNotFoundEror):
            handler.get_project_description_vector("project_a", "goal")
        assert handler.get_project_description_vector("project_a", "scope").tolist() == vec(3).tolist()

    def test_the_same_description_id_across_projects_is_separate(self, handler):
        handler.add_project_description_vector("project_a", "goal", vec(1))
        handler.add_project_description_vector("project_b", "goal", vec(2))
        assert handler.get_project_description_vector("project_a", "goal").tolist() == vec(1).tolist()
        assert handler.get_project_description_vector("project_b", "goal").tolist() == vec(2).tolist()


class TestBatchDescriptionVectors:
    def test_writes_them_all(self, handler):
        handler.add_project_description_vectors(
            "project_a", [("goal", vec(1)), ("scope", vec(2)), ("risk", vec(3))]
        )
        assert len(FakeVectorRepository.instances[0].store) == 3

    def test_is_idempotent(self, handler):
        """batch_insert ignores ids already held, so a retry after a partial
        failure re-runs cleanly."""
        handler.add_project_description_vectors("project_a", [("goal", vec(1))])
        handler.add_project_description_vectors("project_a", [("goal", vec(1))])
        assert len(FakeVectorRepository.instances[0].store) == 1

    def test_a_repeated_id_is_refused(self, handler):
        with pytest.raises(MisMatchCount):
            handler.add_project_description_vectors(
                "project_a", [("goal", vec(1)), ("goal", vec(2))]
            )

    def test_a_bad_pair_writes_nothing(self, handler):
        with pytest.raises(InvalidVectorDimension):
            handler.add_project_description_vectors(
                "project_a", [("goal", vec(1)), ("scope", [1.0])]
            )
        assert FakeVectorRepository.instances == []

    def test_an_empty_batch_is_a_no_op(self, handler):
        handler.add_project_description_vectors("project_a", [])
        assert FakeVectorRepository.instances == []


class TestReadingForTheRouter:
    def test_returns_rows_in_the_order_of_the_ids_given(self, handler):
        """Row i matches vector_ids[i] — what lets a routing score be traced
        back to the vector, and through it the project, that produced it."""
        handler.add_project_summary_vector("project_a", vec(1))
        handler.add_project_description_vectors(
            "project_a", [("goal", vec(2)), ("scope", vec(3))]
        )
        ids = [
            description_vector_id("project_a", "scope"),
            summary_vector_id("project_a"),
            description_vector_id("project_a", "goal"),
        ]
        rows = handler.get_project_vectors("project_a", ids)
        assert rows.shape == (3, DIMENSIONS)
        assert [row[0] for row in rows] == [3.0, 1.0, 2.0]

    def test_no_ids_reads_nothing(self, handler):
        rows = handler.get_project_vectors("project_a", [])
        assert rows.shape == (0, DIMENSIONS)
        assert FakeVectorRepository.instances == []

    def test_an_unknown_id_raises(self, handler):
        handler.add_project_summary_vector("project_a", vec(1))
        with pytest.raises(VectorNotFoundEror):
            handler.get_project_vectors("project_a", [summary_vector_id("project_b")])

    @pytest.mark.parametrize("bad", [None, "", 7])
    def test_an_unusable_project_id_is_refused(self, handler, bad):
        with pytest.raises(ValueError):
            handler.get_project_vectors(bad, [1])

    @pytest.mark.parametrize("bad", [None, "", "  ", 7])
    def test_an_unusable_description_id_is_refused(self, handler, bad):
        with pytest.raises(ValueError):
            handler.add_project_description_vector("project_a", bad, vec(1))

