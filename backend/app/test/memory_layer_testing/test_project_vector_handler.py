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
from memory.topic_pool.project_pool.project_data_repo.project_vector_handler import (
    ProjectVectorHandler,
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
