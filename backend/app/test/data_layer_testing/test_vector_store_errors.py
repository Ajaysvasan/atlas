"""Error contracts for the two vector stores and the exceptions they raise.

Each class here pins one of four defects found while auditing
`docs/data_layer_docs/`: an exception whose attribute held two different kinds
of value, a batch path that wrapped nothing, a message that described the wrong
condition, and an exception nothing ever raised.
"""

import sys
import unittest.mock as mock

import numpy as np
import pytest

sys.modules.setdefault("diskannpy", mock.MagicMock())


def _stub_psycopg():
    """psycopg is not installed for every interpreter that runs this suite.

    UniqueViolation has to be a real exception class, not a MagicMock, because
    the code under test names it in an `except` clause.
    """
    if "psycopg" in sys.modules:
        return sys.modules["psycopg"]
    stub = mock.MagicMock()
    stub.errors.UniqueViolation = type("UniqueViolation", (Exception,), {})
    sys.modules["psycopg"] = stub
    sys.modules.setdefault("dotenv", mock.MagicMock())
    return stub


psycopg = _stub_psycopg()

from data_layer.datalayer_exceptions.datalayer_exceptions import (
    DuplicateVectorException,
    InvalidVectorDimension,
    InvalidVectorID,
    VectorInsertionError,
)
from data_layer.vector_db_manager.vectorDB_diskann import VectorDb_diskann


@pytest.fixture
def driver():
    db = VectorDb_diskann("l2", np.float32, 128, 1000, 100, 120, 4)
    db.dynamic_dann = mock.MagicMock()
    return db


class TestVectorInsertionErrorCarriesBoth:
    """`vector_id` held the id when raised from DiskANN and the psycopg error
    when raised from the repository, so reading it gave one or the other."""

    def test_id_and_cause_are_separate(self):
        cause = RuntimeError("native failure")
        error = VectorInsertionError(42, cause)
        assert error.vector_id == 42
        assert error.cause is cause

    def test_cause_is_optional(self):
        error = VectorInsertionError(42)
        assert error.cause is None
        assert str(error) == "An Error occured while inserting the vector : 42"

    def test_message_names_the_cause(self):
        text = str(VectorInsertionError(42, ValueError("bad dtype")))
        assert "42" in text
        assert "ValueError" in text
        assert "bad dtype" in text

    def test_batch_ids_are_truncated(self):
        """A batch of 4000 ids must not produce a 100 KB exception message."""
        text = str(VectorInsertionError(list(range(4000)), RuntimeError("boom")))
        assert "(4000 ids)" in text
        assert len(text) < 200

    def test_short_batch_is_listed_in_full(self):
        assert "1, 2, 3" in str(VectorInsertionError([1, 2, 3]))

    def test_numpy_id_is_not_treated_as_a_batch(self):
        assert str(VectorInsertionError(np.uint32(7))).endswith(": 7")


class TestDiskannWrapsBothInsertPaths:
    """`insert` wrapped driver failures, `batch_insert` did not — so
    `except VectorInsertionError` around a batch caught nothing."""

    def test_insert_wraps(self, driver):
        driver.dynamic_dann.insert.side_effect = RuntimeError("native")
        with pytest.raises(VectorInsertionError) as caught:
            driver.insert(np.zeros(128, dtype=np.float32), 1)
        assert caught.value.vector_id == 1

    def test_batch_insert_wraps(self, driver):
        driver.dynamic_dann.batch_insert.side_effect = RuntimeError("native")
        with pytest.raises(VectorInsertionError) as caught:
            driver.batch_insert(np.zeros((2, 128), dtype=np.float32), [1, 2])
        assert caught.value.vector_id == [1, 2]

    @pytest.mark.parametrize("failure", [ValueError("bad"), RuntimeError("native")])
    def test_both_driver_error_types_are_wrapped(self, driver, failure):
        driver.dynamic_dann.batch_insert.side_effect = failure
        with pytest.raises(VectorInsertionError):
            driver.batch_insert(np.zeros((1, 128), dtype=np.float32), [1])

    def test_original_exception_is_chained(self, driver):
        """`raise ... from e` keeps the driver traceback reachable."""
        failure = RuntimeError("native")
        driver.dynamic_dann.insert.side_effect = failure
        with pytest.raises(VectorInsertionError) as caught:
            driver.insert(np.zeros(128, dtype=np.float32), 1)
        assert caught.value.__cause__ is failure


class TestInvalidVectorIDMessage:
    """The message read "There is arrtibute of id 7" — a typo, and it described
    no condition. It is raised when a metadata row is missing."""

    def test_message_states_the_condition(self):
        text = str(InvalidVectorID(7))
        assert "7" in text
        assert "arrtibute" not in text
        assert "no vector meta data" in text.lower()

    def test_id_is_still_exposed(self):
        assert InvalidVectorID(7).vectorId == 7


class TestDuplicateVectorExceptionIsRaised:
    """The class was defined but never constructed anywhere; a duplicate
    single insert surfaced as an indistinguishable VectorInsertionError."""

    @pytest.fixture
    def repository(self, monkeypatch):
        for key, value in [
            ("DBNAME", "db"), ("DB_USER", "u"), ("PASSWORD", "p"),
            ("HOST", "h"), ("PORT", "5432"),
        ]:
            monkeypatch.setenv(key, value)
        from data_layer.vector_db_manager.repository import vectorRepository as module

        monkeypatch.setattr(module, "load_dotenv", lambda *a, **k: None)
        monkeypatch.setattr(module.psycopg, "connect", mock.MagicMock())
        repository = module.VectorRepository("project")
        # The constructor issues its own CREATE EXTENSION / CREATE TABLE and
        # commits them; clear those so each test observes only its own calls.
        repository.curr.reset_mock()
        repository.conn.reset_mock()
        yield repository

    def test_unique_violation_becomes_duplicate(self, repository):
        repository.curr.execute.side_effect = psycopg.errors.UniqueViolation("dup")
        with pytest.raises(DuplicateVectorException) as caught:
            repository.insert(7, np.zeros(128, dtype=np.float32))
        assert caught.value.vector_id == 7
        repository.conn.rollback.assert_called_once()

    def test_other_failures_stay_insertion_errors(self, repository):
        repository.curr.execute.side_effect = RuntimeError("connection lost")
        with pytest.raises(VectorInsertionError):
            repository.insert(7, np.zeros(128, dtype=np.float32))

    def test_duplicate_is_not_an_insertion_error(self, repository):
        """The two must be distinguishable, which is the point of the split."""
        repository.curr.execute.side_effect = psycopg.errors.UniqueViolation("dup")
        with pytest.raises(DuplicateVectorException):
            try:
                repository.insert(7, np.zeros(128, dtype=np.float32))
            except VectorInsertionError:
                pytest.fail("duplicate was swallowed as a generic insertion failure")

    def test_batch_insert_tolerates_duplicates(self, repository):
        """batch_insert is `on conflict do nothing` by design and must not raise."""
        repository.batch_insert([1, 2], np.zeros((2, 128), dtype=np.float32))
        repository.conn.commit.assert_called()

    def test_dimension_check_runs_before_the_write(self, repository):
        with pytest.raises(InvalidVectorDimension):
            repository.insert(7, np.zeros(64, dtype=np.float32))
        repository.curr.execute.assert_not_called()
