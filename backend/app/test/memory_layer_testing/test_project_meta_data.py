"""Tests for ProjectMetaData — the project registry and its summary vectors.

The vector store is a fake rather than a MagicMock. The bugs that mattered in
this layer were all about the two stores disagreeing (a vector written with no
mapping row, a mapping row pointing at nothing), and a MagicMock records the
calls without ever holding state, so it cannot show a disagreement. FakeVector-
Repository keeps a dict and reproduces the real repository's failure modes:
insert rejects duplicates, batch_insert ignores them, update raises when the id
is absent.
"""

import sqlite3
import time

import numpy as np
import pytest

from config import Config
from data_layer.datalayer_exceptions.datalayer_exceptions import VectorNotFoundEror
from memory.memory_pool_exceptions import InvalidVectorId, MisMatchCount
from memory.topic_pool.project_pool.project_data_repo.project_meta_data import (
    ProjectMetaData,
    as_timestamp,
    utc_now,
)

PROJECT_ID = "unit_test_project"
DIMENSIONS = Config.EMBEDDING_DIMENSIONS


class FakeVectorRepository:
    """Stateful stand-in for VectorRepository, same contract and failure modes."""

    def __init__(self, project_id=PROJECT_ID):
        self.project_id = project_id
        self.store = {}
        self.closed = False
        self.delete_fails = False

    def insert(self, vector_id, vector):
        if int(vector_id) in self.store:
            raise RuntimeError("duplicate key value violates unique constraint")
        self.store[int(vector_id)] = np.asarray(vector, dtype=np.float32)

    def batch_insert(self, vector_ids, vectors):
        for vector_id, vector in zip(vector_ids, vectors):
            self.store.setdefault(int(vector_id), np.asarray(vector, dtype=np.float32))

    def update(self, vector_id, vector):
        if int(vector_id) not in self.store:
            raise VectorNotFoundEror(vector_id)
        self.store[int(vector_id)] = np.asarray(vector, dtype=np.float32)

    def search(self, vector_id):
        if int(vector_id) not in self.store:
            raise VectorNotFoundEror(vector_id)
        return self.store[int(vector_id)]

    def batch_search(self, vector_ids):
        return np.array([self.search(v) for v in vector_ids])

    def delete(self, vector_id):
        self.store.pop(int(vector_id), None)

    def batch_delete(self, vector_ids):
        if self.delete_fails:
            raise RuntimeError("connection to server was lost")
        for vector_id in vector_ids:
            self.store.pop(int(vector_id), None)

    def close(self):
        self.closed = True


class Boom(Exception):
    """Distinct from every exception the code under test raises on its own."""


def vec(fill: float) -> np.ndarray:
    return np.full(DIMENSIONS, float(fill), dtype=np.float32)


@pytest.fixture
def fake_vectors():
    return FakeVectorRepository()


@pytest.fixture
def db_path(tmp_path):
    # Deliberately nested: the real default path lives under a directory that
    # does not exist on a fresh checkout.
    return tmp_path / "project_db" / "project.sql"


@pytest.fixture
def meta(db_path, fake_vectors):
    m = ProjectMetaData(
        PROJECT_ID, db_path=db_path, vector_repository=fake_vectors
    )
    yield m
    m.close()


def break_meta_writes(meta):
    """Make the SQLite half of a write fail, leaving the vector half done."""
    meta._ProjectMetaData__write_meta_data = _raise_boom


def _raise_boom(*args, **kwargs):
    raise Boom()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_creates_missing_parent_directory(self, db_path, fake_vectors):
        assert not db_path.parent.exists()
        m = ProjectMetaData(PROJECT_ID, db_path=db_path, vector_repository=fake_vectors)
        assert db_path.exists()
        m.close()

    def test_both_tables_created(self, meta):
        with sqlite3.connect(meta.db_path) as conn:
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        assert {"project_table", "project_mapping_table"} <= tables

    def test_mapping_table_primary_key_is_composite(self, meta):
        with sqlite3.connect(meta.db_path) as conn:
            key = [
                row[1]
                for row in conn.execute("PRAGMA table_info(project_mapping_table)")
                if row[5]
            ]
        assert set(key) == {"project_id", "project_summary_vector_id"}

    def test_orphan_mapping_row_rejected(self, meta):
        """The FK must be enforced, not merely declared."""
        with sqlite3.connect(meta.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO project_mapping_table VALUES ('ghost', 1, 'now')"
                )

    def test_init_is_idempotent(self, db_path, fake_vectors):
        first = ProjectMetaData(PROJECT_ID, db_path=db_path, vector_repository=fake_vectors)
        first.add_project_vector(vec(1), 1, "Alpha")
        first.close()
        second = ProjectMetaData(PROJECT_ID, db_path=db_path, vector_repository=fake_vectors)
        assert second.get_all_summary_vector_id() == [1]
        second.close()


# ---------------------------------------------------------------------------
# add_project_vector
# ---------------------------------------------------------------------------


class TestAddProjectVector:
    def test_writes_all_three_stores(self, meta, fake_vectors):
        meta.add_project_vector(vec(1), 11, "Alpha", user_id="u1")

        assert fake_vectors.store[11].tolist() == vec(1).tolist()
        assert meta.get_all_summary_vector_id() == [11]
        project = meta.get_project()
        assert project[0] == PROJECT_ID
        assert project[1] == "Alpha"
        assert project[4] == "u1"

    def test_timestamps_default_to_now(self, meta):
        before = utc_now()
        meta.add_project_vector(vec(1), 11, "Alpha")
        project = meta.get_project()
        assert project[2] >= before
        assert project[3] >= before

    def test_explicit_timestamps_are_stored(self, meta):
        meta.add_project_vector(
            vec(1), 11, "Alpha", created_at="2026-01-01", updated_at="2026-02-02"
        )
        project = meta.get_project()
        assert project[2] == "2026-01-01"
        assert project[3] == "2026-02-02"

    def test_second_vector_reuses_the_project_row(self, meta):
        meta.add_project_vector(vec(1), 11, "Alpha")
        meta.add_project_vector(vec(2), 12, "Alpha")
        with sqlite3.connect(meta.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM project_table").fetchone()[0]
        assert count == 1
        assert meta.get_all_summary_vector_id() == [11, 12]

    def test_duplicate_id_is_rejected(self, meta):
        """Unlike the batch path: the single insert has no on-conflict clause."""
        meta.add_project_vector(vec(1), 11, "Alpha")
        with pytest.raises(RuntimeError):
            meta.add_project_vector(vec(2), 11, "Alpha")


class TestProjectUpsert:
    @pytest.fixture
    def renamed(self, meta):
        meta.add_project_vector(vec(1), 11, "Alpha", user_id="owner")
        first = meta.get_project()
        time.sleep(0.01)
        meta.add_project_vector(vec(2), 12, "Alpha Renamed", user_id="someone_else")
        return first, meta.get_project()

    def test_rename_lands(self, renamed):
        _, second = renamed
        assert second[1] == "Alpha Renamed"

    def test_created_at_is_preserved(self, renamed):
        first, second = renamed
        assert second[2] == first[2]

    def test_updated_at_advances(self, renamed):
        first, second = renamed
        assert second[3] > first[3]

    def test_ownership_is_not_transferred(self, renamed):
        """Recording a summary vector must not reassign the project."""
        _, second = renamed
        assert second[4] == "owner"


# ---------------------------------------------------------------------------
# add_batch_project_vector
# ---------------------------------------------------------------------------


class TestAddBatchProjectVector:
    def test_writes_every_vector_and_mapping(self, meta, fake_vectors):
        meta.add_batch_project_vector([vec(1), vec(2), vec(3)], [11, 12, 13], "Alpha")
        assert sorted(fake_vectors.store) == [11, 12, 13]
        assert meta.get_all_summary_vector_id() == [11, 12, 13]

    def test_accepts_a_plain_list_of_lists(self, meta, fake_vectors):
        """VectorRepository.batch_insert calls .tolist() on each row."""
        meta.add_batch_project_vector([[1.0] * DIMENSIONS], [11], "Alpha")
        assert fake_vectors.store[11].tolist() == [1.0] * DIMENSIONS

    def test_accepts_a_two_dimensional_array(self, meta, fake_vectors):
        meta.add_batch_project_vector(np.vstack([vec(1), vec(2)]), [11, 12], "Alpha")
        assert sorted(fake_vectors.store) == [11, 12]

    def test_re_adding_an_existing_id_is_a_no_op(self, meta):
        meta.add_batch_project_vector([vec(1)], [11], "Alpha")
        meta.add_batch_project_vector([vec(1)], [11], "Alpha")
        assert meta.get_all_summary_vector_id() == [11]

    def test_empty_batch_writes_nothing(self, meta, fake_vectors):
        meta.add_batch_project_vector([], [], "Alpha")
        assert fake_vectors.store == {}
        assert meta.get_project() is None

    def test_empty_batch_does_not_touch_updated_at(self, meta):
        meta.add_project_vector(vec(1), 11, "Alpha")
        before = meta.get_project()[3]
        time.sleep(0.01)
        meta.add_batch_project_vector([], [], "Alpha")
        assert meta.get_project()[3] == before

    def test_count_mismatch_raises(self, meta, fake_vectors):
        with pytest.raises(MisMatchCount):
            meta.add_batch_project_vector([vec(1), vec(2)], [11], "Alpha")
        assert fake_vectors.store == {}

    def test_repeated_id_inside_one_batch_raises(self, meta, fake_vectors):
        """Otherwise the store silently deduplicates and the count is wrong."""
        with pytest.raises(MisMatchCount):
            meta.add_batch_project_vector([vec(1), vec(2)], [11, 11], "Alpha")
        assert fake_vectors.store == {}


# ---------------------------------------------------------------------------
# Vector id validation
# ---------------------------------------------------------------------------


class TestVectorIdValidation:
    @pytest.mark.parametrize(
        "bad_id", [-1, Config.VECTOR_ID_MASK + 1, 1 << 64, "abc", "11", None, 1.5, 1e9]
    )
    def test_rejected_before_anything_is_written(self, meta, fake_vectors, bad_id):
        with pytest.raises(InvalidVectorId):
            meta.add_project_vector(vec(1), bad_id, "Alpha")
        assert fake_vectors.store == {}
        assert meta.get_project() is None

    def test_largest_allowed_id_is_accepted(self, meta):
        meta.add_project_vector(vec(1), Config.VECTOR_ID_MASK, "Alpha")
        assert meta.get_all_summary_vector_id() == [Config.VECTOR_ID_MASK]

    def test_numpy_unsigned_id_is_accepted(self, meta):
        meta.add_project_vector(vec(1), np.uint32(11), "Alpha")
        assert meta.get_all_summary_vector_id() == [11]

    def test_batch_rejects_a_bad_id_without_writing_the_good_ones(
        self, meta, fake_vectors
    ):
        with pytest.raises(InvalidVectorId):
            meta.add_batch_project_vector([vec(1), vec(2)], [11, -5], "Alpha")
        assert fake_vectors.store == {}


# ---------------------------------------------------------------------------
# update_summary_vector
# ---------------------------------------------------------------------------


class TestUpdateSummaryVector:
    def test_replaces_the_embedding(self, meta):
        meta.add_project_vector(vec(1), 11, "Alpha")
        meta.update_summary_vector(11, vec(99))
        assert meta.get_summary_vector(11).tolist() == vec(99).tolist()

    def test_mapping_is_unchanged(self, meta):
        meta.add_project_vector(vec(1), 11, "Alpha")
        meta.update_summary_vector(11, vec(99))
        assert meta.get_all_summary_vector_id() == [11]

    def test_bumps_updated_at(self, meta):
        meta.add_project_vector(vec(1), 11, "Alpha")
        before = meta.get_project()[3]
        time.sleep(0.01)
        meta.update_summary_vector(11, vec(99))
        assert meta.get_project()[3] > before

    def test_unknown_id_raises(self, meta):
        meta.add_project_vector(vec(1), 11, "Alpha")
        with pytest.raises(VectorNotFoundEror):
            meta.update_summary_vector(12, vec(99))

    def test_invalid_id_raises_before_reaching_the_store(self, meta):
        with pytest.raises(InvalidVectorId):
            meta.update_summary_vector(-1, vec(99))


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


class TestReads:
    @pytest.fixture
    def seeded(self, meta):
        for i in range(1, 6):
            meta.add_project_vector(vec(i), 10 + i, "Alpha")
        return meta

    def test_get_summary_vector(self, seeded):
        assert seeded.get_summary_vector(13).tolist() == vec(3).tolist()

    def test_get_summary_vector_unknown_id_raises(self, seeded):
        with pytest.raises(VectorNotFoundEror):
            seeded.get_summary_vector(99)

    def test_ids_come_back_oldest_first(self, seeded):
        assert seeded.get_all_summary_vector_id() == [11, 12, 13, 14, 15]

    def test_vectors_align_with_ids(self, seeded):
        vectors = seeded.get_all_summary_vector()
        assert vectors.shape == (5, DIMENSIONS)
        for index, vector_id in enumerate(seeded.get_all_summary_vector_id()):
            assert vectors[index].tolist() == seeded.get_summary_vector(vector_id).tolist()

    def test_sub_second_writes_keep_their_order(self, meta):
        """datetime() truncates to seconds; the raw TEXT tiebreaker recovers it."""
        ids = list(range(20, 40))
        for vector_id in ids:
            meta.add_project_vector(vec(1), vector_id, "Alpha")
        assert meta.get_all_summary_vector_id() == ids

    def test_empty_project_returns_no_ids(self, meta):
        assert meta.get_all_summary_vector_id() == []

    def test_empty_project_returns_a_correctly_shaped_array(self, meta):
        """np.array([]) is 1-D; callers stacking the result need the width."""
        vectors = meta.get_all_summary_vector()
        assert vectors.shape == (0, DIMENSIONS)

    def test_get_project_before_any_write(self, meta):
        assert meta.get_project() is None


# ---------------------------------------------------------------------------
# Cross-store consistency
# ---------------------------------------------------------------------------


class TestCompensation:
    def test_single_add_rolls_the_vector_back(self, meta, fake_vectors):
        break_meta_writes(meta)
        with pytest.raises(Boom):
            meta.add_project_vector(vec(1), 11, "Alpha")
        assert fake_vectors.store == {}

    def test_batch_add_rolls_every_vector_back(self, meta, fake_vectors):
        break_meta_writes(meta)
        with pytest.raises(Boom):
            meta.add_batch_project_vector([vec(1), vec(2)], [11, 12], "Alpha")
        assert fake_vectors.store == {}

    def test_no_mapping_row_survives_a_failed_write(self, meta):
        break_meta_writes(meta)
        with pytest.raises(Boom):
            meta.add_project_vector(vec(1), 11, "Alpha")
        assert meta.get_all_summary_vector_id() == []

    def test_earlier_vectors_are_untouched(self, meta, fake_vectors):
        meta.add_project_vector(vec(1), 11, "Alpha")
        break_meta_writes(meta)
        with pytest.raises(Boom):
            meta.add_project_vector(vec(2), 12, "Alpha")
        assert sorted(fake_vectors.store) == [11]

    def test_failed_cleanup_does_not_mask_the_original_error(
        self, meta, fake_vectors
    ):
        fake_vectors.delete_fails = True
        break_meta_writes(meta)
        with pytest.raises(Boom):
            meta.add_project_vector(vec(1), 11, "Alpha")
        # The orphan remains — that is the point of logging it.
        assert sorted(fake_vectors.store) == [11]

    def test_sqlite_failure_leaves_neither_table_half_written(self, meta):
        """Both metadata tables are one transaction, not two commits."""
        meta.add_project_vector(vec(1), 11, "Alpha")
        meta._ProjectMetaData__connection.execute(
            "DROP TABLE project_mapping_table"
        )
        with pytest.raises(sqlite3.Error):
            meta.add_project_vector(vec(2), 12, "Renamed")
        assert meta.get_project()[1] == "Alpha"


# ---------------------------------------------------------------------------
# Multi-project isolation
# ---------------------------------------------------------------------------


class TestIsolation:
    def test_two_projects_may_share_a_vector_id(self, db_path):
        """Content-derived ids collide across projects; the PK is composite."""
        first = ProjectMetaData(
            "project_a", db_path=db_path, vector_repository=FakeVectorRepository("project_a")
        )
        second = ProjectMetaData(
            "project_b", db_path=db_path, vector_repository=FakeVectorRepository("project_b")
        )
        first.add_project_vector(vec(1), 11, "Alpha")
        second.add_project_vector(vec(2), 11, "Beta")

        assert first.get_all_summary_vector_id() == [11]
        assert second.get_all_summary_vector_id() == [11]
        first.close()
        second.close()

    def test_one_project_does_not_see_another_s_vectors(self, db_path):
        first = ProjectMetaData(
            "project_a", db_path=db_path, vector_repository=FakeVectorRepository("project_a")
        )
        second = ProjectMetaData(
            "project_b", db_path=db_path, vector_repository=FakeVectorRepository("project_b")
        )
        first.add_batch_project_vector([vec(1), vec(2)], [11, 12], "Alpha")
        assert second.get_all_summary_vector_id() == []
        assert second.get_project() is None
        first.close()
        second.close()

    def test_state_survives_reopen(self, db_path, fake_vectors):
        first = ProjectMetaData(PROJECT_ID, db_path=db_path, vector_repository=fake_vectors)
        first.add_batch_project_vector([vec(1), vec(2)], [11, 12], "Alpha", user_id="u1")
        first.close()

        second = ProjectMetaData(PROJECT_ID, db_path=db_path, vector_repository=fake_vectors)
        assert second.get_all_summary_vector_id() == [11, 12]
        assert second.get_project()[1] == "Alpha"
        assert second.get_project()[4] == "u1"
        second.close()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_close_leaves_an_injected_repository_open(self, meta, fake_vectors):
        meta.close()
        assert fake_vectors.closed is False

    def test_close_is_repeatable(self, meta):
        meta.close()
        meta.close()

    def test_close_does_not_build_a_vector_repository(self, db_path):
        """Constructing one opens PostgreSQL — closing must not require it."""
        m = ProjectMetaData(PROJECT_ID, db_path=db_path)
        m.close()

    def test_construction_does_not_build_a_vector_repository(self, db_path):
        m = ProjectMetaData(PROJECT_ID, db_path=db_path)
        assert m._ProjectMetaData__project_vector_handler is None
        m.close()

    def test_reads_work_without_a_vector_repository(self, db_path):
        m = ProjectMetaData(PROJECT_ID, db_path=db_path)
        assert m.get_all_summary_vector_id() == []
        assert m.get_project() is None
        assert m._ProjectMetaData__project_vector_handler is None
        m.close()

    def test_context_manager_closes(self, db_path, fake_vectors):
        with ProjectMetaData(
            PROJECT_ID, db_path=db_path, vector_repository=fake_vectors
        ) as m:
            m.add_project_vector(vec(1), 11, "Alpha")
        assert m._ProjectMetaData__connection is None

    def test_del_on_a_half_built_object_does_not_raise(self, tmp_path):
        """A failure in __init__ must surface as itself, not AttributeError."""
        broken = tmp_path / "a_file"
        broken.write_text("not a directory")
        with pytest.raises(OSError):
            ProjectMetaData(PROJECT_ID, db_path=broken / "nested" / "project.sql")


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


class TestTimestamps:
    def test_utc_now_is_sortable_as_text(self):
        first = utc_now()
        time.sleep(0.001)
        assert utc_now() > first

    def test_utc_now_keeps_sub_second_precision(self):
        assert "." in utc_now()

    def test_as_timestamp_defaults_to_now(self):
        assert as_timestamp(None) >= "2026"

    def test_as_timestamp_passes_strings_through(self):
        assert as_timestamp("2026-01-01") == "2026-01-01"

    def test_as_timestamp_normalises_dates(self):
        from datetime import date

        assert as_timestamp(date(2026, 1, 1)) == "2026-01-01"
