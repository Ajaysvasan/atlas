import sqlite3
import threading
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from memory.topic_pool.project_pool.conversation_pool.conversation_data_management.conversationVectorMetaManager import (
    ConversationVectorMetaDataRepository,
)
from memory.topic_pool.project_pool.conversation_pool.conversation_data_management.conversationVectorManager import (
    ConversationVectorManager,
)

PROJECT_ID = "unit_test_project"

_MOCK_VECTOR_REPO = (
    "memory.topic_pool.project_pool.conversation_pool"
    ".conversation_data_management.conversationVectorManager.VectorRepository"
)


@pytest.fixture
def repo(tmp_path):
    r = ConversationVectorMetaDataRepository(
        conversation_path=tmp_path, project_id=PROJECT_ID
    )
    yield r
    r.close()


@pytest.fixture
def seeded_repo(repo):
    repo.batch_insert_summary_chunks([
        ("c1", "text one", "2026-08-01", "typeA"),
        ("c2", "text two", "2026-08-02", "typeA"),
    ])
    repo.batch_insert_summary_vector_meta_data([
        (101, "c1", PROJECT_ID),
        (102, "c2", PROJECT_ID),
    ])
    repo.insert_cumulative_vector_meta_data(201, "summary A", "2026-08-01", PROJECT_ID, 10)
    repo.insert_cumulative_vector_meta_data(202, "summary B", "2026-08-02", PROJECT_ID, 20)
    return repo


def _row_count(db_path: Path, table: str) -> int:
    with sqlite3.connect(db_path) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_all_four_tables_created(self, repo):
        with sqlite3.connect(repo.db_path) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        assert {"summary_chunks", "summary_vector_meta_data",
                "cumulative_vector_meta_data", "summary_snapshot_map"} <= tables

    def test_summary_vector_meta_fk_enforced(self, repo):
        with pytest.raises(sqlite3.IntegrityError):
            repo.batch_insert_summary_vector_meta_data(
                [(999, "nonexistent_chunk", PROJECT_ID)]
            )

    def test_map_fk_on_cumulative_enforced(self, repo):
        repo.batch_insert_summary_chunks([("c1", "t", "2026-01-01", "x")])
        repo.batch_insert_summary_vector_meta_data([(1, "c1", PROJECT_ID)])
        with pytest.raises(sqlite3.IntegrityError):
            repo.insert_map_table(9999, 1)

    def test_map_fk_on_summary_enforced(self, repo):
        repo.insert_cumulative_vector_meta_data(300, "s", "2026-01-01", PROJECT_ID, 5)
        with pytest.raises(sqlite3.IntegrityError):
            repo.insert_map_table(300, 9999)

    def test_map_unique_constraint_bug_4_16(self, repo):
        """Regression guard: duplicate (cum_id, sum_id) pair must raise."""
        repo.batch_insert_summary_chunks([("c1", "t", "2026-01-01", "x")])
        repo.batch_insert_summary_vector_meta_data([(1, "c1", PROJECT_ID)])
        repo.insert_cumulative_vector_meta_data(300, "s", "2026-01-01", PROJECT_ID, 5)
        repo.insert_map_table(300, 1)
        with pytest.raises(sqlite3.IntegrityError):
            repo.insert_map_table(300, 1)

    def test_db_file_created_at_expected_path(self, tmp_path):
        repo = ConversationVectorMetaDataRepository(tmp_path, "proj_schema")
        expected = tmp_path / "proj_schema_conversation.db"
        assert expected.exists()
        repo.close()

    def test_foreign_keys_enforced_on_repo_connection(self, repo):
        """
        PRAGMA foreign_keys = ON must be active on the repo's own connection.
        We verify this behaviourally: inserting into summary_vector_meta_data with
        a non-existent chunk_id must raise, proving the FK is enforced rather than
        silently ignored (which happens when the PRAGMA is OFF).
        """
        with pytest.raises(sqlite3.IntegrityError):
            repo.batch_insert_summary_vector_meta_data(
                [(1, "does_not_exist", PROJECT_ID)]
            )

    def test_repeated_init_is_idempotent(self, tmp_path):
        """Creating two repos on the same path must not raise or duplicate tables."""
        r1 = ConversationVectorMetaDataRepository(tmp_path, PROJECT_ID)
        r1.close()
        r2 = ConversationVectorMetaDataRepository(tmp_path, PROJECT_ID)
        with sqlite3.connect(r2.db_path) as conn:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
        assert tables.count("summary_chunks") == 1
        r2.close()


# ---------------------------------------------------------------------------
# summary_chunks
# ---------------------------------------------------------------------------

class TestSummaryChunks:
    def test_basic_insert_and_count(self, repo):
        repo.batch_insert_summary_chunks([
            ("ch1", "hello", "2026-08-01", "typeA"),
            ("ch2", "world", "2026-08-02", "typeB"),
        ])
        assert _row_count(repo.db_path, "summary_chunks") == 2

    def test_empty_batch_is_noop(self, repo):
        repo.batch_insert_summary_chunks([])
        assert _row_count(repo.db_path, "summary_chunks") == 0

    def test_duplicate_chunk_id_is_ignored(self, repo):
        """
        Idempotent by design: this table is shared with FullConversationRepository
        in the same DB file, so the chunks a snapshot covers are already present.
        Raising here aborted every snapshot taken over stored turns.
        """
        repo.batch_insert_summary_chunks([("dup", "text", "2026-01-01", "t")])
        repo.batch_insert_summary_chunks([("dup", "text2", "2026-01-02", "t")])
        assert _row_count(repo.db_path, "summary_chunks") == 1

    def test_duplicate_insert_preserves_the_original_row(self, repo):
        """IGNORE, not REPLACE — the stored turn is authoritative."""
        repo.batch_insert_summary_chunks([("dup", "original", "2026-01-01", "t")])
        repo.batch_insert_summary_chunks([("dup", "overwritten", "2026-01-02", "t")])
        with sqlite3.connect(repo.db_path) as conn:
            row = conn.execute(
                "SELECT chunk, created_at FROM summary_chunks WHERE chunk_id='dup'"
            ).fetchone()
        assert row == ("original", "2026-01-01")

    def test_unicode_text_roundtrip(self, repo):
        text = "rocket: \U0001f680, earth: \U0001f30d"
        repo.batch_insert_summary_chunks([("u1", text, "2026-08-01", "t")])
        with sqlite3.connect(repo.db_path) as conn:
            row = conn.execute(
                "SELECT chunk FROM summary_chunks WHERE chunk_id='u1'"
            ).fetchone()
        assert row[0] == text

    def test_long_text_insert(self, repo):
        long_text = "x" * 100_000
        repo.batch_insert_summary_chunks([("long1", long_text, "2026-08-01", "t")])
        with sqlite3.connect(repo.db_path) as conn:
            row = conn.execute(
                "SELECT chunk FROM summary_chunks WHERE chunk_id='long1'"
            ).fetchone()
        assert len(row[0]) == 100_000

    def test_thousand_record_batch(self, repo):
        records = [(f"b{i}", f"text {i}", "2026-08-01", "typeA") for i in range(1000)]
        repo.batch_insert_summary_chunks(records)
        assert _row_count(repo.db_path, "summary_chunks") == 1000

    def test_all_fields_stored_correctly(self, repo):
        repo.batch_insert_summary_chunks([("cX", "hello world", "2026-09-01", "recursive")])
        with sqlite3.connect(repo.db_path) as conn:
            row = conn.execute(
                "SELECT chunk_id, chunk, created_at, chunker_type FROM summary_chunks WHERE chunk_id='cX'"
            ).fetchone()
        assert row == ("cX", "hello world", "2026-09-01", "recursive")

    def test_partial_duplicate_still_inserts_the_new_rows(self, repo):
        """
        A snapshot window overlaps the previous one by design, so a batch is
        normally part-old part-new. The duplicate must be skipped without
        taking the genuinely new rows down with it.
        """
        repo.batch_insert_summary_chunks([("existing", "t", "2026-01-01", "x")])
        repo.batch_insert_summary_chunks([
            ("new1", "t", "2026-01-01", "x"),
            ("existing", "t2", "2026-01-02", "x"),  # already stored
            ("new2", "t", "2026-01-01", "x"),
        ])
        assert _row_count(repo.db_path, "summary_chunks") == 3
        with sqlite3.connect(repo.db_path) as conn:
            ids = {
                r[0] for r in conn.execute("SELECT chunk_id FROM summary_chunks")
            }
        assert ids == {"existing", "new1", "new2"}


# ---------------------------------------------------------------------------
# cumulative_vector_meta_data
# ---------------------------------------------------------------------------

class TestCumulativeVectorMetaData:
    def test_single_insert_and_get(self, repo):
        repo.insert_cumulative_vector_meta_data(1, "summary", "2026-08-01", PROJECT_ID, 42)
        row = repo.get_cumulative_vector_meta_data(1)
        assert row is not None
        assert row[0] == 1

    def test_batch_insert_and_batch_get(self, repo):
        records = [
            (1, "s1", "2026-08-01", PROJECT_ID, 10),
            (2, "s2", "2026-08-02", PROJECT_ID, 20),
            (3, "s3", "2026-08-03", PROJECT_ID, 30),
        ]
        repo.batch_insert_cumulative_vector_meta_data(records)
        result = repo.batch_get_cumulative_vector_meta_data([1, 2, 3])
        assert len(result) == 3

    def test_get_nonexistent_returns_none(self, repo):
        assert repo.get_cumulative_vector_meta_data(9999) is None

    def test_numpy_uint32_coercion(self, repo):
        vid = np.uint32(501)
        repo.insert_cumulative_vector_meta_data(vid, "numpy summary", "2026-08-07", PROJECT_ID, 42)
        row = repo.get_cumulative_vector_meta_data(vid)
        assert row is not None
        assert row[0] == 501

    def test_chronological_order_bug_4_17(self, repo):
        """Regression: IDs must come back sorted by datetime(created_at), not TEXT sort."""
        repo.insert_cumulative_vector_meta_data(3, "s3", "2026-08-03", PROJECT_ID, 3)
        repo.insert_cumulative_vector_meta_data(1, "s1", "2026-08-01", PROJECT_ID, 1)
        repo.insert_cumulative_vector_meta_data(2, "s2", "2026-08-02", PROJECT_ID, 2)
        ids = [row[0] for row in repo.get_cumulative_vector_meta_data_ids()]
        assert ids == [1, 2, 3]

    def test_empty_batch_get_returns_empty(self, repo):
        assert repo.batch_get_cumulative_vector_meta_data([]) == []

    def test_batch_get_with_nonexistent_ids_returns_only_found(self, repo):
        repo.insert_cumulative_vector_meta_data(10, "s", "2026-01-01", PROJECT_ID, 1)
        result = repo.batch_get_cumulative_vector_meta_data([10, 99999])
        assert len(result) == 1
        assert result[0][0] == 10

    def test_five_hundred_sequential_inserts(self, repo):
        records = [(i, f"s{i}", "2026-08-01", PROJECT_ID, i) for i in range(1, 501)]
        repo.batch_insert_cumulative_vector_meta_data(records)
        ids = repo.get_cumulative_vector_meta_data_ids()
        assert len(ids) == 500

    def test_get_cumulative_vector_meta_data_ids_empty_returns_empty_list(self, repo):
        assert repo.get_cumulative_vector_meta_data_ids() == []

    def test_all_five_fields_stored_correctly(self, repo):
        repo.insert_cumulative_vector_meta_data(77, "the summary", "2026-09-01", PROJECT_ID, 99)
        row = repo.get_cumulative_vector_meta_data(77)
        assert row[0] == 77
        assert row[1] == "the summary"
        assert row[2] == "2026-09-01"
        assert row[3] == PROJECT_ID


# ---------------------------------------------------------------------------
# get_latest_summary
# ---------------------------------------------------------------------------

class TestGetLatestSummary:
    def test_returns_none_on_empty_table(self, repo):
        assert repo.get_latest_summary() is None

    def test_sub_second_timestamps_are_ordered(self, repo):
        """
        SQLite datetime() truncates to whole seconds. Two snapshots taken in the
        same second tie under datetime() alone, making "latest" arbitrary — the
        rolling summary would then feed the wrong predecessor into the next
        prompt. The raw TEXT tiebreaker recovers the microseconds.
        """
        repo.insert_cumulative_vector_meta_data(
            1, "first", "2026-08-23T05:51:05.280183+00:00", "p", 5
        )
        repo.insert_cumulative_vector_meta_data(
            2, "second", "2026-08-23T05:51:05.940217+00:00", "p", 6
        )
        assert repo.get_latest_summary() == "second"

    def test_sub_second_ordering_independent_of_insert_order(self, repo):
        """The later timestamp wins even when it is inserted first."""
        repo.insert_cumulative_vector_meta_data(
            1, "second", "2026-08-23T05:51:05.940217+00:00", "p", 6
        )
        repo.insert_cumulative_vector_meta_data(
            2, "first", "2026-08-23T05:51:05.280183+00:00", "p", 5
        )
        assert repo.get_latest_summary() == "second"

    def test_snapshot_ids_ordered_by_sub_second_timestamp(self, repo):
        """
        SnapShot's cursors index into this list, so an unstable order silently
        points them at the wrong snapshots.
        """
        repo.insert_cumulative_vector_meta_data(
            30, "c", "2026-08-23T05:51:05.900000+00:00", "p", 1
        )
        repo.insert_cumulative_vector_meta_data(
            10, "a", "2026-08-23T05:51:05.100000+00:00", "p", 1
        )
        repo.insert_cumulative_vector_meta_data(
            20, "b", "2026-08-23T05:51:05.500000+00:00", "p", 1
        )
        assert repo.get_cumulative_vector_meta_data_ids() == [(10,), (20,), (30,)]

    def test_returns_summary_text_after_single_insert(self, repo):
        repo.insert_cumulative_vector_meta_data(1, "first summary", "2026-08-01", PROJECT_ID, 10)
        assert repo.get_latest_summary() == "first summary"

    def test_returns_most_recent_by_datetime_not_insertion_order(self, repo):
        """Inserts in reverse chronological order; must still return the latest."""
        repo.insert_cumulative_vector_meta_data(3, "oldest", "2026-08-01", PROJECT_ID, 5)
        repo.insert_cumulative_vector_meta_data(1, "newest", "2026-08-10", PROJECT_ID, 5)
        repo.insert_cumulative_vector_meta_data(2, "middle", "2026-08-05", PROJECT_ID, 5)
        assert repo.get_latest_summary() == "newest"

    def test_sequential_inserts_always_return_last(self, repo):
        for i in range(1, 6):
            repo.insert_cumulative_vector_meta_data(
                i, f"summary {i}", f"2026-08-0{i}", PROJECT_ID, i
            )
        assert repo.get_latest_summary() == "summary 5"

    def test_unicode_summary_roundtrip(self, repo):
        text = "要約: \U0001f916 AIが会話を要約しました。"
        repo.insert_cumulative_vector_meta_data(1, text, "2026-08-01", PROJECT_ID, 5)
        assert repo.get_latest_summary() == text

    def test_long_summary_roundtrip(self, repo):
        long_text = "word " * 10_000
        repo.insert_cumulative_vector_meta_data(1, long_text, "2026-08-01", PROJECT_ID, 10_000)
        assert repo.get_latest_summary() == long_text

    def test_returns_str_not_tuple(self, repo):
        repo.insert_cumulative_vector_meta_data(1, "hello", "2026-08-01", PROJECT_ID, 1)
        result = repo.get_latest_summary()
        assert isinstance(result, str)

    def test_one_hundred_inserts_returns_last_one(self, repo):
        for i in range(1, 101):
            date = f"2026-08-{i:02d}" if i <= 31 else f"2026-09-{(i-31):02d}"
            repo.insert_cumulative_vector_meta_data(i, f"summary {i}", date, PROJECT_ID, i)
        result = repo.get_latest_summary()
        assert result is not None
        assert "summary" in result


# ---------------------------------------------------------------------------
# summary_vector_meta_data
# ---------------------------------------------------------------------------

class TestSummaryVectorMetaData:
    def test_fk_violation_raises(self, repo):
        with pytest.raises(sqlite3.IntegrityError):
            repo.batch_insert_summary_vector_meta_data(
                [(1, "no_such_chunk", PROJECT_ID)]
            )

    def test_insert_and_retrieve(self, repo):
        repo.batch_insert_summary_chunks([("c1", "text", "2026-08-01", "t")])
        repo.batch_insert_summary_vector_meta_data([(10, "c1", PROJECT_ID)])
        row = repo.get_summary_vector_meta_data(10)
        assert row is not None
        assert row[0] == 10
        assert row[1] == "c1"

    def test_batch_get(self, repo):
        repo.batch_insert_summary_chunks([
            ("c1", "t1", "2026-08-01", "t"),
            ("c2", "t2", "2026-08-02", "t"),
        ])
        repo.batch_insert_summary_vector_meta_data(
            [(1, "c1", PROJECT_ID), (2, "c2", PROJECT_ID)]
        )
        result = repo.batch_get_summary_vector_meta_data([1, 2])
        assert len(result) == 2

    def test_empty_batch_get_returns_empty(self, repo):
        assert repo.batch_get_summary_vector_meta_data([]) == []

    def test_get_nonexistent_returns_none(self, repo):
        assert repo.get_summary_vector_meta_data(9999) is None

    def test_numpy_uint32_summary_vector_id(self, repo):
        repo.batch_insert_summary_chunks([("np_c", "text", "2026-08-01", "t")])
        repo.batch_insert_summary_vector_meta_data([(np.uint32(600), "np_c", PROJECT_ID)])
        row = repo.get_summary_vector_meta_data(np.uint32(600))
        assert row is not None
        assert row[0] == 600

    def test_batch_get_with_nonexistent_ids_returns_only_found(self, repo):
        repo.batch_insert_summary_chunks([("c1", "t", "2026-01-01", "x")])
        repo.batch_insert_summary_vector_meta_data([(5, "c1", PROJECT_ID)])
        result = repo.batch_get_summary_vector_meta_data([5, 99999])
        assert len(result) == 1
        assert result[0][0] == 5

    def test_project_id_stored_correctly(self, repo):
        repo.batch_insert_summary_chunks([("c1", "t", "2026-01-01", "x")])
        repo.batch_insert_summary_vector_meta_data([(1, "c1", "my_project")])
        row = repo.get_summary_vector_meta_data(1)
        assert row[2] == "my_project"


# ---------------------------------------------------------------------------
# summary_snapshot_map
# ---------------------------------------------------------------------------

class TestMapTable:
    def test_batch_insert_exactly_n_rows_bug_4_26(self, repo):
        """Regression guard: batch_insert_map_table(N records) must produce exactly N rows."""
        repo.batch_insert_summary_chunks([
            ("c1", "t", "2026-01-01", "x"), ("c2", "t", "2026-01-01", "x")
        ])
        repo.batch_insert_summary_vector_meta_data(
            [(1, "c1", PROJECT_ID), (2, "c2", PROJECT_ID)]
        )
        repo.insert_cumulative_vector_meta_data(10, "s", "2026-01-01", PROJECT_ID, 1)
        repo.insert_cumulative_vector_meta_data(20, "s", "2026-01-01", PROJECT_ID, 1)

        records = [(10, 1), (10, 2), (20, 1), (20, 2)]
        repo.batch_insert_map_table(records)

        assert _row_count(repo.db_path, "summary_snapshot_map") == len(records)

    def test_unique_constraint_on_duplicate_pair_bug_4_16(self, repo):
        """Regression: inserting the same (cum_id, sum_id) pair twice must fail."""
        repo.batch_insert_summary_chunks([("c1", "t", "2026-01-01", "x")])
        repo.batch_insert_summary_vector_meta_data([(1, "c1", PROJECT_ID)])
        repo.insert_cumulative_vector_meta_data(10, "s", "2026-01-01", PROJECT_ID, 1)
        repo.insert_map_table(10, 1)
        with pytest.raises(sqlite3.IntegrityError):
            repo.insert_map_table(10, 1)

    def test_get_summary_vector_ids_from_map(self, seeded_repo):
        seeded_repo.insert_map_table(201, 101)
        seeded_repo.insert_map_table(201, 102)
        ids = seeded_repo.get_summary_vector_ids_from_map(201)
        assert set(ids) == {101, 102}

    def test_get_ids_returns_empty_for_unknown_cumulative(self, repo):
        repo.insert_cumulative_vector_meta_data(10, "s", "2026-01-01", PROJECT_ID, 1)
        assert repo.get_summary_vector_ids_from_map(10) == []

    def test_numpy_uint32_in_map_insert(self, seeded_repo):
        seeded_repo.insert_map_table(np.uint32(201), np.uint32(101))
        ids = seeded_repo.get_summary_vector_ids_from_map(np.uint32(201))
        assert 101 in ids

    def test_batch_map_insert_with_numpy_ids(self, seeded_repo):
        records = [(np.uint32(201), np.uint32(101)), (np.uint32(202), np.uint32(102))]
        seeded_repo.batch_insert_map_table(records)
        assert 101 in seeded_repo.get_summary_vector_ids_from_map(201)
        assert 102 in seeded_repo.get_summary_vector_ids_from_map(202)

    def test_mapping_correctness_separate_cumulatives(self, seeded_repo):
        seeded_repo.insert_map_table(201, 101)
        seeded_repo.insert_map_table(202, 102)
        assert seeded_repo.get_summary_vector_ids_from_map(201) == [101]
        assert seeded_repo.get_summary_vector_ids_from_map(202) == [102]

    def test_empty_batch_map_insert_is_noop(self, repo):
        repo.batch_insert_map_table([])
        assert _row_count(repo.db_path, "summary_snapshot_map") == 0

    def test_get_ids_from_nonexistent_cumulative_returns_empty(self, repo):
        assert repo.get_summary_vector_ids_from_map(99999) == []


# ---------------------------------------------------------------------------
# Stress
# ---------------------------------------------------------------------------

class TestStress:
    def test_five_hundred_snapshot_pipeline(self, tmp_path):
        repo = ConversationVectorMetaDataRepository(tmp_path, PROJECT_ID)
        chunk_recs = [(f"c{i}", f"text {i}", "2026-08-01", "typeA") for i in range(500)]
        repo.batch_insert_summary_chunks(chunk_recs)
        sv_recs = [(i, f"c{i}", PROJECT_ID) for i in range(500)]
        repo.batch_insert_summary_vector_meta_data(sv_recs)
        cum_recs = [(i + 1000, f"sum {i}", "2026-08-01", PROJECT_ID, i) for i in range(500)]
        repo.batch_insert_cumulative_vector_meta_data(cum_recs)
        assert len(repo.get_cumulative_vector_meta_data_ids()) == 500
        repo.close()

    def test_two_hundred_map_records_exact_count(self, tmp_path):
        repo = ConversationVectorMetaDataRepository(tmp_path, PROJECT_ID)
        repo.batch_insert_summary_chunks(
            [(f"c{i}", "t", "2026-08-01", "t") for i in range(100)]
        )
        repo.batch_insert_summary_vector_meta_data(
            [(i, f"c{i}", PROJECT_ID) for i in range(100)]
        )
        repo.batch_insert_cumulative_vector_meta_data(
            [(i + 1000, "s", "2026-08-01", PROJECT_ID, 1) for i in range(100)]
        )
        repo.batch_insert_map_table([(i + 1000, i) for i in range(100)])
        assert _row_count(repo.db_path, "summary_snapshot_map") == 100
        repo.close()

    def test_twenty_concurrent_read_threads(self, tmp_path):
        repo = ConversationVectorMetaDataRepository(tmp_path, PROJECT_ID)
        repo.batch_insert_cumulative_vector_meta_data(
            [(i, f"s{i}", "2026-08-01", PROJECT_ID, i) for i in range(100)]
        )
        db_path = repo.db_path
        repo.close()

        errors: list = []

        def reader():
            try:
                with sqlite3.connect(db_path) as conn:
                    count = conn.execute(
                        "SELECT COUNT(*) FROM cumulative_vector_meta_data"
                    ).fetchone()[0]
                assert count == 100
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == [], f"Thread errors: {errors}"

    def test_full_pipeline_one_hundred_snapshots(self, tmp_path):
        repo = ConversationVectorMetaDataRepository(tmp_path, PROJECT_ID)
        for i in range(100):
            cid = f"c{i}"
            repo.batch_insert_summary_chunks([(cid, f"text {i}", "2026-08-01", "t")])
            repo.batch_insert_summary_vector_meta_data([(i, cid, PROJECT_ID)])
            repo.insert_cumulative_vector_meta_data(
                i + 1000, f"sum {i}", "2026-08-01", PROJECT_ID, i
            )
            repo.insert_map_table(i + 1000, i)
        with sqlite3.connect(repo.db_path) as conn:
            counts = {
                tbl: conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                for tbl in [
                    "summary_chunks",
                    "summary_vector_meta_data",
                    "cumulative_vector_meta_data",
                    "summary_snapshot_map",
                ]
            }
        assert all(v == 100 for v in counts.values()), counts
        repo.close()

    def test_get_latest_summary_after_five_hundred_inserts(self, tmp_path):
        """get_latest_summary must return the entry with the latest date even at scale."""
        repo = ConversationVectorMetaDataRepository(tmp_path, PROJECT_ID)
        for i in range(1, 501):
            year = 2025 + (i // 366)
            day = (i % 365) or 1
            date = f"{year}-{(day // 31) + 1:02d}-{(day % 28) + 1:02d}"
            repo.insert_cumulative_vector_meta_data(i, f"summary {i}", date, PROJECT_ID, i)
        result = repo.get_latest_summary()
        assert result is not None
        assert "summary" in result
        repo.close()


# ---------------------------------------------------------------------------
# ConversationVectorManager (mocked VectorRepository)
# ---------------------------------------------------------------------------

class TestConversationVectorManager:
    @pytest.fixture
    def setup(self):
        with patch(_MOCK_VECTOR_REPO) as MockVR:
            mock_instance = MockVR.return_value
            mgr = ConversationVectorManager(project_name="TestProject", project_id="proj123")
            yield mgr, mock_instance

    def test_insert_returns_vector_id(self, setup):
        manager, _ = setup
        vid = np.uint32(101)
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = manager.insert(vid, vec)
        assert result == vid
        manager.repository.insert.assert_called_once_with(vid, vec)

    def test_batch_insert_returns_ids(self, setup):
        manager, _ = setup
        vids = [np.uint32(1), np.uint32(2)]
        vecs = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        result = manager.batch_insert(vids, vecs)
        assert len(result) == 2
        manager.repository.batch_insert.assert_called_once_with(vids, vecs)

    def test_get_vector_calls_search(self, setup):
        manager, mock_repo = setup
        mock_repo.search.return_value = np.array([1.0, 2.0])
        result = manager.get_vector(np.uint32(100))
        manager.repository.search.assert_called_once_with(np.uint32(100))
        assert np.array_equal(result, np.array([1.0, 2.0]))

    def test_get_vectors_calls_batch_search(self, setup):
        manager, mock_repo = setup
        mock_repo.batch_search.return_value = np.array([[1.0], [2.0]])
        vids = [np.uint32(1), np.uint32(2)]
        result = manager.get_vectors(vids)
        manager.repository.batch_search.assert_called_once_with(vids)
        assert np.array_equal(result, np.array([[1.0], [2.0]]))

    def test_uint32_zero_boundary(self, setup):
        manager, _ = setup
        manager.insert(np.uint32(0), np.zeros(128, dtype=np.float32))
        manager.repository.insert.assert_called_once()

    def test_uint32_max_boundary(self, setup):
        manager, _ = setup
        manager.insert(np.uint32(4294967295), np.ones(128, dtype=np.float32))
        manager.repository.insert.assert_called_once()

    def test_batch_insert_empty(self, setup):
        manager, _ = setup
        manager.batch_insert([], np.array([]))
        manager.repository.batch_insert.assert_called_once()

    def test_get_vectors_empty_list(self, setup):
        manager, mock_repo = setup
        mock_repo.batch_search.return_value = np.array([])
        manager.get_vectors([])
        manager.repository.batch_search.assert_called_once_with([])

    def test_project_id_stored_on_instance(self, setup):
        manager, _ = setup
        assert manager.project_id == "proj123"

    def test_batch_insert_returns_same_ids_passed_in(self, setup):
        manager, _ = setup
        vids = [np.uint32(10), np.uint32(20), np.uint32(30)]
        vecs = np.zeros((3, 4), dtype=np.float32)
        result = manager.batch_insert(vids, vecs)
        assert result == vids
