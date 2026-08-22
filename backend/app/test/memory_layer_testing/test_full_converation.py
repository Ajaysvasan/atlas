"""
Production-grade tests for FullConversationRepository and FullConversation.

Known bugs in current code (P2 — not yet fixed):
  Bug 4.30: __get_last_n_chunks uses ORDER BY f.created_at ASC LIMIT n, returning
            the OLDEST n rows instead of the most recent n. Tests that call
            get_n_chunks document current behaviour and are marked accordingly.
  Bug 4.31: __get_ranged_chunks and __get_messages_after order by f.created_at
            instead of f.sequence_number. In tests using a uniform created_at
            timestamp the order is deterministic (SQLite rowid order equals
            insertion order), so ordering assertions still pass incidentally.

Regression guards for fixed bugs are annotated with their bug ID.
"""

import sqlite3
from pathlib import Path

import pytest

from memory.topic_pool.project_pool.conversation_pool.fullconversation_repository.fullconversation_repository import (
    FullConversationRepository,
)
from memory.topic_pool.project_pool.conversation_pool.full_conversation_bucket import (
    FullConversation,
)

PROJECT_ID = "proj_123"
PROJECT_NAME = "test_project"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_meta(project_id, seq, chunk_id, role="user", created_at="2026-08-10"):
    return (project_id, seq, chunk_id, role, created_at)


def _make_chunk(chunk_id, text, created_at="2026-08-10", chunker_type="text"):
    return (chunk_id, text, created_at, chunker_type)


# ---------------------------------------------------------------------------
# FullConversationRepository
# ---------------------------------------------------------------------------

@pytest.fixture
def repo(tmp_path):
    return FullConversationRepository(
        conversation_path=tmp_path,
        project_id=PROJECT_ID,
        project_name=PROJECT_NAME,
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_db_file_created(self, repo, tmp_path):
        db_path = tmp_path / f"{PROJECT_ID}_conversation.db"
        assert db_path.exists()

    def test_both_tables_exist(self, repo, tmp_path):
        db_path = tmp_path / f"{PROJECT_ID}_conversation.db"
        with sqlite3.connect(db_path) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        assert "summary_chunks" in tables
        assert "full_conversation" in tables

    def test_summary_chunks_created_before_full_conversation(self, repo, tmp_path):
        """Regression guard Bug 4.5: summary_chunks must exist for FK to work."""
        db_path = tmp_path / f"{PROJECT_ID}_conversation.db"
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='summary_chunks'"
            ).fetchone()
        assert row is not None

    def test_full_conversation_has_fk_on_chunk_id(self, repo, tmp_path):
        """FK from full_conversation.chunk_id → summary_chunks.chunk_id must be enforced."""
        db_path = tmp_path / f"{PROJECT_ID}_conversation.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO full_conversation(project_id, sequence_number, chunk_id, role, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("p", 1, "nonexistent_chunk", "user", "2026-01-01"),
                )

    def test_repeated_construction_same_path_is_idempotent(self, tmp_path):
        """Two repos on the same path must not duplicate tables."""
        r1 = FullConversationRepository(tmp_path, PROJECT_ID, PROJECT_NAME)
        r2 = FullConversationRepository(tmp_path, PROJECT_ID, PROJECT_NAME)
        db_path = tmp_path / f"{PROJECT_ID}_conversation.db"
        with sqlite3.connect(db_path) as conn:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
        assert tables.count("summary_chunks") == 1
        assert tables.count("full_conversation") == 1


# ---------------------------------------------------------------------------
# add / fetch_all
# ---------------------------------------------------------------------------

class TestAddAndFetch:
    def test_add_two_chunks_and_fetch_all(self, repo):
        meta = [
            _make_meta(PROJECT_ID, 1, "c1"),
            _make_meta(PROJECT_ID, 2, "c2"),
        ]
        chunks = [
            _make_chunk("c1", "Hello"),
            _make_chunk("c2", "World"),
        ]
        repo.add(meta, chunks)
        result = repo.fetch_all()
        assert len(result) == 2
        texts = [r[0] for r in result]
        assert "Hello" in texts
        assert "World" in texts

    def test_fetch_all_ordered_by_sequence_number(self, repo):
        """Regression guard Bug 4.15: fetch_all must respect sequence order."""
        meta = [
            _make_meta(PROJECT_ID, 3, "c3"),
            _make_meta(PROJECT_ID, 1, "c1"),
            _make_meta(PROJECT_ID, 2, "c2"),
        ]
        chunks = [
            _make_chunk("c3", "Third"),
            _make_chunk("c1", "First"),
            _make_chunk("c2", "Second"),
        ]
        repo.add(meta, chunks)
        result = repo.fetch_all()
        assert [r[0] for r in result] == ["First", "Second", "Third"]

    def test_add_empty_lists_is_noop(self, repo):
        repo.add([], [])
        assert repo.fetch_all() == []

    def test_add_unicode_chunk(self, repo):
        text = "unicode: éàü\U0001f680"
        repo.add(
            [_make_meta(PROJECT_ID, 1, "u1")],
            [_make_chunk("u1", text)],
        )
        result = repo.fetch_all()
        assert result[0][0] == text

    def test_add_raises_on_duplicate_sequence_number(self, repo):
        repo.add(
            [_make_meta(PROJECT_ID, 1, "c1")],
            [_make_chunk("c1", "First")],
        )
        with pytest.raises(sqlite3.IntegrityError):
            repo.add(
                [_make_meta(PROJECT_ID, 1, "c2")],
                [_make_chunk("c2", "Duplicate seq")],
            )

    def test_fetch_all_returns_list_of_tuples(self, repo):
        repo.add([_make_meta(PROJECT_ID, 1, "c1")], [_make_chunk("c1", "x")])
        result = repo.fetch_all()
        assert isinstance(result, list)
        assert isinstance(result[0], tuple)

    def test_add_long_text(self, repo):
        long_text = "a" * 100_000
        repo.add(
            [_make_meta(PROJECT_ID, 1, "long")],
            [_make_chunk("long", long_text)],
        )
        result = repo.fetch_all()
        assert len(result[0][0]) == 100_000

    def test_multiple_separate_add_calls_accumulate(self, repo):
        repo.add([_make_meta(PROJECT_ID, 1, "c1")], [_make_chunk("c1", "A")])
        repo.add([_make_meta(PROJECT_ID, 2, "c2")], [_make_chunk("c2", "B")])
        assert repo.get_size() == 2


# ---------------------------------------------------------------------------
# get_sequence_number
# ---------------------------------------------------------------------------

class TestGetSequenceNumber:
    def test_returns_correct_int(self, repo):
        """Regression guard Bug 4.11: must return int, not tuple."""
        repo.add(
            [_make_meta(PROJECT_ID, 42, "c42")],
            [_make_chunk("c42", "Life")],
        )
        seq = repo.get_sequence_number("c42")
        assert seq == 42
        assert isinstance(seq, int)

    def test_returns_none_for_unknown_chunk(self, repo):
        result = repo.get_sequence_number("nonexistent")
        assert result is None

    def test_sequence_number_one(self, repo):
        repo.add([_make_meta(PROJECT_ID, 1, "first")], [_make_chunk("first", "x")])
        assert repo.get_sequence_number("first") == 1

    def test_sequence_number_large(self, repo):
        repo.add([_make_meta(PROJECT_ID, 99999, "big")], [_make_chunk("big", "x")])
        assert repo.get_sequence_number("big") == 99999


# ---------------------------------------------------------------------------
# get_n_chunks
# Known Bug 4.30: returns OLDEST n chunks (ORDER BY created_at ASC), not newest.
# Tests document current behaviour; when Bug 4.30 is fixed, the ordering
# assertions in test_get_n_chunks_returns_oldest_n will need inverting.
# ---------------------------------------------------------------------------

class TestGetNChunks:
    def test_get_returns_correct_count(self, repo):
        meta = [_make_meta(PROJECT_ID, i, f"c{i}") for i in range(1, 4)]
        chunks = [_make_chunk(f"c{i}", f"Msg {i}") for i in range(1, 4)]
        repo.add(meta, chunks)
        result = repo.get_n_chunks(2)
        assert len(result) == 2

    def test_get_more_than_available_returns_all(self, repo):
        repo.add(
            [_make_meta(PROJECT_ID, 1, "c1")],
            [_make_chunk("c1", "Only one")],
        )
        result = repo.get_n_chunks(100)
        assert len(result) == 1

    def test_get_zero_returns_empty(self, repo):
        repo.add(
            [_make_meta(PROJECT_ID, 1, "c1")],
            [_make_chunk("c1", "something")],
        )
        result = repo.get_n_chunks(0)
        assert result == []

    def test_get_n_chunks_returns_oldest_n_known_bug_4_30(self, repo):
        """
        Known Bug 4.30: get_n_chunks returns the OLDEST n entries (ascending
        created_at + LIMIT), not the most recent n.
        Documenting current behaviour as a regression guard.
        All messages share the same created_at so SQLite returns them in rowid
        (insertion) order — first n rows == oldest n.
        """
        meta = [_make_meta(PROJECT_ID, i, f"c{i}") for i in range(1, 6)]
        chunks = [_make_chunk(f"c{i}", f"Msg {i}") for i in range(1, 6)]
        repo.add(meta, chunks)
        result = repo.get_n_chunks(2)
        texts = [r[0] for r in result]
        # Current (buggy) behaviour: returns first 2 inserted rows
        assert "Msg 1" in texts
        assert "Msg 2" in texts

    def test_get_n_chunks_empty_db_returns_empty(self, repo):
        assert repo.get_n_chunks(5) == []


# ---------------------------------------------------------------------------
# get_ranged_chunks
# Note (Bug 4.31): ordered by f.created_at instead of f.sequence_number.
# Tests use a uniform created_at so SQLite insertion order is deterministic.
# ---------------------------------------------------------------------------

class TestGetRangedChunks:
    @pytest.fixture
    def filled_repo(self, repo):
        meta = [_make_meta(PROJECT_ID, i, f"c{i}") for i in range(1, 6)]
        chunks = [_make_chunk(f"c{i}", f"Text {i}") for i in range(1, 6)]
        repo.add(meta, chunks)
        return repo

    def test_range_2_to_4_returns_three(self, filled_repo):
        result = filled_repo.get_ranged_chunks(2, 4)
        assert len(result) == 3

    def test_range_inclusive_both_ends(self, filled_repo):
        result = filled_repo.get_ranged_chunks(1, 5)
        assert len(result) == 5

    def test_range_single_element(self, filled_repo):
        result = filled_repo.get_ranged_chunks(3, 3)
        assert len(result) == 1

    def test_range_out_of_bounds_returns_empty(self, filled_repo):
        result = filled_repo.get_ranged_chunks(10, 20)
        assert result == []

    def test_range_returns_correct_text_content(self, filled_repo):
        result = filled_repo.get_ranged_chunks(2, 3)
        texts = {r[0] for r in result}
        assert "Text 2" in texts
        assert "Text 3" in texts
        assert "Text 1" not in texts
        assert "Text 4" not in texts

    def test_range_start_equals_end_returns_one(self, filled_repo):
        result = filled_repo.get_ranged_chunks(5, 5)
        assert len(result) == 1
        assert result[0][0] == "Text 5"

    def test_reversed_range_returns_empty(self, filled_repo):
        """start > end: no rows match the WHERE clause."""
        result = filled_repo.get_ranged_chunks(5, 1)
        assert result == []


# ---------------------------------------------------------------------------
# get_sequence_after
# Note (Bug 4.31): ordered by f.created_at — uniform timestamp keeps tests stable.
# ---------------------------------------------------------------------------

class TestGetSequenceAfter:
    @pytest.fixture
    def filled_repo(self, repo):
        meta = [_make_meta(PROJECT_ID, i, f"c{i}") for i in range(1, 6)]
        chunks = [_make_chunk(f"c{i}", f"Text {i}") for i in range(1, 6)]
        repo.add(meta, chunks)
        return repo

    def test_after_2_returns_three_items(self, filled_repo):
        result = filled_repo.get_sequence_after(2)
        assert len(result) == 3

    def test_after_last_returns_empty(self, filled_repo):
        result = filled_repo.get_sequence_after(5)
        assert result == []

    def test_after_0_returns_all(self, filled_repo):
        result = filled_repo.get_sequence_after(0)
        assert len(result) == 5

    def test_after_returns_correct_text_content(self, filled_repo):
        result = filled_repo.get_sequence_after(3)
        texts = {r[0] for r in result}
        assert "Text 4" in texts
        assert "Text 5" in texts
        assert "Text 3" not in texts

    def test_after_negative_sequence_returns_all(self, filled_repo):
        result = filled_repo.get_sequence_after(-1)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# get_size
# ---------------------------------------------------------------------------

class TestGetSize:
    def test_get_size_returns_int(self, repo):
        """Regression guard: get_size() must return int, not a tuple."""
        repo.add(
            [_make_meta(PROJECT_ID, 1, "c1")],
            [_make_chunk("c1", "Hello")],
        )
        size = repo.get_size()
        assert size == 1
        assert isinstance(size, int)

    def test_empty_db_size_is_zero(self, repo):
        assert repo.get_size() == 0

    def test_size_matches_inserted_count(self, repo):
        n = 10
        meta = [_make_meta(PROJECT_ID, i, f"c{i}") for i in range(1, n + 1)]
        chunks = [_make_chunk(f"c{i}", f"t{i}") for i in range(1, n + 1)]
        repo.add(meta, chunks)
        assert repo.get_size() == n

    def test_size_after_multiple_separate_adds(self, repo):
        for i in range(1, 6):
            repo.add(
                [_make_meta(PROJECT_ID, i, f"c{i}")],
                [_make_chunk(f"c{i}", f"msg {i}")],
            )
        assert repo.get_size() == 5


# ---------------------------------------------------------------------------
# Stress: FullConversationRepository
# ---------------------------------------------------------------------------

class TestRepositoryStress:
    def test_five_hundred_sequential_adds(self, tmp_path):
        repo = FullConversationRepository(
            conversation_path=tmp_path,
            project_id=PROJECT_ID,
            project_name=PROJECT_NAME,
        )
        meta = [_make_meta(PROJECT_ID, i, f"c{i}") for i in range(1, 501)]
        chunks = [_make_chunk(f"c{i}", f"text {i}") for i in range(1, 501)]
        repo.add(meta, chunks)
        assert repo.get_size() == 500

    def test_fetch_all_preserves_order_at_scale(self, tmp_path):
        """Regression guard Bug 4.15: ORDER BY sequence_number must hold at scale."""
        repo = FullConversationRepository(
            conversation_path=tmp_path,
            project_id=PROJECT_ID,
            project_name=PROJECT_NAME,
        )
        n = 200
        meta = [_make_meta(PROJECT_ID, i, f"c{i}") for i in range(1, n + 1)]
        chunks = [_make_chunk(f"c{i}", str(i)) for i in range(1, n + 1)]
        repo.add(meta, chunks)
        result = repo.fetch_all()
        texts = [int(r[0]) for r in result]
        assert texts == list(range(1, n + 1))

    def test_range_query_at_scale(self, tmp_path):
        repo = FullConversationRepository(
            conversation_path=tmp_path,
            project_id=PROJECT_ID,
            project_name=PROJECT_NAME,
        )
        meta = [_make_meta(PROJECT_ID, i, f"c{i}") for i in range(1, 101)]
        chunks = [_make_chunk(f"c{i}", f"text {i}") for i in range(1, 101)]
        repo.add(meta, chunks)
        result = repo.get_ranged_chunks(51, 100)
        assert len(result) == 50

    def test_sequence_after_at_scale(self, tmp_path):
        repo = FullConversationRepository(
            conversation_path=tmp_path,
            project_id=PROJECT_ID,
            project_name=PROJECT_NAME,
        )
        meta = [_make_meta(PROJECT_ID, i, f"c{i}") for i in range(1, 201)]
        chunks = [_make_chunk(f"c{i}", f"text {i}") for i in range(1, 201)]
        repo.add(meta, chunks)
        result = repo.get_sequence_after(100)
        assert len(result) == 100

    def test_fetch_all_out_of_order_insertion_still_ordered(self, tmp_path):
        """Inserting in reverse seq order must still return in seq order via fetch_all."""
        repo = FullConversationRepository(
            conversation_path=tmp_path,
            project_id=PROJECT_ID,
            project_name=PROJECT_NAME,
        )
        meta = [_make_meta(PROJECT_ID, i, f"c{i}") for i in range(50, 0, -1)]
        chunks = [_make_chunk(f"c{i}", str(i)) for i in range(50, 0, -1)]
        repo.add(meta, chunks)
        result = repo.fetch_all()
        seqs = [int(r[0]) for r in result]
        assert seqs == list(range(1, 51))


# ---------------------------------------------------------------------------
# FullConversation bucket
# ---------------------------------------------------------------------------

@pytest.fixture
def bucket(tmp_path):
    return FullConversation(
        full_conversation_dir=tmp_path,
        project_id="proj_bucket",
        project_name="test_bucket",
    )


class TestFullConversationBucket:
    def test_append_and_get_full_conversation(self, bucket):
        meta = [_make_meta("proj_bucket", 1, "c1")]
        chunks = [_make_chunk("c1", "Hello from bucket")]
        bucket.append_chunks(meta, chunks)
        full = bucket.get_full_conversation()
        assert len(full) == 1
        assert full[0][0] == "Hello from bucket"

    def test_size_returns_int(self, bucket):
        meta = [_make_meta("proj_bucket", 1, "c1")]
        chunks = [_make_chunk("c1", "one")]
        bucket.append_chunks(meta, chunks)
        assert bucket.size() == 1
        assert isinstance(bucket.size(), int)

    def test_get_chunk_order_returns_int(self, bucket):
        meta = [_make_meta("proj_bucket", 7, "c7")]
        chunks = [_make_chunk("c7", "seven")]
        bucket.append_chunks(meta, chunks)
        order = bucket.get_chunk_order("c7")
        assert order == 7
        assert isinstance(order, int)

    def test_get_last_n_chunks(self, bucket):
        meta = [_make_meta("proj_bucket", i, f"c{i}") for i in range(1, 6)]
        chunks = [_make_chunk(f"c{i}", f"msg {i}") for i in range(1, 6)]
        bucket.append_chunks(meta, chunks)
        result = bucket.get_last_n_chunks(3)
        assert len(result) == 3

    def test_get_context_range(self, bucket):
        meta = [_make_meta("proj_bucket", i, f"c{i}") for i in range(1, 6)]
        chunks = [_make_chunk(f"c{i}", f"msg {i}") for i in range(1, 6)]
        bucket.append_chunks(meta, chunks)
        result = bucket.get_context(2, 4)
        assert len(result) == 3

    def test_get_context_text_content(self, bucket):
        meta = [_make_meta("proj_bucket", i, f"c{i}") for i in range(1, 4)]
        chunks = [_make_chunk(f"c{i}", f"word{i}") for i in range(1, 4)]
        bucket.append_chunks(meta, chunks)
        result = bucket.get_context(2, 3)
        texts = {r[0] for r in result}
        assert "word2" in texts
        assert "word3" in texts
        assert "word1" not in texts

    def test_get_conversation_since(self, bucket):
        meta = [_make_meta("proj_bucket", i, f"c{i}") for i in range(1, 6)]
        chunks = [_make_chunk(f"c{i}", f"msg {i}") for i in range(1, 6)]
        bucket.append_chunks(meta, chunks)
        result = bucket.get_conversation_since(2)
        assert len(result) == 3

    def test_get_conversation_since_text_content(self, bucket):
        meta = [_make_meta("proj_bucket", i, f"c{i}") for i in range(1, 4)]
        chunks = [_make_chunk(f"c{i}", f"item{i}") for i in range(1, 4)]
        bucket.append_chunks(meta, chunks)
        result = bucket.get_conversation_since(1)
        texts = {r[0] for r in result}
        assert "item2" in texts
        assert "item3" in texts
        assert "item1" not in texts

    def test_empty_bucket_size_is_zero(self, bucket):
        assert bucket.size() == 0

    def test_get_context_empty_range_returns_empty(self, bucket):
        meta = [_make_meta("proj_bucket", i, f"c{i}") for i in range(1, 4)]
        chunks = [_make_chunk(f"c{i}", f"x{i}") for i in range(1, 4)]
        bucket.append_chunks(meta, chunks)
        assert bucket.get_context(10, 20) == []

    def test_get_chunk_order_unknown_returns_none(self, bucket):
        assert bucket.get_chunk_order("no_such_chunk") is None

    def test_stress_five_hundred_via_bucket(self, tmp_path):
        b = FullConversation(
            full_conversation_dir=tmp_path,
            project_id="stress",
            project_name="stress_proj",
        )
        meta = [_make_meta("stress", i, f"c{i}") for i in range(1, 501)]
        chunks = [_make_chunk(f"c{i}", f"text {i}") for i in range(1, 501)]
        b.append_chunks(meta, chunks)
        assert b.size() == 500

    def test_full_conversation_ordering_at_scale(self, tmp_path):
        """fetch_all must respect sequence_number even at 300 records."""
        b = FullConversation(
            full_conversation_dir=tmp_path,
            project_id="scale",
            project_name="scale_proj",
        )
        n = 300
        meta = [_make_meta("scale", i, f"c{i}") for i in range(1, n + 1)]
        chunks = [_make_chunk(f"c{i}", str(i)) for i in range(1, n + 1)]
        b.append_chunks(meta, chunks)
        result = b.get_full_conversation()
        assert [int(r[0]) for r in result] == list(range(1, n + 1))
