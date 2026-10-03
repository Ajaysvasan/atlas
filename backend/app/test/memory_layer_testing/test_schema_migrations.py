"""Bug 4.65: a database written before `conversation_id` must still open.

The schema changes arrived as `CREATE TABLE IF NOT EXISTS`, which does nothing
to a table that already exists, so every pre-change database failed its next
insert. `full_conversation` also needs a new primary key, and SQLite cannot
alter one in place — hence a rebuild rather than an ALTER.

Version 1 introduced `conversation_id` on the turn tables; version 2 introduced
it, plus the monotonic `seq`, on `cumulative_vector_meta_data`.
"""

import sqlite3

import pytest

from memory.topic_pool.project_pool.conversation_pool.conversation_data_management.conversationVectorMetaManager import (
    ConversationVectorMetaDataRepository,
)
from memory.topic_pool.project_pool.conversation_pool.fullconversation_repository.fullconversation_repository import (
    FullConversationRepository,
)
from memory.topic_pool.project_pool.conversation_pool.schema_migrations import (
    LEGACY_CONVERSATION_ID,
    SCHEMA_VERSION,
    migrate,
)

PROJECT_ID = "p1"

_OLD_SCHEMA = """
    create table summary_chunks(
        chunk_id text primary key, chunk text not null,
        created_at date not null, chunker_type text not null);
    create table full_conversation(
        project_id text not null, sequence_number int primary key,
        chunk_id text not null, role text not null, created_at date not null,
        foreign key (chunk_id) references summary_chunks (chunk_id));
    create table cumulative_vector_meta_data(
        cumulative_vector_id integer primary key,
        cumulative_summary text not null, created_at date not null,
        project_id text not null, len_of_the_summary integer not null);
"""


def _old_database(tmp_path, turns=3):
    """A database exactly as the code before these changes left it."""
    path = tmp_path / f"{PROJECT_ID}_conversation.db"
    conn = sqlite3.connect(path)
    conn.executescript(_OLD_SCHEMA)
    conn.executemany(
        "insert into summary_chunks values (?,?,?,?)",
        [(f"c{i}", f"old turn {i}", "2026-01-01", "turn") for i in range(1, turns + 1)],
    )
    conn.executemany(
        "insert into full_conversation values (?,?,?,?,?)",
        [(PROJECT_ID, i, f"c{i}", "user", "2026-01-01") for i in range(1, turns + 1)],
    )
    conn.executemany(
        "insert into cumulative_vector_meta_data values (?,?,?,?,?)",
        [(30, "c", "2026-08-01T00:00:03", PROJECT_ID, 1),
         (10, "a", "2026-08-01T00:00:01", PROJECT_ID, 1),
         (20, "b", "2026-08-01T00:00:02", PROJECT_ID, 1)],
    )
    conn.commit()
    conn.close()
    return path


def _repo(tmp_path, conversation_id):
    return FullConversationRepository(
        conversation_path=tmp_path, project_id=PROJECT_ID,
        project_name="n", conversation_id=conversation_id,
    )


def _columns(path, table):
    with sqlite3.connect(path) as conn:
        return [row[1] for row in conn.execute(f"pragma table_info({table})")]


def test_the_legacy_id_is_pinned_to_its_literal_value():
    """Asserted against the literal on purpose.

    This string is written into real databases. Changing it does not migrate
    anything — it orphans every row a previous migration assigned, because the
    readers' `conversation_id = ?` filters would no longer match them. Every
    other test here refers to the constant, so without this one the value could
    be changed and they would all still pass.
    """
    assert LEGACY_CONVERSATION_ID == "legacy"


class TestAnOldDatabaseOpens:
    def test_opening_it_is_enough(self, tmp_path):
        """No migration command to remember: the repository runs it."""
        path = _old_database(tmp_path)
        assert "conversation_id" not in _columns(path, "full_conversation")

        _repo(tmp_path, "conv_new")

        assert "conversation_id" in _columns(path, "full_conversation")
        assert "conversation_id" in _columns(path, "summary_chunks")

    def test_the_old_turns_survive(self, tmp_path):
        _old_database(tmp_path, turns=3)
        legacy = _repo(tmp_path, LEGACY_CONVERSATION_ID)

        assert [t.text for t in legacy.get_all_turns()] == [
            "old turn 1", "old turn 2", "old turn 3",
        ]

    def test_the_old_turns_become_one_conversation(self, tmp_path):
        """They were one conversation all along; the column just had no name."""
        path = _old_database(tmp_path)
        _repo(tmp_path, "conv_new")

        with sqlite3.connect(path) as conn:
            ids = {r[0] for r in conn.execute(
                "select distinct conversation_id from full_conversation"
            )}
        assert ids == {LEGACY_CONVERSATION_ID}

    def test_a_new_conversation_can_reuse_the_old_sequence_numbers(self, tmp_path):
        """The point of the composite key: legacy holds 1..3, the new one starts at 1."""
        _old_database(tmp_path, turns=3)
        fresh = _repo(tmp_path, "conv_new")

        assert fresh.append_turns([("user", "brand new")]) == [1]
        assert [t.text for t in fresh.get_all_turns()] == ["brand new"]

    def test_the_two_conversations_stay_separate(self, tmp_path):
        _old_database(tmp_path, turns=2)
        legacy = _repo(tmp_path, LEGACY_CONVERSATION_ID)
        fresh = _repo(tmp_path, "conv_new")
        fresh.append_turns([("user", "brand new")])

        assert [t.text for t in legacy.get_all_turns()] == ["old turn 1", "old turn 2"]
        assert [t.text for t in fresh.get_all_turns()] == ["brand new"]

    def test_no_foreign_key_is_left_dangling(self, tmp_path):
        path = _old_database(tmp_path)
        _repo(tmp_path, "conv_new")

        with sqlite3.connect(path) as conn:
            assert conn.execute("pragma foreign_key_check;").fetchall() == []


class TestTheSnapshotRowsGetSeq:
    def test_the_column_arrives(self, tmp_path):
        path = _old_database(tmp_path)
        _repo(tmp_path, "conv_new")

        columns = _columns(path, "cumulative_vector_meta_data")
        assert "seq" in columns
        assert "conversation_id" in columns

    def test_seq_is_backfilled_in_the_order_the_rows_were_already_read_in(
        self, tmp_path
    ):
        """So the existing snapshot ordering carries over rather than being invented."""
        path = _old_database(tmp_path)
        _repo(tmp_path, "conv_new")

        with sqlite3.connect(path) as conn:
            pairs = conn.execute(
                "select cumulative_vector_id, seq from cumulative_vector_meta_data "
                "order by seq"
            ).fetchall()
        assert pairs == [(10, 1), (20, 2), (30, 3)]

    def test_the_rows_are_still_readable(self, tmp_path):
        _old_database(tmp_path)
        _repo(tmp_path, "conv_new")
        meta = ConversationVectorMetaDataRepository(
            tmp_path, PROJECT_ID, LEGACY_CONVERSATION_ID
        )

        assert meta.get_cumulative_vector_meta_data_ids() == [(10,), (20,), (30,)]
        assert meta.get_highest_snapshot_seq() == 3
        meta.close()


class TestItRunsOnceAndOnceOnly:
    def test_the_version_is_stamped(self, tmp_path):
        path = _old_database(tmp_path)
        _repo(tmp_path, "conv_new")

        with sqlite3.connect(path) as conn:
            assert conn.execute("pragma user_version;").fetchone()[0] == SCHEMA_VERSION

    def test_a_second_open_does_not_migrate_again(self, tmp_path):
        """Without the guard the rebuild would reset every row to legacy."""
        _old_database(tmp_path, turns=2)
        fresh = _repo(tmp_path, "conv_new")
        fresh.append_turns([("user", "written after the migration")])

        _repo(tmp_path, "conv_new")

        assert [t.text for t in fresh.get_all_turns()] == [
            "written after the migration"
        ]

    def test_migrate_is_idempotent_when_called_directly(self, tmp_path):
        path = _old_database(tmp_path)
        assert migrate(path) == SCHEMA_VERSION
        assert migrate(path) == SCHEMA_VERSION

    def test_the_stamped_version_is_authoritative(self, tmp_path):
        """user_version wins over inspecting the columns.

        Pins the short-circuit itself: the other tests here pass with or without
        it, because the 'does this table already have the column' check
        independently prevents a second rebuild. That check cannot tell version 1
        from version 2, which is what the stamp is for — so a database claiming
        to be current is left alone even when its tables look old.
        """
        path = _old_database(tmp_path)
        conn = sqlite3.connect(path)
        conn.execute(f"pragma user_version = {SCHEMA_VERSION};")
        conn.commit()
        conn.close()

        assert migrate(path) == SCHEMA_VERSION
        assert "conversation_id" not in _columns(path, "full_conversation")

    def test_a_fresh_database_is_stamped_without_a_rebuild(self, tmp_path):
        repo = _repo(tmp_path, "conv_new")
        repo.append_turns([("user", "hello")])

        path = tmp_path / f"{PROJECT_ID}_conversation.db"
        with sqlite3.connect(path) as conn:
            assert conn.execute("pragma user_version;").fetchone()[0] == SCHEMA_VERSION
        assert [t.text for t in repo.get_all_turns()] == ["hello"]


class TestTheMigratedShapeMatchesAFreshOne:
    def test_the_columns_are_in_the_same_order(self, tmp_path):
        """Positional INSERTs exist, so column order has to agree, not just the set."""
        migrated_dir = tmp_path / "migrated"
        migrated_dir.mkdir()
        _old_database(migrated_dir)
        _repo(migrated_dir, "conv_new")

        fresh_dir = tmp_path / "fresh"
        fresh_dir.mkdir()
        _repo(fresh_dir, "conv_new")
        ConversationVectorMetaDataRepository(fresh_dir, PROJECT_ID, "conv_new").close()

        for table in ("summary_chunks", "full_conversation",
                      "cumulative_vector_meta_data"):
            assert _columns(migrated_dir / f"{PROJECT_ID}_conversation.db", table) == \
                   _columns(fresh_dir / f"{PROJECT_ID}_conversation.db", table)

    def test_the_primary_key_is_the_composite_one(self, tmp_path):
        path = _old_database(tmp_path)
        _repo(tmp_path, "conv_new")

        with sqlite3.connect(path) as conn:
            key = [r[1] for r in conn.execute("pragma table_info(full_conversation)")
                   if r[5]]
        assert key == ["conversation_id", "sequence_number"]

    def test_the_chunk_index_survives_the_rebuild(self, tmp_path):
        """The rebuild drops the table, so the index has to be recreated."""
        path = _old_database(tmp_path)
        _repo(tmp_path, "conv_new")

        with sqlite3.connect(path) as conn:
            names = {r[0] for r in conn.execute(
                "select name from sqlite_master where type='index'"
            )}
        assert "idx_full_conversation_chunk" in names
