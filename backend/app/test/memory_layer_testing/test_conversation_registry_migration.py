"""Moving the SQLite conversation registry into PostgreSQL loses nothing and runs once."""

import sqlite3

import pytest

from memory.conversation_registry import ConversationRecord, migrate_from_sqlite


def old_registry(path, rows):
    with sqlite3.connect(path) as db:
        db.execute("create table conversations(conversation_id text primary key, user_id text not null, "
                   "conversation_name text not null, created_at text not null, "
                   "unique (user_id, conversation_name))")
        db.executemany("insert into conversations values (?, ?, ?, ?)", rows)


ROWS = [("c1", "alice", "PostgreSQL MVCC", "2026-11-09T06:15:21.389318+00:00"),
        ("c2", "alice", "WAL", "2026-11-09T06:45:30.862713+00:00"),
        ("c3", "bob", "PostgreSQL MVCC", "2026-11-09T06:49:49.665010+00:00")]


def test_every_row_is_copied_verified_and_the_old_table_kept_as_a_backup(conversation_registry, tmp_path):
    path = tmp_path / "memory.db"
    old_registry(path, ROWS)
    report = migrate_from_sqlite(conversation_registry, path)
    assert (report.found, report.copied, report.already_present, report.renamed) == (3, 3, 0, ())
    for row in ROWS:
        assert conversation_registry.get(row[0], row[1]) == ConversationRecord(*row)
    with sqlite3.connect(path) as db:
        assert {r[0] for r in db.execute("select name from sqlite_master where type='table'")} == {
            "conversations_migrated"}


def test_running_it_again_does_nothing(conversation_registry, tmp_path):
    path = tmp_path / "memory.db"
    old_registry(path, ROWS)
    migrate_from_sqlite(conversation_registry, path)
    assert migrate_from_sqlite(conversation_registry, path) is None
    assert conversation_registry.count() == 3


def test_an_interrupted_migration_resumes_without_duplicates(conversation_registry, tmp_path):
    """The copy committed but the rename never happened: the rerun finds the rows and finishes."""
    path = tmp_path / "memory.db"
    old_registry(path, ROWS)
    conversation_registry.import_records([ConversationRecord(*ROWS[0])])
    report = migrate_from_sqlite(conversation_registry, path)
    assert (report.copied, report.already_present) == (2, 1)
    assert conversation_registry.count() == 3


def test_a_name_already_taken_here_is_suffixed_and_reported(conversation_registry, tmp_path):
    conversation_registry.create("alice", "WAL")
    path = tmp_path / "memory.db"
    old_registry(path, ROWS)
    report = migrate_from_sqlite(conversation_registry, path)
    assert report.renamed == (("c2", "WAL", "WAL (2)"),)
    assert conversation_registry.get("c2", "alice").conversation_name == "WAL (2)"


def test_nothing_to_migrate(conversation_registry, tmp_path):
    assert migrate_from_sqlite(conversation_registry, tmp_path / "absent.db") is None
    with sqlite3.connect(tmp_path / "empty.db") as db:
        db.execute("create table other(x)")
    assert migrate_from_sqlite(conversation_registry, tmp_path / "empty.db") is None


def test_a_failed_copy_keeps_the_sqlite_table(conversation_registry, tmp_path, monkeypatch):
    path = tmp_path / "memory.db"
    old_registry(path, ROWS)

    def fail(records):
        raise ConnectionError("PostgreSQL went away")

    monkeypatch.setattr(conversation_registry, "import_records", fail)
    with pytest.raises(ConnectionError):
        migrate_from_sqlite(conversation_registry, path)
    with sqlite3.connect(path) as db:
        assert db.execute("select count(*) from conversations").fetchone()[0] == 3
