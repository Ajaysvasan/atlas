"""The conversation registry against the real PostgreSQL: types, the unique constraint, races, migration."""

import sqlite3
import threading
import uuid
from datetime import datetime

import pytest

pytestmark = pytest.mark.live


@pytest.fixture
def user():
    from memory.conversation_registry import ConversationRegistry

    name = f"__live_test__{uuid.uuid4().hex}"
    yield name
    registry = ConversationRegistry()
    with registry.conn.cursor() as cursor:
        cursor.execute("delete from conversations where user_id = %s", (name,))
    registry.conn.commit()
    registry.close()


@pytest.fixture
def registry():
    from memory.conversation_registry import ConversationRegistry

    registry = ConversationRegistry()
    yield registry
    registry.close()


def test_it_lives_in_the_database_that_holds_the_vectors(registry):
    with registry.conn.cursor() as cursor:
        cursor.execute("select to_regclass('conversations'), to_regclass('vectors'), "
                       "to_regclass('conversation_table')")
        assert all(cursor.fetchone())


def test_a_record_round_trips_with_a_timezone_aware_start(registry, user):
    created = registry.create(user, "Raft Leader Election")
    assert registry.get(created.conversation_id, user) == created
    assert datetime.fromisoformat(created.created_at).tzinfo is not None
    assert registry.get(created.conversation_id, "someone-else") is None


def test_names_take_the_next_free_suffix(registry, user):
    names = [registry.create(user, "Topic").conversation_name for _ in range(3)]
    assert names == ["Topic", "Topic (2)", "Topic (3)"]
    assert registry.create(user, "Topic (2)").conversation_name == "Topic (4)"


def test_like_wildcards_in_names_are_literal(registry, user):
    registry.create(user, "100% done")
    registry.create(user, "100X done (2)")
    assert registry.create(user, "100% done").conversation_name == "100% done (2)"


def test_the_database_refuses_a_duplicate_name_for_one_user(registry, user):
    import psycopg

    registry.create(user, "Dup")
    with pytest.raises(psycopg.errors.UniqueViolation):
        with registry.conn.cursor() as cursor:
            cursor.execute("insert into conversations values (%s, %s, 'Dup', now())",
                           (uuid.uuid4().hex, user))
    registry.conn.rollback()


def test_concurrent_writers_on_separate_connections_get_distinct_names(user):
    from memory.conversation_registry import ConversationRegistry

    names, errors = [], []
    barrier = threading.Barrier(8)

    def create():
        registry = ConversationRegistry()
        try:
            barrier.wait(timeout=10)
            names.append(registry.create(user, "Same Name").conversation_name)
        except Exception as error:  # pragma: no cover - reported by the assertion
            errors.append(error)
        finally:
            registry.close()

    threads = [threading.Thread(target=create) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert sorted(names) == sorted(["Same Name"] + [f"Same Name ({n})" for n in range(2, 9)])


def test_listing_is_per_user_newest_first(registry, user):
    a = registry.create(user, "A")
    b = registry.create(user, "B")
    assert [r.conversation_id for r in registry.list(user)] == [b.conversation_id, a.conversation_id]
    assert registry.find_by_prefix(user, a.conversation_id[:8]) == [a]


def test_sqlite_rows_migrate_once_with_ids_names_and_starts_intact(registry, user, tmp_path):
    from memory.conversation_registry import migrate_from_sqlite

    path = tmp_path / "memory.db"
    rows = [(uuid.uuid4().hex, user, "Old One", "2026-11-01T10:00:00.000001+00:00"),
            (uuid.uuid4().hex, user, "Old Two", "2026-11-02T10:00:00.000002+00:00")]
    with sqlite3.connect(path) as db:
        db.execute("create table conversations(conversation_id text primary key, user_id text, "
                   "conversation_name text, created_at text)")
        db.executemany("insert into conversations values (?, ?, ?, ?)", rows)

    report = migrate_from_sqlite(registry, path)
    assert (report.found, report.copied, report.already_present) == (2, 2, 0)
    for conversation_id, owner, name, created in rows:
        stored = registry.get(conversation_id, owner)
        assert stored.conversation_name == name
        assert datetime.fromisoformat(stored.created_at) == datetime.fromisoformat(created)
    with sqlite3.connect(path) as db:
        tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
        assert tables == {"conversations_migrated"}
        assert db.execute("select count(*) from conversations_migrated").fetchone()[0] == 2
    assert migrate_from_sqlite(registry, path) is None
