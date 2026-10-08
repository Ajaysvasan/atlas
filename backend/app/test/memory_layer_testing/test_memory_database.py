"""The one database every memory-layer table lives in.

It owns no table — only the connection, the transaction discipline and the
order tables are created in. So these tests are about those three, and about
what happens when several owners share one connection: a failure in one must
not leave the connection unusable for the rest.
"""

import sqlite3
import threading

import pytest

from config import Config
from memory.memory_database import SCHEMA_VERSION, MemoryDatabase, Schema
from memory.memory_pool_exceptions import NewerMemorySchema


@pytest.fixture
def db(tmp_path):
    database = MemoryDatabase(tmp_path / "memory.db")
    yield database
    database.close()


def parent_schema(created=None):
    def create(cursor):
        if created is not None:
            created.append("parent")
        cursor.execute("create table if not exists parent(id text primary key)")
    return Schema("parent", create)


def child_schema(parent, created=None):
    def create(cursor):
        if created is not None:
            created.append("child")
        cursor.execute(
            "create table if not exists child("
            "id text primary key, parent_id text not null references parent(id))"
        )
    return Schema("child", create, requires=(parent,))


def other_connection(database):
    return sqlite3.connect(database.path)


class TestOpening:
    def test_the_file_is_created_where_asked(self, tmp_path):
        database = MemoryDatabase(tmp_path / "nested" / "dir" / "memory.db")
        assert database.path.exists()
        database.close()

    def test_it_runs_in_wal(self, db):
        assert db.journal_mode.lower() == "wal"

    def test_foreign_keys_are_enforced(self, db):
        with db.reading() as cursor:
            assert cursor.execute("pragma foreign_keys").fetchone()[0] == 1

    def test_a_busy_database_is_waited_for(self, db):
        with db.reading() as cursor:
            assert cursor.execute("pragma busy_timeout").fetchone()[0] >= 1000

    def test_a_new_database_is_stamped_with_the_schema_version(self, db):
        with db.reading() as cursor:
            assert cursor.execute("pragma user_version").fetchone()[0] == SCHEMA_VERSION

    def test_a_newer_schema_is_refused(self, tmp_path):
        """Older code writing into a newer layout could corrupt it."""
        path = tmp_path / "newer.db"
        with sqlite3.connect(path) as conn:
            conn.execute(f"pragma user_version = {SCHEMA_VERSION + 1}")
        with pytest.raises(NewerMemorySchema, match=str(SCHEMA_VERSION + 1)):
            MemoryDatabase(path)

    def test_the_default_path_is_read_when_called_not_when_imported(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Config, "MEMORY_DB", tmp_path / "late.db")
        database = MemoryDatabase()
        assert database.path == tmp_path / "late.db"
        database.close()


class TestCreatingTables:
    def test_a_schema_creates_its_tables(self, db):
        db.ensure(parent_schema())
        with db.reading() as cursor:
            names = {r[0] for r in cursor.execute("select name from sqlite_master")}
        assert "parent" in names

    def test_the_tables_a_schema_references_are_created_first(self, db):
        """Otherwise inserting into the child fails on a missing parent table,
        depending on which owner happened to be constructed first."""
        created = []
        parent = parent_schema(created)
        db.ensure(child_schema(parent, created))
        assert created == ["parent", "child"]

    def test_a_schema_is_created_once_per_database(self, db):
        created = []
        parent = parent_schema(created)
        db.ensure(parent)
        db.ensure(parent)
        db.ensure(child_schema(parent))
        assert created.count("parent") == 1

    def test_each_database_creates_its_own(self, tmp_path):
        created = []
        parent = parent_schema(created)
        first, second = MemoryDatabase(tmp_path / "a.db"), MemoryDatabase(tmp_path / "b.db")
        first.ensure(parent)
        second.ensure(parent)
        assert created == ["parent", "parent"]
        first.close()
        second.close()

    def test_a_reference_to_a_missing_parent_row_is_refused(self, db):
        parent = parent_schema()
        db.ensure(child_schema(parent))
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            with db.writing() as cursor:
                cursor.execute("insert into child values ('c', 'nobody')")


class TestWriting:
    @pytest.fixture
    def table(self, db):
        db.ensure(parent_schema())
        return db

    def rows(self, database):
        with other_connection(database) as conn:
            return sorted(r[0] for r in conn.execute("select id from parent"))

    def test_a_write_is_committed(self, table):
        with table.writing() as cursor:
            cursor.execute("insert into parent values ('a')")
        assert self.rows(table) == ["a"]

    def test_a_failed_write_leaves_nothing(self, table):
        with pytest.raises(RuntimeError):
            with table.writing() as cursor:
                cursor.execute("insert into parent values ('a')")
                raise RuntimeError("halfway")
        assert self.rows(table) == []

    def test_a_nested_write_commits_with_the_outer_one_only(self, table):
        with table.writing() as outer:
            with table.writing() as inner:
                inner.execute("insert into parent values ('inner')")
            assert self.rows(table) == []
            outer.execute("insert into parent values ('outer')")
        assert self.rows(table) == ["inner", "outer"]

    def test_a_caught_nested_failure_undoes_only_the_nested_part(self, table):
        """A savepoint, not a flag: the outer transaction keeps its own work."""
        with table.writing() as outer:
            outer.execute("insert into parent values ('kept')")
            try:
                with table.writing() as inner:
                    inner.execute("insert into parent values ('undone')")
                    raise ValueError("inner step failed")
            except ValueError:
                pass
            outer.execute("insert into parent values ('after')")
        assert self.rows(table) == ["after", "kept"]

    def test_an_uncaught_nested_failure_undoes_everything(self, table):
        with pytest.raises(ValueError):
            with table.writing() as outer:
                outer.execute("insert into parent values ('outer')")
                with table.writing() as inner:
                    inner.execute("insert into parent values ('inner')")
                    raise ValueError("boom")
        assert self.rows(table) == []

    def test_a_failed_commit_does_not_leave_the_connection_stuck(self, db):
        """A deferred constraint fails at COMMIT; without a rollback the shared
        connection would stay inside that transaction for every later caller."""
        db.ensure(parent_schema())
        with db.writing() as cursor:
            cursor.execute(
                "create table deferred(id text, parent_id text references parent(id) "
                "deferrable initially deferred)"
            )
        with pytest.raises(sqlite3.IntegrityError):
            with db.writing() as cursor:
                cursor.execute("insert into deferred values ('d', 'nobody')")
        with db.writing() as cursor:
            cursor.execute("insert into parent values ('still works')")
        assert self.rows(db) == ["still works"]

    def test_the_depth_is_restored_after_a_failure(self, table):
        with pytest.raises(RuntimeError):
            with table.writing():
                raise RuntimeError("x")
        with table.writing() as cursor:
            cursor.execute("insert into parent values ('a')")
        assert self.rows(table) == ["a"]


class TestOneConnectionForEveryOwner:
    def test_concurrent_writers_are_serialised(self, db):
        """One connection used from many threads: the lock is what makes it
        correct, and BEGIN inside an open transaction is what breaks it."""
        db.ensure(parent_schema())
        errors = []

        def write(worker):
            try:
                for i in range(50):
                    with db.writing() as cursor:
                        cursor.execute("insert into parent values (?)", (f"{worker}-{i}",))
            except BaseException as error:
                errors.append(error)

        threads = [threading.Thread(target=write, args=(w,)) for w in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == []
        with db.reading() as cursor:
            assert cursor.execute("select count(*) from parent").fetchone()[0] == 400


class TestTheSharedInstance:
    def test_one_instance_per_path(self, tmp_path):
        assert MemoryDatabase.shared(tmp_path / "x.db") is MemoryDatabase.shared(tmp_path / "x.db")

    def test_the_same_file_by_another_spelling_is_the_same_instance(self, tmp_path):
        (tmp_path / "sub").mkdir()
        assert MemoryDatabase.shared(tmp_path / "x.db") is \
            MemoryDatabase.shared(tmp_path / "sub" / ".." / "x.db")

    def test_different_files_get_different_instances(self, tmp_path):
        assert MemoryDatabase.shared(tmp_path / "a.db") is not MemoryDatabase.shared(tmp_path / "b.db")

    def test_the_default_follows_the_configured_path(self):
        assert MemoryDatabase.shared().path == Config.MEMORY_DB.resolve()

    def test_closing_the_shared_instances_closes_them(self, tmp_path):
        database = MemoryDatabase.shared(tmp_path / "x.db")
        MemoryDatabase.close_shared()
        assert database.connection is None

    def test_after_closing_a_fresh_instance_is_opened(self, tmp_path):
        first = MemoryDatabase.shared(tmp_path / "x.db")
        first.close()
        second = MemoryDatabase.shared(tmp_path / "x.db")
        assert second is not first and second.connection is not None


class TestClosing:
    def test_a_closed_database_refuses_work(self, db):
        db.close()
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            with db.reading():
                pass

    def test_closing_twice_is_harmless(self, db):
        db.close()
        db.close()


class TestTheSuiteNeverTouchesTheRealDatabase:
    def test_the_configured_path_is_a_test_file(self, tmp_path_factory):
        """The root conftest points it at a per-test file; this is the guard
        that the redirect is in force."""
        real = Config.ABS_PATH / "data" / "memory" / "memory_layer" / "memory_layer.db"
        assert Config.MEMORY_DB != real
        assert str(tmp_path_factory.getbasetemp()) in str(Config.MEMORY_DB)


class TestAnOlderFile:
    def test_a_rollback_journal_database_is_switched_to_wal(self, tmp_path):
        path = tmp_path / "old.db"
        with sqlite3.connect(path) as conn:
            conn.execute("pragma journal_mode = delete").fetchone()
            conn.execute("create table t(i)")
        database = MemoryDatabase(path)
        assert database.journal_mode.lower() == "wal"
        database.close()
