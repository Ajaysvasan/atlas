"""Tests for TopicManager and TopicPoolMetaHandler.

SQLite is real; there is nothing else to stub at this layer. The three
regressions guarded here all failed on the first run of topic_manager.py.
"""

import pathlib
import sqlite3
import subprocess
import sys
import threading

import pytest

from memory.topic_pool.topic_manager import TopicManager
from memory.topic_pool.topic_pool_repo.topic_pool_meta_handler import (
    TopicPoolMetaHandler,
)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "topic_db" / "topic.sql"


@pytest.fixture
def manager(db_path):
    m = TopicManager("retrieval", "how does routing work?", db_path)
    yield m
    m.close()


class TestConstruction:
    @pytest.mark.parametrize("bad", ["", None])
    def test_an_empty_topic_is_refused(self, db_path, bad):
        with pytest.raises(ValueError):
            TopicManager(bad, "a query", db_path)

    @pytest.mark.parametrize("bad", ["", None])
    def test_an_empty_query_is_refused(self, db_path, bad):
        with pytest.raises(ValueError):
            TopicManager("retrieval", bad, db_path)

    def test_the_supplied_database_is_the_one_used(self, db_path):
        """The path used to be accepted and then dropped on the floor, so every
        caller silently shared the default registry."""
        with TopicManager("retrieval", "q", db_path) as m:
            m.create_new_topic()
        assert db_path.exists()
        with sqlite3.connect(db_path) as conn:
            assert conn.execute("select count(*) from topics_mapping_table").fetchone()[0] == 1

    def test_the_default_path_is_used_when_none_is_given(self, tmp_path, monkeypatch):
        from memory.topic_pool.topic_pool_repo import topic_pool_meta_handler as module

        monkeypatch.setattr(module.Config, "DATA_DIR", str(tmp_path))
        handler = TopicPoolMetaHandler(None)
        assert handler.topic_db_path.is_relative_to(tmp_path)
        handler.close()


class TestCreate:
    def test_a_new_topic_is_created_and_readable(self, manager):
        """Regression: the INSERT named four columns and bound three values, so
        every create died with '3 values for 4 columns'."""
        manager.create_new_topic()
        assert isinstance(manager.get_topic_id(), str)

    def test_every_column_is_written(self, manager, db_path):
        manager.create_new_topic()
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "select topic_id, topic_name, created_at, is_active "
                "from topics_mapping_table"
            ).fetchone()
        assert row[1] == "retrieval"
        assert row[3] == "t"
        assert all(value is not None for value in row)

    def test_creating_twice_is_refused(self, manager):
        manager.create_new_topic()
        with pytest.raises(Exception, match="already exists"):
            manager.create_new_topic()

    def test_two_topics_get_different_ids(self, db_path):
        with TopicManager("alpha", "q", db_path) as a, TopicManager("beta", "q", db_path) as b:
            a.create_new_topic()
            b.create_new_topic()
            assert a.get_topic_id() != b.get_topic_id()

    def test_created_at_is_iso_utc(self, manager, db_path):
        from datetime import datetime

        manager.create_new_topic()
        with sqlite3.connect(db_path) as conn:
            stamp = conn.execute("select created_at from topics_mapping_table").fetchone()[0]
        assert datetime.fromisoformat(stamp).utcoffset().total_seconds() == 0


class TestGetTopicId:
    def test_an_absent_topic_raises(self, manager):
        with pytest.raises(Exception, match="doesn't exists"):
            manager.get_topic_id()

    def test_the_id_is_stable_across_reads(self, manager):
        manager.create_new_topic()
        assert manager.get_topic_id() == manager.get_topic_id()

    def test_a_reopened_manager_sees_the_topic(self, db_path):
        with TopicManager("retrieval", "q", db_path) as first:
            first.create_new_topic()
            expected = first.get_topic_id()
        with TopicManager("retrieval", "q", db_path) as second:
            assert second.get_topic_id() == expected


class TestSoftDelete:
    def test_it_hides_the_topic(self, manager):
        """Regression: the UPDATE set a `soft_delete` column that does not
        exist, so every delete died with 'no such column'."""
        manager.create_new_topic()
        manager.soft_delete()
        with pytest.raises(Exception, match="doesn't exists"):
            manager.get_topic_id()

    def test_the_row_is_kept(self, manager, db_path):
        """Soft: the history stays, only the flag moves."""
        manager.create_new_topic()
        manager.soft_delete()
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "select topic_name, is_active from topics_mapping_table"
            ).fetchall()
        assert rows == [("retrieval", "f")]

    def test_deleting_an_absent_topic_raises(self, manager):
        with pytest.raises(Exception, match="doesn't exists"):
            manager.soft_delete()

    def test_the_name_can_be_used_again_afterwards(self, manager):
        manager.create_new_topic()
        first = manager.get_topic_id()
        manager.soft_delete()
        manager.create_new_topic()
        assert manager.get_topic_id() != first

    def test_deleting_an_unknown_id_at_the_repository_raises(self, db_path):
        """An UPDATE matching nothing is not an error to SQLite."""
        handler = TopicPoolMetaHandler(db_path)
        with pytest.raises(ValueError):
            handler.soft_delete("no_such_id")
        handler.close()


class TestLifecycle:
    def test_close_is_repeatable(self, db_path):
        handler = TopicPoolMetaHandler(db_path)
        handler.close()
        handler.close()

    def test_a_write_from_another_thread_works(self, db_path):
        """The connection is opened with check_same_thread=False. Without it the
        RLock and the cursor context managers protect something no other thread
        could reach: SQLite rejects the call outright."""
        handler = TopicPoolMetaHandler(db_path)
        errors = []

        def writer():
            try:
                handler.create_new_topic("from a thread", "id1", None)
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")

        thread = threading.Thread(target=writer)
        thread.start()
        thread.join()
        assert errors == []
        assert handler.get_topic_id("from a thread") == "id1"
        handler.close()

    def test_concurrent_writers_all_land(self, db_path):
        handler = TopicPoolMetaHandler(db_path)
        errors = []
        barrier = threading.Barrier(8)

        def writer(index):
            try:
                barrier.wait()
                handler.create_new_topic(f"topic {index}", f"id{index}", None)
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == []
        with sqlite3.connect(db_path) as conn:
            assert conn.execute("select count(*) from topics_mapping_table").fetchone()[0] == 8
        handler.close()

    def test_close_does_not_crash_a_write_in_flight(self, tmp_path):
        """Closing the connection under a writing thread does not raise — it
        takes the interpreter down, so it cannot be caught in process. The race
        runs in a subprocess and the assertion is on how that process died.
        """
        script = tmp_path / "race.py"
        script.write_text(
            "import sys, threading, time\n"
            f"sys.path.insert(0, {str(pathlib.Path.cwd())!r})\n"
            "from memory.topic_pool.topic_pool_repo.topic_pool_meta_handler import "
            "TopicPoolMetaHandler\n"
            f"handler = TopicPoolMetaHandler({str(tmp_path / 'race.sql')!r})\n"
            "stop = False\n"
            "def writer():\n"
            "    index = 0\n"
            "    while not stop:\n"
            "        try:\n"
            "            handler.create_new_topic(f't{index}', f'id{index}', None)\n"
            "            index += 1\n"
            "        except Exception:\n"
            "            return\n"
            "t = threading.Thread(target=writer, daemon=True)\n"
            "t.start()\n"
            "time.sleep(0.05)\n"
            "handler.close()\n"
            "time.sleep(0.1)\n"
            "stop = True\n"
        )
        for _ in range(4):
            finished = subprocess.run(
                [sys.executable, str(script)], capture_output=True, timeout=60
            )
            assert finished.returncode >= 0, (
                f"the interpreter crashed with signal {-finished.returncode}: "
                "close() released the connection while a statement was running"
            )

    def test_the_module_imports_without_psycopg(self):
        """A stray `from psycopg import cursor` made the whole topic layer
        unimportable wherever psycopg is absent — including the test runner."""
        import ast
        import pathlib

        source = pathlib.Path(
            "memory/topic_pool/topic_pool_repo/topic_pool_meta_handler.py"
        ).read_text()
        imported = {
            node.module.split(".")[0]
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert "psycopg" not in imported
