"""`MemoryMappingHandler`: (conversation_id, user_id) -> topic, project, snapshot.

A row is created when a conversation starts (`populate_conversation_id`),
filled in once routing has resolved a topic and project
(`insert_into_mapping_table`), and read back by `search`. The latest project
snapshot is not stored here: it is read from the project's snapshot chain on
every search. The cache it replaced had to be rewritten by hand whenever a
project moved on.

Regression guards for Bugs 4.73-4.83. The constructor takes no `conversation_id`
and no `query`: neither was ever stored, and every method names the conversation
it acts on, so one handler serves the whole table.
"""

import sqlite3

import pytest

from memory.memory_mapping_handler import MemoryMappingHandler
from memory.memory_pool_exceptions import (
    InvalidIdentifier,
    ProjectInAnotherTopic,
    ProjectNotFound,
)
from memory.topic_pool.project_pool.project_data_repo.project_snapshot_repo import (
    ProjectSnapshotRepository,
)

USER = "user_1"
CONVERSATION = "conv_1"


@pytest.fixture
def db(tmp_path, seed_projects):
    path = tmp_path / "memory.db"
    seed_projects(path, "topic_1", "project_1")
    seed_projects(path, "topic_2", "project_2")
    return path


@pytest.fixture
def handler(db):
    h = MemoryMappingHandler(db)
    yield h
    h.close()


def _rows(handler):
    with sqlite3.connect(handler.database.path) as conn:
        return conn.execute("select * from memory_mapping_table").fetchall()


def route(handler, conversation=CONVERSATION, user=USER, topic="topic_1",
          project="project_1"):
    handler.insert_into_mapping_table(
        conversation_id=conversation, user_id=user, topic_id=topic,
        project_id=project, created_at="2026-10-04",
    )


class TestTheSchema:
    def test_the_table_is_created_on_construction(self, handler):
        with sqlite3.connect(handler.database.path) as conn:
            tables = {r[0] for r in conn.execute(
                "select name from sqlite_master where type='table'"
            )}
        assert "memory_mapping_table" in tables

    def test_a_conversation_appears_at_most_once(self, handler):
        """The mapping answers 'which project is this conversation in'.

        Two rows for one conversation make that question ambiguous, and `search`
        picks one of them arbitrarily.
        """
        handler.populate_conversation_id(CONVERSATION, USER)
        with pytest.raises(sqlite3.IntegrityError):
            handler.populate_conversation_id(CONVERSATION, USER)

    def test_no_snapshot_pointer_is_stored(self, handler):
        """Read from the chain instead, so it cannot go stale."""
        with sqlite3.connect(handler.database.path) as conn:
            columns = [r[1] for r in conn.execute("pragma table_info(memory_mapping_table)")]
        assert "latest_project_snapshot_id" not in columns


class TestStartingAConversation:
    def test_it_creates_one_row(self, handler):
        handler.populate_conversation_id(CONVERSATION, USER)
        assert len(_rows(handler)) == 1

    def test_the_routing_columns_start_empty(self, handler):
        handler.populate_conversation_id(CONVERSATION, USER)
        assert handler.search(CONVERSATION, USER) == (None, None, None)


class TestFillingInTheRouting:
    def test_the_topic_and_project_read_back(self, handler):
        handler.populate_conversation_id(CONVERSATION, USER)
        route(handler)
        assert handler.search(CONVERSATION, USER)[:2] == ("topic_1", "project_1")

    def test_it_touches_only_the_conversation_it_names(self, handler):
        """Bug 4.78 drops the WHERE values, so the update would hit every row."""
        handler.populate_conversation_id(CONVERSATION, USER)
        handler.populate_conversation_id("conv_2", USER)
        route(handler)
        assert handler.search("conv_2", USER) == (None, None, None)

    @pytest.mark.parametrize("field", ["topic_id", "project_id"])
    def test_routing_needs_both_a_topic_and_a_project(self, handler, field):
        handler.populate_conversation_id(CONVERSATION, USER)
        values = dict(conversation_id=CONVERSATION, user_id=USER, topic_id="topic_1",
                      project_id="project_1", created_at="now")
        values[field] = "  "
        with pytest.raises(InvalidIdentifier):
            handler.insert_into_mapping_table(**values)


class TestTheSnapshotComesFromTheChain:
    def test_a_project_with_no_snapshot_has_none(self, handler):
        handler.populate_conversation_id(CONVERSATION, USER)
        route(handler)
        assert handler.search(CONVERSATION, USER).latest_project_snapshot_id is None

    def test_the_latest_snapshot_is_found(self, handler, db):
        handler.populate_conversation_id(CONVERSATION, USER)
        route(handler)
        snapshot = ProjectSnapshotRepository("project_1", db).add_snapshot(
            "about one", last_seq_included=1)
        assert handler.search(CONVERSATION, USER).latest_project_snapshot_id == snapshot

    def test_a_newer_snapshot_is_seen_without_touching_the_mapping(self, handler, db):
        """What the cache needed an explicit update call for."""
        handler.populate_conversation_id(CONVERSATION, USER)
        route(handler)
        chain = ProjectSnapshotRepository("project_1", db)
        chain.add_snapshot("first", last_seq_included=1)
        newer = chain.add_snapshot("second", last_seq_included=2)
        assert handler.search(CONVERSATION, USER).latest_project_snapshot_id == newer

    def test_another_projects_snapshot_does_not_leak(self, handler, db):
        handler.populate_conversation_id(CONVERSATION, USER)
        route(handler)
        ProjectSnapshotRepository("project_2", db).add_snapshot("elsewhere", 5)
        assert handler.search(CONVERSATION, USER).latest_project_snapshot_id is None

    def test_the_id_is_the_integer_the_snapshot_is_stored_under(self, handler, db):
        """The cached column was text against an integer key."""
        handler.populate_conversation_id(CONVERSATION, USER)
        route(handler)
        ProjectSnapshotRepository("project_1", db).add_snapshot("about one", 1)
        assert isinstance(handler.search(CONVERSATION, USER).latest_project_snapshot_id, int)


class TestTheDatabaseKeepsTheRelationships:
    def test_routing_to_an_unregistered_project_is_refused(self, handler):
        handler.populate_conversation_id(CONVERSATION, USER)
        with pytest.raises(ProjectNotFound):
            route(handler, project="no_such_project")
        assert handler.search(CONVERSATION, USER) == (None, None, None)

    def test_routing_to_a_project_under_another_topic_is_refused(self, handler):
        handler.populate_conversation_id(CONVERSATION, USER)
        with pytest.raises(ProjectInAnotherTopic) as caught:
            route(handler, topic="topic_1", project="project_2")
        assert caught.value.actual_topic_id == "topic_2"

    def test_the_refusal_keeps_the_database_error_as_its_cause(self, handler):
        handler.populate_conversation_id(CONVERSATION, USER)
        with pytest.raises(ProjectNotFound) as caught:
            route(handler, project="no_such_project")
        assert isinstance(caught.value.__cause__, sqlite3.IntegrityError)

    def test_a_half_routed_row_is_refused(self, handler):
        """A null in the pair skips the foreign key; the CHECK is what holds."""
        handler.populate_conversation_id(CONVERSATION, USER)
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            with handler.database.writing() as cursor:
                cursor.execute(
                    "update memory_mapping_table set topic_id = 'topic_1' "
                    "where conversation_id = ?", (CONVERSATION,))

    def test_moving_a_project_moves_its_conversations(self, handler):
        """ON UPDATE CASCADE carries the routing record with the project."""
        handler.populate_conversation_id(CONVERSATION, USER)
        route(handler)
        with handler.database.writing() as cursor:
            cursor.execute(
                "update project_table set topic_id = 'topic_2' where project_id = 'project_1'")
        assert handler.search(CONVERSATION, USER)[:2] == ("topic_2", "project_1")


class TestSearching:
    def test_an_unknown_conversation_reads_back_as_none(self, handler):
        """`return row[0] if not None else None` raises IndexError instead.

        `not None` is `True` whatever `row` holds, so the guard never fires and
        the empty list is indexed.
        """
        handler.populate_conversation_id(CONVERSATION, USER)
        assert handler.search("no_such_conversation", USER) is None

    def test_an_empty_table_reads_back_as_none(self, handler):
        assert handler.search(CONVERSATION, USER) is None

    def test_a_user_cannot_read_another_user_s_conversation(self, handler):
        handler.populate_conversation_id(CONVERSATION, USER)
        assert handler.search(CONVERSATION, "user_2") is None


class TestTheConstructor:
    def test_the_database_is_optional(self):
        from config import Config

        assert MemoryMappingHandler().database.path == Config.MEMORY_DB.resolve()

    def test_closing_leaves_the_shared_database_open(self, handler):
        handler.close()
        assert handler.database.connection is not None

    @pytest.mark.parametrize("value", ["", "   ", None])
    def test_a_blank_conversation_id_is_refused(self, handler, value):
        with pytest.raises(InvalidIdentifier):
            handler.populate_conversation_id(value, USER)
