"""`MemoryMappingHandler`: (conversation_id, user_id) -> topic, project, snapshot.

The contract these tests describe, read off the module: a row is created when a
conversation starts (`populate_conversation_id`), filled in once routing has
resolved a topic and project (`insert_into_mapping_table`), has its snapshot
pointer overwritten as the project's description moves on
(`update_latest_project_snapshot_id`), and is read back by `search`.

Regression guards for Bugs 4.73-4.83. The constructor takes no `conversation_id`
and no `query`: neither was ever stored, and every method names the conversation
it acts on, so one handler serves the whole table.
"""

import sqlite3

import pytest

from memory.memory_mapping_handler import MemoryMapping, MemoryMappingHandler

USER = "user_1"
CONVERSATION = "conv_1"


@pytest.fixture
def handler(tmp_path):
    h = MemoryMappingHandler(tmp_path / "memory_mapping.sql")
    yield h
    close = getattr(h, "close", None)
    if close is not None:
        close()


def _rows(tmp_path):
    with sqlite3.connect(tmp_path / "memory_mapping.sql") as conn:
        return conn.execute("select * from memory_mapping_table").fetchall()


class TestTheSchema:
    def test_the_table_is_created_on_construction(self, handler, tmp_path):
        with sqlite3.connect(tmp_path / "memory_mapping.sql") as conn:
            tables = {r[0] for r in conn.execute(
                "select name from sqlite_master where type='table'"
            )}
        assert "memory_mapping_table" in tables

    def test_a_conversation_appears_at_most_once(self, handler, tmp_path):
        """The mapping answers 'which project is this conversation in'.

        Two rows for one conversation make that question ambiguous, and `search`
        picks one of them arbitrarily.
        """
        handler.populate_conversation_id(CONVERSATION, USER)
        with pytest.raises(sqlite3.IntegrityError):
            handler.populate_conversation_id(CONVERSATION, USER)


class TestStartingAConversation:
    def test_it_creates_one_row(self, handler, tmp_path):
        handler.populate_conversation_id(CONVERSATION, USER)
        assert len(_rows(tmp_path)) == 1

    def test_the_routing_columns_start_empty(self, handler):
        handler.populate_conversation_id(CONVERSATION, USER)
        assert handler.search(CONVERSATION, USER) == (None, None, None)


class TestFillingInTheRouting:
    def test_the_topic_project_and_snapshot_read_back(self, handler):
        handler.populate_conversation_id(CONVERSATION, USER)
        handler.insert_into_mapping_table(
            conversation_id=CONVERSATION, user_id=USER, topic_id="topic_1",
            project_id="project_1", new_latest_project_snapshot_id="snap_1",
            created_at="2026-10-04",
        )
        assert handler.search(CONVERSATION, USER) == (
            "topic_1", "project_1", "snap_1",
        )

    def test_it_touches_only_the_conversation_it_names(self, handler):
        """Bug 4.78 drops the WHERE values, so the update would hit every row."""
        handler.populate_conversation_id(CONVERSATION, USER)
        handler.populate_conversation_id("conv_2", USER)
        handler.insert_into_mapping_table(
            conversation_id=CONVERSATION, user_id=USER, topic_id="topic_1",
            project_id="project_1", new_latest_project_snapshot_id="snap_1",
            created_at="2026-10-04",
        )
        assert handler.search("conv_2", USER) == (None, None, None)


class TestMovingTheSnapshotPointer:
    def test_the_newest_snapshot_replaces_the_previous_one(self, handler):
        handler.populate_conversation_id(CONVERSATION, USER)
        handler.insert_into_mapping_table(
            conversation_id=CONVERSATION, user_id=USER, topic_id="topic_1",
            project_id="project_1", new_latest_project_snapshot_id="snap_1",
            created_at="2026-10-04",
        )
        handler.update_latest_project_snapshot_id(USER, "project_1", "snap_2")

        assert handler.search(CONVERSATION, USER)[2] == "snap_2"

    def test_another_user_on_the_same_project_is_untouched(self, handler):
        handler.populate_conversation_id(CONVERSATION, USER)
        handler.populate_conversation_id("conv_other", "user_2")
        handler.insert_into_mapping_table(
            conversation_id="conv_other", user_id="user_2", topic_id="topic_1",
            project_id="project_1", new_latest_project_snapshot_id="snap_1",
            created_at="2026-10-04",
        )
        handler.update_latest_project_snapshot_id(USER, "project_1", "snap_2")

        assert handler.search("conv_other", "user_2")[2] == "snap_1"


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
    def test_the_database_path_is_optional(self, tmp_path, monkeypatch):
        """The None-handling in __init__ implies it is, but the signature does not."""
        monkeypatch.chdir(tmp_path)
        MemoryMappingHandler()

    def test_the_connection_can_be_released(self, tmp_path):
        h = MemoryMappingHandler(tmp_path / "memory_mapping.sql")
        h.close()

    @pytest.mark.parametrize("value", ["", "   ", None])
    def test_a_blank_conversation_id_is_refused(self, tmp_path, value):
        from memory.memory_pool_exceptions import InvalidIdentifier
        with pytest.raises(InvalidIdentifier):
            MemoryMappingHandler(tmp_path / "memory_mapping.sql").populate_conversation_id(
                value, USER
            )
