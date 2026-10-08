"""The project snapshot: incremental, and about the project rather than a chat.

Two decisions drive these tests. It summarises the *previous project snapshot*
plus whatever conversation summaries are new since, so cost tracks what changed
rather than the project's size. And it reads across every conversation in the
project, ignoring conversation_id, because the result is meant to describe the
project and not any one conversation.
"""

from unittest import mock

import numpy as np
import pytest

from memory.topic_pool.project_pool.conversation_pool.conversation_data_management.conversationVectorMetaManager import (
    ConversationVectorMetaDataRepository,
)
from memory.topic_pool.project_pool.project_snapshot import (
    PROJECT_SNAPSHOT_SYSTEM,
    ProjectSnapshot,
    render_prompt,
)

PROJECT_ID = "proj_1"
DB = "memory.db"


@pytest.fixture
def summariser():
    """Records every user prompt and returns a numbered description."""
    calls = []

    def summarise(system, user):
        calls.append(user)
        return f"description v{len(calls)}"

    summarise.calls = calls
    return summarise


def _embed(_text):
    return np.ones(128, dtype=np.float32)


@pytest.fixture(autouse=True)
def _projects_registered(tmp_path, seed_projects):
    """Snapshot rows reference their project, so the ones used here are registered."""
    seed_projects(tmp_path / DB, "topic", PROJECT_ID, "proj_2")


@pytest.fixture
def conversations(tmp_path):
    """Two conversations writing snapshots into one project's database."""
    a = ConversationVectorMetaDataRepository(PROJECT_ID, "conv_A", tmp_path / DB)
    b = ConversationVectorMetaDataRepository(PROJECT_ID, "conv_B", tmp_path / DB)
    yield a, b
    a.close()
    b.close()


@pytest.fixture
def project(tmp_path, conversations):
    a, _ = conversations
    snapshot = ProjectSnapshot(
        PROJECT_ID, "Proj", meta_repo=a, embed=_embed,
    )
    # The vectors live in PostgreSQL; this covers the flow, not the store.
    snapshot.snap_shot._vector_manager = mock.MagicMock()
    yield snapshot
    snapshot.close()


class TestThePrompt:
    def test_the_first_one_says_there_is_no_description_yet(self):
        prompt = render_prompt(None, ["a conversation summary"])
        assert "no description yet" in prompt
        assert "a conversation summary" in prompt

    def test_a_later_one_carries_the_previous_description(self):
        prompt = render_prompt("what we knew", ["what is new"])
        assert "what we knew" in prompt
        assert "what is new" in prompt

    def test_the_system_half_asks_for_the_project_not_the_conversation(self):
        """Otherwise the result reads as a chat recap rather than a description."""
        assert "not the conversations" in PROJECT_SNAPSHOT_SYSTEM
        assert "no speaker labels" in PROJECT_SNAPSHOT_SYSTEM


class TestNothingToDo:
    def test_an_untouched_project_takes_no_snapshot(self, project, summariser):
        assert project.take(summariser) is None

    def test_a_second_call_with_nothing_new_takes_none(self, project, conversations,
                                                       summariser):
        a, _ = conversations
        a.insert_cumulative_vector_meta_data(101, "A on auth", "2026-10-01",
                                             PROJECT_ID, 5)
        assert project.take(summariser) is not None
        assert project.take(summariser) is None

    def test_the_summariser_is_not_called_when_nothing_is_new(self, project,
                                                              summariser):
        project.take(summariser)
        assert summariser.calls == []


class TestItSpansEveryConversation:
    def test_pending_covers_both_conversations(self, project, conversations):
        a, b = conversations
        a.insert_cumulative_vector_meta_data(101, "A on auth", "2026-10-01",
                                             PROJECT_ID, 5)
        b.insert_cumulative_vector_meta_data(201, "B on caching", "2026-10-01",
                                             PROJECT_ID, 5)

        assert [text for _, _, text in project.pending()] == [
            "A on auth", "B on caching",
        ]

    def test_the_first_prompt_carries_both(self, project, conversations, summariser):
        a, b = conversations
        a.insert_cumulative_vector_meta_data(101, "A on auth", "2026-10-01",
                                             PROJECT_ID, 5)
        b.insert_cumulative_vector_meta_data(201, "B on caching", "2026-10-01",
                                             PROJECT_ID, 5)
        project.take(summariser)

        assert "A on auth" in summariser.calls[0]
        assert "B on caching" in summariser.calls[0]

    def test_another_project_is_not_pulled_in(self, project, tmp_path,
                                              conversations):
        a, _ = conversations
        other = ConversationVectorMetaDataRepository("proj_2", "conv_X", tmp_path / DB)
        other.insert_cumulative_vector_meta_data(301, "someone else", "2026-10-01",
                                                 "proj_2", 5)
        a.insert_cumulative_vector_meta_data(101, "ours", "2026-10-01",
                                             PROJECT_ID, 5)

        assert [text for _, _, text in project.pending()] == ["ours"]
        other.close()


class TestItIsIncremental:
    def test_the_watermark_advances(self, project, conversations, summariser):
        a, _ = conversations
        a.insert_cumulative_vector_meta_data(101, "first", "2026-10-01",
                                             PROJECT_ID, 5)
        project.take(summariser)
        assert project.snapshot_repo.last_seq_included() == 1

    def test_a_later_prompt_carries_only_what_is_new(self, project, conversations,
                                                     summariser):
        a, b = conversations
        a.insert_cumulative_vector_meta_data(101, "A on auth", "2026-10-01",
                                             PROJECT_ID, 5)
        project.take(summariser)
        b.insert_cumulative_vector_meta_data(201, "B on caching", "2026-10-02",
                                             PROJECT_ID, 5)
        project.take(summariser)

        second = summariser.calls[1]
        assert "B on caching" in second
        assert "A on auth" not in second

    def test_a_later_prompt_carries_the_previous_description(self, project,
                                                             conversations,
                                                             summariser):
        a, b = conversations
        a.insert_cumulative_vector_meta_data(101, "A on auth", "2026-10-01",
                                             PROJECT_ID, 5)
        project.take(summariser)
        b.insert_cumulative_vector_meta_data(201, "B on caching", "2026-10-02",
                                             PROJECT_ID, 5)
        project.take(summariser)

        assert "description v1" in summariser.calls[1]

    def test_each_take_appends_to_the_chain(self, project, conversations,
                                            summariser):
        a, b = conversations
        a.insert_cumulative_vector_meta_data(101, "one", "2026-10-01",
                                             PROJECT_ID, 5)
        first = project.take(summariser)
        b.insert_cumulative_vector_meta_data(201, "two", "2026-10-02",
                                             PROJECT_ID, 5)
        second = project.take(summariser)

        assert first != second
        assert [r.summary for r in project.snapshot_repo.history()] == [
            "description v1", "description v2",
        ]


class TestAnEmptySummary:
    def test_it_does_not_advance_the_watermark(self, tmp_path, conversations):
        """Advancing would drop those summaries from every future snapshot."""
        a, _ = conversations
        a.insert_cumulative_vector_meta_data(101, "A on auth", "2026-10-01",
                                             PROJECT_ID, 5)
        snapshot = ProjectSnapshot(
            PROJECT_ID, "Proj", meta_repo=a, embed=_embed,
        )
        snapshot.snap_shot._vector_manager = mock.MagicMock()

        assert snapshot.take(lambda system, user: "   ") is None
        assert snapshot.snapshot_repo.last_seq_included() == 0
        assert [text for _, _, text in snapshot.pending()] == ["A on auth"]
        snapshot.close()


class TestTheVector:
    def test_it_is_stored_under_the_snapshot_id(self, project, conversations,
                                                summariser):
        """One id for both stores, so a row needs no lookup to find its vector."""
        a, _ = conversations
        a.insert_cumulative_vector_meta_data(101, "one", "2026-10-01",
                                             PROJECT_ID, 5)
        snapshot_id = project.take(summariser)

        stored_id, _vector = project.snap_shot._vector_manager.insert.call_args[0]
        assert stored_id == snapshot_id
