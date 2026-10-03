"""Bug 4.69: an id that scopes rows must not be empty or absent.

`conversation_id` is written into `not null` columns and every reader filters on
it. `None` reached SQLite and failed with a constraint error naming the column
rather than the caller; `""` was accepted silently and partitioned nothing.
"""

import pytest

from memory.identifiers import require_identifier
from memory.memory_pool_exceptions import InvalidIdentifier
from memory.snapshot import SnapShot
from memory.topic_pool.project_pool.conversation_pool.conversation_data_management.conversationVectorMetaManager import (
    ConversationVectorMetaDataRepository,
)
from memory.topic_pool.project_pool.conversation_pool.full_conversation_bucket import (
    FullConversation,
)
from memory.topic_pool.project_pool.conversation_pool.fullconversation_repository.fullconversation_repository import (
    FullConversationRepository,
)

REJECTED = ["", "   ", "\t\n", None, 0, 123, [], {}, object()]


class TestTheValidator:
    @pytest.mark.parametrize("value", REJECTED)
    def test_it_rejects(self, value):
        with pytest.raises(InvalidIdentifier):
            require_identifier(value, "conversation_id")

    def test_it_accepts_a_real_id(self):
        assert require_identifier("conv_1", "conversation_id") == "conv_1"

    def test_it_strips_surrounding_whitespace(self):
        """Otherwise ' c1' and 'c1' would be two conversations."""
        assert require_identifier("  conv_1  ", "conversation_id") == "conv_1"

    def test_the_message_names_the_field_and_shows_the_value(self):
        with pytest.raises(InvalidIdentifier) as caught:
            require_identifier("", "conversation_id")
        assert "conversation_id" in str(caught.value)
        assert repr("") in str(caught.value)


class TestTheConversationLayerRefusesABadId:
    """Each of these writes or filters on conversation_id, so each must check."""

    @pytest.mark.parametrize("value", ["", "   ", None])
    def test_the_repository(self, tmp_path, value):
        with pytest.raises(InvalidIdentifier):
            FullConversationRepository(
                conversation_path=tmp_path, project_id="p",
                project_name="n", conversation_id=value,
            )

    @pytest.mark.parametrize("value", ["", "   ", None])
    def test_the_bucket(self, tmp_path, value):
        """A pass-through, but it must still fail at construction."""
        with pytest.raises(InvalidIdentifier):
            FullConversation(
                full_conversation_dir=tmp_path, project_id="p",
                project_name="n", conversation_id=value,
            )

    @pytest.mark.parametrize("value", ["", "   ", None])
    def test_the_metadata_repository(self, tmp_path, value):
        with pytest.raises(InvalidIdentifier):
            ConversationVectorMetaDataRepository(tmp_path, "p", value)

    @pytest.mark.parametrize("value", ["", "   ", None])
    def test_the_snapshot(self, tmp_path, value):
        with pytest.raises(InvalidIdentifier):
            SnapShot(
                conversation_dir=tmp_path, project_id="p",
                project_name="n", conversation_id=value,
            )


class TestAGoodIdStillWorks:
    def test_the_repository_accepts_one(self, tmp_path):
        repo = FullConversationRepository(
            conversation_path=tmp_path, project_id="p",
            project_name="n", conversation_id="conv_1",
        )
        assert repo.conversation_id == "conv_1"
        assert repo.append_turns([("user", "hello")]) == [1]

    def test_a_padded_id_is_stored_stripped(self, tmp_path):
        """So the value written to the column matches what readers filter on."""
        repo = FullConversationRepository(
            conversation_path=tmp_path, project_id="p",
            project_name="n", conversation_id="  conv_1  ",
        )
        repo.append_turns([("user", "hello")])

        same = FullConversationRepository(
            conversation_path=tmp_path, project_id="p",
            project_name="n", conversation_id="conv_1",
        )
        assert [t.text for t in same.get_all_turns()] == ["hello"]
