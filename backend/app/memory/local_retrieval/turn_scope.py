"""Which turns a local retrieval may return."""

from typing import NamedTuple

from memory.identifiers import require_identifier
from memory.snapshot import CONVERSATION, PROJECT, SCOPES
from retrieval_layer.retrieval_exceptions import InvalidRetrievalSetting

__all__ = ["CONVERSATION", "PROJECT", "SCOPES", "TurnScope"]


class TurnScope(NamedTuple):
    """The conversation or project searched, less the current conversation's turns from `before_sequence` on."""

    project_id: str
    conversation_id: str
    scope: str = CONVERSATION
    before_sequence: int | None = None

    @classmethod
    def of(
        cls,
        project_id: str,
        conversation_id: str,
        scope: str = CONVERSATION,
        before_sequence: int | None = None,
    ) -> "TurnScope":
        """A validated scope."""
        if scope not in SCOPES:
            raise InvalidRetrievalSetting(
                "scope", scope, "one of " + ", ".join(sorted(SCOPES))
            )
        if before_sequence is not None and (
            isinstance(before_sequence, bool)
            or not isinstance(before_sequence, int)
            or before_sequence < 1
        ):
            raise InvalidRetrievalSetting(
                "before_sequence", before_sequence, "a sequence number of at least 1, or None"
            )
        return cls(
            require_identifier(project_id, "project_id"),
            require_identifier(conversation_id, "conversation_id"),
            scope,
            before_sequence,
        )

    @property
    def conversation_only(self) -> str | None:
        """The one conversation searched, or None when it is the whole project."""
        return self.conversation_id if self.scope == CONVERSATION else None

    @property
    def key(self) -> str:
        return f"{self.scope}|{self.project_id}|{self.conversation_id}|{self.before_sequence}"

    def admits(self, conversation_id: str, sequence_number: int) -> bool:
        """Whether the turn at this position may be returned."""
        if self.scope == CONVERSATION and conversation_id != self.conversation_id:
            return False
        return not (
            self.before_sequence is not None
            and conversation_id == self.conversation_id
            and sequence_number >= self.before_sequence
        )
