from pathlib import Path
from typing import Iterable, List, Tuple

from memory.memory_pool_exceptions import EmptyTurnContent, InvalidRole

from .fullconversation_repository.fullconversation_repository import (
    FullConversationRepository,
)

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"

# role is a free TEXT column in SQLite. Constraining it here keeps a typo
# ("assistent") from silently becoming a fourth role that no later filter or
# prompt-builder will ever match.
VALID_ROLES = frozenset({ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM})


class FullConversation:

    def __init__(
        self, full_conversation_dir: str | Path, project_id: str, project_name: str
    ) -> None:
        self.fullConversationoRep = FullConversationRepository(
            project_id=project_id,
            conversation_path=full_conversation_dir,
            project_name=project_name,
        )

    @staticmethod
    def __validate_turn(role: str, text: str) -> Tuple[str, str]:
        if not isinstance(role, str):
            raise InvalidRole(role, VALID_ROLES)
        normalized_role = role.strip().lower()
        if normalized_role not in VALID_ROLES:
            raise InvalidRole(role, VALID_ROLES)
        if not isinstance(text, str) or not text.strip():
            raise EmptyTurnContent(normalized_role)
        return normalized_role, text

    def append_turn(self, role: str, text: str) -> int:
        """Append one conversation turn. Returns its sequence number."""
        return self.append_turns([(role, text)])[0]

    def append_turns(self, turns: Iterable[Tuple[str, str]]) -> List[int]:
        """Append several turns in one transaction, in the order given.

        Every turn is validated before any is written, so a bad role partway
        through a batch cannot leave the first half committed.
        """
        validated = [self.__validate_turn(role, text) for role, text in turns]
        return self.fullConversationoRep.append_turns(validated)

    def next_sequence_number(self) -> int:
        return self.fullConversationoRep.next_sequence_number()

    def append_chunks(
        self,
        full_conversaton_meta_datas: List[Tuple[str, int, str, str]],
        chunks: List[Tuple[str, str, str, str]],
    ) -> None:
        """Low-level insert with caller-supplied ids, sequences and timestamps.

        Prefer append_turn/append_turns; this stays for callers that already
        hold fully-formed rows (bulk import, tests, migrations).
        """
        self.fullConversationoRep.add(full_conversaton_meta_datas, chunks)

    def get_chunk_order(self, chunk_id: str) -> int:
        return self.fullConversationoRep.get_sequence_number(chunk_id)

    def get_last_n_chunks(self, n: int):
        return self.fullConversationoRep.get_n_chunks(n)

    def get_context(self, start: int, end: int):
        return self.fullConversationoRep.get_ranged_chunks(start, end)

    def get_context_rows(self, start: int, end: int):
        """[(chunk_id, chunk, created_at, chunker_type)] — what a snapshot records."""
        return self.fullConversationoRep.get_ranged_rows(start, end)

    def get_conversation_since(self, sequence: int):
        return self.fullConversationoRep.get_sequence_after(sequence)

    def get_full_conversation(self):
        return self.fullConversationoRep.fetch_all()

    def size(self):
        return self.fullConversationoRep.get_size()
