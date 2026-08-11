from pathlib import Path
from typing import List, Tuple

from .fullconversation_repository.fullconversation_repository import (
    FullConversationRepository,
)


class FullConversation:

    def __init__(
        self, full_conversation_dir: str | Path, project_id: str, project_name: str
    ) -> None:
        self.fullConversationoRep = FullConversationRepository(
            project_id=project_id,
            conversation_path=full_conversation_dir,
            project_name=project_name,
        )

    def append_chunks(
        self,
        full_conversaton_meta_datas: List[Tuple[str, int, str, str]],
        chunks: List[Tuple[str, str, str, str]],
    ) -> None:
        self.fullConversationoRep.add(full_conversaton_meta_datas, chunks)

    def get_chunk_order(self, chunk_id: str) -> int:
        return self.fullConversationoRep.get_sequence_number(chunk_id)

    def get_last_n_chunks(self, n: int):
        return self.fullConversationoRep.get_n_chunks(n)

    def get_context(self, start: int, end: int):
        return self.fullConversationoRep.get_ranged_chunks(start, end)

    def get_conversation_since(self, sequence: int):
        return self.fullConversationoRep.get_sequence_after(sequence)

    def get_full_conversation(self):
        return self.fullConversationoRep.fetch_all()

    def size(self):
        return self.fullConversationoRep.get_size()
