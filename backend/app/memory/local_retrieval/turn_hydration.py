"""Turning local search results back into the turns they came from."""

from pathlib import Path
from typing import List, Sequence, Tuple

from config import get_logger
from memory.local_retrieval.turn_keyword_search import TurnKeywordSearch
from memory.memory_database import MemoryDatabase
from memory.topic_pool.project_pool.conversation_pool.fullconversation_repository.fullconversation_repository import (
    turns_for_chunks,
)
from retrieval_layer.models import Passage, ScoredId

logger = get_logger(__name__)


class TurnHydration:
    """Reads a page of results' turns in two queries, whatever the page size, keeping who said each."""

    def __init__(
        self,
        index: TurnKeywordSearch,
        database: MemoryDatabase | str | Path | None = None,
    ) -> None:
        self.index = index
        self.database = MemoryDatabase.of(database)

    def hydrate(self, scored: Sequence[ScoredId]) -> Tuple[List[Passage], int]:
        """Passages in the order given, and how many keys had no turn."""
        if not scored:
            return [], 0
        chunk_of = self.index.chunk_ids_for([item.vector_id for item in scored])
        turns = {
            turn.chunk_id: turn
            for turn in turns_for_chunks(sorted(set(chunk_of.values())), self.database)
        }
        passages: List[Passage] = []
        for item in scored:
            turn = turns.get(chunk_of.get(item.vector_id))
            if turn is None:
                continue
            passages.append(Passage(
                vector_id=item.vector_id,
                chunk_id=turn.chunk_id,
                text=turn.text,
                score=item.score,
                source=item.source,
                document_id=turn.conversation_id,
                start_offset=turn.sequence_number,
                end_offset=turn.sequence_number,
                role=turn.role,
            ))
        dropped = len(scored) - len(passages)
        if dropped:
            logger.debug("%d search result(s) had no turn behind them", dropped)
        return passages, dropped
