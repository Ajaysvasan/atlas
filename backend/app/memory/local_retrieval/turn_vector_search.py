"""Exact nearest-neighbour search over a project's turn vectors."""

import threading
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from numpy import float32, ndarray

from config import get_logger
from knowledge_sufficiency.similarity import cosine_scores
from memory.local_retrieval.turn_scope import TurnScope
from memory.memory_database import MemoryDatabase
from memory.topic_pool.project_pool.conversation_pool.conversation_data_management.conversationVectorManager import (
    ConversationVectorManager,
)
from memory.topic_pool.project_pool.conversation_pool.conversation_data_management.conversationVectorMetaManager import (
    turn_vector_ids,
)
from retrieval_layer.models import VECTOR, QueryPlan, ScoredId, as_ranked

logger = get_logger(__name__)


class TurnVectorSearch:
    """The dense half of local retrieval: cosine against every summarised turn in scope."""

    def __init__(
        self,
        project_id: str,
        project_name: str,
        database: MemoryDatabase | str | Path | None = None,
        vector_store=None,
    ) -> None:
        self.project_id = project_id
        self.project_name = project_name
        self.database = MemoryDatabase.of(database)
        self.__vector_store = vector_store
        self.__owns_store = vector_store is None
        self.__vectors: Dict[int, ndarray] = {}
        self.__lock = threading.Lock()

    def __load(self, vector_ids: Sequence[int]) -> None:
        """Fetch the vectors not yet held. A turn's vector never changes, so none is fetched twice."""
        with self.__lock:
            missing = [vector_id for vector_id in vector_ids if vector_id not in self.__vectors]
            if not missing:
                return
            if self.__vector_store is None:
                self.__vector_store = ConversationVectorManager(
                    self.project_name, self.project_id
                )
            try:
                found = self.__vector_store.vectors_for(missing)
            except Exception:
                # A connection that failed once stays failed; the next search
                # has to open a new one or the dense half never comes back.
                self.__drop_store()
                raise
            for vector_id, vector in found.items():
                self.__vectors[int(vector_id)] = np.asarray(vector, dtype=float32)

    def __drop_store(self) -> None:
        if not self.__owns_store or self.__vector_store is None:
            return
        try:
            self.__vector_store.close()
        except Exception as error:
            logger.debug("Closing the failed vector store raised: %s", error)
        self.__vector_store = None

    def search(self, plan: QueryPlan, k: int, scope: TurnScope) -> List[ScoredId]:
        """The k turn vectors in scope nearest the query, nearest first."""
        if k < 1:
            return []
        vector_ids = [
            row.vector_id
            for row in turn_vector_ids(scope.project_id, scope.conversation_only, self.database)
            if scope.admits(row.conversation_id, row.sequence_number)
        ]
        if not vector_ids:
            return []
        try:
            self.__load(vector_ids)
        except Exception as error:
            logger.warning(
                "Turn vectors could not be read (%s); keyword search carries on", error
            )
            return []
        held = [vector_id for vector_id in vector_ids if vector_id in self.__vectors]
        if not held:
            return []
        matrix = np.stack([self.__vectors[vector_id] for vector_id in held])
        scores = cosine_scores(np.asarray(plan.vector, dtype=float32), matrix)
        nearest = np.argpartition(-scores, k - 1)[:k] if len(held) > k else range(len(held))
        ranked = as_ranked([(held[i], float(scores[i])) for i in nearest], VECTOR)
        logger.debug("Vector search returned %d of %d turn(s)", len(ranked), len(held))
        return ranked

    def close(self) -> None:
        """Release the vector store this search opened. One handed in stays open."""
        self.__drop_store()
