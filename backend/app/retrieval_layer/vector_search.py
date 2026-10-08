"""Approximate nearest neighbour search over the DiskANN index."""

from pathlib import Path
from typing import List, Sequence

from config import Config, get_logger
from retrieval_layer.models import QueryPlan, ScoredId, VECTOR, as_ranked
from retrieval_layer.retrieval_exceptions import IndexUnavailable

logger = get_logger(__name__)


class VectorSearch:
    """The dense half of retrieval.

    Builds its index from the vectors stored beside the chunk labels, not from
    DiskANN's own files: diskannpy 0.7.0 cannot load a dynamic index it saved
    (bug 5.20). An index can still be handed in, which is what tests do.
    """

    def __init__(self, index=None, chunk_store_path: str | Path | None = None) -> None:
        self.chunk_store_path = Path(
            chunk_store_path if chunk_store_path is not None else Config.DB_PATH
        )
        self.__index = index
        self.__owns_index = index is None
        self.__indexed_through = 0

    @property
    def index(self):
        if self.__index is None:
            self.__index = self.__load()
        return self.__index

    def catch_up(self) -> None:
        """Index whatever has been ingested since the index was built.

        One handed in belongs to whoever handed it in, who inserts into it.
        """
        if self.__owns_index and self.__index is not None:
            self.__indexed_through = self.__restore(
                self.__index, self.__indexed_through
            )

    def __restore(self, index, after: int) -> int:
        from data_layer.vector_db_manager.repository.vectorMetaDataRepository import (
            VectorMetaDataRepository,
        )

        store = VectorMetaDataRepository(str(self.chunk_store_path))
        try:
            return index.restore(store, after)
        finally:
            store.close()

    def __load(self):
        from data_layer.vector_db_manager.vectorDbManager import VectorDbManager

        if not self.chunk_store_path.exists():
            raise IndexUnavailable(str(self.chunk_store_path), "no chunk store")
        index = VectorDbManager(
            distance_metrics=Config.DISTANCE_METRIC,
            vector_dtype=Config.VECTOR_DTYPE,
            dimensions=Config.EMBEDDING_DIMENSIONS,
            max_vectors=Config.MAX_VECTORS,
            complexity=Config.COMPLEXITY,
            graph_degree=Config.GRAPH_DEGREE,
            num_threads=Config.NUM_THREADS,
            k_neighbors=Config.K_NEIGHBORS,
        )
        try:
            self.__indexed_through = self.__restore(index, 0)
        except Exception as error:
            raise IndexUnavailable(str(self.chunk_store_path), str(error)) from error
        return index

    def search(self, plan: QueryPlan, k: int) -> List[ScoredId]:
        """The k nearest vectors, nearest first."""
        if k < 1:
            return []
        # Resolved outside the guard below: a missing index is a setup problem
        # the caller has to hear about, while a failed search is one the caller
        # can survive on the lexical results alone.
        index = self.index
        try:
            labels, distances = index.search_vector(plan.vector, k)
        except Exception as error:
            logger.warning("Vector search failed: %s", error)
            return []
        # DiskANN returns a distance, where smaller is closer — the one result
        # in this layer that ranks the opposite way to every score.
        ranked = as_ranked(
            list(zip(labels, distances))[:k], VECTOR, descending=False
        )
        logger.debug("Vector search returned %d candidate(s)", len(ranked))
        return ranked

    def search_many(self, plans: Sequence[QueryPlan], k: int) -> List[List[ScoredId]]:
        """Several queries in one call, which DiskANN threads internally."""
        return [self.search(plan, k) for plan in plans]
