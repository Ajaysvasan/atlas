"""Approximate nearest neighbour search over the DiskANN index."""

from pathlib import Path
from typing import List, Sequence

from config import Config, get_logger
from retrieval_layer.models import QueryPlan, ScoredId, VECTOR, as_ranked
from retrieval_layer.retrieval_exceptions import IndexUnavailable

logger = get_logger(__name__)


class VectorSearch:
    """The dense half of retrieval.

    Takes an index rather than building one: the ingestion pipeline owns
    writing, this owns reading, and sharing the object keeps a freshly ingested
    corpus searchable without a round trip through disk.
    """

    def __init__(self, index=None, index_path: str | Path = Config.INDEX_PATH) -> None:
        self.index_path = Path(index_path)
        self.__index = index

    @property
    def index(self):
        if self.__index is None:
            self.__index = self.__load()
        return self.__index

    def __load(self):
        from data_layer.vector_db_manager.vectorDbManager import VectorDbManager

        if not self.index_path.exists():
            raise IndexUnavailable(str(self.index_path), "no index on disk")
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
            index.load(str(self.index_path))
        except Exception as error:
            raise IndexUnavailable(str(self.index_path), str(error)) from error
        logger.info("Loaded the vector index from %s", self.index_path)
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
            labels, distances = index.search_vector(plan.vector)
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
