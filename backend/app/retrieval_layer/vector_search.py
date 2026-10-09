"""Approximate nearest neighbour search over the DiskANN index."""

from pathlib import Path
from typing import List, Sequence

from config import Config, get_logger
from retrieval_layer.models import QueryPlan, ScoredId, VECTOR, as_ranked
from retrieval_layer.retrieval_exceptions import IndexUnavailable

logger = get_logger(__name__)


class VectorSearch:
    """The dense half of retrieval.

    Opens the built index generation and loads only what was ingested after it;
    any resource it cannot have leaves the dense half smaller or empty rather
    than failing the query. An index can still be handed in, which tests do.
    """

    def __init__(
        self,
        index=None,
        chunk_store_path: str | Path | None = None,
        index_path: str | Path | None = None,
        vector_store=None,
    ) -> None:
        self.chunk_store_path = Path(
            chunk_store_path if chunk_store_path is not None else Config.DB_PATH
        )
        self.index_path = Path(index_path if index_path is not None else Config.INDEX_PATH)
        self.__vector_store = vector_store
        self.__index = index
        self.__owns_index = index is None
        self.__attempted = index is not None
        self.__indexed_through = 0
        self.__generation = None
        self.__seen = None

    @property
    def index(self):
        if not self.__attempted:
            if not self.chunk_store_path.exists():
                raise IndexUnavailable(str(self.chunk_store_path), "no chunk store")
            self.__attempted = True
            self.__index = self.__assemble()
        return self.__index

    @property
    def generation(self) -> str | None:
        """The built generation being searched, if any."""
        return self.__generation

    def catch_up(self) -> None:
        """Index whatever has been ingested since, or move to a newer built generation.

        One handed in belongs to whoever handed it in, who inserts into it.
        """
        if not self.__owns_index or not self.__attempted:
            return
        from data_layer.vector_db_manager.index_generations import IndexGenerations

        current = IndexGenerations(self.index_path).current()
        unopened = current is not None and self.__generation is None
        if self.__index is None or current != self.__seen or unopened:
            # The old index goes first, so the new one is never held beside it.
            self.__index = None
            self.__index = self.__assemble()
            return
        self.__indexed_through = self.__fill(self.__index, self.__indexed_through)

    def __assemble(self):
        """The built generation plus the recent vectors, as much of it as memory allows."""
        from data_layer.vector_db_manager import memory_guard
        from data_layer.vector_db_manager.index_generations import IndexGenerations
        from data_layer.vector_db_manager.vectorDbManager import VectorDbManager

        generations = IndexGenerations(self.index_path)
        self.__seen = generations.current()
        dims, degree = Config.EMBEDDING_DIMENSIONS, Config.GRAPH_DEGREE
        capacity = Config.RECENT_VECTOR_CAPACITY
        recent = memory_guard.recent_index_bytes(capacity, dims, degree)
        if not memory_guard.can_hold(recent):
            spare = memory_guard.spare_memory() or 0
            logger.warning(
                "Vector search is off: the recent index needs %d MB and only %d MB "
                "are free. Keyword search carries on; refresh() tries again",
                recent // memory_guard.MiB, spare // memory_guard.MiB,
            )
            self.__generation = None
            return None
        try:
            index = VectorDbManager(
                distance_metrics=Config.DISTANCE_METRIC,
                vector_dtype=Config.VECTOR_DTYPE,
                dimensions=dims,
                max_vectors=capacity,
                complexity=Config.COMPLEXITY,
                graph_degree=degree,
                num_threads=Config.NUM_THREADS,
                k_neighbors=Config.K_NEIGHBORS,
            )
        except Exception as error:
            logger.warning("Vector search is off: the recent index could not be made: %s", error)
            self.__generation = None
            return None
        base = generations.open(memory_guard.budget() - recent)
        index.use_base(base)
        self.__generation = None if base is None else base.name
        self.__indexed_through = self.__fill(index, index.through)
        return index

    def __fill(self, index, after: int) -> int:
        from data_layer.vector_db_manager import stored_vectors
        from data_layer.vector_db_manager.repository.vectorMetaDataRepository import (
            VectorMetaDataRepository,
        )

        mapping = VectorMetaDataRepository(str(self.chunk_store_path))
        try:
            if self.__vector_store is None:
                self.__vector_store = stored_vectors.chunk_vector_store()
            source = stored_vectors.StoredVectors(mapping, self.__vector_store)
            return index.restore(source, after)
        except Exception as error:
            logger.warning(
                "Vectors ingested since the last index build could not be loaded "
                "(%s); searching the built index only until refresh()",
                error,
            )
            return after
        finally:
            mapping.close()

    def search(self, plan: QueryPlan, k: int) -> List[ScoredId]:
        """The k nearest vectors, nearest first."""
        if k < 1:
            return []
        # Resolved outside the guard below: a missing chunk store is a setup
        # problem the caller has to hear about, while a failed search is one
        # the caller can survive on the lexical results alone.
        index = self.index
        if index is None:
            return []
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
        """Several queries in one call."""
        return [self.search(plan, k) for plan in plans]
