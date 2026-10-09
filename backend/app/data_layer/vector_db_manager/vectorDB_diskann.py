import diskannpy as dann

from config import get_logger
from data_layer.datalayer_exceptions.datalayer_exceptions import VectorInsertionError

logger = get_logger(__name__)

# diskannpy defaults this to False, and the dynamic index then builds a graph so
# sparse it cannot find a vector that was inserted into it: recall@10 of 0.09
# against 0.96 with it on (bugs.md 5.19). Not a tuning knob.
SATURATE_GRAPH = True


class VectorDb_diskann:

    def __init__(
        self,
        distance_metrics: str,
        vector_dtype,
        dimensions: int,
        max_vectors: int,
        complexity: int,
        graph_degree: int,
        num_threads: int,
    ) -> None:
        self.distance_metrics = distance_metrics
        self.vector_dtype = vector_dtype
        self.dimensions = dimensions
        self.max_vectors = max_vectors
        self.complexity = complexity
        self.graph_degree = graph_degree
        self.num_threads = num_threads
        self.dynamic_dann = dann.DynamicMemoryIndex(
            distance_metric=self.distance_metrics,
            vector_dtype=self.vector_dtype,
            dimensions=self.dimensions,
            max_vectors=self.max_vectors,
            graph_degree=self.graph_degree,
            saturate_graph=SATURATE_GRAPH,
            search_threads=self.num_threads,
            complexity=self.complexity,
        )

    def __insert_vector(self, vector, vector_id):
        try:
            self.dynamic_dann.insert(vector, vector_id)
        except (ValueError, RuntimeError) as e:
            raise VectorInsertionError(vector_id, e) from e

    def __insert_vectors(self, vectors, vector_ids):
        try:
            self.dynamic_dann.batch_insert(vectors, vector_ids)
        except (ValueError, RuntimeError) as e:
            raise VectorInsertionError(vector_ids, e) from e

    def __delete(self, vector_id):
        self.dynamic_dann.mark_deleted(vector_id)

    def __delete_all(self, vector_ids):
        for vector_id in vector_ids:
            self.__delete(vector_id)

    def delete_vector(self, id):
        self.__delete(id)
        self.dynamic_dann.consolidate_delete()

    def delete_vectors(self, ids):
        self.__delete_all(ids)
        self.dynamic_dann.consolidate_delete()

    def insert(self, vector, vector_id):
        self.__insert_vector(vector, vector_id)

    def batch_insert(self, vectors, vector_ids):
        self.__insert_vectors(vectors, vector_ids)

    def count(self) -> int:
        """Vectors in the index now; diskannpy exposes this only on its native object."""
        return int(self.dynamic_dann._index.num_points())

    def search_vector(self, query, k_neighbors, complexity):
        return self.dynamic_dann.search(
            query=query, k_neighbors=k_neighbors, complexity=complexity
        )

    def batch_search_vector(self, queries, k_neighbors, complexity):
        return self.dynamic_dann.batch_search(
            queries, k_neighbors, complexity, self.num_threads
        )
