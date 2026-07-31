from pathlib import Path

import diskannpy as dann

from config import Config, get_logger
from data_layer.datalayer_exceptions.datalayer_exceptions import (
    IndexDirectoryDoesNotExists,
    VectorInsertionError,
)

logger = get_logger(__name__)


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
            search_threads=self.num_threads,
            complexity=self.complexity,
        )

    def __insert_vector(self, vector, vector_id):
        try:
            self.dynamic_dann.insert(vector, vector_id)
        except ValueError | RuntimeError as e:
            raise VectorInsertionError(vector_id)

    def __insert_vectors(self, vectors, vector_ids):
        self.dynamic_dann.batch_insert(vectors, vector_ids)

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

    def search_vector(self, query, k_neighbors, complexity):
        return self.dynamic_dann.search(
            query=query, k_neighbors=k_neighbors, complexity=complexity
        )

    def batch_search_vector(self, queries, k_neighbors, complexity):
        return self.dynamic_dann.batch_search(
            queries, k_neighbors, complexity, self.num_threads
        )

    def save(self, save_path=Config.INDEX_PATH):
        path = Path(save_path)
        if not path.exists():
            logger.info(f"Creating DiskANN index directory '{save_path}'...")
            path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving DiskANN dynamic index to '{save_path}'...")
        self.dynamic_dann.save(save_path)
        logger.debug("DiskANN dynamic index saved successfully to disk.")

    def load(self, load_path=Config.INDEX_PATH) -> dann.DynamicMemoryIndex | None:
        path = Path(load_path)
        if path.exists():
            logger.info(f"Loading DiskANN index from '{load_path}'...")
            index = self.dynamic_dann.from_file(
                index_directory=load_path,
                max_vectors=self.max_vectors,
                complexity=self.complexity,
                graph_degree=self.graph_degree,
                search_threads=self.num_threads,
                distance_metric=self.distance_metrics,
                vector_dtype=self.vector_dtype,
                dimensions=self.dimensions,
            )
            logger.info("DiskANN index loaded successfully.")
            return index
        logger.error(
            f"Failed to load DiskANN index: Directory '{load_path}' does not exist."
        )
        raise IndexDirectoryDoesNotExists(load_path)
