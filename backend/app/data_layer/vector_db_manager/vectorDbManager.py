import threading
from typing import List, Type, Union

import numpy

from config import Config, get_logger
from data_layer.datalayer_exceptions.datalayer_exceptions import (
    IndexDirectoryDoesNotExists,
)
from data_layer.ingestion.nodes.nodes import EmbeddedChunk
from data_layer.vector_db_manager.vectorDB_diskann import VectorDb_diskann as vdap

logger = get_logger(__name__)


class VectorDbManager:

    def __init__(
        self,
        distance_metrics: str,
        vector_dtype: Union[Type[numpy.float32], Type[numpy.int8], Type[numpy.uint8]],
        dimensions: int,
        max_vectors: int,
        complexity: int,
        graph_degree: int,
        num_threads: int,
        k_neighbors: int,
    ) -> None:
        self.distance_metrics = distance_metrics
        self.vector_dtype = vector_dtype
        self.dimensions = dimensions
        self.max_vectors = max_vectors
        self.complexity = complexity
        self.graph_degree = graph_degree
        self.num_threads = num_threads
        self.k_neighbors = k_neighbors
        logger.info(
            f"Initializing VectorDbManager (distance={distance_metrics}, dims={dimensions}, max_vectors={max_vectors}, threads={num_threads})"
        )
        self.vector_db = vdap(
            self.distance_metrics,
            self.vector_dtype,
            self.dimensions,
            self.max_vectors,
            self.complexity,
            self.graph_degree,
            self.num_threads,
        )
        self.lock = threading.Lock()

    def __insert_vector(self, vector, vector_id) -> None:
        with self.lock:
            self.vector_db.insert(vector, vector_id)

    def __insert_vectors_in_batch(self, vectors, vector_ids) -> None:
        with self.lock:
            self.vector_db.batch_insert(vectors, vector_ids)

    def insert(self, embedded_chunk_obj: EmbeddedChunk) -> None:
        vector = embedded_chunk_obj.vector
        vector_id = embedded_chunk_obj.vector_id
        logger.debug(f"Inserting vector with vector_id='{vector_id}'")
        self.__insert_vector(vector, vector_id)

    def batch_insert(self, embedded_chunk_objs: List[EmbeddedChunk]):
        vectors = []
        vector_ids = []
        for embedded_chunk_obj in embedded_chunk_objs:
            vectors.append(embedded_chunk_obj.vector)
            vector_ids.append(embedded_chunk_obj.vector_id)
        logger.info(f"Batch inserting {len(vector_ids)} vectors into index...")
        self.__insert_vectors_in_batch(numpy.array(vectors, dtype=numpy.float32), vector_ids)

    def search_vector(self, query):
        return self.vector_db.search_vector(query, self.k_neighbors, self.complexity)

    def batch_search_vectors(self, queries):
        return self.vector_db.batch_search_vector(
            queries, self.k_neighbors, self.complexity
        )

    def delete_vector(self, vector_id) -> None:
        with self.lock:
            self.vector_db.delete_vector(vector_id)

    def delete_vectors(self, vector_ids) -> None:
        with self.lock:
            self.vector_db.delete_vectors(vector_ids)

    def save(self, save_path=Config.INDEX_PATH):
        logger.info(f"Saving VectorDbManager index to path '{save_path}'...")
        with self.lock:
            self.vector_db.save(save_path)
        logger.info("VectorDbManager index saved successfully.")

    def load(self, load_path=Config.INDEX_PATH):
        try:
            logger.info(f"Loading VectorDbManager index from path '{load_path}'...")
            with self.lock:
                idx = self.vector_db.load(load_path)
            logger.info("VectorDbManager index loaded successfully.")
            return self.vector_db.dynamic_dann
        except IndexDirectoryDoesNotExists:
            logger.warning(
                f"Index directory '{load_path}' does not exist. Returning None."
            )
            return None
