"""The vector index: a built generation, plus what was ingested since it was built."""

import threading
from typing import List, Type, Union

import numpy

from config import get_logger
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
            "Initializing VectorDbManager (distance=%s, dims=%s, recent capacity=%s, threads=%s)",
            distance_metrics,
            dimensions,
            max_vectors,
            num_threads,
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
        self.base = None
        self.lock = threading.Lock()

    def use_base(self, base) -> None:
        """Search a built generation alongside the recent vectors."""
        self.base = base

    @property
    def through(self) -> int:
        """The highest label the built generation holds; 0 without one."""
        return self.base.through if self.base is not None else 0

    def __insert_vector(self, vector, vector_id) -> None:
        with self.lock:
            self.vector_db.insert(vector, vector_id)

    def __insert_vectors_in_batch(self, vectors, vector_ids) -> None:
        with self.lock:
            self.vector_db.batch_insert(vectors, vector_ids)

    def insert(self, embedded_chunk_obj: EmbeddedChunk, vector_id=None) -> None:
        vector = embedded_chunk_obj.vector
        vector_id = (
            embedded_chunk_obj.vector_id if vector_id is None else vector_id
        )
        logger.debug("Inserting vector with vector_id='%s'", vector_id)
        self.__insert_vector(vector, numpy.uint32(vector_id))

    def batch_insert(self, embedded_chunk_objs: List[EmbeddedChunk], vector_ids=None):
        vectors = [obj.vector for obj in embedded_chunk_objs]
        if vector_ids is None:
            vector_ids = [obj.vector_id for obj in embedded_chunk_objs]
        logger.info("Batch inserting %s vectors into index...", len(vector_ids))
        # Both have to be arrays, and the labels have to be uint32: diskannpy
        # indexes its own identifier space and refuses anything wider.
        self.__insert_vectors_in_batch(
            numpy.array(vectors, dtype=numpy.float32),
            numpy.array(vector_ids, dtype=numpy.uint32),
        )

    def count(self) -> int:
        built = self.base.count if self.base is not None else 0
        return built + self.recent_count()

    def recent_count(self) -> int:
        return self.vector_db.count()

    def restore(self, source, after: int = 0) -> int:
        """Index the source's vectors labelled above `after`, up to capacity; returns the last label taken.

        Labels only grow, so passing back what this returned indexes just what
        was ingested since. Whatever does not fit waits for the next build.
        """
        through, restored = int(after), 0
        for page in source.pages(after=through):
            room = self.max_vectors - self.recent_count()
            if len(page.labels) > room:
                if room > 0:
                    with self.lock:
                        self.vector_db.batch_insert(page.vectors[:room], page.labels[:room])
                    restored += room
                    through = int(page.labels[room - 1])
                logger.warning(
                    "The recent index is full: %d vector(s) wait for the next index "
                    "build and are found by keyword search only until then",
                    source.pending(through),
                )
                break
            if len(page.labels):
                with self.lock:
                    self.vector_db.batch_insert(page.vectors, page.labels)
                restored += len(page.labels)
            through = page.through
        if restored:
            logger.info("Indexed %d stored vector(s), through label %d", restored, through)
        if source.missing:
            logger.warning(
                "%d label(s) have no stored vector and cannot be found by meaning "
                "until their documents are ingested again",
                source.missing,
            )
        return through

    def search_vector(self, query, k_neighbors: int | None = None):
        k = int(k_neighbors or self.k_neighbors)
        if k < 1:
            return numpy.empty(0, numpy.uint32), numpy.empty(0, numpy.float32)
        found = []
        base = self.base
        if base is not None:
            found.append(base.search(query, k, self.complexity))
        # Never more than the recent index holds. diskannpy returns k slots
        # regardless and fills the surplus from uninitialised memory: labels
        # that can be real ones, at distance 0.0, sorted ahead of every true result.
        recent = min(k, self.recent_count())
        if recent > 0:
            labels, distances = self.vector_db.search_vector(query, recent, self.complexity)
            found.append((numpy.asarray(labels, numpy.uint32),
                          numpy.asarray(distances, numpy.float32)))
        return self.__nearest(found, k)

    @staticmethod
    def __nearest(found, k: int):
        if not found:
            return numpy.empty(0, numpy.uint32), numpy.empty(0, numpy.float32)
        labels = numpy.concatenate([part[0] for part in found]).astype(numpy.uint32)
        distances = numpy.concatenate([part[1] for part in found]).astype(numpy.float32)
        order = numpy.argsort(distances, kind="stable")[:k]
        return labels[order], distances[order]

    def batch_search_vectors(self, queries, k_neighbors: int | None = None):
        rows = [self.search_vector(query, k_neighbors) for query in queries]
        width = min((len(labels) for labels, _ in rows), default=0)
        labels = numpy.array([r[0][:width] for r in rows], numpy.uint32).reshape(len(rows), width)
        distances = numpy.array([r[1][:width] for r in rows], numpy.float32).reshape(len(rows), width)
        return labels, distances

    def delete_vector(self, vector_id) -> None:
        with self.lock:
            self.vector_db.delete_vector(vector_id)

    def delete_vectors(self, vector_ids) -> None:
        with self.lock:
            self.vector_db.delete_vectors(vector_ids)
