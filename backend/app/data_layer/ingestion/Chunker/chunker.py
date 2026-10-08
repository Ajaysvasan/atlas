from typing import List, Tuple

from config import Config, get_logger
from data_layer.ingestion.nodes.nodes import HChunk, NormalizedContent, RChunk

from .HierarchicalChunker import HierarchicalChunker
from .RecursiveChunker import RecursiveChunker

logger = get_logger(__name__)


class Chunker:
    def __init__(self, chunk_size=256, overlap=20, db_path=None):
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Read when called, not at import: a default bound at import time is the
        # real chunk store whatever Config says by then (bugs.md 7.5).
        self.db_path = db_path if db_path is not None else Config.DB_PATH

    def _call_hierarchical_chunker(
        self, hierarchical_chunker_list: List[NormalizedContent]
    ) -> List[HChunk]:
        hierarchical_chunker = HierarchicalChunker(
            self.overlap, self.chunk_size, self.db_path, hierarchical_chunker_list
        )
        return hierarchical_chunker.process_doc()

    def __call_recursive_chunker(
        self, recursive_chunker_list: List[NormalizedContent]
    ) -> List[RChunk]:
        recursive_chunker = RecursiveChunker(
            recursive_chunker_list, self.chunk_size, self.overlap, db_path=self.db_path
        )
        return recursive_chunker.recursive_chunker()

    def __hierarchical_and_recursive_objects(
        self, normalised_content: List[NormalizedContent]
    ) -> Tuple[List[NormalizedContent], List[NormalizedContent]]:
        hierarchical_chunker_list: List[NormalizedContent] = []
        recursive_chunker_list: List[NormalizedContent] = []
        for content in normalised_content:
            if content.has_section:
                hierarchical_chunker_list.append(content)
            else:
                recursive_chunker_list.append(content)
        return hierarchical_chunker_list, recursive_chunker_list

    def chunk_per_document(
        self, normalised_content: List[NormalizedContent]
    ) -> Tuple[List[HChunk], List[RChunk]]:
        hierarchical_chunker_list, recursive_chunker_list = (
            self.__hierarchical_and_recursive_objects(normalised_content)
        )
        logger.debug(
            "Routing %d document(s): %d hierarchical, %d recursive",
            len(normalised_content),
            len(hierarchical_chunker_list),
            len(recursive_chunker_list),
        )
        h_chunks = (
            self._call_hierarchical_chunker(hierarchical_chunker_list)
            if hierarchical_chunker_list
            else []
        )
        r_chunks = (
            self.__call_recursive_chunker(recursive_chunker_list)
            if recursive_chunker_list
            else []
        )
        logger.info(
            "Chunked %d document(s) into %d hierarchical and %d recursive chunk(s)",
            len(normalised_content),
            len(h_chunks),
            len(r_chunks),
        )
        return h_chunks, r_chunks

    def chunk_document(
        self, normalised_content: NormalizedContent
    ) -> List[HChunk] | List[RChunk]:
        h_chunks, r_chunks = self.chunk_per_document([normalised_content])
        return h_chunks if normalised_content.has_section else r_chunks
