from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from data_layer.datalayer_exceptions.datalayer_exceptions import VectorStoreUnavailable
from data_layer.vector_db_manager import stored_vectors
from data_layer.vector_db_manager.index_generations import BuildOutcome, IndexGenerations
from data_layer.vector_db_manager.repository.vectorMetaDataRepository import (
    VectorMetaDataRepository,
    checked_vector_id,
)

from config import Config, get_logger, log_timing

from .Chunker.chunker import Chunker
from .embedding.EmbeddingManager import EmbeddingManager
from .nodes.nodes import EmbeddedChunk, HChunk, NormalizedContent, RChunk
from .normalizer.normalizer import NormalizationProfiles
from .TextFileProcessor.file_loader import FileLoader
from .TextFileProcessor.text_extractor import TextExtractor


logger = get_logger(__name__)


# I Don't need any more abstraction here since I am not mutating the data rather just providing an simplified interface that I can use
class IngestionPipeline:
    def __init__(self, vector_store=None, index_path: str | Path | None = None):
        logger.info("Building ingestion pipeline")
        self.f_loader = FileLoader()
        self.t_extractor = TextExtractor()
        self.t_normalizer = NormalizationProfiles.rag_ingestion()
        self.chunker = Chunker()
        self.embedder = EmbeddingManager()
        # Without this, a search result is a label and nothing else (bug 5.3).
        # It shares the chunk store's database file so retrieval can reach the
        # text in one join.
        self.vector_meta = VectorMetaDataRepository(self.chunker.db_path)
        self.vector_store = vector_store
        self._owns_store = vector_store is None
        self.index_path = index_path
        logger.debug("Ingestion pipeline ready")

    def load_file(self, folder_path) -> Dict[str, List[Path]]:
        """Returns the files that are within that specified path"""
        return self.f_loader.load_files(folder_path)

    def extract_text_from_file(self, file_path: str) -> Tuple[str, str]:
        """Returns the extract texts from a loaded file"""
        return self.t_extractor.extract_text_from_file(file_path)

    def extract_text_from_files(
        self, loaded_files: Dict[str, List[Path]]
    ) -> Dict[str, str]:
        """Returns the extract texts from a set of loaded files"""
        with log_timing(logger, "extraction", files=sum(len(v) for v in loaded_files.values())):
            return self.t_extractor.extract_all(loaded_files)

    def normalize_doc(self, file_path, text) -> NormalizedContent:
        return self.t_normalizer.normalize_text(file_path, text)

    def normalize_docs(
        self, extracted_texts: Dict[str, str]
    ) -> List[NormalizedContent]:
        with log_timing(logger, "normalisation", documents=len(extracted_texts)):
            return self.t_normalizer.normalize_all(extracted_texts)

    def chunk_text(
        self, normalized_content: NormalizedContent
    ) -> List[HChunk] | List[RChunk]:
        """Chunk a single document, routed the same way a batch would be."""
        return self.chunker.chunk_document(normalized_content)

    def chunk_texts(
        self, normalized_contents: List[NormalizedContent]
    ) -> Tuple[List[HChunk], List[RChunk]]:
        with log_timing(logger, "chunking", documents=len(normalized_contents)):
            return self.chunker.chunk_per_document(normalized_contents)

    def embed(
        self, arg: HChunk | RChunk | List[HChunk | RChunk]
    ) -> EmbeddedChunk | List[EmbeddedChunk]:
        if not isinstance(arg, list):
            return self.embedder.embed(arg)
        with log_timing(logger, "embedding", chunks=len(arg)):
            return self.embedder.embed(arg)

    def ingest_vector(self, embedded_value: EmbeddedChunk) -> int:
        return self.batch_insert_vectors([embedded_value])[0]

    def batch_insert_vectors(self, embedded_objs: List[EmbeddedChunk]) -> List[int]:
        """Store the vectors under their vector ids, then hand each its DiskANN label.

        The vector goes to PostgreSQL first, so a label never exists for a
        vector that was not stored. Building the index from them is
        `maintain_index()`'s, which runs once enough have accumulated.
        """
        if not embedded_objs:
            return []
        chunk_ids = [e.meta_data.chunk_id for e in embedded_objs]
        vector_ids = [checked_vector_id(e.vector_id, c) for e, c in zip(embedded_objs, chunk_ids)]
        vectors = np.stack([np.asarray(e.vector, dtype=np.float32) for e in embedded_objs])
        with log_timing(logger, "vector store", vectors=len(vector_ids)):
            self.__store().batch_insert(vector_ids, vectors)
        labels = self.vector_meta.batch_insert(vector_ids, chunk_ids)
        self.maintain_index()
        return labels

    def maintain_index(self, force: bool = False) -> BuildOutcome | None:
        """Rebuild the index once INDEX_REBUILD_AT vectors wait outside it, or now if forced."""
        generations = IndexGenerations(self.index_path)
        waiting = self.vector_meta.pending(generations.through())
        if waiting == 0 or (waiting < Config.INDEX_REBUILD_AT and not force):
            return None
        with log_timing(logger, "index build", waiting=waiting):
            return generations.build(
                stored_vectors.StoredVectors(self.vector_meta, self.__store())
            )

    def __store(self):
        if self.vector_store is None:
            try:
                self.vector_store = stored_vectors.chunk_vector_store()
            except Exception as error:
                raise VectorStoreUnavailable(error) from error
        return self.vector_store

    def close(self) -> None:
        """Release what this pipeline opened. A vector store handed in stays open."""
        self.vector_meta.close()
        if self._owns_store and self.vector_store is not None:
            self.vector_store.close()
            self.vector_store = None
