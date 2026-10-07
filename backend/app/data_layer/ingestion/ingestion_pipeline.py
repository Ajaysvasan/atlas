from pathlib import Path
from typing import Dict, List, Tuple

from data_layer.vector_db_manager.repository.vectorMetaDataRepository import (
    VectorMetaDataRepository,
)
from data_layer.vector_db_manager.vectorDbManager import VectorDbManager

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
    def __init__(self):
        logger.info("Building ingestion pipeline")
        self.f_loader = FileLoader()
        self.t_extractor = TextExtractor()
        self.t_normalizer = NormalizationProfiles.rag_ingestion()
        self.chunker = Chunker()
        self.embedder = EmbeddingManager()
        self.vector_db = VectorDbManager(
            distance_metrics=Config.DISTANCE_METRIC,
            vector_dtype=Config.VECTOR_DTYPE,
            dimensions=Config.EMBEDDING_DIMENSIONS,
            max_vectors=Config.MAX_VECTORS,
            complexity=Config.COMPLEXITY,
            graph_degree=Config.GRAPH_DEGREE,
            num_threads=Config.NUM_THREADS,
            k_neighbors=Config.K_NEIGHBORS,
        )
        # Without this, a search result is a vector id and nothing else: the id
        # is a one-way hash of the chunk id, so there is no way back to the text
        # (bug 5.3). It shares the chunk store's database file so retrieval can
        # reach the text in one join.
        self.vector_meta = VectorMetaDataRepository(self.chunker.db_path)
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

    def ingest_vector(self, embedded_value: EmbeddedChunk) -> None:
        label = self.vector_meta.allocate(embedded_value.meta_data.chunk_id)
        self.vector_db.insert(embedded_value, vector_id=label)

    def batch_insert_vectors(self, embedded_objs: List[EmbeddedChunk]) -> List[int]:
        """Index the vectors under labels the mapping table hands out.

        The label is allocated rather than taken from the EmbeddedChunk: that id
        is masked into 63 bits for pgvector, and DiskANN indexes uint32 labels.
        Allocating also means the mapping row exists before the vector does, so
        a hit can never arrive for a chunk the table has not heard of.
        """
        if not embedded_objs:
            return []
        labels = self.vector_meta.allocate_many(
            [e.meta_data.chunk_id for e in embedded_objs]
        )
        with log_timing(logger, "vector insert", vectors=len(embedded_objs)):
            self.vector_db.batch_insert(embedded_objs, vector_ids=labels)
        return labels

    def persist_index(self, save_path: str = Config.INDEX_PATH) -> None:
        """Write the DiskANN index to disk.

        Nothing called this, so the index lived only as long as the process and
        a later search started from an empty one (the open P3 entry).
        """
        with log_timing(logger, "index persist"):
            self.vector_db.save(save_path)

    def close(self) -> None:
        self.vector_meta.close()
