from pathlib import Path
from typing import Dict, List, Tuple

from data_layer.vector_db_manager.vectorDbManager import VectorDbManager

from config import Config

from .Chunker.chunker import Chunker
from .embedding.EmbeddingManager import EmbeddingManager
from .nodes.nodes import EmbeddedChunk, HChunk, NormalizedContent, RChunk
from .normalizer.normalizer import NormalizationProfiles
from .TextFileProcessor.file_loader import FileLoader
from .TextFileProcessor.text_extractor import TextExtractor


# I Don't need any more abstraction here since I am not mutating the data rather just providing an simplified interface that I can use
class IngestionPipeline:
    def __init__(self):
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

    def load_file(self, folder_path) -> Dict[str, List[Path]]:
        """Returns the files that are within that specified path
        This function can be used when there are multiple files that are
        present in a same directory
        """
        return self.f_loader.load_files(folder_path)

    def extract_text_from_file(self, file_path: str) -> Tuple[str, str]:
        """Returns the extract texts from a loaded file"""
        return self.t_extractor.extract_text_from_file(file_path)

    def extract_text_from_files(
        self, loaded_files: Dict[str, List[Path]]
    ) -> Dict[str, str]:
        """Returns the extract texts from a set of loaded files"""
        return self.t_extractor.extract_all(loaded_files)

    def normalize_doc(self, file_path, text) -> NormalizedContent:
        return self.t_normalizer.normalize_text(file_path, text)

    def normalize_docs(
        self, extracted_texts: Dict[str, str]
    ) -> List[NormalizedContent]:
        return self.t_normalizer.normalize_all(extracted_texts)

    def chunk_text(
        self, normalized_content: NormalizedContent
    ) -> List[HChunk] | List[RChunk]:
        """Chunk a single document, routed the same way a batch would be."""
        return self.chunker.chunk_document(normalized_content)

    def chunk_texts(
        self, normalized_contents: List[NormalizedContent]
    ) -> Tuple[List[HChunk], List[RChunk]]:
        return self.chunker.chunk_per_document(normalized_contents)

    def embed(
        self, arg: HChunk | RChunk | List[HChunk | RChunk]
    ) -> EmbeddedChunk | List[EmbeddedChunk]:
        return self.embedder.embed(arg)

    def ingest_vector(self, embedded_value: EmbeddedChunk) -> None:
        self.vector_db.insert(embedded_value)

    def batch_insert_vectors(self, embedded_objs: List[EmbeddedChunk]) -> None:
        self.vector_db.batch_insert(embedded_objs)
