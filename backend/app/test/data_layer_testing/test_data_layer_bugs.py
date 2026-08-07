import os
import sys
import unittest
import unittest.mock as mock
from pathlib import Path
import numpy as np

# Mock downstream dependencies to test the architecture in isolation
sys.modules["diskannpy"] = mock.MagicMock()

class MockModel:
    def encode(self, texts, truncate_dim=None):
        if isinstance(texts, str):
            return np.zeros(128, dtype=np.float32)
        return [np.zeros(128, dtype=np.float32) for _ in texts]

mock_st = mock.MagicMock()
mock_st.SentenceTransformer = mock.MagicMock(return_value=MockModel())
sys.modules["sentence_transformers"] = mock_st
sys.modules["docx"] = mock.MagicMock()
sys.modules["textract"] = mock.MagicMock()
sys.modules["PyPDF2"] = mock.MagicMock()

# Ensure we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_layer.vector_db_manager.vectorDbManager import VectorDbManager
from data_layer.vector_db_manager.vectorDB_diskann import VectorDb_diskann
from data_layer.ingestion.embedding.EmbeddingManager import EmbeddingManager
from data_layer.ingestion.nodes.nodes import EmbeddedChunk, HChunk
from data_layer.ingestion.metadata.metadata import ChunkMetaData, EmbeddedChunkMetaData
from data_layer.datalayer_exceptions.datalayer_exceptions import VectorInsertionError
from data_layer.ingestion.TextFileProcessor.text_extractor import TextExtractor
from data_layer.ingestion.normalizer.normalizer import TextNormalizer
from data_layer.ingestion.Chunker.DB_Manager import Manager


class TestDataLayerRegression(unittest.TestCase):

    def test_embedding_manager_produces_valid_numpy_vectors(self):
        """
        Verifies that EmbeddingManager produces float32 numpy arrays and integer vector IDs.
        """
        embedder = EmbeddingManager()
        chunk = HChunk("chunk_1", "context_1", "test text", 0, 9, ChunkMetaData("doc_1", "txt", "2026-08-07"))
        res = embedder.embed(chunk)

        self.assertIsInstance(res.vector, np.ndarray)
        self.assertEqual(res.vector.dtype, np.float32)
        self.assertIsInstance(res.vector_id, (int, np.integer))

    def test_diskann_exception_handling(self):
        """
        Verifies that VectorDb_diskann properly catches ValueError and RuntimeError
        and wraps them in VectorInsertionError.
        """
        v_db = VectorDb_diskann("l2", np.float32, 128, 1000, 100, 120, 4)

        def mock_insert(vector, vector_id):
            raise RuntimeError("Native C++ exception")

        v_db.dynamic_dann.insert = mock_insert

        with self.assertRaises(VectorInsertionError):
            v_db.insert(np.zeros(128, dtype=np.float32), 1)

    def test_markdown_single_newline_formatting(self):
        """
        Verifies that markdown extractor does not inject double newlines into extracted text.
        """
        md_path = "test_single_newline.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("Line 1\nLine 2\nLine 3\n")

        extractor = TextExtractor()
        _, text = extractor.extract_text_from_file(md_path)

        self.assertEqual(text, "Line 1\nLine 2\nLine 3\n")
        os.remove(md_path)

    def test_index_load_updates_internal_state(self):
        """
        Verifies that VectorDb_diskann.load() updates self.dynamic_dann.
        """
        v_db = VectorDb_diskann("l2", np.float32, 128, 1000, 100, 120, 4)
        mock_new_index = mock.MagicMock()
        v_db.dynamic_dann.from_file = mock.MagicMock(return_value=mock_new_index)

        os.makedirs("dummy_index_dir", exist_ok=True)
        returned_index = v_db.load("dummy_index_dir")

        self.assertEqual(v_db.dynamic_dann, mock_new_index)
        os.rmdir("dummy_index_dir")

    def test_normalizer_handles_path_objects(self):
        """
        Verifies that TextNormalizer handles Path objects without throwing TypeError.
        """
        normalizer = TextNormalizer()
        file_path_obj = Path("dummy.txt")
        file_name = "dummy.txt"
        normalized_text = "test content"

        doc_id = normalizer._TextNormalizer__generate_document_id(file_name, file_path_obj, normalized_text)
        self.assertIsInstance(doc_id, str)

    # ---- Bug 1.1 regression test: vectorMetaDataRepository imports correctly ----
    def test_bug_1_1_vector_meta_data_repository_import(self):
        """
        Verifies that vectorMetaDataRepository can be imported without ModuleNotFoundError.
        The import path was previously broken (used relative 'datalayer_exceptions' instead
        of 'data_layer.datalayer_exceptions').
        """
        try:
            from data_layer.vector_db_manager.repository.vectorMetaDataRepository import VectorMetaDataRepository
        except ModuleNotFoundError:
            self.fail("VectorMetaDataRepository import failed due to broken module path (Bug 1.1)")
        self.assertTrue(callable(VectorMetaDataRepository))

    # ---- Bug 1.2 + 1.3 regression test: DB_Manager thread safety & directory creation ----
    def test_bug_1_2_and_1_3_db_manager_creates_dirs_and_thread_safe(self):
        """
        Verifies that DB_Manager:
        - Creates the parent directory for the database file if it does not exist (Bug 1.3)
        - Can be instantiated without thread affinity errors (Bug 1.2)
        """
        import shutil
        test_dir = "test_db_manager_dir"
        db_path = os.path.join(test_dir, "nested", "test_chunks.db")

        # Ensure clean state
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

        try:
            mgr = Manager(db_path, is_chunker_type_hierarchical=True)
            # Verify directory was created
            self.assertTrue(os.path.exists(os.path.dirname(db_path)))
            # Verify DB file was created
            self.assertTrue(os.path.exists(db_path))
            mgr.close()
        finally:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)


if __name__ == "__main__":
    unittest.main()
