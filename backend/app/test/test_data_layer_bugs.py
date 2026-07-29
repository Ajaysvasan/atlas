import os
import sys
import unittest
import unittest.mock as mock
from pathlib import Path

# Mock downstream dependencies to test the architecture in isolation
sys.modules["diskannpy"] = mock.MagicMock()

class MockModel:
    def encode(self, texts, truncate_dim=None):
        import numpy as np
        if isinstance(texts, str):
            return np.zeros(128)
        return [np.zeros(128) for _ in texts]

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
from data_layer.ingestion.nodes.nodes import EmbeddedChunk, RChunk
from data_layer.ingestion.metadata.metadata import ChunkMetaData, EmbeddedChunkMetaData
from data_layer.datalayer_exceptions.datalayer_exceptions import VectorInsertionError
from data_layer.ingestion.TextFileProcessor.text_extractor import TextExtractor
from data_layer.ingestion.normalizer.normalizer import TextNormalizer
import numpy as np

class TestDataLayerBugs(unittest.TestCase):

    def test_bug_7_1_and_7_2_invalid_datatype_passed_to_diskann(self):
        """
        Tests Bug 7.1 (String UUID to DiskANN) and Bug 7.2 (List instead of Numpy Array).
        We simulate DiskANN's strict native datatype requirements by mocking it to raise TypeError
        if a string is passed as vector_id or a List is passed as a vector instead of a numpy array.
        """
        def mock_batch_insert(vectors, vector_ids):
            # diskannpy requires vector_ids to be integers (uint32/uint64)
            for vid in vector_ids:
                if not isinstance(vid, (int, np.integer)):
                    raise TypeError(f"incompatible function arguments: expected int, got {type(vid)}")
            
            # diskannpy requires vectors to be a numpy array
            if not isinstance(vectors, np.ndarray):
                raise TypeError(f"incompatible function arguments: expected numpy.ndarray, got {type(vectors)}")

        # Initialize
        v_manager = VectorDbManager("l2", np.float32, 128, 1000, 100, 120, 4, 9)
        v_manager.vector_db.dynamic_dann.batch_insert = mock_batch_insert

        # Simulate EmbeddingManager's output (which produces string chunk_ids and List[float] vectors)
        embedded_chunk = EmbeddedChunk(
            vector=[0.0] * 128,  # Bug 7.2: It's a python list, not a numpy array
            meta_data=EmbeddedChunkMetaData(
                chunk_id="uuid-string-1234", # Bug 7.1: It's a string, not an int
                chunk="test",
                modelUsedForChunking="test_model"
            )
        )

        with self.assertRaises(TypeError) as context:
            v_manager.batch_insert([embedded_chunk])
        
        self.assertTrue("incompatible function arguments" in str(context.exception))

    def test_bug_7_3_invalid_custom_exception_catching(self):
        """
        Tests Bug 7.3 where VectorDb_diskann attempts to catch our custom VectorInsertionError
        from the native C++ library, which would instead throw RuntimeError or ValueError.
        """
        v_db = VectorDb_diskann("l2", np.float32, 128, 1000, 100, 120, 4)
        
        # Mock diskannpy throwing a native C++ runtime error
        def mock_insert(vector, vector_id):
            raise RuntimeError("Native C++ exception: Dimension mismatch")
            
        v_db.dynamic_dann.insert = mock_insert
        
        # The custom exception block will fail to catch the native RuntimeError,
        # proving that the exception handling is bypassed.
        with self.assertRaises(RuntimeError):
            v_db.insert(np.zeros(128, dtype=np.float32), 1)

    def test_bug_7_4_markdown_double_newline_corruption(self):
        """
        Tests Bug 7.4 where the markdown extractor doubles newlines.
        """
        # Create a dummy markdown file
        md_path = "test_corruption.md"
        with open(md_path, "w") as f:
            f.write("Line 1\nLine 2\nLine 3\n")
            
        extractor = TextExtractor()
        _, text = extractor.extract_text_from_file(md_path)
        
        # Due to joining readlines() with \n, it creates double newlines.
        self.assertEqual(text, "Line 1\n\nLine 2\n\nLine 3\n")
        
        # Clean up
        os.remove(md_path)

    def test_bug_7_5_index_load_state_desynchronization(self):
        """
        Tests Bug 7.5 where loading an index does not update internal state.
        """
        v_db = VectorDb_diskann("l2", np.float32, 128, 1000, 100, 120, 4)
        
        mock_new_index = mock.MagicMock()
        v_db.dynamic_dann.from_file = mock.MagicMock(return_value=mock_new_index)
        
        # Create dummy path to bypass exists() check
        os.makedirs("dummy_index_dir", exist_ok=True)
        
        returned_index = v_db.load("dummy_index_dir")
        
        # The returned index is the newly loaded one
        self.assertEqual(returned_index, mock_new_index)
        
        # But the internal state is NOT updated!
        self.assertNotEqual(v_db.dynamic_dann, mock_new_index)
        
        os.rmdir("dummy_index_dir")

    def test_bug_7_6_typeerror_posixpath_join(self):
        """
        Tests Bug 7.6 where passing a Path object to __generate_document_id causes TypeError.
        """
        normalizer = TextNormalizer()
        
        # Simulate TextExtractor passing a Path object instead of a string
        file_path_obj = Path("dummy.txt")
        file_name = "dummy.txt"
        normalized_text = "test content"
        
        with self.assertRaises(TypeError) as context:
            # We bypass the private mangling by calling it like this
            normalizer._TextNormalizer__generate_document_id(file_name, file_path_obj, normalized_text)
            
        self.assertTrue("expected str instance, PosixPath found" in str(context.exception))

if __name__ == "__main__":
    unittest.main()
