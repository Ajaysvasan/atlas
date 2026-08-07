import os
import sys
import unittest
import unittest.mock as mock
import numpy as np
from pathlib import Path

# Mock dependencies for isolated testing
sys.modules["diskannpy"] = mock.MagicMock()

class MockDiskANN:
    def __init__(self, *args, **kwargs):
        pass
    def insert(self, vector, vector_id):
        if not isinstance(vector_id, (int, np.integer)):
            raise TypeError("DiskANN vector_id must be integer")
        if not isinstance(vector, np.ndarray):
            raise TypeError("DiskANN vector must be numpy.ndarray")
    def batch_insert(self, vectors, vector_ids):
        for vid in vector_ids:
            if not isinstance(vid, (int, np.integer)):
                raise TypeError("DiskANN vector_id must be integer")
        if not isinstance(vectors, np.ndarray):
            raise TypeError("DiskANN vectors must be numpy.ndarray")
    def search(self, *args, **kwargs):
        return []
    def save(self, *args, **kwargs):
        pass
    def from_file(self, *args, **kwargs):
        return MockDiskANN()

sys.modules["diskannpy"].DynamicMemoryIndex = MockDiskANN

class MockModel:
    def encode(self, texts, truncate_dim=None):
        if isinstance(texts, str):
            return np.zeros(128, dtype=np.float32)
        return np.array([np.zeros(128, dtype=np.float32) for _ in texts])

mock_st = mock.MagicMock()
mock_st.SentenceTransformer = mock.MagicMock(return_value=MockModel())
sys.modules["sentence_transformers"] = mock_st
sys.modules["docx"] = mock.MagicMock()
sys.modules["textract"] = mock.MagicMock()
sys.modules["PyPDF2"] = mock.MagicMock()

# Setup import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_layer.vector_db_manager.vectorDbManager import VectorDbManager
from data_layer.ingestion.embedding.EmbeddingManager import EmbeddingManager
from data_layer.ingestion.nodes.nodes import EmbeddedChunk, RChunk, ChunkMetaData
from data_layer.ingestion.metadata.metadata import EmbeddedChunkMetaData
from data_layer.datalayer_exceptions.datalayer_exceptions import VectorInsertionError
from data_layer.ingestion.TextFileProcessor.text_extractor import TextExtractor
from data_layer.ingestion.normalizer.normalizer import TextNormalizer
from data_layer.ingestion.ingestion_pipeline import IngestionPipeline
from config import Config


class TestDataLayerProduction(unittest.TestCase):
    
    def setUp(self):
        os.makedirs(os.path.dirname(Config.DB_PATH), exist_ok=True)
        self.md_file_path = "test_doc.md"
        with open(self.md_file_path, "w") as f:
            f.write("Line 1\nLine 2\nLine 3\n")

    def tearDown(self):
        if os.path.exists(self.md_file_path):
            os.remove(self.md_file_path)

    def test_text_extractor_markdown_formatting(self):
        """
        Tests that markdown extraction preserves formatting without double spacing.
        Expected: "Line 1\nLine 2\nLine 3\n"
        """
        extractor = TextExtractor()
        _, extracted_text = extractor.extract_text_from_file(self.md_file_path)
        
        # We expect exact preservation of the text content
        self.assertEqual(extracted_text, "Line 1\nLine 2\nLine 3\n")

    def test_normalizer_batch_processing(self):
        """
        Tests that Normalizer can process batch extracted dictionaries,
        which natively use pathlib.Path objects as keys.
        """
        normalizer = TextNormalizer()
        extractor = TextExtractor()
        
        # Simulate FileLoader output
        loaded_files = {"md": [Path(self.md_file_path)]}
        extracted_texts = extractor.extract_all(loaded_files)
        
        # This should process seamlessly without crashing
        normalized_content = normalizer.normalize_all(extracted_texts)
        self.assertTrue(len(normalized_content) > 0)

    def test_embedding_manager_output_types(self):
        """
        Tests that EmbeddingManager produces numpy arrays natively to be compatible with DiskANN.
        """
        embedder = EmbeddingManager()
        chunks = [RChunk("test chunk", ChunkMetaData("doc", "doc123", "recursive"), "chunk123")]
        
        embedded_chunks = embedder.embed(chunks)
        
        # DiskANN expects numpy arrays, so EmbeddedChunk.vector should be one.
        self.assertIsInstance(embedded_chunks[0].vector, np.ndarray)

    def test_vector_db_manager_batch_insert(self):
        """
        Tests that VectorDbManager successfully inserts EmbeddedChunk batches into DiskANN.
        """
        v_manager = VectorDbManager("l2", np.float32, 128, 1000, 100, 120, 4, 9)
        
        # Create a perfectly valid production embedded chunk
        embedded_chunk = EmbeddedChunk(
            vector=np.zeros(128, dtype=np.float32),
            vector_id=1,
            meta_data=EmbeddedChunkMetaData(
                chunk_id="1",  # Must be a string for metadata
                chunk="test",
                modelUsedForChunking="test_model"
            )
        )
        
        # This should insert without any TypeError
        try:
            v_manager.batch_insert([embedded_chunk])
        except Exception as e:
            self.fail(f"batch_insert crashed unexpectedly: {e}")

    def test_vector_db_manager_index_load_persistence(self):
        """
        Tests that loading an index successfully updates the internal state 
        so that subsequent inserts go to the loaded index.
        """
        v_manager = VectorDbManager("l2", np.float32, 128, 1000, 100, 120, 4, 9)
        
        # Create a mock directory
        os.makedirs("test_index_dir", exist_ok=True)
        
        v_manager.load("test_index_dir")
        
        # After loading, the internal dynamic_dann should be the one returned from from_file
        # Our mock from_file returns a new MockDiskANN object. Let's verify type/identity if possible.
        # A simple check: the object should have changed.
        original_db = v_manager.vector_db.dynamic_dann
        v_manager.load("test_index_dir")
        new_db = v_manager.vector_db.dynamic_dann
        
        self.assertNotEqual(original_db, new_db, "The internal DiskANN instance was not updated after load()")
        
        os.rmdir("test_index_dir")
        
    def test_full_ingestion_pipeline_end_to_end(self):
        """
        Tests the entire ingestion pipeline to ensure no fatal exceptions exist 
        between component integrations.
        """
        pipeline = IngestionPipeline()
        
        # Mock load files
        loaded_files = {"md": [Path(self.md_file_path)]}
        
        try:
            # Step 1: Extract
            extracted_texts = pipeline.extract_text_from_files(loaded_files)
            
            # Step 2: Normalize
            normalized_contents = pipeline.normalize_docs(extracted_texts)
            
            # Step 3: Chunk (Assuming DB_Manager mock exists or we skip actual DB insert in mock)
            # Actually, Chunker uses a real SQLite DB_Manager. It will write to DB_PATH.
            h_chunks, r_chunks = pipeline.chunk_texts(normalized_contents)
            
            # Step 4: Embed
            embedded_chunks = pipeline.embed(r_chunks)
            
            # Step 5: Ingest
            pipeline.batch_insert_vectors(embedded_chunks)
            
        except Exception as e:
            self.fail(f"Pipeline crashed entirely during end-to-end processing: {e}")

if __name__ == "__main__":
    unittest.main()
