import os
import sys
import unittest.mock as mock

sys.modules["diskannpy"] = mock.MagicMock()


class MockModel:
    def encode(self, texts):
        import numpy as np

        return [np.zeros(128) for _ in texts]


mock_st = mock.MagicMock()
mock_st.SentenceTransformer = mock.MagicMock(return_value=MockModel())
sys.modules["sentence_transformers"] = mock_st
sys.modules["docx"] = mock.MagicMock()
sys.modules["textract"] = mock.MagicMock()
sys.modules["PyPDF2"] = mock.MagicMock()

import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data_layer.datalayer_exceptions.datalayer_exceptions import (
    IndexDirectoryDoesNotExists,
)
from data_layer.ingestion.Chunker.DB_Manager import Manager
from data_layer.ingestion.Chunker.HierarchicalChunker import HierarchicalChunker
from data_layer.ingestion.Chunker.RecursiveChunker import RecursiveChunker
from data_layer.ingestion.embedding.EmbeddingManager import EmbeddingManager
from data_layer.ingestion.ingestion_pipeline import IngestionPipeline
from data_layer.ingestion.nodes.nodes import (
    ChunkMetaData,
    Context,
    Document,
    HChunk,
    NormalizedContent,
    RChunk,
    Section,
)
from data_layer.ingestion.normalizer.normalizer import (
    NormalizationProfiles,
    TextNormalizer,
)
from data_layer.ingestion.TextFileProcessor.text_extractor import TextExtractor
from data_layer.vector_db_manager.vectorDbManager import VectorDbManager


class TestNonMemoryBugs(unittest.TestCase):

    def test_bug_1_1_config_paths(self):
        self.assertTrue(os.path.isabs(Config.DB_PATH))
        self.assertTrue(os.path.isabs(Config.INDEX_PATH))

    def test_bug_2_1_and_2_2_normalizer(self):
        # Normalizer should not match blank lines as sections
        normalizer = NormalizationProfiles.rag_ingestion()
        text = "\n   \n\nHELLO WORLD\n"
        # Since it uses r"^(?=.*[A-Z])[A-Z\s]+$", "   " won't match, but "HELLO WORLD" will.
        norm_content = normalizer.normalize_text("test.txt", text)
        self.assertTrue(norm_content.has_section)

        # Test lowercasing does not break has_section
        normalizer_lower = TextNormalizer(lowercase=True)
        text_lower = "HELLO SECTION\n\nSome content."
        norm_content_lower = normalizer_lower.normalize_text("test2.txt", text_lower)
        self.assertTrue(norm_content_lower.has_section)

    def test_bug_2_3_text_extractor(self):
        # Markup is stripped rather than passed through: tags and inline
        # scripts used to be chunked and embedded along with the prose.
        with open("test.csv", "w") as f:
            f.write("a,b,c\n1,2,3")
        with open("test.html", "w") as f:
            f.write("<html><body><script>evil()</script><p>Hello</p></body></html>")
        with open("test.xml", "w") as f:
            f.write("<note><body>Reminder</body></note>")

        extractor = TextExtractor()
        _, csv_text = extractor.extract_text_from_file("test.csv")
        self.assertEqual(csv_text, "a,b,c\n1,2,3")

        _, html_text = extractor.extract_text_from_file("test.html")
        self.assertIn("Hello", html_text)
        self.assertNotIn("<p>", html_text)
        self.assertNotIn("evil()", html_text)

        _, xml_text = extractor.extract_text_from_file("test.xml")
        self.assertIn("Reminder", xml_text)
        self.assertNotIn("<body>", xml_text)

        os.remove("test.csv")
        os.remove("test.html")
        os.remove("test.xml")

    def test_bug_2_4_exception_format(self):
        e = IndexDirectoryDoesNotExists("fake_dir")
        self.assertTrue("fake_dir" in str(e))

    def test_bug_3_1_db_manager(self):
        # Should not throw UnboundLocalError when list is empty
        m = Manager(":memory:", True)
        m.insert_chunks([])
        m.insert_contexts([])
        m.insert_documents([])
        m.insert_sections([])
        m.close()

    def test_bug_3_2_and_3_3_hierarchical_chunker(self):
        # Preamble text should not be discarded
        text = "This is a preamble.\n\nMY SECTION\nThis is content."
        normalizer = NormalizationProfiles.rag_ingestion()
        norm_content = normalizer.normalize_text("test.txt", text)
        chunker = HierarchicalChunker(
            chunkOverlap=10,
            chunkSize=50,
            db_path=":memory:",
            normalizedDocumentsContents=[norm_content],
        )
        chunks = chunker.process_doc()
        # Should contain chunks for preamble and section
        self.assertTrue(any("This is a preamble" in c.chunk for c in chunks))
        self.assertTrue(any("This is content" in c.chunk for c in chunks))

    def test_bug_3_4_and_3_5_recursive_chunker(self):
        text = "A" * 100 + "\n\n" + "B" * 100
        normalizer = NormalizationProfiles.rag_ingestion()
        norm_content = normalizer.normalize_text("test.txt", text)
        # chunkSize=10, overlap=5
        chunker = RecursiveChunker([norm_content], chunk_size=10, overlap=5)
        chunks = chunker.recursive_chunker()
        for chunk in chunks:
            self.assertLessEqual(len(chunk.chunk), 10)

    def test_bug_4_1_embedding_manager(self):
        embedder = EmbeddingManager()
        # Use dummy chunks
        chunks = [
            RChunk(f"text {i}", ChunkMetaData("test", "test", "r"), str(i))
            for i in range(100)
        ]
        embedded = embedder.embed(chunks)
        self.assertEqual(len(embedded), 100)

    def test_bug_4_2_ingestion_pipeline(self):
        pipeline = IngestionPipeline()
        self.assertEqual(pipeline.vector_db.graph_degree, Config.GRAPH_DEGREE)
        self.assertEqual(pipeline.vector_db.k_neighbors, Config.K_NEIGHBORS)

    def test_bug_5_1_vector_db_manager_lock(self):
        v = VectorDbManager("l2", np.float32, 128, 1000, 100, 120, 4, 9)
        self.assertTrue(hasattr(v, "lock"))


if __name__ == "__main__":
    unittest.main()
