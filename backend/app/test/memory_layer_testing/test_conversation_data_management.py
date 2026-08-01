import unittest
import sys
from unittest.mock import patch, MagicMock
sys.modules["psycopg"] = MagicMock()
sys.modules["dotenv"] = MagicMock()

from pathlib import Path
import os
import shutil
import numpy as np

from memory.topic_pool.project_pool.conversation_pool.conversation_data_management.conversationVectorMetaManager import ConversationVectorMetaDataManager
from memory.topic_pool.project_pool.conversation_pool.conversation_data_management.conversatoinVectorManager import ConversationVectorManager


class TestConversationVectorMetaDataManager(unittest.TestCase):
    def setUp(self):
        self.project_id = "test_project_123"
        self.full_dir = Path("./test_full_conversation")
        self.summary_dir = Path("./test_summary")
        self.manager = ConversationVectorMetaDataManager(
            full_conversation_dir=self.full_dir,
            summary_dir=self.summary_dir,
            project_id=self.project_id
        )

    def tearDown(self):
        if self.full_dir.exists():
            shutil.rmtree(self.full_dir)
        if self.summary_dir.exists():
            shutil.rmtree(self.summary_dir)

    def test_initialization_creates_directories_and_dbs(self):
        self.assertTrue(self.full_dir.exists())
        self.assertTrue(self.summary_dir.exists())
        self.assertTrue(self.manager.full_db_path.exists())
        self.assertTrue(self.manager.summary_db_path.exists())

    def test_single_insert_and_get_full_conversation(self):
        vector_id = 101
        chunk_id = "chunk_101"
        chunk_text = "Hello world full conversation"
        
        self.manager.insert_full_conversation_chunk(vector_id, chunk_id, chunk_text)
        result = self.manager.get_full_conversation_chunk(vector_id)
        
        self.assertIsNotNone(result)
        self.assertEqual(result[0], vector_id)
        self.assertEqual(result[1], chunk_id)
        self.assertEqual(result[2], chunk_text)

    def test_single_insert_and_get_summary(self):
        vector_id = 202
        chunk_id = "chunk_202"
        chunk_text = "Hello world summary"
        
        self.manager.insert_summary_chunk(vector_id, chunk_id, chunk_text)
        result = self.manager.get_summary_chunk(vector_id)
        
        self.assertIsNotNone(result)
        self.assertEqual(result[0], vector_id)
        self.assertEqual(result[1], chunk_id)
        self.assertEqual(result[2], chunk_text)

    def test_batch_insert(self):
        records = [
            (301, "c301", "chunk text 1"),
            (302, "c302", "chunk text 2"),
            (303, "c303", "chunk text 3"),
        ]
        
        self.manager.batch_insert_full_conversation_chunks(records)
        
        res1 = self.manager.get_full_conversation_chunk(301)
        res3 = self.manager.get_full_conversation_chunk(303)
        
        self.assertEqual(res1[2], "chunk text 1")
        self.assertEqual(res3[2], "chunk text 3")


class TestConversationVectorManager(unittest.TestCase):
    @patch('memory.topic_pool.project_pool.conversation_pool.conversation_data_management.conversatoinVectorManager.VectorRepository')
    def setUp(self, MockVectorRepository):
        self.mock_repo = MockVectorRepository.return_value
        self.manager = ConversationVectorManager(project_name="TestProject", project_id="proj123")

    def test_generate_vector_id(self):
        vid1 = self.manager.generate_vector_id("chunk1")
        vid2 = self.manager.generate_vector_id("chunk2")
        vid1_again = self.manager.generate_vector_id("chunk1")
        
        self.assertIsInstance(vid1, int)
        self.assertNotEqual(vid1, vid2)
        self.assertEqual(vid1, vid1_again)

    def test_insert_vector(self):
        chunk_id = "chunk_1"
        vector = np.array([0.1, 0.2, 0.3])
        
        vid = self.manager.insert(chunk_id, vector)
        self.assertIsInstance(vid, int)
        self.manager.repository.insert.assert_called_once_with(vid, vector)

    def test_batch_insert(self):
        chunk_ids = ["chunk_1", "chunk_2"]
        vectors = np.array([[0.1], [0.2]])
        
        vids = self.manager.batch_insert(chunk_ids, vectors)
        self.assertEqual(len(vids), 2)
        self.manager.repository.batch_insert.assert_called_once_with(vids, vectors)

    def test_get_vector(self):
        self.mock_repo.search.return_value = np.array([1.0, 2.0])
        result = self.manager.get_vector(100)
        
        self.manager.repository.search.assert_called_once_with(100)
        self.assertTrue(np.array_equal(result, np.array([1.0, 2.0])))

    def test_get_vectors(self):
        self.mock_repo.batch_search.return_value = np.array([[1.0], [2.0]])
        result = self.manager.get_vectors([100, 200])
        
        self.manager.repository.batch_search.assert_called_once_with([100, 200])
        self.assertTrue(np.array_equal(result, np.array([[1.0], [2.0]])))

if __name__ == '__main__':
    unittest.main()
