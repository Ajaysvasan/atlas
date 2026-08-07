import unittest
import sys
from unittest.mock import patch, MagicMock
sys.modules["psycopg"] = MagicMock()
sys.modules["dotenv"] = MagicMock()

from pathlib import Path
import os
import shutil
import numpy as np

from memory.topic_pool.project_pool.conversation_pool.conversation_data_management.conversationVectorMetaManager import ConversationVectorMetaDataRepository
from memory.topic_pool.project_pool.conversation_pool.conversation_data_management.conversationVectorManager import ConversationVectorManager


class TestConversationVectorMetaDataManager(unittest.TestCase):
    def setUp(self):
        self.project_id = "test_project_123"
        self.full_dir = Path("./test_full_conversation")
        self.summary_dir = Path("./test_summary")
        self.manager = ConversationVectorMetaDataRepository(
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
        self.assertTrue(self.manager.db_path.exists())

    def test_summary_chunks_batch_insert(self):
        records = [
            ("chunk_1", "text 1", "2026-08-01", "typeA"),
            ("chunk_2", "text 2", "2026-08-01", "typeB"),
        ]
        self.manager.batch_insert_summary_chunks(records)
        # Verify via manual query
        import sqlite3
        with sqlite3.connect(self.manager.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM summary_chunks")
            self.assertEqual(len(c.fetchall()), 2)

    def test_summary_vector_meta_data(self):
        # Insert chunk first due to foreign key
        self.manager.batch_insert_summary_chunks([("chunk_1", "text 1", "2026-08-01", "typeA")])
        records = [
            (101, "chunk_1", self.project_id),
            (102, "chunk_1", self.project_id)
        ]
        self.manager.batch_insert_summary_vector_meta_data(records)
        
        single = self.manager.get_summary_vector_meta_data(101)
        self.assertEqual(single[0], 101)
        
        batch = self.manager.batch_get_summary_vector_meta_data([101, 102])
        self.assertEqual(len(batch), 2)

    def test_cumulative_vector_meta_data(self):
        self.manager.insert_cumulative_vector_meta_data(201, "cum summary 1", "2026-08-01", self.project_id, "13")
        self.manager.batch_insert_cumulative_vector_meta_data([
            (202, "cum summary 2", "2026-08-02", self.project_id, "13"),
            (203, "cum summary 3", "2026-08-03", self.project_id, "13")
        ])
        
        single = self.manager.get_cumulative_vector_meta_data(201)
        self.assertEqual(single[0], 201)
        
        batch = self.manager.batch_get_cumulative_vector_meta_data([201, 202])
        self.assertEqual(len(batch), 2)

    def test_map_table(self):
        # Insert parent records to satisfy foreign keys
        self.manager.batch_insert_summary_chunks([("chunk_1", "text 1", "2026-08-01", "typeA")])
        self.manager.batch_insert_summary_vector_meta_data([(101, "chunk_1", self.project_id), (102, "chunk_1", self.project_id)])
        self.manager.insert_cumulative_vector_meta_data(201, "cum summary 1", "2026-08-01", self.project_id, "13")
        self.manager.insert_cumulative_vector_meta_data(202, "cum summary 2", "2026-08-01", self.project_id, "13")
        
        # Test map table logic
        self.manager.insert_map_table(201, 101)
        self.manager.batch_insert_map_table([
            (201, 102),
            (202, 101)
        ])
        
        ids_201 = self.manager.get_summary_vector_ids_from_map(201)
        self.assertEqual(set(ids_201), {101, 102})
        
        ids_202 = self.manager.get_summary_vector_ids_from_map(202)
        self.assertEqual(ids_202, [101])

    def test_bug_2_1_numpy_uint32_type_coercion(self):
        """
        Bug 2.1 regression: Verifies that numpy.uint32 scalar values are properly
        coerced to native Python int before being bound to SQLite INTEGER columns.
        Previously, passing numpy.uint32 directly caused sqlite3.IntegrityError: datatype mismatch.
        """
        cum_id = np.uint32(501)
        summary_vec_id = np.uint32(601)

        # Insert cumulative with numpy.uint32 cumulative_vector_id and integer len
        self.manager.insert_cumulative_vector_meta_data(
            cum_id, "numpy uint32 summary", "2026-08-07", self.project_id, 42
        )

        # Read it back to verify
        result = self.manager.get_cumulative_vector_meta_data(cum_id)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 501)

        # Insert chunks + summary vector with numpy.uint32 summary_vector_id
        self.manager.batch_insert_summary_chunks([("np_chunk", "text", "2026-08-07", "typeX")])
        self.manager.batch_insert_summary_vector_meta_data([(int(summary_vec_id), "np_chunk", self.project_id)])

        # Read back summary vector meta data with numpy.uint32
        sv_result = self.manager.get_summary_vector_meta_data(summary_vec_id)
        self.assertIsNotNone(sv_result)
        self.assertEqual(sv_result[0], 601)

        # Insert map entry with numpy.uint32 on both sides
        self.manager.insert_map_table(cum_id, summary_vec_id)
        map_ids = self.manager.get_summary_vector_ids_from_map(cum_id)
        self.assertIn(601, map_ids)



class TestConversationVectorManager(unittest.TestCase):
    @patch('memory.topic_pool.project_pool.conversation_pool.conversation_data_management.conversationVectorManager.VectorRepository')
    def setUp(self, MockVectorRepository):
        self.mock_repo = MockVectorRepository.return_value
        self.manager = ConversationVectorManager(project_name="TestProject", project_id="proj123")

    def test_insert_vector(self):
        vid = np.uint32(101)
        vector = np.array([0.1, 0.2, 0.3])
        
        result_vid = self.manager.insert(vid, vector)
        self.assertEqual(result_vid, vid)
        self.manager.repository.insert.assert_called_once_with(vid, vector)

    def test_batch_insert(self):
        vids = [np.uint32(101), np.uint32(102)]
        vectors = np.array([[0.1], [0.2]])
        
        result_vids = self.manager.batch_insert(vids, vectors)
        self.assertEqual(len(result_vids), 2)
        self.manager.repository.batch_insert.assert_called_once_with(vids, vectors)

    def test_get_vector(self):
        self.mock_repo.search.return_value = np.array([1.0, 2.0])
        result = self.manager.get_vector(np.uint32(100))
        
        self.manager.repository.search.assert_called_once_with(np.uint32(100))
        self.assertTrue(np.array_equal(result, np.array([1.0, 2.0])))

    def test_get_vectors(self):
        self.mock_repo.batch_search.return_value = np.array([[1.0], [2.0]])
        result = self.manager.get_vectors([np.uint32(100), np.uint32(200)])
        
        self.manager.repository.batch_search.assert_called_once_with([np.uint32(100), np.uint32(200)])
        self.assertTrue(np.array_equal(result, np.array([[1.0], [2.0]])))

if __name__ == '__main__':
    unittest.main()
