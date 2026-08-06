import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import sys
import os

sys.modules["psycopg"] = MagicMock()
sys.modules["dotenv"] = MagicMock()

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "memory"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "memory", "topic_pool", "project_pool", "conversation_pool"))

from memory.topic_pool.project_pool.conversation_pool.snapshot import SnapShot
from numpy import uint32

class TestSnapShot(unittest.TestCase):
    def setUp(self):
        self.meta_patcher = patch('memory.topic_pool.project_pool.conversation_pool.snapshot.ConversationVectorMetaDataRepository')
        self.vector_patcher = patch('memory.topic_pool.project_pool.conversation_pool.snapshot.ConversationVectorManager')
        
        self.MockMetaDataRepository = self.meta_patcher.start()
        self.MockVectorManager = self.vector_patcher.start()
        
        self.mock_meta_repo = self.MockMetaDataRepository.return_value
        self.mock_vector_manager = self.MockVectorManager.return_value
        
        self.snapshot = SnapShot()
        self.project_id = "test_proj"
        self.project_name = "TestProject"
        
        # Setup mock return values for getting snapshot ids
        self.mock_meta_repo.get_cumulative_vector_meta_data_ids = MagicMock(return_value=[1, 2, 3])
        self.snapshot._SnapShot__get_snap_shot = MagicMock(return_value=[1, 2, 3])
        self.mock_vector_manager.get_vector = MagicMock(return_value=np.array([0.1, 0.2, 0.3]))
        
    def tearDown(self):
        self.meta_patcher.stop()
        self.vector_patcher.stop()

    def test_add_snapshot_success(self):
        """
        Test that adding a snapshot properly invokes all dependencies.
        Assumes memory module is production ready.
        """
        # Call add with appropriate arguments
        self.snapshot.add(
            project_name=self.project_name,
            project_id=self.project_id,
            time_of_snapshot="2026-08-01",
            len_of_the_summary=100,
            summary_vector_ids=[uint32(101), uint32(102)],
            summary_vectors=np.array([[0.1], [0.2]]),
            chunk_ids=["chunk_1", "chunk_2"],
            summary="This is a summary",
            cumulative_summary_vector_id=uint32(1),
            cumulative_summary_vector=np.array([0.1, 0.2]),
            reset_right_pointer=True,
            reset_left_pointer=True
        )

        # In a bug-free production implementation, pointers should be properly reset
        self.assertEqual(self.snapshot._SnapShot__left_cursor, 0)
        self.assertEqual(self.snapshot._SnapShot__right_cursor, 2) # Since len of list is 3 (index 2)

    def test_search_returns_best_snapshot(self):
        """
        Tests that searching for a vector query returns the most relevant snapshot index.
        Assumes memory module is production ready.
        """
        query = np.array([0.1, 0.2, 0.3])
        
        # Manually configure the cursor pointers to ensure they are valid before search
        self.snapshot._SnapShot__left_cursor = 0
        self.snapshot._SnapShot__right_cursor = 2
        
        # Best snapshot index (search currently fails due to the tensor size bug internally in snapshot.py)
        # We assume it should run properly here
        best_snapshot = self.snapshot.search(
            query=query, 
            project_id=self.project_id, 
            project_name=self.project_name
        )
        self.assertIsNotNone(best_snapshot)

    def test_advance_and_prev_cursors(self):
        """
        Tests advancing cursors correctly traverses without invalid exceptions.
        """
        self.snapshot._SnapShot__left_cursor = 0
        self.snapshot._SnapShot__right_cursor = 2
        
        self.snapshot.advance(self.project_id)
        self.assertEqual(self.snapshot._SnapShot__left_cursor, 1)
        
        self.snapshot.prev()
        self.assertEqual(self.snapshot._SnapShot__right_cursor, 1)

if __name__ == '__main__':
    unittest.main()
