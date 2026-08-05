import unittest
import torch
import sys
import os

# Add the parent directory to sys.path to resolve imports correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "memory"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "memory/topic_pool/project_pool/conversation_pool"))

from memory.topic_pool.project_pool.conversation_pool.snapshot import SnapShot
from memory.topic_pool.project_pool.conversation_pool.snapShotNodes import SnapShotNode
from numpy import uint32

class TestSnapShotBugs(unittest.TestCase):
    def setUp(self):
        self.snapshot = SnapShot()
        
        # Add a couple of snapshots to populate the list
        self.snapshot.add(
            snapshot_id="snap-1",
            time_of_snapshot="2026-08-01",
            size_of_the_summary=100,
            len_of_the_summary=10,
            summary_vector_ids=[],
            conversation_id="conv-1",
            cumulative_summary_vector_id=uint32(1),
            project_id="test_proj"
        )
        self.snapshot.add(
            snapshot_id="snap-2",
            time_of_snapshot="2026-08-01",
            size_of_the_summary=100,
            len_of_the_summary=10,
            summary_vector_ids=[],
            conversation_id="conv-1",
            cumulative_summary_vector_id=uint32(2),
            project_id="test_proj"
        )

    def test_bug_1_1_cosine_similarity_crash(self):
        """
        Tests Bug 1.1: `__find_best_snapshot` uses empty tensor([]) and crashes during cosine_similarity.
        """
        query = [0.1, 0.2, 0.3]
        with self.assertRaises(RuntimeError) as context:
            self.snapshot.search(query)
            
        self.assertIn("The size of tensor a (0) must match the size of tensor b", str(context.exception))

    def test_bug_1_2_cursor_clobbering(self):
        """
        Tests Bug 1.2: `add()` clobbers the active cursors because reset is default True.
        """
        self.snapshot._SnapShot__left_cursor = 1
        
        self.snapshot.add(
            snapshot_id="snap-3",
            time_of_snapshot="2026-08-01",
            size_of_the_summary=100,
            len_of_the_summary=10,
            summary_vector_ids=[],
            conversation_id="conv-1",
            cumulative_summary_vector_id=uint32(3),
            project_id="test_proj"
        )
        
        # Left cursor gets reset to 0 even though we set it to 1!
        self.assertEqual(self.snapshot._SnapShot__left_cursor, 0)

if __name__ == '__main__':
    unittest.main()
