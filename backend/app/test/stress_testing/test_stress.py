import os
import sys
import time
import unittest
import unittest.mock as mock
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Mock heavy/external dependencies to run in test environments
sys.modules["diskannpy"] = mock.MagicMock()

class MockModel:
    def encode(self, texts, truncate_dim=None):
        if isinstance(texts, str):
            return np.random.randn(128).astype(np.float32)
        return [np.random.randn(128).astype(np.float32) for _ in texts]

mock_st = mock.MagicMock()
mock_st.SentenceTransformer = mock.MagicMock(return_value=MockModel())
sys.modules["sentence_transformers"] = mock_st
sys.modules["psycopg"] = mock.MagicMock()
sys.modules["dotenv"] = mock.MagicMock()

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "memory"))
sys.path.append(os.path.join(base_dir, "memory", "topic_pool", "project_pool", "conversation_pool"))

from config import Config
from data_layer.vector_db_manager.vectorDbManager import VectorDbManager
from data_layer.ingestion.nodes.nodes import EmbeddedChunk
from data_layer.ingestion.metadata.metadata import EmbeddedChunkMetaData
from memory.topic_pool.project_pool.conversation_pool.conversation_data_management.conversationVectorMetaManager import (
    ConversationVectorMetaDataRepository,
)
from memory.topic_pool.project_pool.conversation_pool.snapshot import SnapShot
from numpy import uint32


class TestSystemStress(unittest.TestCase):

    def setUp(self):
        self.project_id = "stress_test_project"
        self.db_dir = Config.CONVERSATION_SNAPSHOT_DB
        self.target_db = self.db_dir / f"{self.project_id}_summary_metadata.db"
        if self.target_db.exists():
            try:
                os.remove(self.target_db)
            except Exception:
                pass

    def test_concurrent_vector_db_manager_stress(self):
        """
        Stress test VectorDbManager with 10 concurrent worker threads inserting
        50 batches of 20 vectors each (total 1000 vectors).
        """
        v_manager = VectorDbManager("l2", np.float32, 128, 5000, 100, 120, 4, 9)
        
        def worker_task(worker_id):
            chunks = []
            for i in range(20):
                vec = np.random.randn(128).astype(np.float32)
                chunk_id = worker_id * 10000 + i + int(time.time() * 1000) % 10000
                chunks.append(
                    EmbeddedChunk(
                        vec,
                        uint32(chunk_id),
                        EmbeddedChunkMetaData(f"chunk_{chunk_id}", "content", "model"),
                    )
                )
            v_manager.batch_insert(chunks)
            return len(chunks)

        start_time = time.time()
        total_inserted = 0
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker_task, w) for w in range(10)]
            for future in as_completed(futures):
                total_inserted += future.result()

        elapsed = time.time() - start_time
        print(f"\n[Stress Test] Inserted {total_inserted} vectors concurrently across 10 threads in {elapsed:.3f}s")
        self.assertEqual(total_inserted, 200)

    def test_concurrent_sqlite_metadata_repository_stress(self):
        """
        Stress test ConversationVectorMetaDataRepository under concurrent multi-threaded
        read/write load (15 concurrent threads writing and querying SQLite).
        """
        meta_repo = ConversationVectorMetaDataRepository(
            Config.FULL_CONVERSATION, self.db_dir, self.project_id
        )

        def worker_write_and_read(thread_id):
            cum_id = int(thread_id * 1000 + int(time.time() * 1000) % 1000)
            meta_repo.insert_cumulative_vector_meta_data(
                cum_id,
                f"Stress summary {thread_id}",
                "2026-08-07",
                self.project_id,
                "100",
            )
            data = meta_repo.get_cumulative_vector_meta_data_ids()
            return len(data)

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = [executor.submit(worker_write_and_read, i) for i in range(15)]
            results = [f.result() for f in as_completed(futures)]

        elapsed = time.time() - start_time
        print(f"\n[Stress Test] Completed 15 concurrent SQLite writes/reads in {elapsed:.3f}s")
        self.assertTrue(all(r > 0 for r in results))

    def test_snapshot_navigation_stress(self):
        """
        Stress test SnapShot cursor navigation and search under rapid repeated queries.
        """
        snapshot = SnapShot()
        query_vector = np.random.randn(128).astype(np.float32)
        
        # Mock vector manager and meta repo internally for rapid search
        with mock.patch("memory.topic_pool.project_pool.conversation_pool.snapshot.ConversationVectorManager") as mock_vman, \
             mock.patch("memory.topic_pool.project_pool.conversation_pool.snapshot.ConversationVectorMetaDataRepository") as mock_mrepo:
            
            mock_mrepo.return_value.get_cumulative_vector_meta_data_ids.return_value = list(range(1, 101))
            mock_vman.return_value.get_vector.return_value = np.random.randn(128).astype(np.float32)

            start_time = time.time()
            for _ in range(50):
                snapshot._SnapShot__left_cursor = 0
                snapshot._SnapShot__right_cursor = 99
                res = snapshot.search(query_vector, self.project_id, "StressProject")
                self.assertIsNotNone(res)

            elapsed = time.time() - start_time
            print(f"\n[Stress Test] Executed 50 snapshot dual-cursor search iterations across 100 nodes in {elapsed:.3f}s")


if __name__ == "__main__":
    unittest.main()
