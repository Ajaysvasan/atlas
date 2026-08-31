"""
Stress tests: the layers that carry real load, put under volume and concurrency.

Nothing here reaches an external service. diskannpy and psycopg are stubbed
(neither is installed for every interpreter that runs this suite) and
SentenceTransformer is replaced with a random-vector stand-in, so what these
tests exercise is the project's own locking, id derivation, SQLite access and
offset arithmetic — not the third-party index or model.

Every test writes into a fresh temporary directory. These tests used to write
into the real data/ tree through Config, so a stress run polluted the working
corpus and consecutive runs interfered with each other.
"""

import os
import sqlite3
import sys
import tempfile
import time
import unittest
import unittest.mock as mock
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


class _StubModel:
    """Stands in for SentenceTransformer: the shape of the output is all the
    pipeline cares about, and loading the real model would download weights."""

    def encode(self, texts, truncate_dim=None):
        dimensions = truncate_dim or 128
        if isinstance(texts, str):
            return np.random.randn(dimensions).astype(np.float32)
        return [np.random.randn(dimensions).astype(np.float32) for _ in texts]


_stub_sentence_transformers = mock.MagicMock()
_stub_sentence_transformers.SentenceTransformer = mock.MagicMock(
    return_value=_StubModel()
)

sys.modules.setdefault("diskannpy", mock.MagicMock())
sys.modules.setdefault("psycopg", mock.MagicMock())
sys.modules.setdefault("dotenv", mock.MagicMock())
sys.modules["sentence_transformers"] = _stub_sentence_transformers

_APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from config import Config
from data_layer.ingestion.Chunker.chunker import Chunker
from data_layer.ingestion.metadata.metadata import EmbeddedChunkMetaData
from data_layer.ingestion.nodes.nodes import EmbeddedChunk
from data_layer.ingestion.embedding.EmbeddingManager import EmbeddingManager
from data_layer.ingestion.normalizer.normalizer import NormalizationProfiles
from data_layer.vector_db_manager.vectorDbManager import VectorDbManager
from memory.topic_pool.project_pool.conversation_pool.conversation_data_management.conversationVectorMetaManager import (
    ConversationVectorMetaDataRepository,
)
from memory.topic_pool.project_pool.conversation_pool.fullconversation_repository.fullconversation_repository import (
    FullConversationRepository,
)
from memory.topic_pool.project_pool.conversation_pool.snapshot import SnapShot

CHUNK_SIZE = 256
CHUNK_OVERLAP = 20
EMBEDDING_DIMENSIONS = 128


def report(message: str) -> None:
    print(f"\n[stress] {message}")


# Repeated verbatim in every synthetic document, the way a licence header or a
# boilerplate disclaimer repeats across a real corpus. It is what gives the
# vector-id assertions teeth: deriving an id from chunk text collapses every
# copy of this paragraph onto one id, and all but one copy stops being
# retrievable.
BOILERPLATE = " ".join(f"boilerplate{t}" for t in range(60))


def sectioned_document(index: int, sections: int = 8, paragraphs: int = 5) -> str:
    body = [f"# Document {index}\n", BOILERPLATE + "\n\n"]
    for section in range(sections):
        body.append(f"## Section {section}\n")
        body.append(BOILERPLATE + "\n\n")
        for paragraph in range(paragraphs):
            words = " ".join(f"w{index}s{section}p{paragraph}t{t}" for t in range(60))
            body.append(words + "\n\n")
    return "".join(body)


def flat_document(index: int, paragraphs: int = 10) -> str:
    body = [BOILERPLATE]
    body.extend(
        " ".join(f"f{index}p{paragraph}t{t}" for t in range(80))
        for paragraph in range(paragraphs)
    )
    return "\n\n".join(body)


def synthetic_corpus(sectioned: int, flat: int) -> dict:
    corpus = {f"/synthetic/doc_{i}.md": sectioned_document(i) for i in range(sectioned)}
    corpus.update({f"/synthetic/flat_{i}.txt": flat_document(i) for i in range(flat)})
    return corpus


def table_counts(db_path: str) -> dict:
    connection = sqlite3.connect(db_path)
    try:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        ]
        return {
            table: connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in tables
        }
    finally:
        connection.close()


class StressTestCase(unittest.TestCase):
    """Gives every test its own temporary directory, removed afterwards."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="stress_")
        self.tmp_dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def db_file(self, name: str) -> str:
        return str(self.tmp_dir / name)


class RecordingIndex:
    """Stand-in for diskannpy's DynamicMemoryIndex that remembers what it got.

    The sleep is what makes the serialization assertion meaningful: it widens
    the window in which a second thread could enter, so a VectorDbManager that
    stopped taking its lock trips `concurrent_entries` rather than passing by
    luck.
    """

    def __init__(self):
        self.ids = []
        self.concurrent_entries = 0
        self._inside = 0

    def _enter(self):
        self._inside += 1
        if self._inside > 1:
            self.concurrent_entries += 1

    def _leave(self):
        self._inside -= 1

    def insert(self, vector, vector_id):
        self._enter()
        time.sleep(0.001)
        self.ids.append(int(vector_id))
        self._leave()

    def batch_insert(self, vectors, vector_ids):
        self._enter()
        time.sleep(0.001)
        self.ids.extend(int(vector_id) for vector_id in vector_ids)
        self._leave()


class TestVectorStoreUnderConcurrency(StressTestCase):

    WORKERS = 12
    BATCHES_PER_WORKER = 5
    VECTORS_PER_BATCH = 50
    SINGLE_INSERTS_PER_WORKER = 20

    def make_manager(self):
        manager = VectorDbManager(
            "l2", np.float32, EMBEDDING_DIMENSIONS, 100000, 100, 120, 4, 9
        )
        index = RecordingIndex()
        manager.vector_db.dynamic_dann = index
        return manager, index

    def test_concurrent_batch_inserts_are_serialised_and_complete(self):
        """Every vector reaches the index exactly once, one writer at a time."""
        manager, index = self.make_manager()

        def worker(worker_id):
            # A generator per thread: numpy's is not thread-safe, and sharing
            # one would make the test itself the race it is looking for.
            rng = np.random.default_rng(worker_id)
            written = []
            for batch in range(self.BATCHES_PER_WORKER):
                chunks = []
                for position in range(self.VECTORS_PER_BATCH):
                    vector_id = worker_id * 1000000 + batch * 1000 + position
                    chunks.append(
                        EmbeddedChunk(
                            rng.standard_normal(EMBEDDING_DIMENSIONS).astype(np.float32),
                            vector_id,
                            EmbeddedChunkMetaData(
                                f"chunk_{vector_id}", "content", "stub-model"
                            ),
                        )
                    )
                    written.append(vector_id)
                manager.batch_insert(chunks)
            return written

        started = time.time()
        submitted = []
        with ThreadPoolExecutor(max_workers=self.WORKERS) as executor:
            futures = [executor.submit(worker, w) for w in range(self.WORKERS)]
            for future in as_completed(futures):
                submitted.extend(future.result())
        elapsed = time.time() - started

        report(
            f"{len(submitted)} vectors inserted by {self.WORKERS} threads in {elapsed:.3f}s"
        )
        self.assertEqual(
            len(submitted),
            self.WORKERS * self.BATCHES_PER_WORKER * self.VECTORS_PER_BATCH,
        )
        self.assertEqual(index.concurrent_entries, 0)
        self.assertEqual(sorted(index.ids), sorted(submitted))

    def test_concurrent_single_inserts_lose_nothing(self):
        manager, index = self.make_manager()
        total = self.WORKERS * self.SINGLE_INSERTS_PER_WORKER

        def worker(worker_id):
            rng = np.random.default_rng(worker_id)
            for position in range(self.SINGLE_INSERTS_PER_WORKER):
                vector_id = worker_id * 1000 + position
                manager.insert(
                    EmbeddedChunk(
                        rng.standard_normal(EMBEDDING_DIMENSIONS).astype(np.float32),
                        vector_id,
                        EmbeddedChunkMetaData(f"chunk_{vector_id}", "c", "stub-model"),
                    )
                )

        started = time.time()
        with ThreadPoolExecutor(max_workers=self.WORKERS) as executor:
            for future in [executor.submit(worker, w) for w in range(self.WORKERS)]:
                future.result()
        elapsed = time.time() - started

        report(f"{total} single inserts across {self.WORKERS} threads in {elapsed:.3f}s")
        self.assertEqual(index.concurrent_entries, 0)
        self.assertEqual(len(index.ids), total)
        self.assertEqual(len(set(index.ids)), total)


class TestIngestionUnderLoad(StressTestCase):

    SECTIONED_DOCUMENTS = 150
    FLAT_DOCUMENTS = 150
    CHUNKER_THREADS = 6

    def setUp(self):
        super().setUp()
        self.normalizer = NormalizationProfiles.rag_ingestion()
        self.corpus = synthetic_corpus(self.SECTIONED_DOCUMENTS, self.FLAT_DOCUMENTS)

    def test_full_pipeline_holds_its_invariants_over_a_large_corpus(self):
        """Normalize, chunk and embed the whole synthetic corpus in one pass.

        The assertions are the pipeline's structural contract: offsets index
        back into the normalized document, no chunk exceeds chunk_size, and
        neither chunk ids nor vector ids collide. A vector id collision is the
        quiet one — colliding ids overwrite each other in the index and the
        losing chunks become permanently unsearchable.
        """
        started = time.time()
        normalized = self.normalizer.normalize_all(self.corpus)
        normalize_seconds = time.time() - started

        chunker = Chunker(CHUNK_SIZE, CHUNK_OVERLAP, self.db_file("chunks.db"))
        started = time.time()
        h_chunks, r_chunks = chunker.chunk_per_document(normalized)
        chunk_seconds = time.time() - started

        chunks = h_chunks + r_chunks
        started = time.time()
        embedded = EmbeddingManager().embed(chunks)
        embed_seconds = time.time() - started

        report(
            f"{len(normalized)} documents -> {len(chunks)} chunks "
            f"(normalize {normalize_seconds:.2f}s, chunk {chunk_seconds:.2f}s, "
            f"embed {embed_seconds:.2f}s)"
        )

        self.assertEqual(len(normalized), len(self.corpus))
        sectioned = {n.meta_data.document_id for n in normalized if n.has_section}
        flat = {n.meta_data.document_id for n in normalized if not n.has_section}
        self.assertEqual({c.meta_data.document_id for c in h_chunks}, sectioned)
        self.assertEqual({c.meta_data.document_id for c in r_chunks}, flat)

        content_by_document = {
            document.meta_data.document_id: document.content for document in normalized
        }
        for chunk in chunks:
            self.assertLessEqual(len(chunk.chunk), CHUNK_SIZE)
            content = content_by_document[chunk.meta_data.document_id]
            self.assertEqual(
                content[chunk.start_off_set : chunk.end_off_set], chunk.chunk
            )

        texts = Counter(chunk.chunk for chunk in chunks)
        self.assertGreater(max(texts.values()), 1)

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        self.assertEqual(len(set(chunk_ids)), len(chunk_ids))

        vector_ids = [item.vector_id for item in embedded]
        self.assertEqual(len(set(vector_ids)), len(vector_ids))
        self.assertTrue(all(0 <= i <= Config.VECTOR_ID_MASK for i in vector_ids))
        self.assertTrue(
            all(item.vector.shape == (EMBEDDING_DIMENSIONS,) for item in embedded)
        )

    def test_concurrent_chunkers_share_one_database(self):
        """Disjoint slices of the corpus, one chunker per thread, one SQLite file."""
        normalized = self.normalizer.normalize_all(self.corpus)
        db_path = self.db_file("shared.db")
        slices = [
            normalized[offset :: self.CHUNKER_THREADS]
            for offset in range(self.CHUNKER_THREADS)
        ]

        def worker(documents):
            return Chunker(CHUNK_SIZE, CHUNK_OVERLAP, db_path).chunk_per_document(
                documents
            )

        started = time.time()
        produced = 0
        with ThreadPoolExecutor(max_workers=self.CHUNKER_THREADS) as executor:
            futures = [executor.submit(worker, documents) for documents in slices]
            for future in as_completed(futures):
                h_chunks, r_chunks = future.result()
                produced += len(h_chunks) + len(r_chunks)
        elapsed = time.time() - started

        counts = table_counts(db_path)
        report(
            f"{produced} chunks written by {self.CHUNKER_THREADS} concurrent chunkers "
            f"in {elapsed:.3f}s -> {counts}"
        )
        self.assertEqual(counts["Documents"], len(normalized))
        self.assertEqual(counts["Chunks"] + counts["RecursiveChunks"], produced)

    def test_re_chunking_the_same_corpus_adds_no_rows(self):
        """Re-ingesting an unchanged corpus is a no-op, not a UNIQUE failure."""
        normalized = self.normalizer.normalize_all(self.corpus)
        db_path = self.db_file("idempotent.db")

        Chunker(CHUNK_SIZE, CHUNK_OVERLAP, db_path).chunk_per_document(normalized)
        first_pass = table_counts(db_path)

        started = time.time()
        for _ in range(3):
            Chunker(CHUNK_SIZE, CHUNK_OVERLAP, db_path).chunk_per_document(normalized)
        elapsed = time.time() - started

        report(f"3 re-ingestions of {len(normalized)} documents in {elapsed:.3f}s")
        self.assertEqual(table_counts(db_path), first_pass)


class TestConversationStoreUnderConcurrency(StressTestCase):

    PROJECT_ID = "stress_project"
    PROJECT_NAME = "StressProject"
    APPEND_WORKERS = 10
    TURNS_PER_WORKER = 20
    META_WORKERS = 12
    ROWS_PER_WORKER = 25
    SNAPSHOTS_PER_WORKER = 12
    CHUNKS_PER_SNAPSHOT = 5
    READERS = 6
    READS_PER_READER = 150
    TURNS_WHILE_READING = 150

    def test_concurrent_appends_allocate_unique_contiguous_sequences(self):
        """append_turns() takes the write lock before reading MAX(sequence_number).

        Without BEGIN IMMEDIATE two writers read the same maximum and hand out
        the same sequence_number, which is a primary key — one turn is lost and
        the conversation silently develops a hole.
        """
        repository = FullConversationRepository(
            self.tmp_dir, self.PROJECT_ID, self.PROJECT_NAME
        )

        def worker(worker_id):
            return repository.append_turns(
                [
                    ("user" if turn % 2 == 0 else "assistant", f"turn {worker_id}-{turn}")
                    for turn in range(self.TURNS_PER_WORKER)
                ]
            )

        started = time.time()
        sequences = []
        with ThreadPoolExecutor(max_workers=self.APPEND_WORKERS) as executor:
            futures = [executor.submit(worker, w) for w in range(self.APPEND_WORKERS)]
            for future in as_completed(futures):
                sequences.extend(future.result())
        elapsed = time.time() - started

        expected = self.APPEND_WORKERS * self.TURNS_PER_WORKER
        report(
            f"{expected} turns appended by {self.APPEND_WORKERS} threads in {elapsed:.3f}s"
        )
        self.assertEqual(len(sequences), expected)
        self.assertEqual(sorted(sequences), list(range(1, expected + 1)))
        self.assertEqual(repository.get_size(), expected)
        self.assertEqual(len(repository.fetch_all()), expected)

    def test_separate_repositories_on_one_database_file(self):
        """A repository per thread, each on its own connection to one file.

        Serialised by SQLite's file locking rather than by the repository's own
        lock, so it is the case the repository cannot cover.
        """
        errors = []

        def worker(worker_id):
            repository = ConversationVectorMetaDataRepository(
                self.tmp_dir, self.PROJECT_ID
            )
            try:
                for row in range(self.ROWS_PER_WORKER):
                    repository.insert_cumulative_vector_meta_data(
                        worker_id * 10000 + row,
                        f"summary {worker_id}-{row}",
                        f"2026-09-01T00:00:{row:02d}.000000+00:00",
                        self.PROJECT_ID,
                        100,
                    )
                return len(repository.get_cumulative_vector_meta_data_ids())
            finally:
                repository.close()

        started = time.time()
        with ThreadPoolExecutor(max_workers=self.META_WORKERS) as executor:
            for future in as_completed(
                [executor.submit(worker, w) for w in range(self.META_WORKERS)]
            ):
                try:
                    future.result()
                except Exception as error:
                    errors.append(f"{type(error).__name__}: {error}")
        elapsed = time.time() - started

        expected = self.META_WORKERS * self.ROWS_PER_WORKER
        reader = ConversationVectorMetaDataRepository(self.tmp_dir, self.PROJECT_ID)
        try:
            stored = reader.get_cumulative_vector_meta_data_ids()
        finally:
            reader.close()

        report(
            f"{expected} metadata rows written by {self.META_WORKERS} threads in {elapsed:.3f}s"
        )
        self.assertEqual(errors, [])
        self.assertEqual(len(stored), expected)
        self.assertEqual(len(set(stored)), expected)

    def test_one_repository_shared_by_every_thread(self):
        """The shared-instance case: one repository, one connection, N threads.

        This is how the repository is actually used — SnapShot holds one for its
        lifetime and hands the same instance to ConversationSummary — so it has
        to survive being driven from more than one thread at a time.
        """
        repository = ConversationVectorMetaDataRepository(
            self.tmp_dir, self.PROJECT_ID
        )
        self.addCleanup(repository.close)
        errors = []

        def worker(worker_id):
            for row in range(self.ROWS_PER_WORKER):
                repository.insert_cumulative_vector_meta_data(
                    worker_id * 10000 + row,
                    f"summary {worker_id}-{row}",
                    f"2026-09-01T00:00:{row:02d}.000000+00:00",
                    self.PROJECT_ID,
                    100,
                )
                repository.get_cumulative_vector_meta_data_ids()
                repository.get_latest_summary()

        started = time.time()
        with ThreadPoolExecutor(max_workers=self.META_WORKERS) as executor:
            for future in as_completed(
                [executor.submit(worker, w) for w in range(self.META_WORKERS)]
            ):
                try:
                    future.result()
                except Exception as error:
                    errors.append(f"{type(error).__name__}: {error}")
        elapsed = time.time() - started

        expected = self.META_WORKERS * self.ROWS_PER_WORKER
        stored = repository.get_cumulative_vector_meta_data_ids()
        report(
            f"{expected} interleaved reads and writes on one shared repository "
            f"across {self.META_WORKERS} threads in {elapsed:.3f}s"
        )
        self.assertEqual(errors, [])
        self.assertEqual(len(stored), expected)
        self.assertEqual(len(set(stored)), expected)

    def test_concurrent_snapshot_writes_are_all_or_nothing(self):
        """insert_snapshot is one transaction, and stays one under contention.

        Every third snapshot here reuses an existing cumulative_vector_id and so
        fails on the primary key, after its chunk rows have already been written
        inside the transaction. Those rows must not survive. On a shared
        connection without the lock they do: another thread's commit lands
        mid-transaction and publishes them, which is why turning
        check_same_thread off on its own would be worse than the ProgrammingError
        it silences.
        """
        repository = ConversationVectorMetaDataRepository(
            self.tmp_dir, self.PROJECT_ID
        )
        self.addCleanup(repository.close)
        taken_id = 999999
        repository.insert_cumulative_vector_meta_data(
            taken_id, "taken", "2026-09-01T00:00:00.000000+00:00", self.PROJECT_ID, 1
        )
        unexpected = []

        def worker(worker_id):
            for snapshot in range(self.SNAPSHOTS_PER_WORKER):
                doomed = snapshot % 3 == 0
                cumulative_id = (
                    taken_id if doomed else worker_id * 1000 + snapshot
                )
                tag = f"{'doomed' if doomed else 'good'}_{worker_id}_{snapshot}"
                chunks = [
                    (f"{tag}_c{c}", "text", "2026-09-01T00:00:00", "turn")
                    for c in range(self.CHUNKS_PER_SNAPSHOT)
                ]
                vector_rows = [
                    (
                        worker_id * 100000 + snapshot * 100 + c,
                        f"{tag}_c{c}",
                        self.PROJECT_ID,
                    )
                    for c in range(self.CHUNKS_PER_SNAPSHOT)
                ]
                try:
                    repository.insert_snapshot(
                        chunks,
                        (
                            cumulative_id,
                            "summary",
                            "2026-09-01T00:00:00.000000+00:00",
                            self.PROJECT_ID,
                            1,
                        ),
                        vector_rows,
                        [(cumulative_id, row[0]) for row in vector_rows],
                    )
                    if doomed:
                        unexpected.append(f"{tag} should have collided")
                except sqlite3.IntegrityError:
                    if not doomed:
                        unexpected.append(f"{tag} collided unexpectedly")
                except Exception as error:
                    unexpected.append(f"{tag}: {type(error).__name__}: {error}")

        started = time.time()
        with ThreadPoolExecutor(max_workers=self.META_WORKERS) as executor:
            for future in [
                executor.submit(worker, w) for w in range(self.META_WORKERS)
            ]:
                future.result()
        elapsed = time.time() - started

        doomed_per_worker = len(
            [s for s in range(self.SNAPSHOTS_PER_WORKER) if s % 3 == 0]
        )
        committed = self.META_WORKERS * (
            self.SNAPSHOTS_PER_WORKER - doomed_per_worker
        )
        connection = sqlite3.connect(repository.db_path)
        try:
            chunk_rows = connection.execute(
                "SELECT count(*) FROM summary_chunks"
            ).fetchone()[0]
            rolled_back_rows = connection.execute(
                "SELECT count(*) FROM summary_chunks WHERE chunk_id LIKE 'doomed%'"
            ).fetchone()[0]
        finally:
            connection.close()

        report(
            f"{committed} snapshots committed and "
            f"{self.META_WORKERS * doomed_per_worker} rolled back across "
            f"{self.META_WORKERS} threads in {elapsed:.3f}s"
        )
        self.assertEqual(unexpected, [])
        self.assertEqual(rolled_back_rows, 0)
        self.assertEqual(chunk_rows, committed * self.CHUNKS_PER_SNAPSHOT)
        self.assertEqual(
            len(repository.get_cumulative_vector_meta_data_ids()), committed + 1
        )


    def test_readers_and_a_writer_do_not_block_each_other(self):
        """Both repositories on one file at once, which is the production shape.

        SnapShot reads snapshot metadata while turns are being appended, and in
        the default rollback journal a reader holding a transaction open makes
        every writer fail with `database is locked`. WAL is what removes that;
        this asserts the whole workload runs clean.
        """
        conversation = FullConversationRepository(
            self.tmp_dir, self.PROJECT_ID, self.PROJECT_NAME
        )
        metadata = ConversationVectorMetaDataRepository(self.tmp_dir, self.PROJECT_ID)
        self.addCleanup(metadata.close)
        self.assertEqual(metadata.journal_mode, "wal")

        metadata.batch_insert_cumulative_vector_meta_data(
            [
                (
                    index,
                    f"summary {index}",
                    f"2026-09-01T00:00:{index % 60:02d}.000000+00:00",
                    self.PROJECT_ID,
                    1,
                )
                for index in range(50)
            ]
        )
        errors = []

        def writer():
            for turn in range(self.TURNS_WHILE_READING):
                try:
                    conversation.append_turns([("user", f"turn {turn}")])
                except Exception as error:
                    errors.append(f"writer: {type(error).__name__}: {error}")

        def reader(reader_id):
            for _ in range(self.READS_PER_READER):
                try:
                    metadata.get_cumulative_vector_meta_data_ids()
                    metadata.get_latest_summary()
                    conversation.get_n_chunks(20)
                except Exception as error:
                    errors.append(f"reader {reader_id}: {type(error).__name__}: {error}")

        started = time.time()
        with ThreadPoolExecutor(max_workers=self.READERS + 1) as executor:
            futures = [executor.submit(writer)]
            futures += [executor.submit(reader, r) for r in range(self.READERS)]
            for future in futures:
                future.result()
        elapsed = time.time() - started

        reads = self.READERS * self.READS_PER_READER
        report(
            f"{self.TURNS_WHILE_READING} appends interleaved with {reads} reads "
            f"across {self.READERS + 1} threads in {elapsed:.3f}s"
        )
        self.assertEqual(errors, [])
        self.assertEqual(conversation.get_size(), self.TURNS_WHILE_READING)


class TestSnapshotNavigationUnderLoad(StressTestCase):

    PROJECT_ID = "stress_project"
    PROJECT_NAME = "StressProject"
    SNAPSHOTS = 120
    SEARCHES = 100

    def build_history(self):
        repository = ConversationVectorMetaDataRepository(
            self.tmp_dir, self.PROJECT_ID
        )
        self.addCleanup(repository.close)
        repository.batch_insert_cumulative_vector_meta_data(
            [
                (
                    index,
                    f"cumulative summary {index}",
                    f"2026-09-01T00:{index // 60:02d}:{index % 60:02d}.000000+00:00",
                    self.PROJECT_ID,
                    100,
                )
                for index in range(1, self.SNAPSHOTS + 1)
            ]
        )
        return repository

    def make_snapshot(self, repository, vectors):
        snapshot = SnapShot(
            self.tmp_dir, self.PROJECT_ID, self.PROJECT_NAME, meta_repo=repository
        )
        vector_manager = mock.MagicMock()
        vector_manager.get_vector.side_effect = lambda vector_id: vectors[
            int(vector_id)
        ]
        snapshot._vector_manager = vector_manager
        return snapshot

    def test_repeated_searches_do_not_move_the_cursors(self):
        """Searching is a read: it must not consume the history it walks.

        __find_best_snapshot used to advance the instance cursors as it scanned,
        so they met in the middle on the first call and every later search saw a
        one-snapshot window.
        """
        repository = self.build_history()
        rng = np.random.default_rng(7)
        vectors = {
            index: rng.standard_normal(EMBEDDING_DIMENSIONS).astype(np.float32)
            for index in range(1, self.SNAPSHOTS + 1)
        }
        snapshot = self.make_snapshot(repository, vectors)
        snapshot.sync_cursors()

        cursors_before = (
            snapshot._SnapShot__left_cursor,
            snapshot._SnapShot__right_cursor,
        )
        query = vectors[self.SNAPSHOTS // 2]

        started = time.time()
        results = [snapshot.search(query) for _ in range(self.SEARCHES)]
        elapsed = time.time() - started

        report(
            f"{self.SEARCHES} searches over {self.SNAPSHOTS} snapshots in {elapsed:.3f}s"
        )
        self.assertEqual(cursors_before, (0, self.SNAPSHOTS - 1))
        self.assertEqual(
            (snapshot._SnapShot__left_cursor, snapshot._SnapShot__right_cursor),
            cursors_before,
        )
        self.assertEqual(len(set(results)), 1)
        self.assertEqual(results[0], (self.SNAPSHOTS // 2,))

    def test_search_returns_the_nearest_snapshot_for_every_query(self):
        """One search per stored snapshot, each querying with that snapshot's
        own vector — the exact match has cosine similarity 1.0 and must win."""
        repository = self.build_history()
        rng = np.random.default_rng(11)
        vectors = {
            index: rng.standard_normal(EMBEDDING_DIMENSIONS).astype(np.float32)
            for index in range(1, self.SNAPSHOTS + 1)
        }
        snapshot = self.make_snapshot(repository, vectors)
        snapshot.sync_cursors()

        started = time.time()
        hits = Counter(
            snapshot.search(vectors[index]) == (index,)
            for index in range(1, self.SNAPSHOTS + 1)
        )
        elapsed = time.time() - started

        report(
            f"{self.SNAPSHOTS} nearest-snapshot lookups in {elapsed:.3f}s "
            f"({hits[True]} exact)"
        )
        self.assertEqual(hits[True], self.SNAPSHOTS)


if __name__ == "__main__":
    unittest.main()
