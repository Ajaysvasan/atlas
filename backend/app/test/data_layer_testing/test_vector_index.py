"""The DiskANN index, run against the real library rather than a mock.

The rest of the data-layer suite mocks diskannpy, which is how two defects in
it went unseen: the graph it built could not find its own vectors (5.19), and a
saved index lost every label (5.20). Small indexes hide both — the start node
links to every point — so these tests use enough vectors to need a real graph.
"""

import numpy as np
import pytest

DIMENSIONS = 128


def points(n, seed=0):
    return np.random.default_rng(seed).standard_normal((n, DIMENSIONS)).astype(np.float32)


def index(max_vectors=5000):
    from data_layer.vector_db_manager.vectorDbManager import VectorDbManager

    return VectorDbManager(
        distance_metrics="l2", vector_dtype=np.float32, dimensions=DIMENSIONS,
        max_vectors=max_vectors, complexity=64, graph_degree=32, num_threads=4,
        k_neighbors=10,
    )


def filled(data):
    built = index(len(data) + 10)
    built.vector_db.batch_insert(data, np.arange(1, len(data) + 1, dtype=np.uint32))
    return built


def recall_at_10(built, data, queries):
    found = 0
    for query in queries:
        truth = set((np.argsort(((data - query) ** 2).sum(1))[:10] + 1).tolist())
        labels, _ = built.search_vector(query, 10)
        found += len(truth & set(np.asarray(labels).tolist()))
    return found / (10 * len(queries))


class TestTheGraphCanBeSearched:
    """With diskannpy's default `saturate_graph=False` the insertion path
    builds a graph too sparse to walk: recall@10 measured 0.09."""

    def test_an_inserted_vector_is_its_own_nearest_neighbour(self, real_diskann):
        """Approximate, so not every time: 99.7-99.8% measured with the graph
        saturated, and 0% without it."""
        data = points(2000)
        built = filled(data)
        found = sum(int(built.search_vector(data[i], 1)[0][0]) == i + 1
                    for i in range(2000))
        assert found / 2000 >= 0.98

    def test_recall_at_ten_against_brute_force(self, real_diskann):
        data = points(2000)
        queries = data[::40] + 0.05 * points(50, seed=1)
        assert recall_at_10(filled(data), data, queries) >= 0.9

    def test_the_graph_is_saturated(self):
        """The constant itself, so the reason it is set travels with a test."""
        from data_layer.vector_db_manager.vectorDB_diskann import SATURATE_GRAPH

        assert SATURATE_GRAPH is True


def store_of(tmp_path, data, name="chunks"):
    from data_layer.vector_db_manager.repository.vectorMetaDataRepository import (
        VectorMetaDataRepository,
    )

    store = VectorMetaDataRepository(str(tmp_path / name))
    labels = store.allocate_many([f"c{i}" for i in range(len(data))], list(data))
    return store, labels


class TestSurvivingARestart:
    """Bug 5.20: diskannpy 0.7.0 writes every label as 0 when it saves a
    dynamic index, and loads garbage in a new process even when the labels on
    disk are right. The index is rebuilt from the vectors stored beside their
    labels instead."""

    def test_a_restored_index_finds_what_the_store_holds(self, real_diskann, tmp_path):
        data = points(2000)
        store, labels = store_of(tmp_path, data)
        restored = index(2010)
        restored.restore(store)
        store.close()
        found = sum(int(restored.search_vector(data[i], 1)[0][0]) == labels[i]
                    for i in range(2000))
        assert found / 2000 >= 0.98

    def test_a_restored_index_keeps_its_recall(self, real_diskann, tmp_path):
        data = points(2000)
        store, _ = store_of(tmp_path, data)
        restored = index(2010)
        restored.restore(store)
        store.close()
        queries = data[::40] + 0.05 * points(50, seed=1)
        assert recall_at_10(restored, data, queries) >= 0.9

    def test_it_returns_the_highest_label_it_indexed(self, real_diskann, tmp_path):
        store, labels = store_of(tmp_path, points(5))
        assert index().restore(store) == labels[-1]
        store.close()

    def test_restoring_from_the_last_label_adds_only_what_is_new(self, real_diskann, tmp_path):
        data = points(8)
        store, labels = store_of(tmp_path, data[:5])
        built = index()
        through = built.restore(store)
        more = store.allocate_many(["x", "y", "z"], list(data[5:]))
        assert built.restore(store, after=through) == more[-1]
        assert built.count() == 8
        store.close()

    def test_an_empty_store_restores_nothing(self, real_diskann, tmp_path):
        store, _ = store_of(tmp_path, [])
        built = index()
        assert built.restore(store, after=0) == 0
        assert built.count() == 0
        store.close()

    def test_labels_with_no_usable_vector_are_reported(self, real_diskann, tmp_path, caplog):
        store, _ = store_of(tmp_path, points(2))
        store.allocate("legacy")
        with caplog.at_level("WARNING"):
            index().restore(store)
        store.close()
        assert any("re-embedded" in r.getMessage() for r in caplog.records)

    def test_in_a_new_process(self, tmp_path):
        """The restart itself. Every in-process reload of a saved DiskANN index
        looked correct; only a fresh process showed the labels were gone."""
        import json
        import os
        import subprocess
        import sys
        from pathlib import Path

        data = points(300)
        store, labels = store_of(tmp_path, data)
        store.close()
        picks = [0, 17, 150, 299]
        np.save(tmp_path / "queries.npy", data[picks])

        script = (
            "import json, sys\n"
            "import numpy as np\n"
            "from data_layer.vector_db_manager.repository.vectorMetaDataRepository "
            "import VectorMetaDataRepository\n"
            "from data_layer.vector_db_manager.vectorDbManager import VectorDbManager\n"
            "store = VectorMetaDataRepository(sys.argv[1])\n"
            "built = VectorDbManager(distance_metrics='l2', vector_dtype=np.float32, "
            "dimensions=128, max_vectors=400, complexity=64, graph_degree=32, "
            "num_threads=1, k_neighbors=10)\n"
            "built.restore(store)\n"
            "found = [int(built.search_vector(q, 1)[0][0]) for q in np.load(sys.argv[2])]\n"
            "print('RESULT ' + json.dumps(found))\n"
        )
        app = Path(__file__).resolve().parents[2]
        done = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / "chunks"),
             str(tmp_path / "queries.npy")],
            cwd=app, env={**os.environ, "PYTHONPATH": str(app)},
            capture_output=True, text=True, timeout=300,
        )
        line = next((l for l in done.stdout.splitlines() if l.startswith("RESULT ")), None)
        assert line is not None, done.stderr[-2000:]
        assert json.loads(line[len("RESULT "):]) == [labels[i] for i in picks]


class TestThePipelineStoresWhatItIndexes:
    """The seam: if ingestion indexed a vector without storing it, the next
    restart would silently lose it."""

    def test_batch_insert_stores_every_vector_with_its_label(self, real_diskann, tmp_path):
        from data_layer.ingestion.ingestion_pipeline import IngestionPipeline
        from data_layer.ingestion.metadata.metadata import EmbeddedChunkMetaData
        from data_layer.ingestion.nodes.nodes import EmbeddedChunk
        from data_layer.vector_db_manager.repository.vectorMetaDataRepository import (
            VectorMetaDataRepository,
        )

        pipeline = IngestionPipeline.__new__(IngestionPipeline)
        pipeline.vector_meta = VectorMetaDataRepository(str(tmp_path / "chunks"))
        pipeline.vector_db = index()
        pipeline._indexed_labels = set()
        data = points(4)
        embedded = [EmbeddedChunk(v, 0, EmbeddedChunkMetaData(f"c{i}", "t", "m"))
                    for i, v in enumerate(data)]

        labels = pipeline.batch_insert_vectors(embedded)
        (stored_labels, stored), = list(pipeline.vector_meta.vectors())
        pipeline.vector_meta.close()
        assert stored_labels.tolist() == labels
        np.testing.assert_array_equal(stored, data)

    def test_a_single_insert_stores_its_vector(self, real_diskann, tmp_path):
        from data_layer.ingestion.ingestion_pipeline import IngestionPipeline
        from data_layer.ingestion.metadata.metadata import EmbeddedChunkMetaData
        from data_layer.ingestion.nodes.nodes import EmbeddedChunk
        from data_layer.vector_db_manager.repository.vectorMetaDataRepository import (
            VectorMetaDataRepository,
        )

        pipeline = IngestionPipeline.__new__(IngestionPipeline)
        pipeline.vector_meta = VectorMetaDataRepository(str(tmp_path / "chunks"))
        pipeline.vector_db = index()
        pipeline._indexed_labels = set()
        vector = points(1)[0]
        pipeline.ingest_vector(EmbeddedChunk(vector, 0, EmbeddedChunkMetaData("c", "t", "m")))
        (_, stored), = list(pipeline.vector_meta.vectors())
        pipeline.vector_meta.close()
        np.testing.assert_array_equal(stored[0], vector)


class TestReIngestingKeepsOneLabelPerChunk:
    """Bug 5.21: every re-ingest gave each chunk a new label, and since 5.20
    stores the vectors, every copy was rebuilt into the index."""

    def pipeline(self, tmp_path):
        from data_layer.ingestion.ingestion_pipeline import IngestionPipeline
        from data_layer.vector_db_manager.repository.vectorMetaDataRepository import (
            VectorMetaDataRepository,
        )

        built = IngestionPipeline.__new__(IngestionPipeline)
        built.vector_meta = VectorMetaDataRepository(str(tmp_path / "chunks"))
        built.vector_db = index()
        built._indexed_labels = set()
        return built

    def embedded(self, data, prefix="c"):
        from data_layer.ingestion.metadata.metadata import EmbeddedChunkMetaData
        from data_layer.ingestion.nodes.nodes import EmbeddedChunk

        return [EmbeddedChunk(v, 0, EmbeddedChunkMetaData(f"{prefix}{i}", "t", "m"))
                for i, v in enumerate(data)]

    def test_the_same_chunks_twice_in_one_session(self, real_diskann, tmp_path):
        """DiskANN refuses a label it already holds, so a reused label must
        not be inserted again."""
        pipeline = self.pipeline(tmp_path)
        chunks = self.embedded(points(4))
        first = pipeline.batch_insert_vectors(chunks)
        second = pipeline.batch_insert_vectors(chunks)
        assert second == first
        assert pipeline.vector_meta.count() == 4
        assert pipeline.vector_db.count() == 4

    def test_one_chunk_twice_in_one_batch_is_indexed_once(self, real_diskann, tmp_path):
        pipeline = self.pipeline(tmp_path)
        chunk = self.embedded(points(1))[0]
        labels = pipeline.batch_insert_vectors([chunk, chunk])
        assert labels[0] == labels[1]
        assert pipeline.vector_db.count() == 1

    def test_a_single_insert_repeated(self, real_diskann, tmp_path):
        pipeline = self.pipeline(tmp_path)
        chunk = self.embedded(points(1))[0]
        pipeline.ingest_vector(chunk)
        pipeline.ingest_vector(chunk)
        assert pipeline.vector_meta.count() == 1
        assert pipeline.vector_db.count() == 1

    def test_a_restart_after_re_ingesting_rebuilds_one_vector_per_chunk(self, real_diskann, tmp_path):
        first_session = self.pipeline(tmp_path)
        chunks = self.embedded(points(5))
        first_session.batch_insert_vectors(chunks)
        second_session = self.pipeline(tmp_path)
        second_session.batch_insert_vectors(chunks)

        rebuilt = index()
        rebuilt.restore(second_session.vector_meta)
        assert rebuilt.count() == 5

    def test_new_chunks_beside_old_ones_get_new_labels(self, real_diskann, tmp_path):
        pipeline = self.pipeline(tmp_path)
        data = points(6)
        old = pipeline.batch_insert_vectors(self.embedded(data[:3]))
        mixed = pipeline.batch_insert_vectors(
            self.embedded(data[:3]) + self.embedded(data[3:], prefix="n"))
        assert mixed[:3] == old
        assert len(set(mixed[3:])) == 3 and not set(mixed[3:]) & set(old)
        assert pipeline.vector_db.count() == 6

