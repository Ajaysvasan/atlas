"""The DiskANN index, run against the real library rather than a mock.

The rest of the data-layer suite mocks diskannpy, which is how two defects in
it went unseen: the graph it built could not find its own vectors (5.19), and a
saved index lost every label (5.20). Small indexes hide both — the start node
links to every point — so these tests use enough vectors to need a real graph.
"""

import numpy as np
import pytest

from data_layer.ingestion.embedding.vector_ids import vector_id_for

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


def recall_at_10(built, data, queries, labels=None):
    labels = np.arange(1, len(data) + 1) if labels is None else np.asarray(labels)
    found = 0
    for query in queries:
        truth = set(labels[np.argsort(((data - query) ** 2).sum(1))[:10]].tolist())
        got, _ = built.search_vector(query, 10)
        found += len(truth & set(np.asarray(got).tolist()))
    return found / (10 * len(queries))


class Stored:
    """Labels in a chunk store, vectors in the stand-in for pgvector."""

    def __init__(self, tmp_path, vector_store, name="chunks"):
        from data_layer.vector_db_manager.repository.vectorMetaDataRepository import (
            VectorMetaDataRepository,
        )

        self.mapping = VectorMetaDataRepository(str(tmp_path / name))
        self.vectors = vector_store

    def add(self, data, prefix="c", store_vectors=True):
        chunk_ids = [f"{prefix}{i}" for i in range(len(data))]
        ids = [vector_id_for(c) for c in chunk_ids]
        if store_vectors:
            self.vectors.batch_insert(ids, data)
        return self.mapping.batch_insert(ids, chunk_ids)

    def source(self):
        from data_layer.vector_db_manager.stored_vectors import StoredVectors

        return StoredVectors(self.mapping, self.vectors)

    def close(self):
        self.mapping.close()


@pytest.fixture
def stored(tmp_path, chunk_vectors):
    made = Stored(tmp_path, chunk_vectors)
    yield made
    made.close()


class TestTheGraphCanBeSearched:
    """With diskannpy's default `saturate_graph=False` the insertion path
    builds a graph too sparse to walk: recall@10 measured 0.09."""

    def test_an_inserted_vector_is_its_own_nearest_neighbour(self, real_diskann):
        data = points(2000)
        built = filled(data)
        found = sum(int(built.search_vector(data[i], 1)[0][0]) == i + 1 for i in range(2000))
        assert found / 2000 >= 0.98

    def test_recall_at_ten_against_brute_force(self, real_diskann):
        data = points(2000)
        queries = data[::40] + 0.05 * points(50, seed=1)
        assert recall_at_10(filled(data), data, queries) >= 0.9

    def test_the_graph_is_saturated(self):
        from data_layer.vector_db_manager.vectorDB_diskann import SATURATE_GRAPH

        assert SATURATE_GRAPH is True


class TestRestoringTheRecentVectors:
    """What was ingested after the last build is loaded from the stores at
    startup: labels from SQLite, vectors from PostgreSQL."""

    def test_a_restored_index_finds_what_the_store_holds(self, real_diskann, stored):
        data = points(2000)
        labels = stored.add(data)
        restored = index(2010)
        restored.restore(stored.source())
        found = sum(int(restored.search_vector(data[i], 1)[0][0]) == labels[i] for i in range(2000))
        assert found / 2000 >= 0.98

    def test_it_returns_the_highest_label_it_indexed(self, real_diskann, stored):
        labels = stored.add(points(5))
        assert index().restore(stored.source()) == labels[-1]

    def test_restoring_from_the_last_label_adds_only_what_is_new(self, real_diskann, stored):
        data = points(8)
        stored.add(data[:5])
        built = index()
        through = built.restore(stored.source())
        more = stored.add(data[5:], prefix="n")
        assert built.restore(stored.source(), after=through) == more[-1]
        assert built.count() == 8

    def test_an_empty_store_restores_nothing(self, real_diskann, stored):
        built = index()
        assert built.restore(stored.source(), after=0) == 0
        assert built.count() == 0

    def test_a_label_whose_vector_is_not_stored_is_skipped_and_reported(self, real_diskann, stored, caplog):
        stored.add(points(2))
        lost, = stored.add(points(1, seed=3), prefix="lost", store_vectors=False)
        built = index()
        with caplog.at_level("WARNING"):
            through = built.restore(stored.source())
        assert built.count() == 2
        assert through == lost
        assert any("no stored vector" in r.getMessage() for r in caplog.records)

    def test_a_full_recent_index_stops_rather_than_overflowing(self, real_diskann, stored, caplog):
        labels = stored.add(points(30))
        built = index(max_vectors=20)
        with caplog.at_level("WARNING"):
            through = built.restore(stored.source())
        assert built.count() == 20
        assert through == labels[19]
        assert any("10 vector(s) wait for the next index build" in r.getMessage()
                   for r in caplog.records)

    def test_a_full_recent_index_is_not_filled_further_by_a_later_restore(self, real_diskann, stored):
        stored.add(points(30))
        built = index(max_vectors=20)
        through = built.restore(stored.source())
        assert built.restore(stored.source(), after=through) == through
        assert built.count() == 20


def build(stored, root):
    from data_layer.vector_db_manager.index_generations import IndexGenerations

    generations = IndexGenerations(root)
    return generations, generations.build(stored.source())


def opened(generations, allowance=512 << 20):
    found = generations.open(allowance)
    assert found is not None
    return found


class TestBuiltGenerations:
    """The built index is opened from disk at startup, not rebuilt."""

    def test_a_build_is_found_by_label(self, real_diskann, stored, tmp_path):
        data = points(300)
        labels = stored.add(data)
        generations, outcome = build(stored, tmp_path / "index")
        assert outcome.built and outcome.kind == "memory" and outcome.count == 300
        base = opened(generations)
        found, _ = base.search(data[17], 1, 64)
        assert found.tolist() == [labels[17]]

    def test_it_records_the_last_label_it_was_built_through(self, real_diskann, stored, tmp_path):
        labels = stored.add(points(50))
        generations, _ = build(stored, tmp_path / "index")
        assert generations.through() == labels[-1]
        assert opened(generations).through == labels[-1]

    def test_a_corpus_past_the_budget_is_built_for_disk(self, real_diskann, stored, tmp_path, monkeypatch):
        from data_layer.vector_db_manager import memory_guard

        monkeypatch.setattr(memory_guard, "memory_index_bytes", lambda *a: 1 << 40)
        data = points(300)
        labels = stored.add(data)
        generations, outcome = build(stored, tmp_path / "index")
        assert outcome.built and outcome.kind == "disk"
        base = opened(generations, allowance=64 << 20)
        found, _ = base.search(data[42], 1, 64)
        assert found.tolist() == [labels[42]]

    def test_a_disk_index_never_answers_with_more_than_it_holds(self, real_diskann, stored, tmp_path, monkeypatch):
        """Asked for more, a disk index pads with position 0 — a real label."""
        from data_layer.vector_db_manager import memory_guard

        monkeypatch.setattr(memory_guard, "memory_index_bytes", lambda *a: 1 << 40)
        data = points(3)
        labels = stored.add(data)
        generations, outcome = build(stored, tmp_path / "index")
        assert outcome.kind == "disk"
        found, _ = opened(generations).search(data[0], 9, 64)
        assert sorted(found.tolist()) == sorted(labels)

    def test_the_build_leaves_no_copy_of_the_vectors_behind(self, real_diskann, stored, tmp_path):
        stored.add(points(50))
        generations, outcome = build(stored, tmp_path / "index")
        files = {f.name for f in (tmp_path / "index" / outcome.generation).iterdir()}
        assert "vectors.bin" not in files and "build.log" not in files

    def test_merged_with_the_recent_vectors_it_answers_as_one_index(self, real_diskann, stored, tmp_path):
        data = points(1000)
        labels = stored.add(data[:900])
        generations, _ = build(stored, tmp_path / "index")
        labels += stored.add(data[900:], prefix="new")
        built = index()
        built.use_base(opened(generations))
        built.restore(stored.source(), after=built.through)
        assert built.count() == 1000 and built.recent_count() == 100
        queries = data[::20] + 0.05 * points(50, seed=1)
        assert recall_at_10(built, data, queries, labels) >= 0.9

    def test_the_nearest_wins_whichever_index_holds_it(self, real_diskann, stored, tmp_path):
        """Merged by distance: every built label is smaller than every recent one."""
        data = points(1000)
        labels = stored.add(data[:900])
        generations, _ = build(stored, tmp_path / "index")
        labels += stored.add(data[900:], prefix="new")
        built = index()
        built.use_base(opened(generations))
        built.restore(stored.source(), after=built.through)
        assert int(built.search_vector(data[950], 1)[0][0]) == labels[950]
        assert int(built.search_vector(data[10], 1)[0][0]) == labels[10]
        found, distances = built.search_vector(data[950], 10)
        assert list(distances) == sorted(distances)

    def test_in_a_new_process(self, real_diskann, stored, tmp_path):
        """The restart itself. Every in-process reload of a saved DiskANN index
        looked correct; only a fresh process showed the labels were gone."""
        import json
        import os
        import subprocess
        import sys
        from pathlib import Path

        data = points(300)
        labels = stored.add(data)
        build(stored, tmp_path / "index")
        picks = [0, 17, 150, 299]
        np.save(tmp_path / "queries.npy", data[picks])

        script = (
            "import json, sys\n"
            "import numpy as np\n"
            "from data_layer.vector_db_manager.index_generations import IndexGenerations\n"
            "base = IndexGenerations(sys.argv[1]).open(512 << 20)\n"
            "found = [int(base.search(q, 1, 64)[0][0]) for q in np.load(sys.argv[2])]\n"
            "print('RESULT ' + json.dumps(found))\n"
        )
        app = Path(__file__).resolve().parents[2]
        done = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / "index"), str(tmp_path / "queries.npy")],
            cwd=app, env={**os.environ, "PYTHONPATH": str(app)},
            capture_output=True, text=True, timeout=300,
        )
        line = next((l for l in done.stdout.splitlines() if l.startswith("RESULT ")), None)
        assert line is not None, done.stderr[-2000:]
        assert json.loads(line[len("RESULT "):]) == [labels[i] for i in picks]


class TestThePipelineStoresBeforeItLabels:
    """A label is never written for a vector that was not stored."""

    def pipeline(self, tmp_path, vector_store):
        from data_layer.ingestion.ingestion_pipeline import IngestionPipeline
        from data_layer.vector_db_manager.repository.vectorMetaDataRepository import (
            VectorMetaDataRepository,
        )

        built = IngestionPipeline.__new__(IngestionPipeline)
        built.vector_meta = VectorMetaDataRepository(str(tmp_path / "chunks"))
        built.vector_store = vector_store
        built._owns_store = False
        built.index_path = tmp_path / "index"
        return built

    def embedded(self, data, prefix="c", vector_id=vector_id_for):
        from data_layer.ingestion.metadata.metadata import EmbeddedChunkMetaData
        from data_layer.ingestion.nodes.nodes import EmbeddedChunk

        return [EmbeddedChunk(v, vector_id(f"{prefix}{i}"), EmbeddedChunkMetaData(f"{prefix}{i}", "t", "m"))
                for i, v in enumerate(data)]

    def test_every_vector_is_stored_under_its_vector_id(self, tmp_path, chunk_vectors):
        pipeline = self.pipeline(tmp_path, chunk_vectors)
        data = points(4)
        labels = pipeline.batch_insert_vectors(self.embedded(data))
        assert len(set(labels)) == 4
        for i in range(4):
            np.testing.assert_array_equal(chunk_vectors.rows[vector_id_for(f"c{i}")], data[i])
        pipeline.close()

    def test_a_single_insert_returns_its_label(self, tmp_path, chunk_vectors):
        pipeline = self.pipeline(tmp_path, chunk_vectors)
        label = pipeline.ingest_vector(self.embedded(points(1))[0])
        assert pipeline.vector_meta.chunk_ids_for([label]) == {label: "c0"}
        pipeline.close()

    def test_a_missing_vector_id_writes_nothing_anywhere(self, tmp_path, chunk_vectors):
        from data_layer.datalayer_exceptions.datalayer_exceptions import MissingVectorId

        pipeline = self.pipeline(tmp_path, chunk_vectors)
        chunks = self.embedded(points(2), vector_id=lambda chunk_id: None)
        with pytest.raises(MissingVectorId):
            pipeline.batch_insert_vectors(chunks)
        assert chunk_vectors.rows == {} and pipeline.vector_meta.count() == 0
        pipeline.close()

    def test_an_unreachable_store_writes_no_label(self, tmp_path, chunk_vectors):
        pipeline = self.pipeline(tmp_path, chunk_vectors)
        chunk_vectors.down = True
        with pytest.raises(ConnectionError):
            pipeline.batch_insert_vectors(self.embedded(points(2)))
        assert pipeline.vector_meta.count() == 0
        pipeline.close()

    def test_a_store_that_cannot_be_opened_is_named(self, tmp_path, monkeypatch):
        from data_layer.datalayer_exceptions.datalayer_exceptions import VectorStoreUnavailable
        from data_layer.vector_db_manager import stored_vectors

        def refuse():
            raise OSError("connection refused")

        monkeypatch.setattr(stored_vectors, "chunk_vector_store", refuse)
        pipeline = self.pipeline(tmp_path, None)
        pipeline._owns_store = True
        with pytest.raises(VectorStoreUnavailable, match="connection refused"):
            pipeline.batch_insert_vectors(self.embedded(points(1)))
        assert pipeline.vector_meta.count() == 0
        pipeline.close()

    def test_the_same_chunks_again_keep_their_labels(self, tmp_path, chunk_vectors):
        pipeline = self.pipeline(tmp_path, chunk_vectors)
        chunks = self.embedded(points(4))
        assert pipeline.batch_insert_vectors(chunks) == pipeline.batch_insert_vectors(chunks)
        assert pipeline.vector_meta.count() == 4
        pipeline.close()

    def test_a_vector_store_handed_in_is_left_open(self, tmp_path, chunk_vectors):
        pipeline = self.pipeline(tmp_path, chunk_vectors)
        pipeline.close()
        assert not chunk_vectors.closed


class TestThePipelineRebuildsWhenEnoughWait:
    def setup(self, tmp_path, chunk_vectors, monkeypatch, rebuild_at=50):
        from config import Config

        monkeypatch.setattr(Config, "INDEX_REBUILD_AT", rebuild_at)
        return TestThePipelineStoresBeforeItLabels().pipeline(tmp_path, chunk_vectors)

    def test_below_the_threshold_nothing_is_built(self, tmp_path, chunk_vectors, monkeypatch):
        pipeline = self.setup(tmp_path, chunk_vectors, monkeypatch)
        pipeline.batch_insert_vectors(TestThePipelineStoresBeforeItLabels().embedded(points(10)))
        assert not (tmp_path / "index").exists()
        pipeline.close()

    def test_at_the_threshold_a_generation_is_built(self, real_diskann, tmp_path, chunk_vectors, monkeypatch):
        from data_layer.vector_db_manager.index_generations import IndexGenerations

        pipeline = self.setup(tmp_path, chunk_vectors, monkeypatch)
        labels = pipeline.batch_insert_vectors(TestThePipelineStoresBeforeItLabels().embedded(points(60)))
        assert IndexGenerations(tmp_path / "index").through() == labels[-1]
        assert pipeline.maintain_index() is None
        pipeline.close()

    def test_a_forced_build_ignores_the_threshold(self, real_diskann, tmp_path, chunk_vectors, monkeypatch):
        pipeline = self.setup(tmp_path, chunk_vectors, monkeypatch)
        pipeline.batch_insert_vectors(TestThePipelineStoresBeforeItLabels().embedded(points(10)))
        outcome = pipeline.maintain_index(force=True)
        assert outcome.built and outcome.count == 10
        pipeline.close()
