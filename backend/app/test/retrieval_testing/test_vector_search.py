"""The dense half: DiskANN labels and distances into ranked candidates."""

import numpy as np
import pytest

from retrieval_layer.models import QueryPlan, VECTOR
from retrieval_layer.retrieval_exceptions import IndexUnavailable
from retrieval_layer.vector_search import VectorSearch


class FakeIndex:
    """Behaves as VectorDbManager does: labels and distances in its own order,
    and no more of them than it is asked for — the configured k_neighbors when
    it is not asked at all.
    """

    def __init__(self, labels, distances, raises=None, k_neighbors=9):
        self.labels, self.distances, self.raises = labels, distances, raises
        self.k_neighbors = k_neighbors
        self.calls = 0
        self.asked_for = None

    def search_vector(self, query, k_neighbors=None):
        self.calls += 1
        self.asked_for = k_neighbors
        if self.raises:
            raise self.raises
        k = k_neighbors or self.k_neighbors
        return (np.array(self.labels[:k], dtype=np.uint32),
                np.array(self.distances[:k]))


def plan():
    return QueryPlan("a query", np.ones(128, dtype=np.float32), "a query")


class TestRanking:
    def test_the_nearest_vector_ranks_first(self):
        """Distance, not similarity: smaller is closer."""
        search = VectorSearch(index=FakeIndex([7, 3, 9], [0.9, 0.1, 0.5]))
        assert [h.vector_id for h in search.search(plan(), 3)] == [3, 9, 7]

    def test_ranks_start_at_one(self):
        search = VectorSearch(index=FakeIndex([7], [0.1]))
        assert search.search(plan(), 1)[0].rank == 1

    def test_the_source_is_marked(self):
        """Fusion needs to know which searcher found a candidate."""
        search = VectorSearch(index=FakeIndex([7], [0.1]))
        assert search.search(plan(), 1)[0].source == VECTOR

    def test_labels_come_back_as_ints(self):
        """They arrive as uint32 and go on to index dictionaries."""
        search = VectorSearch(index=FakeIndex([7], [0.1]))
        assert isinstance(search.search(plan(), 1)[0].vector_id, int)


class TestHowMany:
    def test_it_returns_no_more_than_k(self):
        """DiskANN pads a short result set, so the caller's k is the bound."""
        search = VectorSearch(index=FakeIndex([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]))
        assert len(search.search(plan(), 2)) == 2

    def test_fewer_than_k_is_fine(self):
        search = VectorSearch(index=FakeIndex([1], [0.1]))
        assert len(search.search(plan(), 10)) == 1

    @pytest.mark.parametrize("k", [0, -1])
    def test_asking_for_none_searches_nothing(self, k):
        index = FakeIndex([1], [0.1])
        assert VectorSearch(index=index).search(plan(), k) == []
        assert index.calls == 0


class TestWhenTheIndexIsUnhappy:
    def test_a_failing_search_returns_nothing_rather_than_raising(self):
        """A dense failure should not lose the lexical results too."""
        search = VectorSearch(index=FakeIndex([], [], raises=RuntimeError("boom")))
        assert search.search(plan(), 3) == []

    def test_no_chunk_store_says_how_to_make_one(self, tmp_path):
        search = VectorSearch(chunk_store_path=tmp_path / "never_written")
        with pytest.raises(IndexUnavailable, match="Ingest a corpus"):
            search.search(plan(), 3)

    def test_looking_for_a_store_does_not_create_one(self, tmp_path):
        with pytest.raises(IndexUnavailable):
            VectorSearch(chunk_store_path=tmp_path / "absent").search(plan(), 3)
        assert not (tmp_path / "absent").exists()

    def test_an_injected_index_is_not_rebuilt(self, tmp_path):
        index = FakeIndex([1], [0.1])
        search = VectorSearch(index=index, chunk_store_path=tmp_path / "absent")
        assert len(search.search(plan(), 1)) == 1


class TestOverFetching:
    """Reranking and MMR can only choose among what the searchers bring back,
    so the dense half has to return the whole candidate pool, not the index's
    configured default."""

    def test_it_asks_the_index_for_k(self):
        index = FakeIndex(list(range(1, 41)), [i / 100 for i in range(40)])
        VectorSearch(index=index).search(plan(), 32)
        assert index.asked_for == 32

    def test_more_than_the_configured_default_comes_back(self):
        index = FakeIndex(list(range(1, 41)), [i / 100 for i in range(40)],
                          k_neighbors=9)
        assert len(VectorSearch(index=index).search(plan(), 32)) == 32


class TestTheRealManagerPassesKThrough:
    def test_k_reaches_diskann(self):
        from data_layer.vector_db_manager.vectorDbManager import VectorDbManager

        manager = VectorDbManager.__new__(VectorDbManager)
        manager.k_neighbors, manager.complexity, manager.base = 9, 100, None
        seen = {}

        class Backend:
            def search_vector(self, query, k_neighbors, complexity):
                seen["k"] = k_neighbors
                return [], []

            def count(self):
                return 100

        manager.vector_db = Backend()
        manager.search_vector(np.ones(128, dtype=np.float32), 32)
        assert seen["k"] == 32

    def test_without_k_it_keeps_its_configured_default(self):
        from data_layer.vector_db_manager.vectorDbManager import VectorDbManager

        manager = VectorDbManager.__new__(VectorDbManager)
        manager.k_neighbors, manager.complexity, manager.base = 9, 100, None
        seen = {}

        class Backend:
            def search_vector(self, query, k_neighbors, complexity):
                seen["k"] = k_neighbors
                return [], []

            def count(self):
                return 100

        manager.vector_db = Backend()
        manager.search_vector(np.ones(128, dtype=np.float32))
        assert seen["k"] == 9


class Store:
    """Labels in the chunk store, vectors in the stand-in for pgvector."""

    def __init__(self, tmp_path, vectors):
        from data_layer.vector_db_manager.repository.vectorMetaDataRepository import (
            VectorMetaDataRepository,
        )
        from data_layer.vector_db_manager.stored_vectors import chunk_vector_store

        self.mapping = VectorMetaDataRepository(str(tmp_path / "chunks"))
        self.vectors = chunk_vector_store()
        self.add(vectors)

    def add(self, vectors, prefix="c"):
        from data_layer.ingestion.embedding.vector_ids import vector_id_for

        chunk_ids = [f"{prefix}{i}" for i in range(len(vectors))]
        ids = [vector_id_for(c) for c in chunk_ids]
        self.vectors.batch_insert(ids, vectors)
        return self.mapping.batch_insert(ids, chunk_ids)

    def source(self):
        from data_layer.vector_db_manager.stored_vectors import StoredVectors

        return StoredVectors(self.mapping, self.vectors)

    def close(self):
        self.mapping.close()


def store_with(tmp_path, vectors):
    return Store(tmp_path, vectors)


class TestBuildingFromTheStore:
    """The index is rebuilt from the vectors stored beside their labels,
    because diskannpy cannot load one it saved (bugs.md 5.20)."""

    def test_the_stored_vectors_are_searchable(self, real_diskann, tmp_path):
        data = points(8)
        store_with(tmp_path, data).close()
        hits = VectorSearch(chunk_store_path=tmp_path / "chunks").search(
            QueryPlan("q", data[4], "q"), 3)
        assert hits[0].vector_id == 5

    def test_an_empty_store_finds_nothing_rather_than_failing(self, real_diskann, tmp_path):
        store_with(tmp_path, []).close()
        search = VectorSearch(chunk_store_path=tmp_path / "chunks")
        assert search.search(QueryPlan("q", points(1)[0], "q"), 3) == []

    def test_catching_up_adds_what_was_ingested_since(self, real_diskann, tmp_path):
        data = points(12)
        store = store_with(tmp_path, data[:8])
        search = VectorSearch(chunk_store_path=tmp_path / "chunks")
        search.search(QueryPlan("q", data[0], "q"), 1)
        store.add(data[8:], prefix="n")
        store.close()

        search.catch_up()
        assert search.index.count() == 12
        assert search.search(QueryPlan("q", data[10], "q"), 1)[0].vector_id == 11

    def test_catching_up_twice_indexes_nothing_twice(self, real_diskann, tmp_path):
        """Labels only grow, so only what is above the last one is new — the
        last one indexed, which has to move with every catch-up that finds
        something, not only the first build."""
        data = points(12)
        store = store_with(tmp_path, data[:8])
        search = VectorSearch(chunk_store_path=tmp_path / "chunks")
        search.search(QueryPlan("q", data[0], "q"), 1)
        store.add(data[8:], prefix="n")
        search.catch_up()
        search.catch_up()
        store.close()
        assert search.index.count() == 12

    def test_catching_up_before_the_first_search_builds_nothing(self, tmp_path):
        """The first search builds the whole index; there is nothing to add to."""
        search = VectorSearch(chunk_store_path=tmp_path / "absent")
        search.catch_up()
        assert not (tmp_path / "absent").exists()

    def test_an_index_handed_in_is_left_to_its_owner(self):
        live = FakeIndex([1], [0.1])
        live.restore = lambda store, after: pytest.fail("restored a borrowed index")
        search = VectorSearch(index=live)
        search.catch_up()
        search.search(plan(), 1)
        assert live.calls == 1


def real_index(vectors, max_vectors=1000):
    from data_layer.vector_db_manager.vectorDbManager import VectorDbManager

    index = VectorDbManager(
        distance_metrics="l2", vector_dtype=np.float32, dimensions=128,
        max_vectors=max_vectors, complexity=64, graph_degree=32, num_threads=1,
        k_neighbors=9,
    )
    if len(vectors):
        index.vector_db.batch_insert(
            np.asarray(vectors, dtype=np.float32),
            np.arange(1, len(vectors) + 1, dtype=np.uint32),
        )
    return index


def points(n, seed=0):
    v = np.random.default_rng(seed).standard_normal((n, 128)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class TestNeverAskingForMoreThanTheIndexHolds:
    """Asked for k neighbours from an index holding fewer, diskannpy returns k
    slots anyway and fills the surplus from uninitialised memory — labels that
    can be real ones, at distance 0.0, so they sort ahead of every true result.
    On a reloaded index every slot is garbage. A new install's corpus is always
    smaller than the candidate pool, so this is the first thing a user meets."""

    def test_the_count_follows_inserts_deletes_and_restores(self, real_diskann, tmp_path):
        """Pins the private path the count is read through, so a diskannpy
        change breaks here rather than inside a search."""
        index = real_index(points(8))
        assert index.count() == 8
        index.delete_vector(np.uint32(3))
        assert index.count() == 7
        store = store_with(tmp_path, points(8))
        restored = real_index([])
        restored.restore(store.source())
        store.close()
        assert restored.count() == 8

    def test_a_small_live_index_returns_only_what_it_holds(self, real_diskann):
        data = points(8)
        labels, distances = real_index(data).search_vector(data[2], 32)
        assert sorted(labels.tolist()) == list(range(1, 9))
        assert int(labels[0]) == 3
        assert all(d >= 0 for d in distances)

    def test_a_small_restored_index_returns_only_what_it_holds(self, real_diskann, tmp_path):
        data = points(8)
        store = store_with(tmp_path, data)
        restored = real_index([])
        restored.restore(store.source())
        store.close()
        labels, distances = restored.search_vector(data[2], 32)
        assert sorted(labels.tolist()) == list(range(1, 9))
        assert int(labels[0]) == 3

    def test_an_empty_index_returns_nothing(self, real_diskann):
        labels, distances = real_index([]).search_vector(points(1)[0], 32)
        assert len(labels) == 0 and len(distances) == 0

    def test_a_batch_against_an_empty_index_returns_nothing(self, real_diskann):
        labels, distances = real_index([]).batch_search_vectors(points(3), 32)
        assert labels.shape == (3, 0)

    def test_through_vector_search_every_hit_is_real(self, real_diskann):
        data = points(8)
        search = VectorSearch(index=real_index(data))
        hits = search.search(QueryPlan("q", data[5], "q"), 32)
        assert [h.vector_id for h in hits][0] == 6
        assert {h.vector_id for h in hits} <= set(range(1, 9))
