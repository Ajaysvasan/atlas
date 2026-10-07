"""The dense half: DiskANN labels and distances into ranked candidates."""

import numpy as np
import pytest

from retrieval_layer.models import QueryPlan, VECTOR
from retrieval_layer.retrieval_exceptions import IndexUnavailable
from retrieval_layer.vector_search import VectorSearch


class FakeIndex:
    """Returns what DiskANN returns: labels and distances, in its own order."""

    def __init__(self, labels, distances, raises=None):
        self.labels, self.distances, self.raises = labels, distances, raises
        self.calls = 0

    def search_vector(self, query):
        self.calls += 1
        if self.raises:
            raise self.raises
        return np.array(self.labels, dtype=np.uint32), np.array(self.distances)


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

    def test_a_missing_index_on_disk_says_how_to_make_one(self, tmp_path):
        search = VectorSearch(index_path=tmp_path / "never_written")
        with pytest.raises(IndexUnavailable, match="persist_index"):
            search.search(plan(), 3)

    def test_an_injected_index_is_not_loaded_from_disk(self, tmp_path):
        """Sharing the pipeline's index keeps a fresh ingest searchable."""
        index = FakeIndex([1], [0.1])
        search = VectorSearch(index=index, index_path=tmp_path / "absent")
        assert len(search.search(plan(), 1)) == 1
