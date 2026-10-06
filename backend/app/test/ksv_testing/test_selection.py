"""Choosing which trusted sources to ask about a topic.

Without selection every searchable source is queried: a SQLite question also
searches the web platform reference, a preprint server and the RFC index. The
descriptions are the signal, which makes them load-bearing config rather than
documentation.
"""

import numpy as np
import pytest

from knowledge_sufficiency.selection import (
    SOURCE_FLOOR,
    SourceIndex,
    TOP_SOURCES,
)
from knowledge_sufficiency.sources import DOCS, DATASET, TrustedSource
from knowledge_sufficiency.target import AcquisitionTarget

DIMENSIONS = 128


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(DIMENSIONS).astype(np.float32)
    return v / np.linalg.norm(v)


DB, WEB, PAPERS = unit(1), unit(2), unit(3)
VECTORS = {"databases": DB, "web": WEB, "papers": PAPERS}

SOURCES = (
    TrustedSource("sqlite", "sqlite.org", "databases", "https://sqlite.org/s?q={terms}"),
    TrustedSource("mdn", "developer.mozilla.org", "web", "https://mdn/s?q={terms}"),
    TrustedSource("arxiv", "arxiv.org", "papers", "https://arxiv.org/s?q={terms}"),
    TrustedSource("hf", "huggingface.co", "papers", "https://hf/s?q={terms}", DATASET),
)


def embed_one(text):
    return VECTORS.get(text, unit(99))


def target(topic_vector, subtopic_vectors=()):
    return AcquisitionTarget(
        topic="t", query="q", topic_vector=topic_vector,
        query_vector=unit(50), subtopic_vectors=tuple(subtopic_vectors),
        subtopics=tuple("s" for _ in subtopic_vectors),
    )


@pytest.fixture
def index():
    return SourceIndex(embed_one, SOURCES)


class TestRanking:
    def test_the_closest_source_ranks_first(self, index):
        assert index.rank(target(DB))[0].source.name == "sqlite"

    def test_an_unrelated_source_ranks_last(self, index):
        assert index.rank(target(DB))[-1].score < index.rank(target(DB))[0].score

    def test_every_source_is_scored(self, index):
        assert len(index.rank(target(DB))) == len(SOURCES)

    def test_subtopics_count_toward_the_score(self, index):
        """A topic phrased unlike any description can still be placed by them."""
        ranked = index.rank(target(unit(77), [WEB]))
        assert ranked[0].source.name == "mdn"

    def test_the_query_does_not_decide(self, index):
        """A source is chosen for the subject it covers, not the question's wording."""
        chosen = index.rank(target(DB))[0]
        assert chosen.source.name == "sqlite"

    def test_an_empty_index_ranks_nothing(self):
        assert SourceIndex(embed_one, ()).rank(target(DB)) == []


class TestSelection:
    def test_only_the_relevant_source_is_chosen(self, index):
        assert [s.name for s in index.select(target(DB))] == ["sqlite"]

    def test_unrelated_sources_are_not_searched(self, index):
        assert "mdn" not in [s.name for s in index.select(target(DB))]

    def test_the_number_chosen_is_capped(self, index):
        chosen = index.select(target(DB), top_k=1, floor=-1.0)
        assert len(chosen) == 1

    def test_a_lower_floor_admits_more(self, index):
        assert len(index.select(target(DB), floor=-1.0)) > len(index.select(target(DB)))

    def test_the_kind_is_respected(self, index):
        """Datasets are acquired differently, so they are selected separately."""
        assert [s.name for s in index.select(target(PAPERS), kind=DATASET)] == ["hf"]

    def test_docs_selection_excludes_datasets(self, index):
        assert all(s.kind == DOCS for s in index.select(target(PAPERS), kind=DOCS))

    def test_nothing_clearing_the_floor_falls_back_to_the_best(self, index):
        """Better than searching everything, and better than searching nothing."""
        chosen = index.select(target(unit(500)), floor=0.99)
        assert len(chosen) == 1

    def test_no_sources_of_that_kind_selects_nothing(self, index):
        assert index.select(target(DB), kind="nonexistent") == []


class TestTheDefaults:
    def test_the_floor_is_a_cosine(self):
        assert 0.0 <= SOURCE_FLOOR <= 1.0

    def test_the_measured_floor_is_pinned(self):
        """Changing it should fail here and prompt a re-measurement in the docs."""
        assert SOURCE_FLOOR == 0.25

    def test_more_than_one_source_may_be_searched(self):
        assert TOP_SOURCES > 1
