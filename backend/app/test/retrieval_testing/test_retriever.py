"""The orchestrator, which owns no ranking logic of its own — only the order
things happen in and what each stage is handed. Every bug it can have is a
seam: a stage given the wrong count, the wrong scores, or an answer that
outlived the corpus it came from.
"""

import hashlib
import re

import numpy as np
import pytest

from logging_setup import current_context
from retrieval_layer.caching import ResultCache
from retrieval_layer.hydration import Hydration
from retrieval_layer.models import (
    FUSED, KEYWORD, VECTOR, Passage, QueryPlan, RetrievalRequest, ScoredId,
)
from retrieval_layer.query_preparation import STOPWORDS, QueryPreparation
from retrieval_layer.reranking import Reranker
from retrieval_layer.retrieval_exceptions import (
    EmptyQuery, IndexUnavailable, InvalidRetrievalSetting, RerankerUnavailable,
)
from retrieval_layer.retriever import Retriever
from retrieval_layer.settings import RetrievalSettings

DIMENSIONS = 128


def unit(seed):
    v = np.random.default_rng(seed).standard_normal(DIMENSIONS).astype(np.float32)
    return v / np.linalg.norm(v)


QUERY = unit(1)


def toward(cosine, seed):
    """A unit vector at an exact cosine to the query."""
    other = unit(seed)
    other = other - QUERY * float(other @ QUERY)
    other = other / np.linalg.norm(other)
    v = QUERY * cosine + other * np.sqrt(1 - cosine**2)
    return (v / np.linalg.norm(v)).astype(np.float32)


def ranked(ids, source):
    return [ScoredId(i, 1.0 / n, source, n) for n, i in enumerate(ids, start=1)]


class StubSearch:
    def __init__(self, ids, source, raises=None):
        self.ids, self.source, self.raises = ids, source, raises
        self.asked = []
        self.caught_up = 0
        self.synced = 0
        self.closed = 0
        self.context = None

    def search(self, plan, k):
        self.asked.append(k)
        self.context = current_context()
        if self.raises:
            raise self.raises
        return ranked(self.ids[:k], self.source)

    def catch_up(self):
        self.caught_up += 1

    def sync(self):
        self.synced += 1
        return 3

    def close(self):
        self.closed += 1


class StubHydration:
    def __init__(self, texts, missing=()):
        self.texts, self.missing = texts, set(missing)
        self.asked = []

    def hydrate(self, scored):
        self.asked.append([s.vector_id for s in scored])
        passages = [
            Passage(s.vector_id, f"c{s.vector_id}", self.texts[s.vector_id],
                    s.score, s.source, "doc", s.vector_id * 100)
            for s in scored if s.vector_id not in self.missing
        ]
        return passages, len(scored) - len(passages)


def encoder_for(vectors):
    """Each passage text to a chosen vector; anything else is the query."""
    def encode(texts):
        return np.asarray([vectors.get(t, QUERY) for t in texts], dtype=np.float32)
    return encode


def retriever(dense=(1, 2, 3), sparse=(3, 1), texts=None, vectors=None,
              scorer=None, settings=None, cache=None, missing=(), **stages):
    texts = texts or {i: f"passage {i}" for i in range(1, 50)}
    vectors = vectors or {t: unit(100 + i) for i, t in texts.items()}
    return Retriever(
        settings or RetrievalSettings(),
        preparation=stages.get("preparation",
                               QueryPreparation(encoder_for(vectors))),
        vector_search=stages.get("vector_search", StubSearch(list(dense), VECTOR)),
        keyword_search=stages.get("keyword_search",
                                  StubSearch(list(sparse), KEYWORD)),
        hydration=stages.get("hydration", StubHydration(texts, missing)),
        reranker=stages.get("reranker", Reranker(scorer=scorer or (
            lambda pairs: [float(-i) for i in range(len(pairs))]))),
        cache=cache,
    )


class TestWhatItAccepts:
    def test_a_plain_string(self):
        assert retriever().retrieve("a query").passages

    def test_a_request(self):
        assert retriever().retrieve(RetrievalRequest("a query", top_k=2)).passages

    @pytest.mark.parametrize("query", ["", "   ", None])
    def test_nothing_to_search_for(self, query):
        with pytest.raises(EmptyQuery):
            retriever().retrieve(RetrievalRequest(query))

    @pytest.mark.parametrize("top_k", [0, -3])
    def test_asking_for_no_results(self, top_k):
        """Settings validate their own top_k; a request can still override it."""
        with pytest.raises(InvalidRetrievalSetting, match="top_k"):
            retriever().retrieve(RetrievalRequest("q", top_k=top_k))

    def test_a_refused_request_never_reaches_the_cache(self):
        r = retriever()
        with pytest.raises(EmptyQuery):
            r.retrieve("   ")
        assert r.cache.stats.misses == 0


class TestWhatEachStageIsGiven:
    def test_both_searchers_over_fetch(self):
        """Reranking and MMR choose among what the searchers bring back."""
        r = retriever(settings=RetrievalSettings(top_k=8, candidate_multiplier=4))
        r.retrieve("q")
        assert r.vector_search.asked == [32]
        assert r.keyword_search.asked == [32]

    def test_the_pool_follows_the_request_not_the_default(self):
        r = retriever(settings=RetrievalSettings(top_k=8, candidate_multiplier=4))
        r.retrieve(RetrievalRequest("q", top_k=3))
        assert r.vector_search.asked == [12]

    def test_only_the_pool_is_hydrated(self):
        """Fusion can return up to twice the pool; reading all of it is waste."""
        r = retriever(dense=range(1, 21), sparse=range(21, 41),
                      settings=RetrievalSettings(top_k=2, candidate_multiplier=4))
        r.retrieve("q")
        assert len(r.hydration.asked[0]) == 8

    def test_hydration_receives_the_fused_order(self):
        r = retriever(dense=[1, 2, 3], sparse=[3, 1])
        r.retrieve("q")
        assert r.hydration.asked[0][:2] == [1, 3]

    def test_the_reranker_is_skipped_when_not_wanted(self):
        calls = []
        r = retriever(scorer=lambda pairs: calls.append(1) or [0.0] * len(pairs))
        result = r.retrieve(RetrievalRequest("q", rerank=False))
        assert calls == []
        assert "rerank" not in [t.stage for t in result.timings]

    def test_the_reranker_runs_when_wanted(self):
        calls = []
        r = retriever(scorer=lambda pairs: calls.append(1) or [0.0] * len(pairs))
        r.retrieve(RetrievalRequest("q", rerank=True))
        assert calls == [1]


class TestTheReRankersJudgementSurvivesDiversity:
    """MMR runs after reranking. Given cosine instead of the reranker's scores
    it re-ranks by the bi-encoder, and the cross-encoder changes nothing."""

    def test_the_reranked_leader_leads_the_result(self):
        texts = {1: "lexical match", 2: "the answer", 3: "filler a", 4: "filler b"}
        vectors = {"lexical match": toward(0.90, 11), "the answer": toward(0.55, 12),
                   "filler a": toward(0.40, 13), "filler b": toward(0.35, 14)}
        judged = {"the answer": 9.0, "lexical match": -2.0,
                  "filler a": -5.0, "filler b": -6.0}
        r = retriever(dense=[1, 2, 3, 4], sparse=[], texts=texts, vectors=vectors,
                      scorer=lambda pairs: [judged[p] for _, p in pairs])

        result = r.retrieve(RetrievalRequest("q", top_k=1, rerank=True))
        assert result.passages[0].text == "the answer"

    def test_without_reranking_fusion_leads(self):
        """Found by both searchers outranks closest by cosine alone."""
        texts = {1: "closest", 2: "found twice", 3: "other"}
        vectors = {"closest": toward(0.95, 21), "found twice": toward(0.60, 22),
                   "other": toward(0.30, 23)}
        r = retriever(dense=[1, 2, 3], sparse=[2], texts=texts, vectors=vectors)

        result = r.retrieve(RetrievalRequest("q", top_k=1, rerank=False))
        assert result.passages[0].text == "found twice"


class TestTheResult:
    def test_every_stage_is_timed_in_order(self):
        result = retriever().retrieve(RetrievalRequest("q", rerank=True))
        assert [t.stage for t in result.timings] == [
            "cache", "prepare", "vector search", "keyword search", "fusion",
            "hydration", "rerank", "diversity", "assembly",
        ]

    def test_it_counts_what_it_considered(self):
        result = retriever(dense=[1, 2, 3], sparse=[3, 4]).retrieve("q")
        assert result.candidates_considered == 4

    def test_it_counts_what_had_no_text(self):
        result = retriever(dense=[1, 2, 3], sparse=[], missing=[2]).retrieve("q")
        assert result.dropped_unresolvable == 1
        assert 2 not in [p.vector_id for p in result.passages]

    def test_it_reports_the_tokens_spent(self):
        texts = {1: "x" * 40, 2: "y" * 80}
        result = retriever(dense=[1, 2], sparse=[], texts=texts).retrieve(
            RetrievalRequest("q", rerank=False))
        assert result.tokens == (40 + 80) // 4

    def test_it_returns_no_more_than_top_k(self):
        result = retriever(dense=range(1, 30), sparse=[]).retrieve(
            RetrievalRequest("q", top_k=3))
        assert len(result.passages) == 3

    def test_the_budget_is_respected(self):
        texts = {i: f"{i:03d}" + "w" * 397 for i in range(1, 20)}
        r = retriever(dense=range(1, 20), sparse=[], texts=texts,
                      settings=RetrievalSettings(token_budget=250))
        result = r.retrieve("q")
        assert result.tokens <= 250
        assert len(result.passages) == 2

    def test_a_repeat_is_dropped_before_it_can_take_a_slot(self):
        """Overlapping windows repeat text under new ids; a repeat that reached
        MMR would cost one of the k places."""
        texts = {1: "same words", 2: "same  words", 3: "different", 4: "more"}
        result = retriever(dense=[1, 2, 3, 4], sparse=[], texts=texts).retrieve(
            RetrievalRequest("q", top_k=3, rerank=False))
        assert len(result.passages) == 3
        assert sum(1 for p in result.passages if "same" in p.text) == 1

    def test_nothing_found_is_an_empty_result_not_an_error(self):
        result = retriever(dense=[], sparse=[]).retrieve("q")
        assert result.passages == []
        assert result.tokens == 0


class TestCaching:
    def test_the_second_ask_is_served_from_the_cache(self):
        r = retriever()
        first = r.retrieve("q")
        second = r.retrieve("q")
        assert second.cached and not first.cached
        assert [p.vector_id for p in second.passages] == \
            [p.vector_id for p in first.passages]
        assert r.vector_search.asked == [32]

    def test_a_hit_reports_its_own_timing_not_the_original(self):
        """A hit taking 0.01 ms must not report the 200 ms search behind it."""
        r = retriever()
        r.retrieve("q")
        hit = r.retrieve("q")
        assert [t.stage for t in hit.timings] == ["cache"]

    def test_changing_a_returned_result_does_not_change_the_cache(self):
        r = retriever()
        first = r.retrieve("q")
        size = len(first.passages)
        first.passages.clear()
        assert len(r.retrieve("q").passages) == size

    def test_changing_a_hit_does_not_change_the_cache(self):
        r = retriever()
        r.retrieve("q")
        hit = r.retrieve("q")
        size = len(hit.passages)
        hit.passages.append("junk")
        assert len(r.retrieve("q").passages) == size

    def test_a_different_k_is_a_different_answer(self):
        r = retriever()
        r.retrieve(RetrievalRequest("q", top_k=2))
        assert not r.retrieve(RetrievalRequest("q", top_k=3)).cached

    def test_a_default_and_its_explicit_value_share_an_entry(self):
        r = retriever(settings=RetrievalSettings(top_k=8))
        r.retrieve(RetrievalRequest("q"))
        assert r.retrieve(RetrievalRequest("q", top_k=8)).cached

    def test_a_failure_is_not_cached(self):
        vector = StubSearch([1], VECTOR, raises=IndexUnavailable("p", "gone"))
        r = retriever(vector_search=vector)
        with pytest.raises(IndexUnavailable):
            r.retrieve("q")
        vector.raises = None
        assert not r.retrieve("q").cached

    def test_caching_can_be_switched_off(self):
        r = retriever(cache=ResultCache(max_size=0))
        r.retrieve("q")
        assert not r.retrieve("q").cached


class TestFailures:
    def test_a_missing_index_reaches_the_caller(self):
        """A setup problem, not something to answer around."""
        r = retriever(vector_search=StubSearch(
            [], VECTOR, raises=IndexUnavailable("p", "no index")))
        with pytest.raises(IndexUnavailable):
            r.retrieve("q")

    def test_a_reranker_that_cannot_load_reaches_the_caller(self):
        def unloadable(pairs):
            raise RerankerUnavailable("model", "not installed")

        with pytest.raises(RerankerUnavailable):
            retriever(scorer=unloadable).retrieve(RetrievalRequest("q", rerank=True))

    def test_a_reranker_that_fails_mid_batch_keeps_the_fused_order(self):
        def broken(pairs):
            raise RuntimeError("CUDA out of memory")

        result = retriever(scorer=broken).retrieve(RetrievalRequest("q", rerank=True))
        assert result.passages


class TestLogging:
    def test_every_stage_logs_under_one_retrieval_id(self):
        """So one request can be pulled out of an interleaved log."""
        r = retriever()
        r.retrieve("q")
        assert "retrieval_id" in r.vector_search.context
        assert r.keyword_search.context["retrieval_id"] == \
            r.vector_search.context["retrieval_id"]

    def test_each_retrieval_gets_its_own_id(self):
        r = retriever()
        r.retrieve("first")
        first = r.vector_search.context["retrieval_id"]
        r.retrieve("second")
        assert r.vector_search.context["retrieval_id"] != first

    def test_the_id_does_not_leak_past_the_retrieval(self):
        retriever().retrieve("q")
        assert "retrieval_id" not in current_context()


class TestRefreshing:
    def test_refresh_brings_every_store_up_to_date(self):
        r = retriever()
        r.retrieve("q")
        assert r.refresh() == 3
        assert r.keyword_search.synced == 1
        assert r.vector_search.caught_up == 1
        assert not r.retrieve("q").cached


class TestWhatItOwns:
    def test_it_does_not_close_what_it_was_handed(self):
        r = retriever()
        r.close()
        assert r.keyword_search.closed == 0

    def test_it_closes_what_it_opened(self, tmp_path):
        r = Retriever(chunk_store_path=str(tmp_path / "chunks"),
                      keyword_index_path=tmp_path / "kw.sql",
                      preparation=QueryPreparation(encoder_for({})),
                      vector_search=StubSearch([], VECTOR),
                      hydration=StubHydration({}))
        r.close()
        assert r.keyword_search.connection is None

    def test_it_works_as_a_context_manager(self, tmp_path):
        with Retriever(chunk_store_path=str(tmp_path / "chunks"),
                       keyword_index_path=tmp_path / "kw.sql",
                       preparation=QueryPreparation(encoder_for({})),
                       vector_search=StubSearch([], VECTOR),
                       hydration=StubHydration({})) as r:
            keyword = r.keyword_search
        assert keyword.connection is None


# --- End to end: every store real, only the two models stood in for --------

CORPUS = [
    ("wal", "the write ahead log records every change before it reaches the "
            "main database file, and a checkpoint copies those pages back"),
    ("btree", "a b tree keeps keys sorted in pages so a lookup touches only a "
              "few pages from the root down to a leaf"),
    ("vacuum", "vacuum rebuilds the database file to reclaim pages freed by "
               "deleted rows and to defragment the tables"),
    ("fts", "full text search tokenises each document and ranks matches with "
            "bm25 using term frequency and inverse document frequency"),
    ("ann", "approximate nearest neighbour search walks a graph of vectors "
            "toward the query instead of comparing against every vector"),
    ("error", "if the server returns error E4471 the replica has fallen behind "
              "and must be resynchronised from a fresh snapshot"),
    ("cache", "an lru cache evicts the entry used least recently once it is "
              "full, and a ttl expires entries that have grown stale"),
    ("rrf", "reciprocal rank fusion merges ranked lists by summing one over "
            "the constant plus each rank, ignoring raw scores"),
]


def bag_of_words(texts):
    """A stand-in embedder: shared words make close vectors. It cannot see
    identifiers containing digits, as small sentence models largely cannot."""
    out = np.zeros((len(texts), DIMENSIONS), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in re.findall(r"\w+", text.lower()):
            if word in STOPWORDS or any(c.isdigit() for c in word):
                continue
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % DIMENSIONS
            out[row, bucket] += 1.0
        norm = np.linalg.norm(out[row])
        out[row] = out[row] / norm if norm else out[row]
    return out


def overlap_scorer(pairs):
    scores = []
    for query, passage in pairs:
        asked = set(re.findall(r"\w+", query.lower())) - STOPWORDS
        scores.append(float(len(asked & set(re.findall(r"\w+", passage.lower())))))
    return scores


class Corpus:
    """Ingests through the data layer's own stores, as the pipeline does."""

    def __init__(self, tmp_path):
        from data_layer.ingestion.Chunker.DB_Manager import Manager
        from data_layer.ingestion.nodes.nodes import Document
        from data_layer.vector_db_manager.repository.vectorMetaDataRepository import (
            VectorMetaDataRepository,
        )
        from data_layer.vector_db_manager.vectorDbManager import VectorDbManager

        self.path = str(tmp_path / "chunks")
        self.store = Manager(self.path, is_chunker_type_hierarchical=False)
        self.store.insert_documents([Document("d1", "notes.md", "")])
        self.labels = VectorMetaDataRepository(self.path)
        self.index = VectorDbManager(
            distance_metrics="l2", vector_dtype=np.float32, dimensions=DIMENSIONS,
            max_vectors=1000, complexity=64, graph_degree=32, num_threads=1,
            k_neighbors=9,
        )

    def add(self, entries):
        from data_layer.ingestion.metadata.metadata import (
            ChunkMetaData, EmbeddedChunkMetaData,
        )
        from data_layer.ingestion.nodes.nodes import EmbeddedChunk, RChunk

        chunks = [RChunk(text, ChunkMetaData("notes.md", "d1", "recursive"),
                         chunk_id, 0, len(text)) for chunk_id, text in entries]
        self.store.insert_recursive_chunks(chunks)
        vectors = bag_of_words([c.chunk for c in chunks])
        labels = self.labels.allocate_many([c.chunk_id for c in chunks], list(vectors))
        embedded = [EmbeddedChunk(v, label, EmbeddedChunkMetaData(c.chunk_id, c.chunk, "bow"))
                    for c, v, label in zip(chunks, vectors, labels)]
        self.index.batch_insert(embedded, vector_ids=labels)

    def close(self):
        self.store.close()
        self.labels.close()


@pytest.fixture
def corpus(tmp_path, real_diskann):
    built = Corpus(tmp_path)
    built.add(CORPUS)
    yield built
    built.close()


@pytest.fixture
def live(corpus, tmp_path):
    from retrieval_layer.vector_search import VectorSearch

    r = Retriever(
        RetrievalSettings(top_k=3),
        chunk_store_path=corpus.path,
        keyword_index_path=tmp_path / "kw.sql",
        preparation=QueryPreparation(bag_of_words),
        vector_search=VectorSearch(index=corpus.index),
        reranker=Reranker(scorer=overlap_scorer),
    )
    yield r
    r.close()


class TestEndToEnd:
    def test_the_answer_comes_back_as_text(self, live):
        result = live.retrieve("how does the write ahead log checkpoint work")
        assert result.passages[0].chunk_id == "wal"
        assert result.passages[0].text.startswith("the write ahead log")

    def test_both_searchers_contributed(self, live):
        result = live.retrieve("write ahead log checkpoint")
        searched = {t.stage: t.produced for t in result.timings}
        assert searched["vector search"] > 0
        assert searched["keyword search"] > 0

    def test_an_identifier_only_bm25_can_see_is_still_found(self, live):
        """The embedder is blind to E4471; the lexical half carries it through
        fusion, hydration, reranking and MMR to the result."""
        result = live.retrieve(RetrievalRequest("E4471", rerank=False))
        assert "error" in [p.chunk_id for p in result.passages]

    def test_the_result_fits_the_request(self, live):
        result = live.retrieve(RetrievalRequest("database pages", top_k=2))
        assert 1 <= len(result.passages) <= 2
        assert result.tokens == sum(len(p.text) // 4 for p in result.passages)

    def test_no_passage_appears_twice(self, live):
        result = live.retrieve(RetrievalRequest("database pages file", top_k=3))
        ids = [p.chunk_id for p in result.passages]
        assert len(ids) == len(set(ids))

    def test_a_chunk_ingested_later_is_found_after_refresh(self, corpus, live):
        live.retrieve("photosynthesis chlorophyll")
        corpus.add([("plants", "photosynthesis converts light into chemical "
                               "energy using chlorophyll in the leaves")])
        assert live.refresh() == 1
        result = live.retrieve("photosynthesis chlorophyll")
        assert not result.cached
        assert result.passages[0].chunk_id == "plants"

    def test_the_same_retrieval_twice_is_a_cache_hit(self, live):
        live.retrieve("lru cache eviction")
        assert live.retrieve("LRU   cache eviction").cached

    @pytest.mark.parametrize("source", ["live", "restored"])
    def test_the_dense_half_alone_finds_the_answer(self, corpus, tmp_path, source):
        """Every other test here could pass on bm25 alone. With the lexical
        half switched off, the answer has to come from DiskANN labels resolved
        through vector_meta_data into chunk text — from the live index, and
        from one rebuilt out of the stored vectors as after a restart."""
        from retrieval_layer.vector_search import VectorSearch

        if source == "live":
            vectors = VectorSearch(index=corpus.index)
        else:
            vectors = VectorSearch(chunk_store_path=corpus.path)
        r = Retriever(
            RetrievalSettings(top_k=3),
            preparation=QueryPreparation(bag_of_words),
            vector_search=vectors,
            keyword_search=StubSearch([], KEYWORD),
            hydration=Hydration(corpus.path),
            reranker=Reranker(scorer=overlap_scorer),
        )
        result = r.retrieve(RetrievalRequest("lru cache evicts stale entries",
                                             rerank=False))
        assert result.passages[0].chunk_id == "cache"
        assert result.passages[0].text.startswith("an lru cache")

    def test_refresh_finds_a_later_ingest_through_the_store(self, corpus, tmp_path):
        """The production path: the index is built from the store, and refresh
        adds what was ingested since without rebuilding it."""
        from retrieval_layer.vector_search import VectorSearch

        r = Retriever(
            RetrievalSettings(top_k=3),
            preparation=QueryPreparation(bag_of_words),
            vector_search=VectorSearch(chunk_store_path=corpus.path),
            keyword_search=StubSearch([], KEYWORD),
            hydration=Hydration(corpus.path),
            reranker=Reranker(scorer=overlap_scorer),
        )
        r.retrieve(RetrievalRequest("photosynthesis chlorophyll", rerank=False))
        corpus.add([("plants", "photosynthesis converts light into chemical "
                               "energy using chlorophyll in the leaves")])
        r.refresh()
        result = r.retrieve(RetrievalRequest("photosynthesis chlorophyll",
                                             rerank=False))
        assert result.passages[0].chunk_id == "plants"
        assert r.vector_search.index.count() == len(CORPUS) + 1
