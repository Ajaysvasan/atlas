"""Preparing a query for the embedder, bm25 and the reranker at once.

The one failure here that raises nothing is a query embedded differently from
the corpus: every distance the index returns is then measured between two
spaces and the ranking is noise. So the central test runs the real ingestion
embedding path beside this one and requires the same vector from each.
"""

import threading

import numpy as np
import pytest

from retrieval_layer.query_preparation import (
    STOPWORDS,
    QueryPreparation,
    encoder_from,
    lexical_terms,
)
from retrieval_layer.retrieval_exceptions import EmptyQuery
from retrieval_layer.settings import MAX_QUERY_TERMS


class FakeModel:
    """Stands in for SentenceTransformer: deterministic, and records its calls.

    Returns float64 on purpose — the real model's dtype is not something to
    rely on, which is why both paths cast.
    """

    def __init__(self, width=384):
        self.width = width
        self.calls = []

    def encode(self, texts, truncate_dim=None):
        self.calls.append({"texts": texts, "truncate_dim": truncate_dim})
        single = isinstance(texts, str)
        rows = []
        for text in [texts] if single else texts:
            seed = sum(ord(c) for c in text)
            rows.append(np.random.default_rng(seed).standard_normal(self.width))
        out = np.asarray(rows)[:, :truncate_dim] if truncate_dim else np.asarray(rows)
        return out[0] if single else out


class FakeManager:
    def __init__(self, dimension=128):
        self.model = FakeModel()
        self.embedding_dimension = dimension


def fixed_encoder(width=128):
    calls = []

    def encode(texts):
        calls.append(list(texts))
        return np.ones((len(texts), width), dtype=np.float32)

    encode.calls = calls
    return encode


class TestTheLexicalTerms:
    def test_stopwords_are_dropped(self):
        assert lexical_terms("how does the write ahead log work") == \
            "write ahead log work"

    def test_a_query_of_only_stopwords_keeps_them(self):
        """An empty match finds nothing at all, which is worse than slow."""
        assert lexical_terms("what is it") == "what is it"

    def test_stopwords_match_whatever_their_case(self):
        assert lexical_terms("The WAL") == "wal"

    def test_each_term_appears_once(self):
        """A repeated OR term adds a posting-list walk and no information."""
        assert lexical_terms("WAL wal Wal checkpoint") == "wal checkpoint"

    def test_the_order_of_first_appearance_is_kept(self):
        assert lexical_terms("zeta alpha zeta beta") == "zeta alpha beta"

    def test_the_number_of_terms_is_capped(self):
        pasted = " ".join(f"frame{i}" for i in range(500))
        assert len(lexical_terms(pasted).split()) == MAX_QUERY_TERMS

    def test_the_cap_keeps_the_first_terms(self):
        assert lexical_terms("a1 b2 c3 d4", max_terms=2) == "a1 b2"

    def test_punctuation_that_means_something_to_fts5_is_gone(self):
        assert lexical_terms('"write-ahead" AND (log* OR NEAR)') == \
            "write ahead log near"

    def test_words_outside_ascii_survive(self):
        assert lexical_terms("naïve café") == "naïve café"

    def test_the_stopword_list_holds_no_content_words(self):
        for word in ("log", "index", "database", "error", "python", "query"):
            assert word not in STOPWORDS


class TestTheSameSpaceAsTheCorpus:
    def test_a_query_embeds_exactly_as_a_chunk_with_the_same_text(self):
        from data_layer.ingestion.embedding.EmbeddingManager import EmbeddingManager
        from data_layer.ingestion.metadata.metadata import ChunkMetaData
        from data_layer.ingestion.nodes.nodes import RChunk

        manager = EmbeddingManager.__new__(EmbeddingManager)
        manager.model = FakeModel()
        manager.model_name = "fake"
        manager.embedding_dimension = 128

        text = "the write ahead log is checkpointed into the main database"
        chunk = RChunk(text, ChunkMetaData("doc", "d1", "recursive"), "c1")
        as_a_chunk = manager.embed([chunk])[0].vector
        as_a_query = encoder_from(manager)([text])[0]

        assert as_a_query.dtype == as_a_chunk.dtype == np.float32
        assert as_a_query.shape == as_a_chunk.shape == (128,)
        np.testing.assert_array_equal(as_a_query, as_a_chunk)

    def test_it_truncates_to_the_corpus_dimension(self):
        manager = FakeManager(dimension=128)
        encoder_from(manager)(["a query"])
        assert manager.model.calls[0]["truncate_dim"] == 128

    def test_it_returns_float32(self):
        """The index is float32; a float64 query is refused or silently cast."""
        assert encoder_from(FakeManager())(["q"]).dtype == np.float32

    def test_it_encodes_a_batch_in_one_call(self):
        """MMR embeds every candidate; one call per passage is the slow way."""
        manager = FakeManager()
        out = encoder_from(manager)(["a", "b", "c"])
        assert len(manager.model.calls) == 1
        assert out.shape == (3, 128)


class TestPreparingAPlan:
    def test_the_text_is_kept_whole_for_the_reranker(self):
        """A cross-encoder reads language; stripping stopwords would hurt it."""
        plan = QueryPreparation(fixed_encoder()).prepare("how does the WAL work")
        assert plan.text == "how does the WAL work"

    def test_only_bm25_gets_the_stripped_terms(self):
        plan = QueryPreparation(fixed_encoder()).prepare("how does the WAL work")
        assert plan.terms == "wal work"

    def test_the_embedder_sees_the_whole_text(self):
        encode = fixed_encoder()
        QueryPreparation(encode).prepare("how does the WAL work")
        assert encode.calls == [["how does the WAL work"]]

    def test_spacing_is_collapsed(self):
        plan = QueryPreparation(fixed_encoder()).prepare("  how\tdoes \n WAL  ")
        assert plan.text == "how does WAL"

    def test_the_vector_is_one_row(self):
        plan = QueryPreparation(fixed_encoder()).prepare("a query")
        assert plan.vector.shape == (128,)

    @pytest.mark.parametrize("query", ["", "   ", "\n\t", None, 42])
    def test_nothing_to_search_for_is_refused(self, query):
        with pytest.raises(EmptyQuery):
            QueryPreparation(fixed_encoder()).prepare(query)

    def test_a_refused_query_costs_no_embedding(self):
        encode = fixed_encoder()
        with pytest.raises(EmptyQuery):
            QueryPreparation(encode).prepare("   ")
        assert encode.calls == []


class TestLoadingTheModel:
    def test_nothing_loads_until_the_first_query(self, monkeypatch):
        import data_layer.ingestion.embedding.EmbeddingManager as module

        built = []
        monkeypatch.setattr(module, "EmbeddingManager",
                            lambda: built.append(1) or FakeManager())
        preparation = QueryPreparation()
        assert built == []
        preparation.prepare("a query")
        assert built == [1]

    def test_concurrent_first_queries_load_it_once(self, monkeypatch):
        """Each load is a model read from disk; doing it per thread is waste."""
        import time

        import data_layer.ingestion.embedding.EmbeddingManager as module

        built = []

        def slow_manager():
            built.append(1)
            time.sleep(0.05)
            return FakeManager()

        monkeypatch.setattr(module, "EmbeddingManager", slow_manager)
        preparation = QueryPreparation()
        threads = [threading.Thread(target=preparation.prepare, args=("q",))
                   for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert built == [1]
