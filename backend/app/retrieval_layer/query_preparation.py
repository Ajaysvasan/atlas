"""Turning what was asked into what each searcher needs."""

import threading
from typing import Callable, Sequence

from numpy import float32, ndarray

from config import get_logger
from retrieval_layer.keyword_search import WORD
from retrieval_layer.models import QueryPlan
from retrieval_layer.retrieval_exceptions import EmptyQuery
from retrieval_layer.settings import MAX_QUERY_TERMS

logger = get_logger(__name__)

Encoder = Callable[[Sequence[str]], ndarray]

STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "did", "do",
    "does", "for", "from", "has", "have", "how", "i", "if", "in", "into", "is",
    "it", "its", "me", "my", "no", "not", "of", "on", "or", "our", "should",
    "so", "such", "that", "the", "their", "then", "there", "these", "they",
    "this", "to", "was", "we", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "would", "you", "your",
})


def lexical_terms(text: str, max_terms: int = MAX_QUERY_TERMS) -> str:
    """The words worth matching on, each once, at most `max_terms` of them.

    Stopwords go because under OR each one matches nearly every chunk, and bm25
    then scores the whole corpus only to give those words almost no weight —
    127x slower measured at 100k chunks. A query made of nothing else keeps
    them: an empty match finds nothing, which is worse than a slow one.
    """
    words = WORD.findall(text)
    content = [word for word in words if word.casefold() not in STOPWORDS]
    unique = list(dict.fromkeys(word.casefold() for word in content or words))
    return " ".join(unique[:max_terms])


def encoder_from(manager) -> Encoder:
    """Batched text to vectors, by the same call ingestion embeds chunks with.

    A query embedded any other way — another model, another truncation, no
    float32 cast — lands in a different space from the corpus, and every
    distance the index returns is then meaningless without raising anything.
    """
    def encode(texts: Sequence[str]) -> ndarray:
        return manager.model.encode(
            list(texts), truncate_dim=manager.embedding_dimension
        ).astype(float32)

    return encode


class QueryPreparation:
    """Normalises the query once, for the embedder, bm25 and the reranker."""

    def __init__(self, encode: Encoder | None = None) -> None:
        self.__encode = encode
        self.__lock = threading.Lock()

    @property
    def encode(self) -> Encoder:
        """Loaded on first use; locked so concurrent first queries load it once."""
        if self.__encode is None:
            with self.__lock:
                if self.__encode is None:
                    from data_layer.ingestion.embedding.EmbeddingManager import (
                        EmbeddingManager,
                    )

                    self.__encode = encoder_from(EmbeddingManager())
        return self.__encode

    def prepare(self, query: str) -> QueryPlan:
        """The plan both searchers run from. The text is kept whole for the
        embedder and the reranker; only bm25 gets the stripped terms."""
        if not isinstance(query, str) or not query.strip():
            raise EmptyQuery(query)
        text = " ".join(query.split())
        vector = self.encode([text])[0]
        terms = lexical_terms(text)
        logger.debug("Prepared a query of %d term(s)", len(terms.split()))
        return QueryPlan(text=text, vector=vector, terms=terms)
