"""What moves between the retrieval stages."""

from typing import Dict, List, NamedTuple, Sequence, Tuple

from numpy import ndarray

VECTOR = "vector"
KEYWORD = "keyword"
FUSED = "fused"


class RetrievalRequest(NamedTuple):
    query: str
    top_k: int | None = None
    rerank: bool | None = None

    def with_defaults(self, settings) -> "RetrievalRequest":
        return RetrievalRequest(
            query=self.query,
            top_k=settings.top_k if self.top_k is None else self.top_k,
            rerank=settings.rerank if self.rerank is None else self.rerank,
        )


class QueryPlan(NamedTuple):
    """A query prepared for both searchers at once."""

    text: str
    vector: ndarray
    terms: str


class ScoredId(NamedTuple):
    """A candidate, before anyone knows what it says.

    Carries `rank` as well as `score` because fusion works on rank: a bm25
    score and a cosine distance are not on the same scale and averaging them
    means nothing, while their positions in two lists are comparable.
    """

    vector_id: int
    score: float
    source: str
    rank: int


class Passage(NamedTuple):
    """A candidate with its text, once hydration has been there."""

    vector_id: int
    chunk_id: str
    text: str
    score: float
    source: str
    document_id: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    embedding: ndarray | None = None

    def tokens(self, chars_per_token: int) -> int:
        return len(self.text) // chars_per_token

    def rescored(self, score: float, source: str | None = None) -> "Passage":
        """The same passage after a later stage has judged it again."""
        return self._replace(score=score, source=source or self.source)


class StageTiming(NamedTuple):
    stage: str
    milliseconds: float
    produced: int


class RetrievalResult(NamedTuple):
    passages: List[Passage]
    timings: List[StageTiming]
    cached: bool = False
    candidates_considered: int = 0
    dropped_unresolvable: int = 0

    @property
    def text(self) -> str:
        return "\n\n".join(passage.text for passage in self.passages)

    @property
    def total_milliseconds(self) -> float:
        return sum(timing.milliseconds for timing in self.timings)

    def slowest_stage(self) -> StageTiming | None:
        return max(self.timings, key=lambda t: t.milliseconds, default=None)


def rank_of(scored: Sequence[ScoredId]) -> Dict[int, int]:
    """Vector id to its position, for a list already in order."""
    return {item.vector_id: item.rank for item in scored}


def as_ranked(
    pairs: Sequence[Tuple[int, float]], source: str, descending: bool = True
) -> List[ScoredId]:
    """Score pairs into ranked candidates, best first.

    `descending` is False for a distance, where smaller is closer — the one
    place a raw DiskANN result differs from everything else here.
    """
    ordered = sorted(pairs, key=lambda pair: pair[1], reverse=descending)
    return [
        ScoredId(vector_id=int(vector_id), score=float(score), source=source,
                 rank=position)
        for position, (vector_id, score) in enumerate(ordered, start=1)
    ]
