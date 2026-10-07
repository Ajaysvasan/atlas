"""Every knob the retrieval layer has, in one place."""

from dataclasses import dataclass

from retrieval_layer.retrieval_exceptions import InvalidRetrievalSetting

DEFAULT_TOP_K = 8

# Each searcher returns this many times the final k before fusion. Reranking
# can only reorder what it is given, so the candidate pool has to be wider than
# the answer; too wide and the cross-encoder cost grows linearly for nothing.
CANDIDATE_MULTIPLIER = 4

# The smoothing constant from the Reciprocal Rank Fusion paper. It damps the
# top of each list so one searcher's first result cannot dominate the other's
# first three, which is the whole point of fusing by rank instead of by score.
RRF_K = 60

RERANK = True
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_BATCH = 32

# 1.0 is pure relevance, 0.0 pure diversity. Leaning to relevance: a retrieval
# that drops the best passage to avoid repeating itself has failed at its job.
MMR_LAMBDA = 0.7

# Matches the memory layer's estimate so one corpus is not measured two ways.
CHARS_PER_TOKEN = 4
TOKEN_BUDGET = 2048

CACHE_SIZE = 256
CACHE_TTL_SECONDS = 300.0


@dataclass(frozen=True)
class RetrievalSettings:
    """The knobs, bundled so they travel together rather than as ten arguments."""

    top_k: int = DEFAULT_TOP_K
    candidate_multiplier: int = CANDIDATE_MULTIPLIER
    rrf_k: int = RRF_K
    rerank: bool = RERANK
    rerank_model: str = RERANK_MODEL
    rerank_batch: int = RERANK_BATCH
    mmr_lambda: float = MMR_LAMBDA
    chars_per_token: int = CHARS_PER_TOKEN
    token_budget: int = TOKEN_BUDGET
    cache_size: int = CACHE_SIZE
    cache_ttl_seconds: float = CACHE_TTL_SECONDS

    def __post_init__(self) -> None:
        for name, value, ok, expected in (
            ("top_k", self.top_k, self.top_k >= 1, "at least 1"),
            ("candidate_multiplier", self.candidate_multiplier,
             self.candidate_multiplier >= 1, "at least 1"),
            ("rrf_k", self.rrf_k, self.rrf_k > 0, "greater than 0"),
            ("rerank_batch", self.rerank_batch, self.rerank_batch >= 1,
             "at least 1"),
            ("mmr_lambda", self.mmr_lambda, 0.0 <= self.mmr_lambda <= 1.0,
             "in [0, 1]"),
            ("chars_per_token", self.chars_per_token, self.chars_per_token >= 1,
             "at least 1"),
            ("token_budget", self.token_budget, self.token_budget >= 1,
             "at least 1"),
            ("cache_size", self.cache_size, self.cache_size >= 0,
             "0 or more"),
            ("cache_ttl_seconds", self.cache_ttl_seconds,
             self.cache_ttl_seconds > 0, "greater than 0"),
        ):
            if not ok:
                raise InvalidRetrievalSetting(name, value, expected)

    @property
    def candidates(self) -> int:
        """How many each searcher fetches before fusion narrows them."""
        return self.top_k * self.candidate_multiplier
