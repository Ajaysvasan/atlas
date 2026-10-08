"""Choosing a spread of passages rather than the same one several times."""

from typing import Callable, List, Sequence

import numpy as np
from numpy import float32, ndarray

from config import get_logger
from knowledge_sufficiency.similarity import cosine_scores
from retrieval_layer.models import Passage, QueryPlan
from retrieval_layer.settings import MMR_LAMBDA

logger = get_logger(__name__)

Embedder = Callable[[Sequence[str]], Sequence[ndarray]]


def _matrix(passages: Sequence[Passage], embed: Embedder | None) -> ndarray | None:
    """The passage embeddings, from the passages or from the embedder."""
    if all(p.embedding is not None for p in passages):
        return np.asarray([p.embedding for p in passages], dtype=float32)
    if embed is None:
        return None
    return np.asarray(embed([p.text for p in passages]), dtype=float32)


def normalised(scores: Sequence[float]) -> ndarray:
    """Scores rescaled to [0, 1] across the pool, so any scorer can be traded against a cosine."""
    values = np.asarray(scores, dtype=float32)
    low, high = float(values.min()), float(values.max())
    if high == low:
        return np.ones_like(values)
    return (values - low) / (high - low)


def maximal_marginal_relevance(
    plan: QueryPlan,
    passages: Sequence[Passage],
    k: int,
    lambda_: float = MMR_LAMBDA,
    embed: Embedder | None = None,
    relevance: Sequence[float] | None = None,
) -> List[Passage]:
    """The k passages that are relevant without repeating each other.

    Overlapping chunks mean the same sentence can occupy three places in a
    ranking, and a context window filled three times over with one paragraph
    has answered less than one filled once.

    `relevance` is the judgement of whatever ranked the passages before this —
    a cross-encoder, or fusion. Without it relevance falls back to cosine
    against the query, which is the weakest signal in the pipeline: after a
    reranker, it would quietly undo the reranker.

    Without embeddings to compare passages against each other there is no way
    to tell repetition from coverage, so this falls back to the ranking it was
    given rather than guessing.
    """
    passages = list(passages)
    if k < 1 or not passages:
        return []
    if len(passages) <= k:
        return passages

    matrix = _matrix(passages, embed)
    if matrix is None:
        logger.debug("No passage embeddings; keeping the ranking as it stands")
        return passages[:k]

    if relevance is None:
        relevance = cosine_scores(np.asarray(plan.vector, dtype=float32), matrix)
    else:
        relevance = normalised(relevance)

    selected: List[int] = [int(np.argmax(relevance))]
    remaining = [i for i in range(len(passages)) if i not in selected]

    while len(selected) < k and remaining:
        chosen_matrix = matrix[selected]
        best_index, best_value = None, -np.inf
        for candidate in remaining:
            # How much this candidate repeats whatever has already been chosen.
            redundancy = float(
                np.max(cosine_scores(matrix[candidate], chosen_matrix))
            )
            value = lambda_ * float(relevance[candidate]) - (1 - lambda_) * redundancy
            if value > best_value:
                best_index, best_value = candidate, value
        selected.append(best_index)
        remaining.remove(best_index)

    logger.debug("Selected %d of %d passage(s) for diversity", len(selected),
                 len(passages))
    return [passages[i] for i in selected]
