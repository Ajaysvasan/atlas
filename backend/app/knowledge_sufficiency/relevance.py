"""The gate that decides whether collected material is on-subject."""

from typing import List, NamedTuple, Sequence, Tuple

import numpy as np
from numpy import float32, ndarray

from config import get_logger
from knowledge_sufficiency.similarity import cosine_scores
from knowledge_sufficiency.target import AcquisitionTarget

logger = get_logger(__name__)

# Placeholder, like the sufficiency floors: measure before relying on it. Lower
# than SUFFICIENT_FLOOR on purpose — this asks "is this about the subject",
# which is a weaker bar than "does this answer the question".
TOPIC_FLOOR = 0.45

MAX_KEPT_CHUNKS = 200

Candidate = Tuple[object, ndarray]


class Filtered(NamedTuple):
    kept: List[Candidate]
    dropped: int
    capped: int


def _scores(vector: ndarray, candidates: Sequence[Candidate]) -> ndarray:
    matrix = np.asarray([v for _, v in candidates], dtype=float32)
    return cosine_scores(np.asarray(vector, dtype=float32), matrix)


def filter_relevant(
    target: AcquisitionTarget,
    candidates: Sequence[Candidate],
    topic_floor: float = TOPIC_FLOOR,
    max_kept: int = MAX_KEPT_CHUNKS,
) -> Filtered:
    """Keep what is about the target's subject, discard the rest.

    Subtopics narrow rather than broaden: a chunk has to clear the topic floor
    *and* match at least one subtopic. A page fetched for "write-ahead logging"
    carries plenty of material that is about databases generally, and keeping it
    is how an index fills with things nobody asked for.
    """
    candidates = list(candidates)
    if not candidates:
        return Filtered([], 0, 0)

    on_topic = _scores(target.topic_vector, candidates) >= topic_floor
    if target.subtopic_vectors:
        on_subtopic = np.zeros(len(candidates), dtype=bool)
        for vector in target.subtopic_vectors:
            on_subtopic |= _scores(vector, candidates) >= topic_floor
        keep_mask = on_topic & on_subtopic
    else:
        keep_mask = on_topic

    indices = [i for i in range(len(candidates)) if keep_mask[i]]
    dropped = len(candidates) - len(indices)

    capped = 0
    if len(indices) > max_kept:
        # Ranked by the query, not the topic: when there is more on-subject
        # material than the cap allows, the useful ones are those closest to
        # what was actually asked.
        by_query = _scores(target.query_vector, candidates)
        indices.sort(key=lambda i: float(by_query[i]), reverse=True)
        capped = len(indices) - max_kept
        indices = indices[:max_kept]

    if dropped or capped:
        logger.debug(
            "Relevance gate kept %d, dropped %d off-subject, capped %d",
            len(indices), dropped, capped,
        )
    return Filtered([candidates[i] for i in sorted(indices)], dropped, capped)
