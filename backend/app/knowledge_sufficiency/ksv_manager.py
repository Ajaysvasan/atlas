"""Knowledge Sufficiency Verification: can this query be answered from what we have."""

from typing import Iterable, List, Sequence, Tuple

import numpy as np
from numpy import float32, ndarray
from numpy.typing import NDArray

from config import get_logger
from knowledge_sufficiency.acquisition import Acquisition, KnowledgeAcquisition
from knowledge_sufficiency.acquisition_store import AcquisitionStore
from knowledge_sufficiency.similarity import cosine_scores
from knowledge_sufficiency.thresholds import (
    PARTIAL_FLOOR,
    SUFFICIENT_FLOOR,
    classify,
)
from knowledge_sufficiency.relevance import MAX_KEPT_CHUNKS, TOPIC_FLOOR
from knowledge_sufficiency.sources import DEFAULT_SOURCES, TrustedSource
from knowledge_sufficiency.target import AcquisitionTarget, build_target
from knowledge_sufficiency.verdict import Sufficiency, Verdict

logger = get_logger(__name__)

Candidate = Tuple[object, ndarray]


class KSVManager:
    """Scores candidate knowledge against a query and bands the result.

    Takes candidates rather than fetching them, so the same subsystem serves the
    memory layer reading its conversation vectors out of pgvector and the global
    retrieval layer reading its own out of DiskANN. It performs no retrieval, no
    network call and no inference: it answers how much is already here, and the
    caller decides what to do about a shortfall.
    """

    def __init__(
        self,
        sufficient_floor: float = SUFFICIENT_FLOOR,
        partial_floor: float = PARTIAL_FLOOR,
    ) -> None:
        if not 0.0 <= partial_floor <= 1.0 or not 0.0 <= sufficient_floor <= 1.0:
            raise ValueError(
                "floors are cosine scores and must lie in [0, 1]; got "
                f"sufficient_floor={sufficient_floor}, partial_floor={partial_floor}"
            )
        if sufficient_floor < partial_floor:
            raise ValueError(
                f"sufficient_floor ({sufficient_floor}) is below partial_floor "
                f"({partial_floor}); nothing could ever be PARTIAL"
            )
        self.sufficient_floor = sufficient_floor
        self.partial_floor = partial_floor

    @staticmethod
    def __split(candidates: Sequence[Candidate]) -> Tuple[List, NDArray[float32]]:
        ids = [candidate_id for candidate_id, _ in candidates]
        matrix = np.asarray([vector for _, vector in candidates], dtype=float32)
        return ids, matrix

    def __supporting(
        self, ids: Sequence, scores: NDArray[float32]
    ) -> List:
        """The candidates worth handing on, best first.

        Everything at or above the partial floor, not just the best one: on a
        PARTIAL verdict the caller combines these with whatever retrieval brings
        back, so it needs all of what is already here.
        """
        keep = [i for i in range(len(ids)) if scores[i] >= self.partial_floor]
        keep.sort(key=lambda i: float(scores[i]), reverse=True)
        return [ids[i] for i in keep]

    def acquire(
        self,
        target: AcquisitionTarget,
        embed,
        urls: Sequence[str] | None = None,
        store=None,
        fetch=None,
        sources: Sequence[TrustedSource] = DEFAULT_SOURCES,
        max_rounds: int = 3,
        store_on_partial: bool = False,
        topic_floor: float = TOPIC_FLOOR,
        max_kept: int = MAX_KEPT_CHUNKS,
        acquisition_store: AcquisitionStore | None = None,
    ) -> Acquisition:
        """Gather knowledge from trusted sources until it answers the query.

        The second half of this subsystem, used by the global retrieval layer.
        The target's topic bounds what may be collected at all; its query
        decides whether what was collected answers anything. With no `urls`,
        candidates come from searching the trusted sources themselves.

        The memory layer does not call this — a shortfall there is answered by
        retrieval, not by going to the network.
        """
        acquisition = KnowledgeAcquisition(
            verify=self.sufficiency_verification,
            embed=embed,
            store=store,
            sources=sources,
            max_rounds=max_rounds,
            store_on_partial=store_on_partial,
            topic_floor=topic_floor,
            max_kept=max_kept,
            acquisition_store=acquisition_store,
            **({"fetch": fetch} if fetch is not None else {}),
        )
        return acquisition.acquire_until_sufficient(target, urls)

    def sufficiency_verification(
        self, query_vector: ndarray, candidates: Iterable[Candidate]
    ) -> Verdict:
        """How much of this query the supplied candidates already answer."""
        candidates = list(candidates)
        if not candidates:
            logger.debug("No candidates supplied; nothing can be sufficient")
            return Verdict(Sufficiency.NONE, -1.0, [], np.empty(0, dtype=float32))

        ids, matrix = self.__split(candidates)
        query = np.asarray(query_vector, dtype=float32)
        if matrix.ndim != 2 or matrix.shape[1] != query.shape[0]:
            raise ValueError(
                f"query has {query.shape[0]} dimension(s) but candidates have "
                f"{matrix.shape[1] if matrix.ndim == 2 else matrix.shape}"
            )

        scores = cosine_scores(query, matrix)
        best = float(scores.max())
        sufficiency = classify(best, self.sufficient_floor, self.partial_floor)
        supporting = self.__supporting(ids, scores)

        logger.debug(
            "Sufficiency %s over %d candidate(s): best %.3f, %d supporting",
            sufficiency.value,
            len(ids),
            best,
            len(supporting),
        )
        return Verdict(sufficiency, best, supporting, scores)
