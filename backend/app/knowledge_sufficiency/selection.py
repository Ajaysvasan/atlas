"""Choosing which trusted sources to ask about a given topic."""

from typing import Callable, List, NamedTuple, Sequence, Tuple

import numpy as np
from numpy import float32, ndarray

from config import get_logger
from knowledge_sufficiency.similarity import cosine_scores
from knowledge_sufficiency.sources import DEFAULT_SOURCES, DOCS, TrustedSource
from knowledge_sufficiency.target import AcquisitionTarget

logger = get_logger(__name__)

# Measured, not guessed — unlike the other floors in this subsystem. Over six
# topics against the shipped descriptions: 0.10 to 0.30 all selected the right
# source every time, but the number searched fell from 3.0 to 1.5 across that
# range, and 0.35 began missing (two topics cleared nothing and fell back).
# 0.25 searches 1.8 sources of 8 with no misses. Six topics is a small set, so
# re-measure when the source list grows; scripts in the session that recorded
# this are reproducible from README.md.
SOURCE_FLOOR = 0.25
TOP_SOURCES = 3


class ScoredSource(NamedTuple):
    source: TrustedSource
    score: float


class SourceIndex:
    """Trusted sources with their descriptions embedded, ready to be ranked.

    Built once and reused: the descriptions do not change between queries, and
    re-embedding them per acquisition would cost more than the search it saves.
    """

    def __init__(
        self,
        embed_one: Callable[[str], ndarray],
        sources: Sequence[TrustedSource] = DEFAULT_SOURCES,
    ) -> None:
        self.sources = tuple(sources)
        self.vectors = (
            np.asarray([embed_one(s.description) for s in self.sources], dtype=float32)
            if self.sources
            else np.empty((0, 0), dtype=float32)
        )

    def rank(self, target: AcquisitionTarget) -> List[ScoredSource]:
        """Every source scored against the target's subject, best first."""
        if not self.sources:
            return []
        # Scored against the topic and the subtopics, not the query: the
        # question's wording is about the answer, while a source is chosen for
        # the subject it covers.
        subject_vectors = [target.topic_vector, *target.subtopic_vectors]
        best = np.full(len(self.sources), -1.0, dtype=float32)
        for vector in subject_vectors:
            best = np.maximum(best, cosine_scores(np.asarray(vector, dtype=float32),
                                                  self.vectors))
        scored = [
            ScoredSource(source, float(best[i]))
            for i, source in enumerate(self.sources)
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored

    def select(
        self,
        target: AcquisitionTarget,
        kind: str = DOCS,
        top_k: int = TOP_SOURCES,
        floor: float = SOURCE_FLOOR,
    ) -> List[TrustedSource]:
        """The few sources worth asking about this target."""
        ranked = [s for s in self.rank(target) if s.source.kind == kind]
        chosen = [s for s in ranked if s.score >= floor][:top_k]
        if not chosen and ranked:
            # Nothing cleared the floor, which usually means the topic is
            # phrased unlike any description. Falling back to the single best
            # beats searching everything, and beats searching nothing.
            chosen = ranked[:1]
            logger.debug(
                "No source cleared %.2f for topic %r; falling back to %s",
                floor, target.topic, chosen[0].source.name,
            )
        selected = [s.source for s in chosen]
        logger.info(
            "Selected %d of %d %s source(s) for topic %r: %s",
            len(selected), len(ranked), kind, target.topic,
            ", ".join(s.source.name for s in chosen),
        )
        return selected
