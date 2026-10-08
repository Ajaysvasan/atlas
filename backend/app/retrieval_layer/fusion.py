"""Combining the dense and lexical result lists."""

from collections import defaultdict
from typing import Dict, List, Sequence

from config import get_logger
from retrieval_layer.models import FUSED, ScoredId
from retrieval_layer.settings import RRF_K

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    lists: Sequence[Sequence[ScoredId]], rrf_k: int = RRF_K
) -> List[ScoredId]:
    """Fuse ranked lists by position, best first.

    By rank, not by score: a bm25 value and a cosine distance are not on the
    same scale, and normalising them means deciding what a distance of 0.4 is
    worth as a bm25 — a judgement neither number supports. Positions are
    directly comparable, which is the whole argument for this method.

    `rrf_k` damps the head of each list. With it small, the top hit of one
    searcher outweighs the other's first several; at 60 the lists contribute
    more evenly, which is the point of consulting two of them.
    """
    totals: Dict[int, float] = defaultdict(float)
    found_by: Dict[int, set] = defaultdict(set)
    for ranked in lists:
        for item in ranked:
            totals[item.vector_id] += 1.0 / (rrf_k + item.rank)
            found_by[item.vector_id].add(item.source)

    ordered = sorted(totals.items(), key=lambda pair: pair[1], reverse=True)
    fused = [
        ScoredId(vector_id=vector_id, score=score, source=FUSED, rank=position)
        for position, (vector_id, score) in enumerate(ordered, start=1)
    ]
    if fused:
        both = sum(1 for v in found_by.values() if len(v) > 1)
        logger.debug(
            "Fused %d candidate(s); %d found by more than one searcher",
            len(fused), both,
        )
    return fused


def sources_of(
    lists: Sequence[Sequence[ScoredId]]
) -> Dict[int, set]:
    """Which searchers found each candidate, for explaining a result."""
    found: Dict[int, set] = defaultdict(set)
    for ranked in lists:
        for item in ranked:
            found[item.vector_id].add(item.source)
    return dict(found)
