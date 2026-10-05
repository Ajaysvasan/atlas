"""The bands a score falls into, and where their edges are."""

from knowledge_sufficiency.verdict import Sufficiency

# Placeholders until calibrated against real conversation vectors; see
# scripts/calibrate_ksv.py and knowledge_sufficiency/README.md. PARTIAL_FLOOR
# borrows ProjectManager's measured SIMILARITY_FLOOR, which answers a similar
# question over the same embedding model. SUFFICIENT_FLOOR is a guess.
SUFFICIENT_FLOOR = 0.60
PARTIAL_FLOOR = 0.35


def classify(
    best: float,
    sufficient_floor: float = SUFFICIENT_FLOOR,
    partial_floor: float = PARTIAL_FLOOR,
) -> Sufficiency:
    """Which band the best score falls into."""
    if sufficient_floor < partial_floor:
        raise ValueError(
            f"sufficient_floor ({sufficient_floor}) is below partial_floor "
            f"({partial_floor}); nothing could ever be PARTIAL"
        )
    if best >= sufficient_floor:
        return Sufficiency.FULL
    if best >= partial_floor:
        return Sufficiency.PARTIAL
    return Sufficiency.NONE
