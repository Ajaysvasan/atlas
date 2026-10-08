"""Turning a ranked list of passages into the context that gets sent on."""

from typing import List, Sequence, Tuple

from config import get_logger
from retrieval_layer.models import Passage
from retrieval_layer.settings import CHARS_PER_TOKEN, TOKEN_BUDGET

logger = get_logger(__name__)


def deduplicate(passages: Sequence[Passage]) -> Tuple[List[Passage], int]:
    """Drop repeats, keeping the best-scoring copy of each.

    Two kinds of repeat reach here. The same chunk can be found by both
    searchers, and overlapping chunks can carry identical text under different
    ids — the windowing deliberately repeats content at its boundaries, so this
    is normal rather than a fault.
    """
    seen_chunks: set = set()
    seen_text: set = set()
    kept: List[Passage] = []
    for passage in passages:
        normalised = " ".join(passage.text.split())
        if passage.chunk_id in seen_chunks or normalised in seen_text:
            continue
        seen_chunks.add(passage.chunk_id)
        seen_text.add(normalised)
        kept.append(passage)
    return kept, len(passages) - len(kept)


def fit_to_budget(
    passages: Sequence[Passage],
    token_budget: int = TOKEN_BUDGET,
    chars_per_token: int = CHARS_PER_TOKEN,
) -> Tuple[List[Passage], int]:
    """Take passages in order until the budget is spent.

    It stops rather than truncating: half a passage is a sentence cut mid-
    clause, which reads as though the source said something it did not.

    A later passage that happens to fit is **not** taken after a larger one has
    been skipped. Doing so would reorder the context by size, and the ranking
    is the one thing that has been carefully established by this point.
    """
    kept: List[Passage] = []
    spent = 0
    for passage in passages:
        cost = passage.tokens(chars_per_token)
        if spent + cost > token_budget:
            break
        kept.append(passage)
        spent += cost
    return kept, spent


def in_reading_order(passages: Sequence[Passage]) -> List[Passage]:
    """Reorder within each document so adjacent chunks read in sequence."""
    return sorted(
        passages,
        key=lambda p: (p.document_id or "", p.start_offset or 0),
    )


def assemble(
    passages: Sequence[Passage],
    token_budget: int = TOKEN_BUDGET,
    chars_per_token: int = CHARS_PER_TOKEN,
    reading_order: bool = False,
) -> Tuple[List[Passage], int, int]:
    """Deduplicate, fit to the budget, and return the context with its cost."""
    unique, duplicates = deduplicate(passages)
    fitted, spent = fit_to_budget(unique, token_budget, chars_per_token)
    if reading_order:
        fitted = in_reading_order(fitted)
    logger.debug(
        "Assembled %d passage(s), %d token(s), %d duplicate(s) dropped",
        len(fitted), spent, duplicates,
    )
    return fitted, spent, duplicates
