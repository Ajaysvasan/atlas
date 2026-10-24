"""Tidying a query before anything reads it, and refusing one too large to decompose."""

import re
import unicodedata

from query_decomposition.query_decomposition_exceptions import EmptyQuery, QueryTooLarge

CHARS_PER_TOKEN = 4

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_BLANK_RUN = re.compile(r"\n{3,}")


def clean(query) -> str:
    """The query in NFC, with line endings, trailing spaces and blank runs tidied."""
    if not isinstance(query, str):
        raise EmptyQuery(query)
    text = unicodedata.normalize("NFC", query)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    # Spaces inside a line are kept: a query can carry indented code.
    text = _TRAILING_SPACE.sub("", text)
    text = _BLANK_RUN.sub("\n\n", text).strip()
    if not text:
        raise EmptyQuery(query)
    return text


def estimate_tokens(text: str) -> int:
    return -(-len(text) // CHARS_PER_TOKEN)


def require_within(text: str, limit: int) -> int:
    """The text's estimated tokens, or QueryTooLarge past `limit`."""
    tokens = estimate_tokens(text)
    if tokens > limit:
        raise QueryTooLarge(tokens, limit)
    return tokens
