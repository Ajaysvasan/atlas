"""One timestamp format for every row this project writes.

It sits beside sqlite_setup for the same reason: the memory layer and
knowledge_sufficiency both stamp rows, and memory already imports that
subsystem. It existed first to stop utc_now being defined four times over
(bug 4.58), and a copy in another package would be the fifth.
"""

from datetime import date, datetime, timezone


def utc_now() -> str:
    """ISO-8601 UTC, sortable as text, microseconds kept so two writes do not tie."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def as_timestamp(value: str | date | datetime | None) -> str:
    """Normalise a caller-supplied stamp to the stored TEXT form."""
    if value is None:
        return utc_now()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)
