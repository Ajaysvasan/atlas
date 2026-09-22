"""One timestamp format for every row the memory layer writes.

See README.md in this directory.
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
