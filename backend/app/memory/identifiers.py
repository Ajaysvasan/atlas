"""Validation for the ids that scope rows across the memory layer."""

from memory.memory_pool_exceptions import InvalidIdentifier


def require_identifier(value, name: str) -> str:
    """Return `value` stripped, or raise if it cannot scope anything."""
    if not isinstance(value, str) or not value.strip():
        raise InvalidIdentifier(name, value)
    return value.strip()
