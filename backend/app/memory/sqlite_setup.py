"""Connection setup for the conversation database.

Shared by the conversation databases and the project registry; see
memory/README.md.
"""

import sqlite3
from pathlib import Path

from config import get_logger

logger = get_logger(__name__)

SYNCHRONOUS = "NORMAL"


def connect(db_path: str | Path, **kwargs) -> sqlite3.Connection:
    """Open the conversation database with the pragmas every connection needs."""
    conn = sqlite3.connect(db_path, **kwargs)
    conn.execute(f"PRAGMA synchronous = {SYNCHRONOUS};")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def enable_wal(conn: sqlite3.Connection, db_path: str | Path | None = None) -> str:
    """Put the database into WAL mode; return the journal mode now in force."""
    try:
        mode = conn.execute("PRAGMA journal_mode = WAL;").fetchone()[0]
    except sqlite3.OperationalError as error:
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        logger.warning(
            "Could not switch %s to WAL (%s); running in %s mode",
            db_path or "the conversation database",
            error,
            mode,
        )
        return mode
    if mode.lower() != "wal":
        # Not an exception, so the fallback is otherwise silent.
        logger.warning(
            "Requested WAL for %s but it reports %s mode",
            db_path or "the conversation database",
            mode,
        )
    return mode
