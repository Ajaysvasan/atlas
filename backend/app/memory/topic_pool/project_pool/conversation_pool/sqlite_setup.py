"""Connection setup for the conversation database.

Two classes open `<project_id>_conversation.db`: ConversationVectorMetaDataRepository
holds one long-lived connection for snapshot metadata, and FullConversationRepository
opens a short-lived one per call for conversation turns. Both go through `connect()`
so the pragmas cannot drift apart between them.

The two kinds of setting behave differently, which is why they are separate calls:
`journal_mode` is written into the database file and survives every later open, so
`enable_wal()` only has to succeed once; `synchronous` and `foreign_keys` are
properties of a connection and reset to their defaults on every open, so `connect()`
has to set them every time.
"""

import sqlite3
from pathlib import Path

# NORMAL rather than the FULL default, which fsyncs the write-ahead log on every
# single commit. In WAL mode NORMAL is durable against a process crash — the log
# is still on disk and intact — and gives that up only for an OS crash or power
# loss, where the last few committed transactions can be rolled back. The
# database is never corrupted either way. On a long-lived connection this is
# worth roughly two orders of magnitude in commit throughput.
SYNCHRONOUS = "NORMAL"


def connect(db_path: str | Path, **kwargs) -> sqlite3.Connection:
    """Open the conversation database with the pragmas every connection needs."""
    conn = sqlite3.connect(db_path, **kwargs)
    conn.execute(f"PRAGMA synchronous = {SYNCHRONOUS};")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def enable_wal(conn: sqlite3.Connection) -> str:
    """Put the database into WAL mode; return the journal mode now in force.

    Called once, at initialisation. It cannot convert while another connection
    holds a write transaction, and losing that race is not a failure: the
    database stays in its previous journal mode — slower under concurrency,
    equally correct — and the next open tries again. The mode is returned rather
    than swallowed so a caller can see which one it got.
    """
    try:
        return conn.execute("PRAGMA journal_mode = WAL;").fetchone()[0]
    except sqlite3.OperationalError:
        return conn.execute("PRAGMA journal_mode;").fetchone()[0]
