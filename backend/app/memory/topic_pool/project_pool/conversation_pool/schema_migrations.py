"""Schema migrations for the conversation database."""

import sqlite3
from pathlib import Path

from config import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = 2

# Every row that predates `conversation_id` belongs to this conversation. It is
# not a placeholder: before the column existed a project database held exactly
# one conversation, so collapsing its rows into a single id is what they always
# meant. A real id, so the readers' `conversation_id = ?` filters match it and
# the validation in memory/identifiers.py accepts it.
LEGACY_CONVERSATION_ID = "legacy"

_FULL_CONVERSATION = """
    create table full_conversation(
        project_id text not null,
        conversation_id text not null,
        sequence_number int not null,
        chunk_id text not null,
        role text not null,
        created_at date not null,
        primary key (conversation_id, sequence_number),
        foreign key (chunk_id) references summary_chunks (chunk_id)
    )
"""

_SUMMARY_CHUNKS = """
    create table summary_chunks(
        chunk_id text primary key,
        conversation_id text not null,
        chunk text not null,
        created_at date not null,
        chunker_type text not null
    )
"""

_CUMULATIVE = """
    create table cumulative_vector_meta_data(
        cumulative_vector_id integer primary key,
        conversation_id text not null,
        seq integer not null unique,
        cumulative_summary text not null,
        created_at date not null,
        project_id text not null,
        len_of_the_summary integer not null
    )
"""


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"pragma table_info({table})")]


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "select 1 from sqlite_master where type='table' and name=?", (table,)
    ).fetchone() is not None


def _needs(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return _table_exists(conn, table) and column not in _columns(conn, table)


def _rebuild(conn: sqlite3.Connection, table: str, create_sql: str,
             select_sql: str, columns: list[str]) -> None:
    """Recreate `table` in its current shape, carrying the old rows over."""
    staging = f"{table}__migrating"
    conn.execute(create_sql.replace(f"table {table}(", f"table {staging}("))
    conn.execute(
        f"insert into {staging} ({', '.join(columns)}) "
        f"select {select_sql} from {table}"
    )
    conn.execute(f"drop table {table}")
    conn.execute(f"alter table {staging} rename to {table}")
    logger.info("Rebuilt %s carrying %d column(s) over", table, len(columns))


def _to_version_1(conn: sqlite3.Connection) -> None:
    """Introduce `conversation_id` on the two turn tables."""
    for table, create_sql in (
        ("summary_chunks", _SUMMARY_CHUNKS),
        ("full_conversation", _FULL_CONVERSATION),
    ):
        if not _needs(conn, table, "conversation_id"):
            continue
        old = _columns(conn, table)
        selected = ", ".join(old) + f", '{LEGACY_CONVERSATION_ID}'"
        _rebuild(conn, table, create_sql, selected, old + ["conversation_id"])


def _to_version_2(conn: sqlite3.Connection) -> None:
    """Introduce `conversation_id` and the monotonic `seq` on the snapshot rows."""
    if not _needs(conn, "cumulative_vector_meta_data", "seq"):
        return
    old = _columns(conn, "cumulative_vector_meta_data")
    # seq is backfilled in the order the rows were already read in, so the
    # existing snapshot ordering carries over rather than being invented.
    selected = (
        ", ".join(old)
        + f", '{LEGACY_CONVERSATION_ID}'"
        + ", row_number() over (order by datetime(created_at), created_at)"
    )
    _rebuild(
        conn, "cumulative_vector_meta_data", _CUMULATIVE, selected,
        old + ["conversation_id", "seq"],
    )


_STEPS = {1: _to_version_1, 2: _to_version_2}


def migrate(db_path: str | Path) -> int:
    """Bring the conversation database at `db_path` up to SCHEMA_VERSION."""
    # Its own connection, because the rebuild has to turn foreign keys off and
    # a pragma is silently ignored inside a transaction — including the implicit
    # one a caller's connection may already be holding.
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        version = conn.execute("pragma user_version;").fetchone()[0]
        if version >= SCHEMA_VERSION:
            return version

        pending = [v for v in sorted(_STEPS) if v > version]

        conn.execute("pragma foreign_keys = OFF;")
        try:
            conn.execute("begin immediate;")
            for target in pending:
                _STEPS[target](conn)
            conn.execute(f"pragma user_version = {SCHEMA_VERSION};")
            conn.execute("commit;")
        except Exception:
            conn.execute("rollback;")
            raise
        finally:
            conn.execute("pragma foreign_keys = ON;")

        violations = conn.execute("pragma foreign_key_check;").fetchall()
        if violations:
            # Reported rather than raised: the rows are still there and readable,
            # and failing the open would lock the user out of their own history.
            logger.error(
                "Foreign key violations after migrating %s: %d row(s)",
                db_path, len(violations),
            )
        logger.info("Migrated %s from schema %d to %d", db_path, version,
                    SCHEMA_VERSION)
        return SCHEMA_VERSION
    finally:
        conn.close()
