"""The project snapshot registry: metadata and the append-only mapping."""

import hashlib
import sqlite3
from pathlib import Path
from typing import List, NamedTuple

from config import Config, get_logger
from memory.identifiers import require_identifier
from memory.memory_database import MemoryDatabase, Schema
from memory.memory_pool_exceptions import ProjectNotFound
from memory.topic_pool.project_pool.project_data_repo.project_meta_data import (
    SCHEMA as PROJECT_SCHEMA,
)
from storage.timestamps import utc_now

logger = get_logger(__name__)


class ProjectSnapshotRow(NamedTuple):
    project_snapshot_id: int
    project_id: str
    summary: str
    len_of_the_summary: int
    last_seq_included: int
    created_at: str


def project_snapshot_id(project_id: str, created_at: str, summary: str) -> int:
    """The id this snapshot is stored under, in SQLite and in pgvector alike.

    One id for both stores, because it is also the `vector_id` the embedding is
    written under — so nothing has to be looked up to go from a row to its
    vector. Bound to the timestamp as well as the text, since a project
    summarised twice to the same words is still two snapshots.
    """
    payload = f"{project_id}\x00{created_at}\x00{summary}".encode("utf-8")
    packed = int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="little")
    return packed & Config.VECTOR_ID_MASK


def _create_tables(cursor: sqlite3.Cursor) -> None:
    cursor.execute("""
        create table if not exists project_snapshot(
            project_snapshot_id integer primary key,
            project_id text not null references project_table(project_id),
            summary text not null,
            len_of_the_summary integer not null,
            last_seq_included integer not null,
            created_at text not null,
            unique (project_snapshot_id, project_id)
        )
    """)
    cursor.execute("""
        create table if not exists project_snapshot_mapping(
            project_id text not null,
            project_snapshot_id integer not null,
            created_at text not null,
            primary key (project_id, project_snapshot_id),
            foreign key (project_snapshot_id, project_id)
                references project_snapshot(project_snapshot_id, project_id)
        )
    """)
    # project_snapshot is keyed by its id alone, so its reads by project need
    # this. The mapping needs none: its primary key already leads with project_id.
    cursor.execute(
        "create index if not exists idx_project_snapshot_project "
        "on project_snapshot(project_id)"
    )


SCHEMA = Schema("project_snapshots", _create_tables, requires=(PROJECT_SCHEMA,))


class ProjectSnapshotRepository:
    """Snapshot metadata for every project, in the memory database."""

    def __init__(
        self, project_id: str, database: MemoryDatabase | str | Path | None = None
    ) -> None:
        self.project_id = require_identifier(project_id, "project_id")
        self.database = MemoryDatabase.of(database)
        self.database.ensure(SCHEMA)

    def add_snapshot(
        self, summary: str, last_seq_included: int, created_at: str | None = None
    ) -> int:
        """Store one project snapshot and point the mapping at it."""
        created_at = created_at or utc_now()
        snapshot_id = project_snapshot_id(self.project_id, created_at, summary)
        try:
            with self.database.writing() as cursor:
                cursor.execute(
                    "insert or ignore into project_snapshot (project_snapshot_id, "
                    "project_id, summary, len_of_the_summary, last_seq_included, "
                    "created_at) values (?, ?, ?, ?, ?, ?)",
                    (snapshot_id, self.project_id, summary, len(summary),
                     int(last_seq_included), created_at),
                )
                # Append-only: the mapping is the ordered chain of every snapshot
                # this project has had, not a pointer at the current one.
                cursor.execute(
                    "insert or ignore into project_snapshot_mapping (project_id, "
                    "project_snapshot_id, created_at) values (?, ?, ?)",
                    (self.project_id, snapshot_id, created_at),
                )
        except sqlite3.IntegrityError as error:
            # OR IGNORE covers unique keys, never foreign keys: the only refusal
            # left is a project that was never registered.
            if "FOREIGN KEY" in str(error):
                raise ProjectNotFound(self.project_id) from error
            raise
        logger.info(
            "Stored project snapshot %s for %s covering snapshots up to seq %d",
            snapshot_id, self.project_id, last_seq_included,
        )
        return snapshot_id

    def latest(self) -> ProjectSnapshotRow | None:
        """The most recent snapshot for this project, or None if it has none."""
        with self.database.reading() as cursor:
            cursor.execute(
                """
                select s.project_snapshot_id, s.project_id, s.summary,
                       s.len_of_the_summary, s.last_seq_included, s.created_at
                from project_snapshot_mapping as m
                join project_snapshot as s
                  on s.project_snapshot_id = m.project_snapshot_id
                where m.project_id = ?
                order by s.last_seq_included desc, m.created_at desc
                limit 1
                """,
                (self.project_id,),
            )
            row = cursor.fetchone()
        return ProjectSnapshotRow(*row) if row is not None else None

    def last_seq_included(self) -> int:
        """The watermark: the conversation snapshot seq this project has folded in."""
        latest = self.latest()
        return latest.last_seq_included if latest is not None else 0

    def history(self) -> List[ProjectSnapshotRow]:
        """Every snapshot this project has had, oldest first."""
        with self.database.reading() as cursor:
            cursor.execute(
                """
                select s.project_snapshot_id, s.project_id, s.summary,
                       s.len_of_the_summary, s.last_seq_included, s.created_at
                from project_snapshot_mapping as m
                join project_snapshot as s
                  on s.project_snapshot_id = m.project_snapshot_id
                where m.project_id = ?
                order by s.last_seq_included, m.created_at
                """,
                (self.project_id,),
            )
            return [ProjectSnapshotRow(*row) for row in cursor.fetchall()]

    def get_snapshot(self, snapshot_id: int) -> ProjectSnapshotRow | None:
        with self.database.reading() as cursor:
            cursor.execute(
                "select project_snapshot_id, project_id, summary, "
                "len_of_the_summary, last_seq_included, created_at "
                "from project_snapshot where project_snapshot_id = ?",
                (int(snapshot_id),),
            )
            row = cursor.fetchone()
        return ProjectSnapshotRow(*row) if row is not None else None

    def close(self) -> None:
        """Releases nothing: the database is shared and outlives its owners."""
