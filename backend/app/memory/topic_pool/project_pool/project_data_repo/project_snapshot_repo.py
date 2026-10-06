"""The project snapshot registry: metadata and the append-only mapping."""

import hashlib
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, NamedTuple

from config import Config, get_logger
from memory.identifiers import require_identifier
from storage.sqlite_setup import connect
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


class ProjectSnapshotRepository:
    """Snapshot metadata for every project, in the shared project registry."""

    __project_db = Config.DATA_DIR / Path("project_db/project.sql")

    def __init__(self, project_id: str, db_path: str | Path | None = None) -> None:
        self.project_id = require_identifier(project_id, "project_id")
        self._lock = threading.RLock()
        self.db_path = Path(db_path) if db_path is not None else self.__project_db
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.__connection = connect(self.db_path, check_same_thread=False)
        self.__init_db()

    @contextmanager
    def _reading(self) -> Iterator[sqlite3.Cursor]:
        """A cursor held under the lock for as long as the caller needs it."""
        with self._lock:
            yield self.__connection.cursor()

    @contextmanager
    def _writing(self) -> Iterator[sqlite3.Cursor]:
        """A cursor under the lock, committed on success, rolled back on failure."""
        with self._lock:
            cursor = self.__connection.cursor()
            try:
                yield cursor
                self.__connection.commit()
            except BaseException:
                self.__connection.rollback()
                raise

    def __init_db(self) -> None:
        with self._writing() as curr:
            curr.execute("""
                create table if not exists project_snapshot(
                    project_snapshot_id integer primary key,
                    project_id text not null,
                    summary text not null,
                    len_of_the_summary integer not null,
                    last_seq_included integer not null,
                    created_at date not null
                )
            """)
            curr.execute("""
                create table if not exists project_snapshot_mapping(
                    project_id text not null,
                    project_snapshot_id integer not null,
                    created_at date not null,
                    primary key (project_id, project_snapshot_id),
                    foreign key (project_snapshot_id)
                        references project_snapshot(project_snapshot_id)
                )
            """)
            # Every read here filters on project_id, which neither primary key
            # covers on its own, and the latest-snapshot read runs on every
            # conversation snapshot.
            curr.execute(
                "create index if not exists idx_project_snapshot_project "
                "on project_snapshot(project_id);"
            )
            curr.execute(
                "create index if not exists idx_project_snapshot_mapping_project "
                "on project_snapshot_mapping(project_id);"
            )

    def add_snapshot(
        self, summary: str, last_seq_included: int, created_at: str | None = None
    ) -> int:
        """Store one project snapshot and point the mapping at it."""
        created_at = created_at or utc_now()
        snapshot_id = project_snapshot_id(self.project_id, created_at, summary)
        with self._writing() as cursor:
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
        logger.info(
            "Stored project snapshot %s for %s covering snapshots up to seq %d",
            snapshot_id, self.project_id, last_seq_included,
        )
        return snapshot_id

    def latest(self) -> ProjectSnapshotRow | None:
        """The most recent snapshot for this project, or None if it has none."""
        with self._reading() as cursor:
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
        with self._reading() as cursor:
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
        with self._reading() as cursor:
            cursor.execute(
                "select project_snapshot_id, project_id, summary, "
                "len_of_the_summary, last_seq_included, created_at "
                "from project_snapshot where project_snapshot_id = ?",
                (int(snapshot_id),),
            )
            row = cursor.fetchone()
        return ProjectSnapshotRow(*row) if row is not None else None

    def close(self) -> None:
        # getattr, not attribute access: if connect() failed in __init__ the
        # attribute never existed, and __del__ would bury the real error. The
        # lock makes this wait for a write in flight on another thread rather
        # than closing the connection out from under it.
        conn = getattr(self, "_ProjectSnapshotRepository__connection", None)
        if conn is None:
            return
        lock = getattr(self, "_lock", None)
        if lock is None:
            conn.close()
            return
        with lock:
            conn.close()

    def __del__(self):
        self.close()
