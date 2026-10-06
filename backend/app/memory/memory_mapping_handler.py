"""Which topic, project and project snapshot a conversation belongs to."""

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, NamedTuple

from config import Config, get_logger
from memory.identifiers import require_identifier
from storage.sqlite_setup import connect, enable_wal
from storage.timestamps import utc_now

logger = get_logger(__name__)


class MemoryMapping(NamedTuple):
    topic_id: str | None
    project_id: str | None
    latest_project_snapshot_id: str | None


class MemoryMappingHandler:
    """The mapping table, for every conversation — not one conversation.

    Takes no `conversation_id`: every method names the conversation it acts on,
    so one handler serves them all. The original signature also took a `query`,
    which nothing read.
    """

    __mapping_db = Config.DATA_DIR / Path("memory_mapping/memory_mapping.sql")

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.__db_path = Path(db_path) if db_path is not None else self.__mapping_db
        self.__db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.__connection: sqlite3.Connection | None = connect(
            self.__db_path, check_same_thread=False
        )
        self.journal_mode = enable_wal(self.__connection, self.__db_path)
        self.__db_init()
        logger.debug(
            "Memory mapping store ready at %s (journal=%s)",
            self.__db_path,
            self.journal_mode,
        )

    @property
    def db_path(self) -> Path:
        return self.__db_path

    @contextmanager
    def _reading(self) -> Iterator[sqlite3.Cursor]:
        """A cursor held under the lock for as long as the caller needs it."""
        with self._lock:
            assert self.__connection is not None
            yield self.__connection.cursor()

    @contextmanager
    def _writing(self) -> Iterator[sqlite3.Cursor]:
        """A cursor under the lock, committed on success, rolled back on failure."""
        with self._lock:
            assert self.__connection is not None
            cursor = self.__connection.cursor()
            try:
                yield cursor
                self.__connection.commit()
            except BaseException:
                self.__connection.rollback()
                logger.debug("Rolled back a write on %s", self.__db_path)
                raise

    def __db_init(self) -> None:
        with self._writing() as curr:
            # latest_project_snapshot_id is overwritten in place rather than
            # appended to: it is a cache of ProjectSnapshotRepository.latest(),
            # which owns the ordered chain in project_snapshot_mapping. Read it
            # here to resume a conversation without opening the project
            # registry; do not treat it as the history.
            curr.execute("""
                create table if not exists memory_mapping_table (
                    conversation_id text not null,
                    user_id text not null,
                    topic_id text,
                    project_id text,
                    latest_project_snapshot_id text,
                    created_at text,
                    primary key (conversation_id, user_id)
                );
            """)
            # The snapshot pointer is rewritten by (project_id, user_id), which
            # no part of the primary key covers.
            curr.execute(
                "create index if not exists idx_memory_mapping_project "
                "on memory_mapping_table(project_id, user_id);"
            )

    def __populate_conversation_id(
        self, cursor: sqlite3.Cursor, conversation_id: str, user_id: str
    ) -> None:
        cursor.execute(
            "insert into memory_mapping_table (conversation_id, user_id, created_at) "
            "values (?, ?, ?);",
            (conversation_id, user_id, utc_now()),
        )

    def __insert_into_mapping_table(
        self,
        curr: sqlite3.Cursor,
        conversation_id: str,
        user_id: str,
        topic_id: str,
        project_id: str,
        new_latest_project_snapshot_id: str,
        created_at: str,
    ) -> int:
        curr.execute(
            "update memory_mapping_table set topic_id = ?, project_id = ?, "
            "latest_project_snapshot_id = ?, created_at = ? "
            "where conversation_id = ? and user_id = ?;",
            (
                topic_id,
                project_id,
                new_latest_project_snapshot_id,
                created_at,
                conversation_id,
                user_id,
            ),
        )
        return curr.rowcount

    def __search(self, conversation_id: str, user_id: str) -> MemoryMapping | None:
        with self._reading() as cursor:
            cursor.execute(
                "select topic_id, project_id, latest_project_snapshot_id "
                "from memory_mapping_table "
                "where conversation_id = ? and user_id = ?;",
                (conversation_id, user_id),
            )
            row = cursor.fetchone()
        return MemoryMapping(*row) if row is not None else None

    def __update_latest_project_snapshot_id(
        self, user_id: str, project_id: str, new_latest_project_snapshot_id: str
    ) -> int:
        with self._writing() as cursor:
            cursor.execute(
                "update memory_mapping_table set latest_project_snapshot_id = ? "
                "where project_id = ? and user_id = ?;",
                (new_latest_project_snapshot_id, project_id, user_id),
            )
            return cursor.rowcount

    # Public APIs

    def populate_conversation_id(self, conversation_id: str, user_id: str) -> None:
        """Open a row for a conversation that has not been routed yet."""
        conversation_id = require_identifier(conversation_id, "conversation_id")
        user_id = require_identifier(user_id, "user_id")
        with self._writing() as cursor:
            self.__populate_conversation_id(cursor, conversation_id, user_id)
        logger.debug(
            "Opened a mapping row for conversation %s (user %s)",
            conversation_id,
            user_id,
        )

    def insert_into_mapping_table(
        self,
        conversation_id: str,
        user_id: str,
        topic_id: str,
        project_id: str,
        new_latest_project_snapshot_id: str,
        created_at: str,
    ) -> None:
        """Record the topic and project a conversation was routed to."""
        conversation_id = require_identifier(conversation_id, "conversation_id")
        user_id = require_identifier(user_id, "user_id")
        with self._writing() as cursor:
            updated = self.__insert_into_mapping_table(
                curr=cursor,
                conversation_id=conversation_id,
                user_id=user_id,
                topic_id=topic_id,
                project_id=project_id,
                new_latest_project_snapshot_id=new_latest_project_snapshot_id,
                created_at=created_at,
            )
        if updated:
            logger.info(
                "Mapped conversation %s (user %s) to topic %s, project %s",
                conversation_id,
                user_id,
                topic_id,
                project_id,
            )
        else:
            # An UPDATE that matches nothing is not an error to SQLite, so the
            # caller would otherwise read this as success.
            logger.warning(
                "No mapping row for conversation %s (user %s); "
                "populate_conversation_id first",
                conversation_id,
                user_id,
            )

    def update_latest_project_snapshot_id(
        self, user_id: str, project_id: str, new_latest_project_snapshot_id: str
    ) -> None:
        """Point this user's conversations in a project at its newest snapshot."""
        user_id = require_identifier(user_id, "user_id")
        project_id = require_identifier(project_id, "project_id")
        updated = self.__update_latest_project_snapshot_id(
            user_id=user_id,
            project_id=project_id,
            new_latest_project_snapshot_id=new_latest_project_snapshot_id,
        )
        logger.debug(
            "Moved %d conversation(s) in project %s (user %s) to snapshot %s",
            updated,
            project_id,
            user_id,
            new_latest_project_snapshot_id,
        )

    def search(self, conversation_id: str, user_id: str) -> MemoryMapping | None:
        """Where this conversation belongs, or None if it has no row."""
        conversation_id = require_identifier(conversation_id, "conversation_id")
        user_id = require_identifier(user_id, "user_id")
        found = self.__search(conversation_id=conversation_id, user_id=user_id)
        if found is None:
            logger.debug(
                "No mapping for conversation %s (user %s)", conversation_id, user_id
            )
        return found

    def close(self) -> None:
        # getattr, not attribute access: if connect() failed in __init__ the
        # attribute never existed, and __del__ would then raise AttributeError
        # and bury the real construction error. The lock makes this wait for a
        # write in flight on another thread rather than closing the connection
        # out from under it, which segfaults rather than raising (Bug 4.47).
        conn = getattr(self, "_MemoryMappingHandler__connection", None)
        if conn is None:
            return
        lock = getattr(self, "_lock", None)
        if lock is None:
            conn.close()
        else:
            with lock:
                conn.close()
        self.__connection = None

    def __del__(self):
        self.close()
