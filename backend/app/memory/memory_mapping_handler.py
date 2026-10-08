"""Which topic, project and project snapshot a conversation belongs to."""

import sqlite3
from pathlib import Path
from typing import NamedTuple

from config import get_logger
from memory.identifiers import require_identifier
from memory.memory_database import MemoryDatabase, Schema
from memory.memory_pool_exceptions import ProjectInAnotherTopic, ProjectNotFound
from memory.topic_pool.project_pool.project_data_repo.project_meta_data import (
    SCHEMA as PROJECT_SCHEMA,
    project_topic,
)
from memory.topic_pool.project_pool.project_data_repo.project_snapshot_repo import (
    SCHEMA as PROJECT_SNAPSHOT_SCHEMA,
    ProjectSnapshotRepository,
)
from storage.timestamps import utc_now

logger = get_logger(__name__)


class MemoryMapping(NamedTuple):
    topic_id: str | None
    project_id: str | None
    latest_project_snapshot_id: int | None


def _create_tables(cursor: sqlite3.Cursor) -> None:
    # A row is opened before routing and filled in by it, so topic and project
    # are null together or set together. A pair with a null in it is not
    # checked as a foreign key, which is what lets the unrouted row exist; the
    # CHECK is what stops a half-routed one.
    cursor.execute("""
        create table if not exists memory_mapping_table(
            conversation_id text not null,
            user_id text not null,
            topic_id text,
            project_id text,
            created_at text,
            primary key (conversation_id, user_id),
            foreign key (project_id, topic_id)
                references project_table(project_id, topic_id) on update cascade,
            check ((topic_id is null) = (project_id is null))
        )
    """)
    cursor.execute(
        "create index if not exists idx_memory_mapping_project "
        "on memory_mapping_table(project_id, topic_id)"
    )


SCHEMA = Schema(
    "memory_mapping", _create_tables, requires=(PROJECT_SCHEMA, PROJECT_SNAPSHOT_SCHEMA)
)


class MemoryMappingHandler:
    """The mapping table, for every conversation — not one conversation.

    Takes no `conversation_id`: every method names the conversation it acts on,
    so one handler serves them all.
    """

    def __init__(self, database: MemoryDatabase | str | Path | None = None) -> None:
        self.database = MemoryDatabase.of(database)
        self.database.ensure(SCHEMA)

    def __refused(self, error: sqlite3.IntegrityError, topic_id: str, project_id: str) -> Exception:
        if "FOREIGN KEY" not in str(error):
            return error
        registered_under = project_topic(project_id, self.database)
        if registered_under is None:
            return ProjectNotFound(project_id)
        if registered_under != topic_id:
            return ProjectInAnotherTopic(project_id, topic_id, registered_under)
        return error

    def populate_conversation_id(self, conversation_id: str, user_id: str) -> None:
        """Open a row for a conversation that has not been routed yet."""
        conversation_id = require_identifier(conversation_id, "conversation_id")
        user_id = require_identifier(user_id, "user_id")
        with self.database.writing() as cursor:
            cursor.execute(
                "insert into memory_mapping_table (conversation_id, user_id, created_at) "
                "values (?, ?, ?);",
                (conversation_id, user_id, utc_now()),
            )
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
        created_at: str,
    ) -> None:
        """Record the topic and project a conversation was routed to."""
        conversation_id = require_identifier(conversation_id, "conversation_id")
        user_id = require_identifier(user_id, "user_id")
        topic_id = require_identifier(topic_id, "topic_id")
        project_id = require_identifier(project_id, "project_id")
        try:
            with self.database.writing() as cursor:
                cursor.execute(
                    "update memory_mapping_table set topic_id = ?, project_id = ?, "
                    "created_at = ? where conversation_id = ? and user_id = ?;",
                    (topic_id, project_id, created_at, conversation_id, user_id),
                )
                updated = cursor.rowcount
        except sqlite3.IntegrityError as error:
            raise self.__refused(error, topic_id, project_id) from error
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

    def search(self, conversation_id: str, user_id: str) -> MemoryMapping | None:
        """Where this conversation belongs, or None if it has no row.

        The snapshot is asked of the project's snapshot chain on every read
        rather than cached here: a cache had to be rewritten by hand each time
        a project moved on, and was stale whenever that was missed.
        """
        conversation_id = require_identifier(conversation_id, "conversation_id")
        user_id = require_identifier(user_id, "user_id")
        with self.database.reading() as cursor:
            row = cursor.execute(
                "select topic_id, project_id from memory_mapping_table "
                "where conversation_id = ? and user_id = ?;",
                (conversation_id, user_id),
            ).fetchone()
        if row is None:
            logger.debug(
                "No mapping for conversation %s (user %s)", conversation_id, user_id
            )
            return None
        topic_id, project_id = row
        latest = (
            ProjectSnapshotRepository(project_id, self.database).latest()
            if project_id is not None
            else None
        )
        return MemoryMapping(
            topic_id, project_id, latest.project_snapshot_id if latest else None
        )

    def close(self) -> None:
        """Releases nothing: the database is shared and outlives its owners."""
