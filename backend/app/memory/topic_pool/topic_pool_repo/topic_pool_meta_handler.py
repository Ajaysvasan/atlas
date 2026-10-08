import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import List, NamedTuple

from config import get_logger
from memory.memory_database import MemoryDatabase, Schema
from memory.memory_pool_exceptions import TopicAlreadyExists, TopicNotFound
from storage.timestamps import as_timestamp

logger = get_logger(__name__)


class Topic(NamedTuple):
    """One row of topics_mapping_table, as the listing returns it."""

    topic_id: str
    topic_name: str
    created_at: str


def _create_tables(cursor: sqlite3.Cursor) -> None:
    cursor.execute("""
        create table if not exists topics_mapping_table(
            topic_id text primary key not null,
            topic_name text not null,
            created_at text not null,
            is_active text not null default 't' check (is_active in ('t', 'f'))
        )
    """)
    cursor.execute(
        "create unique index if not exists idx_active_topic_name "
        "on topics_mapping_table(topic_name) where is_active = 't'"
    )


SCHEMA = Schema("topics", _create_tables)


class TopicPoolMetaHandler:
    def __init__(self, database: MemoryDatabase | str | Path | None = None):
        self.database = MemoryDatabase.of(database)
        self.database.ensure(SCHEMA)

    def __is_topic_exists(self, topic_name: str):
        with self.database.reading() as curr:
            curr.execute(
                """
                select 1 from topics_mapping_table
                where topic_name = ? and is_active = 't' limit 1;
            """,
                (topic_name,),
            )
            row = curr.fetchone()
            return row[0] if row is not None else None

    def __get_topic_id(self, topic_name: str):
        with self.database.reading() as curr:
            curr.execute(
                """
                select topic_id from topics_mapping_table
                where topic_name = ? and is_active = 't' limit 1;
            """,
                (topic_name,),
            )
            row = curr.fetchone()
            return row[0] if row is not None else None

    def __create_new_topic(
        self, topic_name: str, topic_id: str, created_at: str
    ) -> None:
        try:
            with self.database.writing() as curr:
                curr.execute(
                    """
                insert into topics_mapping_table(topic_id , topic_name , created_at , is_active)
                values (? , ? , ? , ?);
                """,
                    (topic_id, topic_name, created_at, 't'),
                )
        except sqlite3.IntegrityError as error:
            # The partial unique index decides this, not a prior read, so two
            # writers racing the same name cannot both win.
            raise TopicAlreadyExists(topic_name) from error
        logger.debug("Stored topic %s as %s", topic_name, topic_id)

    def __soft_delete_by_name(self, topic_name: str) -> str:
        with self.database.writing() as curr:
            row = curr.execute(
                """
                update topics_mapping_table set is_active = 'f'
                where topic_name = ? and is_active = 't'
                returning topic_id;
                """,
                (topic_name,),
            ).fetchone()
            if row is None:
                raise TopicNotFound(topic_name)
        logger.debug("Marked topic %r inactive (%s)", topic_name, row[0])
        return row[0]

    def __get_all_topics(self) -> List[Topic]:
        with self.database.reading() as curr:
            curr.execute(
                """
                select topic_id, topic_name, created_at from topics_mapping_table
                where is_active = 't'
                order by datetime(created_at), created_at, rowid;
                """
            )
            return [Topic(*row) for row in curr.fetchall()]

    def __soft_delete(self , topic_id):
        with self.database.writing() as curr:
            curr.execute(
                """
                update topics_mapping_table set is_active = ? where topic_id = ?;
                """,
                ('f', topic_id),
            )
            # An UPDATE matching nothing is not an error to SQLite.
            if curr.rowcount == 0:
                raise ValueError(f"No topic with id {topic_id!r}")
        logger.debug("Marked topic %s inactive", topic_id)

    def has_topic_id(self, topic_id: str) -> bool:
        """Whether any row, active or soft-deleted, carries this id."""
        with self.database.reading() as curr:
            return curr.execute(
                "select 1 from topics_mapping_table where topic_id = ? limit 1",
                (topic_id,),
            ).fetchone() is not None

    def is_topic_exists(self, topic: str):
        return True if self.__is_topic_exists(topic) is not None else False

    def get_topic_id(self, topic):
        return self.__get_topic_id(topic)

    def create_new_topic(
        self, topic_name: str, topic_id: str, created_at: date | datetime | str
    ):
        created = as_timestamp(created_at)
        self.__create_new_topic(topic_name, topic_id, created)
    def soft_delete(self , topic_id):
        self.__soft_delete(topic_id)

    def soft_delete_by_name(self, topic_name: str) -> str:
        """Deactivate the active topic with this name; returns its id."""
        return self.__soft_delete_by_name(topic_name)

    def get_all_topics(self) -> List[Topic]:
        """Every active topic, oldest first."""
        return self.__get_all_topics()

    def close(self) -> None:
        """Releases nothing: the database is shared and outlives its owners."""
