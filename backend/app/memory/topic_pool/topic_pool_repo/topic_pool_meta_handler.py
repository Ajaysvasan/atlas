import sqlite3
import threading
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator

from config import Config, get_logger
from memory.sqlite_setup import connect, enable_wal

logger = get_logger(__name__)


def utc_now() -> str:
    """Canonical timestamp for every row this repository writes."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def as_timestamp(value: str | date | datetime | None) -> str:
    """Normalise a caller-supplied stamp to the stored TEXT form."""
    if value is None:
        return utc_now()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


class TopicPoolMetaHandler:
    def __init__(self, topic_pool_path: str | None | Path):
        self._lock = threading.RLock()
        self.__connection: sqlite3.Connection | None = None
        self.topic_db_path = (
            Path(topic_pool_path)
            if topic_pool_path is not None
            else Config.DATA_DIR / Path("topic_db/topic.sql")
        )
        self.topic_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.__connection = connect(self.topic_db_path, check_same_thread=False)
        self.journal_mode = enable_wal(self.__connection, self.topic_db_path)
        self.__db_init()

    @contextmanager
    def __writing(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            assert self.__connection is not None
            cursor = self.__connection.cursor()
            try:
                yield cursor
                self.__connection.commit()
            except BaseException:
                self.__connection.rollback()
                logger.debug("Rolled back a write on %s", self.topic_db_path)
                raise

    @contextmanager
    def __reading(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            assert self.__connection is not None
            yield self.__connection.cursor()

    def __db_init(self):
        with self.__writing() as curr:
            curr.execute("""
                create table if not exists topics_mapping_table(
                    topic_id text primary key not null,
                    topic_name text not null,
                    created_at DATE NOT NULL,
                    is_active CHAR(2)
                    );
                    """)
        logger.debug(
            "Topic registry ready at %s (journal=%s)",
            self.topic_db_path,
            self.journal_mode,
        )

    def __is_topic_exists(self, topic_name: str):
        with self.__reading() as curr:
            curr.execute(
                """
                select 1 from topics_mapping_table where topic_name = ? and is_active = ? limit 1;
            """,
                (topic_name, 't' ,),
            )
            row = curr.fetchone()
            return row[0] if row is not None else None

    def __get_topic_id(self, topic_name: str):
        with self.__reading() as curr:
            curr.execute(
                """
                select topic_id from topics_mapping_table where topic_name = ? and is_active = ? limit 1;
            """,
                (topic_name, 't',),
            )
            row = curr.fetchone()
            return row[0] if row is not None else None

    def __create_new_topic(
        self, topic_name: str, topic_id: str, created_at: str
    ) -> None:
        with self.__writing() as curr:
            curr.execute(
                """
            insert into topics_mapping_table(topic_id , topic_name , created_at , is_active)
            values (? , ? , ? , ?);
            """,
                (topic_id, topic_name, created_at , 't'),
            )
        logger.debug("Stored topic %s as %s", topic_name, topic_id)

    def __soft_delete(self , topic_id):
        with self.__writing() as curr:
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

    def close(self):
        with self._lock:
            if self.__connection is not None:
                logger.debug("Closing the topic registry at %s", self.topic_db_path)
                self.__connection.close()
                self.__connection = None

    def __del__(self):
        self.close()

