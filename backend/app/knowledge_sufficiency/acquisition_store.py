"""What this subsystem has acquired, and where it came from."""

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, NamedTuple, Sequence

from config import Config, get_logger
from storage.sqlite_setup import connect, enable_wal
from storage.timestamps import utc_now

logger = get_logger(__name__)


class AcquiredRecord(NamedTuple):
    id: int
    topic: str
    query: str
    url: str
    created_at: str


class AcquisitionStore:
    """The record of every document KSV has taken knowledge from.

    Owned by this subsystem rather than the memory layer: it answers what was
    acquired and from where, which is a question about acquisition and not
    about any conversation or project.
    """

    __store_db = Config.DATA_DIR / Path("ksv/acquired_knowledge.sql")

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.__db_path = Path(db_path) if db_path is not None else self.__store_db
        self.__db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.__connection: sqlite3.Connection | None = connect(
            self.__db_path, check_same_thread=False
        )
        self.journal_mode = enable_wal(self.__connection, self.__db_path)
        self.__init_db()
        logger.debug(
            "Acquisition store ready at %s (journal=%s)",
            self.__db_path, self.journal_mode,
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

    def __init_db(self) -> None:
        with self._writing() as curr:
            # AUTOINCREMENT, not a bare INTEGER PRIMARY KEY: both assign ids,
            # but only this one refuses to reuse the id of a deleted row. These
            # are provenance records, and an id pointing at two different
            # documents over a table's life is worse than a gap in the numbers.
            curr.execute("""
                create table if not exists acquired_knowledge (
                    id integer primary key autoincrement,
                    topic text not null,
                    query text not null,
                    url text not null,
                    created_at text not null
                );
            """)
            # Both reads filter on one of these, and neither is covered by the
            # key, which is the id.
            curr.execute(
                "create index if not exists idx_acquired_topic "
                "on acquired_knowledge(topic);"
            )
            curr.execute(
                "create index if not exists idx_acquired_url "
                "on acquired_knowledge(url);"
            )

    def record(self, topic: str, query: str, url: str) -> int:
        """Note that knowledge was taken from `url`, and return the new id."""
        with self._writing() as cursor:
            cursor.execute(
                "insert into acquired_knowledge (topic, query, url, created_at) "
                "values (?, ?, ?, ?);",
                (topic, query, url, utc_now()),
            )
            new_id = cursor.lastrowid
        logger.info(
            "Recorded acquisition %d: %s for topic %r", new_id, url, topic
        )
        return new_id

    def record_many(
        self, topic: str, query: str, urls: Sequence[str]
    ) -> List[int]:
        """Note several documents from one acquisition, in one transaction."""
        if not urls:
            return []
        # One statement per row rather than executemany: sqlite3 does not set
        # lastrowid reliably after executemany, and the caller needs the ids.
        # Still one transaction, so either every row lands or none does.
        with self._writing() as cursor:
            # One stamp for the batch, not one per row: these arrived from a
            # single acquisition, and stamping them separately would suggest an
            # ordering between them that the id already carries.
            stamped_at = utc_now()
            new_ids = []
            for url in urls:
                cursor.execute(
                    "insert into acquired_knowledge (topic, query, url, created_at) "
                    "values (?, ?, ?, ?);",
                    (topic, query, url, stamped_at),
                )
                new_ids.append(cursor.lastrowid)
        logger.info(
            "Recorded %d acquisition(s) for topic %r", len(urls), topic
        )
        return new_ids

    def for_topic(self, topic: str) -> List[AcquiredRecord]:
        """Everything acquired for a topic, oldest first."""
        with self._reading() as cursor:
            cursor.execute(
                "select id, topic, query, url, created_at from acquired_knowledge "
                "where topic = ? order by id;",
                (topic,),
            )
            return [AcquiredRecord(*row) for row in cursor.fetchall()]

    def seen(self, url: str) -> bool:
        """Whether knowledge has already been taken from this document."""
        with self._reading() as cursor:
            return cursor.execute(
                "select 1 from acquired_knowledge where url = ? limit 1;", (url,)
            ).fetchone() is not None

    def all_records(self) -> List[AcquiredRecord]:
        with self._reading() as cursor:
            cursor.execute(
                "select id, topic, query, url, created_at from acquired_knowledge order by id;"
            )
            return [AcquiredRecord(*row) for row in cursor.fetchall()]

    def close(self) -> None:
        # getattr, not attribute access: if connect() failed in __init__ the
        # attribute never existed, and __del__ would then bury the real error.
        # The lock makes this wait for a write in flight on another thread
        # rather than closing the connection out from under it, which segfaults
        # rather than raising (Bug 4.47).
        conn = getattr(self, "_AcquisitionStore__connection", None)
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
