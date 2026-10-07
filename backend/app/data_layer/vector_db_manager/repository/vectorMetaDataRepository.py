"""The only way back from a DiskANN hit to the chunk it came from."""

import sqlite3
import threading
from contextlib import contextmanager
from typing import Dict, Iterator, List, Sequence

from numpy import uint32

from config import Config, get_logger
from data_layer.datalayer_exceptions.datalayer_exceptions import (
    InvalidBatchSize,
    InvalidColumnNameException,
    InvalidVectorID,
)
from storage.sqlite_setup import connect, enable_wal

logger = get_logger(__name__)

VALID_COLUMNS = ("vectorId", "chunkId", "embeddingModelUsed", "dimensions")


class VectorMetaDataRepository:
    """`vector_id -> chunk_id`, and the allocator of those vector ids.

    It allocates them because DiskANN labels are `uint32` while this project's
    derived ids are masked into 63 bits, which do not fit; and hashing chunk ids
    down into 32 bits would collide about 116 times at MAX_VECTORS. A sequential
    id from this table cannot collide, and the table is what translates it back
    — which is the whole reason the table exists.

    It lives in the chunk store's own database file, so a search result becomes
    text through a single join against `Chunks` rather than a second connection
    and a second round trip.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or Config.DB_PATH
        self._lock = threading.RLock()
        self.connection: sqlite3.Connection | None = connect(
            self.db_path, check_same_thread=False
        )
        self.journal_mode = enable_wal(self.connection, self.db_path)
        self.__create_meta_data_table()

    @contextmanager
    def _reading(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            assert self.connection is not None
            yield self.connection.cursor()

    @contextmanager
    def _writing(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            assert self.connection is not None
            cursor = self.connection.cursor()
            try:
                yield cursor
                self.connection.commit()
            except BaseException:
                self.connection.rollback()
                raise

    def __create_meta_data_table(self) -> None:
        with self._writing() as cursor:
            # No foreign key on chunkId. A chunk id lives in `Chunks` when the
            # document had sections and in `RecursiveChunks` when it did not,
            # and SQLite cannot reference whichever of two tables holds it. The
            # previous declaration named `Chunks` in a database that did not
            # contain it, so every insert failed (bug 5.2).
            cursor.execute("""
                create table if not exists vector_meta_data(
                    vectorId integer primary key autoincrement,
                    chunkId text not null,
                    embeddingModelUsed text not null,
                    dimensions integer not null
                );
            """)
            # Retrieval reads vectorId -> chunkId, which the key serves. This
            # covers the other direction, for re-embedding a known chunk.
            cursor.execute(
                "create index if not exists idx_vector_meta_chunk "
                "on vector_meta_data(chunkId);"
            )

    def __insert_meta(
        self,
        vectorId: uint32,
        chunkId: str,
        embeddingModelUsed: str,
        dimensions: int = Config.EMBEDDING_DIMENSIONS,
    ) -> None:
        with self._writing() as cursor:
            cursor.execute(
                "insert into vector_meta_data"
                "(vectorId, chunkId, embeddingModelUsed, dimensions) "
                "values (?, ?, ?, ?) on conflict (vectorId) do nothing;",
                (int(vectorId), chunkId, embeddingModelUsed, int(dimensions)),
            )

    def __insert_batch_meta_data(
        self,
        vectorIds: Sequence[uint32],
        chunkIds: Sequence[str],
        embeddingModelUsed: str,
        dimensions: int,
    ) -> None:
        if len(vectorIds) != len(chunkIds):
            raise InvalidBatchSize(
                "The number of vectorIds and number of chunk ids don't match"
            )
        rows = [
            (int(vectorId), chunkId, embeddingModelUsed, int(dimensions))
            for vectorId, chunkId in zip(vectorIds, chunkIds)
        ]
        with self._writing() as cursor:
            cursor.executemany(
                "insert into vector_meta_data"
                "(vectorId, chunkId, embeddingModelUsed, dimensions) "
                "values (?, ?, ?, ?) on conflict (vectorId) do nothing;",
                rows,
            )
        logger.debug("Mapped %d vector(s) to their chunks", len(rows))

    def __get_meta_data(self, vectorId: uint32, columnName: str) -> str | int:
        if columnName not in VALID_COLUMNS:
            raise InvalidColumnNameException(columnName)
        with self._reading() as cursor:
            cursor.execute(
                f"select {columnName} from vector_meta_data where vectorId = ?",
                (int(vectorId),),
            )
            result = cursor.fetchone()
        if result is None:
            raise InvalidVectorID(vectorId)
        return result[0]

    # Public APIs

    def insert(
        self,
        vectorId: uint32,
        chunkId: str,
        embeddingModelUsed: str,
        dimensions: int = Config.EMBEDDING_DIMENSIONS,
    ) -> None:
        self.__insert_meta(vectorId, chunkId, embeddingModelUsed, dimensions)

    def batch_insert(
        self,
        vectorIds: Sequence[uint32],
        chunkIds: Sequence[str],
        embeddingModelUsed: str,
        dimensions: int = Config.EMBEDDING_DIMENSIONS,
    ) -> None:
        self.__insert_batch_meta_data(
            vectorIds, chunkIds, embeddingModelUsed, dimensions
        )

    def get_meta_data(self, vectorId: uint32, columnName: str) -> str | int:
        return self.__get_meta_data(vectorId, columnName)

    def allocate(
        self,
        chunkId: str,
        embeddingModelUsed: str = Config.EMBEDDING_MODEL,
        dimensions: int = Config.EMBEDDING_DIMENSIONS,
    ) -> int:
        """Take the next DiskANN label for this chunk, and return it."""
        with self._writing() as cursor:
            cursor.execute(
                "insert into vector_meta_data"
                "(chunkId, embeddingModelUsed, dimensions) values (?, ?, ?);",
                (chunkId, embeddingModelUsed, int(dimensions)),
            )
            return cursor.lastrowid

    def allocate_many(
        self,
        chunkIds: Sequence[str],
        embeddingModelUsed: str = Config.EMBEDDING_MODEL,
        dimensions: int = Config.EMBEDDING_DIMENSIONS,
    ) -> List[int]:
        """Labels for a batch, in the order given, in one transaction."""
        if not chunkIds:
            return []
        # One statement per row: sqlite3 does not set lastrowid reliably after
        # executemany, and the caller needs each id to label its vector.
        with self._writing() as cursor:
            allocated = []
            for chunkId in chunkIds:
                cursor.execute(
                    "insert into vector_meta_data"
                    "(chunkId, embeddingModelUsed, dimensions) values (?, ?, ?);",
                    (chunkId, embeddingModelUsed, int(dimensions)),
                )
                allocated.append(cursor.lastrowid)
        logger.debug("Allocated %d vector label(s)", len(allocated))
        return allocated

    def chunk_ids_for(self, vectorIds: Sequence[uint32]) -> Dict[int, str]:
        """Every id's chunk in one query, keyed by vector id.

        One statement rather than one per hit: a search returns k ids at once,
        and asking for them individually is the N+1 that makes retrieval slow.
        Ids with no mapping are absent rather than raising — a stale index can
        return a vector whose chunk has since been removed.
        """
        ids = [int(v) for v in vectorIds]
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        with self._reading() as cursor:
            cursor.execute(
                f"select vectorId, chunkId from vector_meta_data "
                f"where vectorId in ({placeholders})",
                tuple(ids),
            )
            return {row[0]: row[1] for row in cursor.fetchall()}

    def vector_ids_for(self, chunkIds: Sequence[str]) -> Dict[str, int]:
        """The reverse, for re-embedding or deleting a known chunk."""
        ids = list(chunkIds)
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        with self._reading() as cursor:
            cursor.execute(
                f"select chunkId, vectorId from vector_meta_data "
                f"where chunkId in ({placeholders})",
                tuple(ids),
            )
            return {row[0]: row[1] for row in cursor.fetchall()}

    def count(self) -> int:
        with self._reading() as cursor:
            return cursor.execute(
                "select count(*) from vector_meta_data"
            ).fetchone()[0]

    def close(self) -> None:
        # getattr, not attribute access: if connect() failed in __init__ the
        # attribute never existed, and __del__ would bury the real error. The
        # lock makes this wait for a write in flight rather than closing the
        # connection out from under it.
        conn = getattr(self, "connection", None)
        if conn is None:
            return
        lock = getattr(self, "_lock", None)
        if lock is None:
            conn.close()
        else:
            with lock:
                conn.close()
        self.connection = None

    def __del__(self):
        self.close()
