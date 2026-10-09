"""The only way back from a DiskANN hit to the chunk it came from."""

import operator
import sqlite3
import threading
from contextlib import contextmanager
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
from numpy import ndarray, uint32

from config import Config, get_logger
from data_layer.datalayer_exceptions.datalayer_exceptions import (
    EmbeddingModelMismatch,
    InvalidBatchSize,
    InvalidColumnNameException,
    InvalidVectorID,
    MalformedVectorId,
    MissingVectorId,
    VectorIdConflict,
)
from data_layer.ingestion.embedding.vector_ids import vector_id_for
from storage.sqlite_setup import connect, enable_wal

logger = get_logger(__name__)

VALID_COLUMNS = ("label", "vectorId", "chunkId", "embeddingModelUsed", "dimensions")

PAGE = 50_000

UINT32_MAX = int(np.iinfo(uint32).max)

CREATE = """
    create table if not exists vector_meta_data(
        label integer primary key autoincrement check (label <= 4294967295),
        vectorId integer not null unique
            check (typeof(vectorId) = 'integer' and vectorId >= 0),
        chunkId text not null unique,
        embeddingModelUsed text not null,
        dimensions integer not null
    );
"""


def checked_vector_id(vector_id, chunk_id: str) -> int:
    """The vector id as an int, or the reason it cannot be stored."""
    if vector_id is None:
        raise MissingVectorId(chunk_id)
    if isinstance(vector_id, (bool, np.bool_)):
        raise MalformedVectorId(chunk_id, vector_id)
    try:
        value = operator.index(vector_id)
    except TypeError:
        raise MalformedVectorId(chunk_id, vector_id) from None
    if not 0 <= value <= Config.VECTOR_ID_MASK:
        raise MalformedVectorId(chunk_id, vector_id)
    return value


class VectorMetaDataRepository:
    """`label -> vector id -> chunk id`, and the allocator of DiskANN labels."""

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
            columns = {
                row[1] for row in cursor.execute("pragma table_info(vector_meta_data)")
            }
            if columns and "label" not in columns:
                self.__rebuild_without_vectors(cursor)
            # No foreign key on chunkId: a chunk id lives in `Chunks` or in
            # `RecursiveChunks`, and SQLite cannot reference whichever holds it.
            cursor.execute(CREATE)

    def __rebuild_without_vectors(self, cursor: sqlite3.Cursor) -> None:
        """Carry a table from before vector ids were required (bug 5.22) into the new one."""
        rows = cursor.execute(
            "select vectorId, chunkId, embeddingModelUsed, dimensions "
            "from vector_meta_data where vectorId is not null and chunkId is not null"
        ).fetchall()
        highest = max((row[0] for row in rows), default=0)
        cursor.execute("alter table vector_meta_data rename to vector_meta_data_old")
        cursor.execute(CREATE)
        # The old vectorId was the label. A chunk keeps one: its oldest under
        # the configured model, else its oldest.
        rows.sort(key=lambda r: (r[2] != Config.EMBEDDING_MODEL, r[0]))
        kept, seen = [], set()
        for label, chunk_id, model, dimensions in rows:
            if chunk_id in seen or not 1 <= label <= UINT32_MAX:
                continue
            seen.add(chunk_id)
            kept.append((label, vector_id_for(chunk_id), chunk_id, model, dimensions))
        cursor.executemany(
            "insert into vector_meta_data"
            "(label, vectorId, chunkId, embeddingModelUsed, dimensions) "
            "values (?, ?, ?, ?, ?)",
            kept,
        )
        cursor.execute("drop table vector_meta_data_old")
        # Labels the dropped rows held are not handed out again.
        if 1 <= highest <= UINT32_MAX:
            raised = cursor.execute(
                "update sqlite_sequence set seq = max(seq, ?) "
                "where name = 'vector_meta_data'",
                (highest,),
            ).rowcount
            if not raised:
                cursor.execute(
                    "insert into sqlite_sequence(name, seq) "
                    "values ('vector_meta_data', ?)",
                    (highest,),
                )
        logger.warning(
            "Rebuilt vector_meta_data for required vector ids: kept %d row(s), "
            "dropped %d. Kept rows have no stored vector until their documents "
            "are ingested again",
            len(kept),
            len(rows) - len(kept),
        )

    @staticmethod
    def __label(cursor, vectorId, chunkId, embeddingModelUsed, dimensions) -> int:
        inserted = cursor.execute(
            "insert into vector_meta_data"
            "(vectorId, chunkId, embeddingModelUsed, dimensions) "
            "values (?, ?, ?, ?) on conflict do nothing returning label;",
            (vectorId, chunkId, embeddingModelUsed, dimensions),
        ).fetchone()
        if inserted is not None:
            return inserted[0]
        stored = cursor.execute(
            "select label, vectorId, chunkId, embeddingModelUsed "
            "from vector_meta_data where chunkId = ? or vectorId = ?",
            (chunkId, vectorId),
        ).fetchall()
        for label, stored_id, stored_chunk, stored_model in stored:
            if stored_chunk == chunkId and stored_id == vectorId:
                if stored_model != embeddingModelUsed:
                    raise EmbeddingModelMismatch(chunkId, stored_model, embeddingModelUsed)
                return label
        _, stored_id, stored_chunk, _ = stored[0]
        raise VectorIdConflict(chunkId, vectorId, stored_chunk, stored_id)

    # Public APIs

    def insert(
        self,
        vectorId: int,
        chunkId: str,
        embeddingModelUsed: str = Config.EMBEDDING_MODEL,
        dimensions: int = Config.EMBEDDING_DIMENSIONS,
    ) -> int:
        """This chunk's DiskANN label: the one it already has, or the next one."""
        vector_id = checked_vector_id(vectorId, chunkId)
        with self._writing() as cursor:
            return self.__label(
                cursor, vector_id, chunkId, embeddingModelUsed, int(dimensions)
            )

    def batch_insert(
        self,
        vectorIds: Sequence[int],
        chunkIds: Sequence[str],
        embeddingModelUsed: str = Config.EMBEDDING_MODEL,
        dimensions: int = Config.EMBEDDING_DIMENSIONS,
    ) -> List[int]:
        """Labels for a batch, in the order given, in one transaction."""
        if len(vectorIds) != len(chunkIds):
            raise InvalidBatchSize(
                "The number of vectorIds and number of chunk ids don't match"
            )
        rows = [
            (checked_vector_id(vector_id, chunk_id), chunk_id)
            for vector_id, chunk_id in zip(vectorIds, chunkIds)
        ]
        # One statement per row: executemany cannot return each row's label.
        with self._writing() as cursor:
            labels = [
                self.__label(cursor, vector_id, chunk_id, embeddingModelUsed, int(dimensions))
                for vector_id, chunk_id in rows
            ]
        logger.debug("Mapped %d vector(s) to their labels", len(labels))
        return labels

    def get_meta_data(self, vectorId: int, columnName: str) -> str | int:
        if columnName not in VALID_COLUMNS:
            raise InvalidColumnNameException(columnName)
        with self._reading() as cursor:
            result = cursor.execute(
                f"select {columnName} from vector_meta_data where vectorId = ?",
                (int(vectorId),),
            ).fetchone()
        if result is None:
            raise InvalidVectorID(vectorId)
        return result[0]

    def labels(
        self,
        after: int = 0,
        batch_size: int = PAGE,
        embeddingModelUsed: str = Config.EMBEDDING_MODEL,
        dimensions: int = Config.EMBEDDING_DIMENSIONS,
    ) -> Iterator[Tuple[ndarray, ndarray]]:
        """`(labels, vector ids)` above `after`, in label order, a page at a time.

        Only this model's at this width: another model's vectors are in a
        different space. Each page is read under the lock and yielded outside it.
        """
        last = int(after)
        while True:
            with self._reading() as cursor:
                rows = cursor.execute(
                    "select label, vectorId from vector_meta_data "
                    "where label > ? and embeddingModelUsed = ? and dimensions = ? "
                    "order by label limit ?",
                    (last, embeddingModelUsed, int(dimensions), int(batch_size)),
                ).fetchall()
            if not rows:
                return
            labels = np.fromiter((r[0] for r in rows), dtype=uint32, count=len(rows))
            vector_ids = np.fromiter((r[1] for r in rows), dtype=np.int64, count=len(rows))
            last = int(labels[-1])
            yield labels, vector_ids

    def pending(
        self,
        after: int = 0,
        embeddingModelUsed: str = Config.EMBEDDING_MODEL,
        dimensions: int = Config.EMBEDDING_DIMENSIONS,
    ) -> int:
        """How many of this model's labels lie above `after`."""
        with self._reading() as cursor:
            return cursor.execute(
                "select count(*) from vector_meta_data where label > ? "
                "and embeddingModelUsed = ? and dimensions = ?",
                (int(after), embeddingModelUsed, int(dimensions)),
            ).fetchone()[0]

    def chunk_ids_for(self, labels: Sequence[int]) -> Dict[int, str]:
        """Every label's chunk in one query; a label with no row is absent, not an error."""
        ids = [int(label) for label in labels]
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        with self._reading() as cursor:
            rows = cursor.execute(
                f"select label, chunkId from vector_meta_data "
                f"where label in ({placeholders})",
                tuple(ids),
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    def labels_for(self, chunkIds: Sequence[str]) -> Dict[str, int]:
        """The reverse, for a keyword hit or a known chunk."""
        ids = list(chunkIds)
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        with self._reading() as cursor:
            rows = cursor.execute(
                f"select chunkId, label from vector_meta_data "
                f"where chunkId in ({placeholders})",
                tuple(ids),
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    def count(self) -> int:
        with self._reading() as cursor:
            return cursor.execute("select count(*) from vector_meta_data").fetchone()[0]

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
