"""Registry of projects and the summary vectors that describe them.

Three stores are kept in step by every write here:

    vectors                (PostgreSQL/pgvector)  the embedding itself
    project_table          (SQLite)               one row per project
    project_mapping_table  (SQLite)               project -> its summary vectors

A ProjectMetaData is scoped to one project_id — the same scoping VectorRepository
uses to partition the vector table — but the SQLite file is a shared registry
holding every project, which is what makes "list all projects" answerable later.

Writes go vectors-first, metadata-second, with a compensating delete on failure.
That is the ordering settled for snapshots (see SnapShot.__add_snap_shot): the
two stores cannot share a transaction, so one of the two failure modes has to be
chosen, and an unreachable vector is recoverable where a mapping row pointing at
a missing embedding is not.

What lives where: project_table holds the project's own metadata, including its
summary *text*; project_description_table holds its descriptions, several per
project; project_mapping_table holds only project_id -> summary vector id. The
summary *embedding* goes to PostgreSQL/pgvector — not to the DiskANN index,
which indexes ingested documents and has nothing to do with project summaries.
"""

import operator
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from numpy import float32, ndarray, uint32
from numpy.typing import NDArray

from config import Config, get_logger
from data_layer.vector_db_manager.repository.vectorRepository import VectorRepository
from memory.memory_pool_exceptions import InvalidVectorId, MisMatchCount

logger = get_logger(__name__)

# Vector ids land in a SQLite INTEGER and a Postgres bigint, both signed 64-bit.
# Config.VECTOR_ID_MASK is what EmbeddingManager folds hash-derived ids into, so
# it is also the largest id this repository can store without wrapping.
MAX_VECTOR_ID = Config.VECTOR_ID_MASK


def utc_now() -> str:
    """Canonical timestamp for every row this repository writes.

    Same format as fullconversation_repository.utc_now: ISO-8601 UTC, sortable
    as text, sub-second precision retained so two writes in the same second do
    not tie.
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def as_timestamp(value: str | date | datetime | None) -> str:
    """Normalise a caller-supplied stamp to the stored TEXT form."""
    if value is None:
        return utc_now()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


class ProjectMetaData:
    __project_db = Config.DATA_DIR / Path("project_db/project.sql")

    def __init__(
        self,
        project_id: str,
        db_path: str | Path | None = None,
        vector_repository: VectorRepository | None = None,
    ) -> None:
        self.__connection: sqlite3.Connection | None = None
        self.__project_vector_handler = vector_repository
        self.__owns_vector_handler = vector_repository is None

        self.project_id = project_id
        self.db_path = Path(db_path) if db_path is not None else self.__project_db
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.__connection = sqlite3.connect(self.db_path)
        self.__connection.execute("PRAGMA foreign_keys = ON")
        self.__db_init()

    @property
    def vector_handler(self) -> VectorRepository:
        """Built on first use — it opens a PostgreSQL connection.

        Every read of the metadata tables works without it, so constructing a
        ProjectMetaData must not require the vector store to be reachable.
        """
        if self.__project_vector_handler is None:
            self.__project_vector_handler = VectorRepository(self.project_id)
        return self.__project_vector_handler

    def __db_init(self) -> None:
        assert self.__connection is not None
        curr = self.__connection.cursor()
        curr.execute("""create table if not exists project_table(
                project_id text primary key,
                project_name text not null,
                created_at date not null,
                updated_at date not null,
                project_summary text not null,
                user_id text
                );""")
        curr.execute("""create table if not exists project_description_table(
                project_id text not null,
                project_description_id text not null,
                project_description text not null,
                created_at date not null,
                primary key (project_id, project_description_id),
                foreign key (project_id) references project_table(project_id)
                );""")
        curr.execute("""create table if not exists project_mapping_table(
                project_id text not null,
                project_summary_vector_id integer not null,
                created_at date not null,
                primary key (project_id, project_summary_vector_id),
                foreign key (project_id) references project_table(project_id)
                );""")
        self.__connection.commit()

    @staticmethod
    def __validate_vector_id(vector_id) -> int:
        try:
            value = operator.index(vector_id)
        except TypeError:
            raise InvalidVectorId(vector_id, MAX_VECTOR_ID)
        if value < 0 or value > MAX_VECTOR_ID:
            raise InvalidVectorId(vector_id, MAX_VECTOR_ID)
        return value

    @staticmethod
    def __validate_summary(project_summary) -> str:
        """project_summary is NOT NULL, so refuse None here rather than at the
        constraint, where the message names the column and not the caller."""
        if not isinstance(project_summary, str):
            raise ValueError(
                f"project_summary must be a str, got {type(project_summary).__name__}"
            )
        return project_summary

    @staticmethod
    def __validate_description(description_id, description) -> Tuple[str, str]:
        if not isinstance(description_id, str) or not description_id.strip():
            raise ValueError("description_id must be a non-empty string")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("description must be a non-empty string")
        return description_id, description

    @staticmethod
    def __as_matrix(vectors) -> NDArray[float32]:
        """Accept a list of vectors or a 2-D array; hand VectorRepository an array.

        VectorRepository.batch_insert calls .tolist() on each row, so a plain
        list of lists would fail there rather than here.
        """
        matrix = np.asarray(vectors, dtype=float32)
        if matrix.ndim == 1 and matrix.size == 0:
            return matrix.reshape(0, Config.EMBEDDING_DIMENSIONS)
        return matrix

    def __upsert_project(
        self,
        cursor: sqlite3.Cursor,
        project_name: str,
        project_summary: str,
        created_at: str,
        updated_at: str,
        user_id: str | None,
    ) -> None:
        """Insert the project row, or refresh it if it already exists.

        created_at and user_id are deliberately left alone on conflict:
        created_at is by definition the first write, and adding a summary vector
        is not a transfer of ownership. Renames, the summary and updated_at do
        land — the summary is regenerated as the project moves on, and the row
        holds the current one.
        """
        cursor.execute(
            """
            insert into project_table
                (project_id, project_name, created_at, updated_at, project_summary, user_id)
            values (?, ?, ?, ?, ?, ?)
            on conflict(project_id) do update set
                project_name    = excluded.project_name,
                project_summary = excluded.project_summary,
                updated_at      = excluded.updated_at
            """,
            (
                self.project_id,
                project_name,
                created_at,
                updated_at,
                project_summary,
                user_id,
            ),
        )

    def __write_meta_data(
        self,
        vector_ids: Sequence[int],
        project_name: str,
        project_summary: str,
        created_at: str,
        updated_at: str,
        user_id: str | None,
    ) -> None:
        """Both metadata tables in one transaction.

        project_mapping_table has a foreign key onto project_table, so the
        project row is upserted first and in the same transaction — a mapping
        row can never be committed for a project that does not exist.

        Idempotent on the mapping: a project's summary vector id repeats
        whenever its summary text repeats, and re-asserting "this project has
        this vector" is a no-op rather than a clash.
        """
        assert self.__connection is not None
        cursor = self.__connection.cursor()
        try:
            self.__upsert_project(
                cursor, project_name, project_summary, created_at, updated_at, user_id
            )
            cursor.executemany(
                """
                insert or ignore into project_mapping_table
                    (project_id, project_summary_vector_id, created_at)
                values (?, ?, ?)
                """,
                [(self.project_id, vector_id, updated_at) for vector_id in vector_ids],
            )
            self.__connection.commit()
        except sqlite3.Error:
            self.__connection.rollback()
            raise

    def __compensate(self, vector_ids: Sequence[int]) -> None:
        """Remove vectors whose metadata failed to land.

        Best effort. If the cleanup itself fails the original error is the one
        worth propagating, so this must not raise — but it is logged, because a
        silent swallow here is how orphaned vectors accumulate unnoticed.
        """
        try:
            self.vector_handler.batch_delete(list(vector_ids))
        except Exception:
            logger.exception(
                "Compensating delete failed for project %s, vector ids %s. "
                "These vectors are now unreachable from project_mapping_table.",
                self.project_id,
                list(vector_ids),
            )

    def __add_project_vector(
        self,
        vector: ndarray,
        vector_id: uint32,
        project_name: str,
        project_summary: str,
        created_at: date,
        updated_at: date,
        user_id: str,
    ) -> None:
        checked_id = self.__validate_vector_id(vector_id)
        summary = self.__validate_summary(project_summary)
        created = as_timestamp(created_at)
        updated = as_timestamp(updated_at)

        self.vector_handler.insert(checked_id, vector)
        try:
            self.__write_meta_data(
                [checked_id], project_name, summary, created, updated, user_id
            )
        except Exception:
            self.__compensate([checked_id])
            raise

    def __add_batch_project_vector(
        self,
        vectors: ndarray,
        vector_ids: Sequence[uint32],
        project_name: str,
        project_summary: str,
        created_at: date,
        updated_at: date,
        user_id: str,
    ) -> None:
        summary = self.__validate_summary(project_summary)
        matrix = self.__as_matrix(vectors)
        if len(matrix) != len(vector_ids):
            raise MisMatchCount(
                f"Got {len(matrix)} vectors and {len(vector_ids)} vector ids. "
                "Every vector must carry exactly one id."
            )
        if len(matrix) == 0:
            return

        checked_ids = [self.__validate_vector_id(v) for v in vector_ids]
        if len(set(checked_ids)) != len(checked_ids):
            raise MisMatchCount(
                "The batch repeats a vector id. Each id may appear at most once."
            )

        created = as_timestamp(created_at)
        updated = as_timestamp(updated_at)

        self.vector_handler.batch_insert(checked_ids, matrix)
        try:
            self.__write_meta_data(
                checked_ids, project_name, summary, created, updated, user_id
            )
        except Exception:
            self.__compensate(checked_ids)
            raise

    def __update_summary_vector(
        self, vector_id: uint32, vector: ndarray, updated_at: date | None = None
    ) -> None:
        checked_id = self.__validate_vector_id(vector_id)
        updated = as_timestamp(updated_at)

        self.vector_handler.update(checked_id, vector)

        assert self.__connection is not None
        cursor = self.__connection.cursor()
        try:
            cursor.execute(
                "update project_table set updated_at = ? where project_id = ?",
                (updated, self.project_id),
            )
            self.__connection.commit()
        except sqlite3.Error:
            self.__connection.rollback()
            raise

    def __write_descriptions(
        self, rows: Sequence[Tuple[str, str]], created_at: str
    ) -> None:
        """Insert descriptions, or refresh the text of ones already stored.

        created_at survives a conflict for the same reason project_table's does:
        it records when the description first appeared, and rewriting the text
        does not change that. The foreign key means a description cannot be
        written for a project that has no row yet.
        """
        assert self.__connection is not None
        cursor = self.__connection.cursor()
        try:
            cursor.executemany(
                """
                insert into project_description_table
                    (project_id, project_description_id, project_description, created_at)
                values (?, ?, ?, ?)
                on conflict(project_id, project_description_id) do update set
                    project_description = excluded.project_description
                """,
                [
                    (self.project_id, description_id, description, created_at)
                    for description_id, description in rows
                ],
            )
            self.__connection.commit()
        except sqlite3.Error:
            self.__connection.rollback()
            raise

    def __get_description(self, description_id: str) -> str | None:
        assert self.__connection is not None
        cursor = self.__connection.cursor()
        cursor.execute(
            """
            select project_description
            from project_description_table
            where project_id = ? and project_description_id = ?
            """,
            (self.project_id, description_id),
        )
        row = cursor.fetchone()
        return row[0] if row is not None else None

    def __get_descriptions(self) -> List[Tuple[str, str, str]]:
        assert self.__connection is not None
        cursor = self.__connection.cursor()
        cursor.execute(
            """
            select project_description_id, project_description, created_at
            from project_description_table
            where project_id = ?
            order by datetime(created_at), created_at, rowid
            """,
            (self.project_id,),
        )
        return cursor.fetchall()

    def __get_summary_vector(self, vector_id: uint32) -> NDArray[float32]:
        return self.vector_handler.search(self.__validate_vector_id(vector_id))

    def __get_all_summary_vector_id(self) -> List[int]:
        assert self.__connection is not None
        cursor = self.__connection.cursor()
        cursor.execute(
            """
            select project_summary_vector_id
            from project_mapping_table
            where project_id = ?
            order by datetime(created_at), created_at, rowid
            """,
            (self.project_id,),
        )
        return [row[0] for row in cursor.fetchall()]

    def __get_project(
        self,
    ) -> Tuple[str, str, str, str, str, str | None] | None:
        assert self.__connection is not None
        cursor = self.__connection.cursor()
        cursor.execute(
            """
            select project_id, project_name, created_at, updated_at,
                   project_summary, user_id
            from project_table where project_id = ?
            """,
            (self.project_id,),
        )
        return cursor.fetchone()

    def add_project_vector(
        self,
        vector: ndarray,
        vector_id: uint32,
        project_name: str,
        project_summary: str,
        created_at: date | None = None,
        updated_at: date | None = None,
        user_id: str | None = None,
    ) -> None:
        """Record one summary vector for this project.

        Writes the embedding, creates or refreshes the project row, and maps the
        two together. project_summary is the text the vector was embedded from,
        and is required because project_table stores it NOT NULL. Timestamps
        default to now; there is no user system yet, so user_id defaults to None.
        """
        self.__add_project_vector(
            vector,
            vector_id,
            project_name,
            project_summary,
            created_at,
            updated_at,
            user_id,
        )

    def add_batch_project_vector(
        self,
        vectors: ndarray,
        vector_ids: Sequence[uint32],
        project_name: str,
        project_summary: str,
        created_at: date | None = None,
        updated_at: date | None = None,
        user_id: str | None = None,
    ) -> None:
        """Record several summary vectors in one pass.

        Same three stores as add_project_vector, with the metadata for the whole
        batch in a single transaction. Unlike the single-vector path this is
        idempotent: the underlying batch insert ignores ids the project already
        holds. An empty batch is a no-op.
        """
        self.__add_batch_project_vector(
            vectors,
            vector_ids,
            project_name,
            project_summary,
            created_at,
            updated_at,
            user_id,
        )

    def update_summary_vector(
        self, vector_id: uint32, vector: ndarray, updated_at: date | None = None
    ) -> None:
        """Replace the embedding stored under an existing id, in place.

        Raises VectorNotFoundEror if this project holds no such vector. The
        mapping row is unchanged — the id still identifies the same summary
        slot — and the project's updated_at is refreshed.
        """
        self.__update_summary_vector(vector_id, vector, updated_at)

    def get_summary_vector(self, vector_id: uint32) -> NDArray[float32]:
        """One summary vector by id. Raises VectorNotFoundEror if absent."""
        return self.__get_summary_vector(vector_id)

    def get_all_summary_vector_id(self) -> List[int]:
        """Every summary vector id for this project, oldest first.

        Answered from SQLite alone, so callers that only need ids — a count, a
        map lookup, a delete list — do not pay a PostgreSQL round trip.
        """
        return self.__get_all_summary_vector_id()

    def get_all_summary_vector(self) -> NDArray[float32]:
        """Every summary vector for this project, oldest first.

        Row i corresponds to get_all_summary_vector_id()[i].
        """
        vector_ids = self.__get_all_summary_vector_id()
        if not vector_ids:
            return np.empty((0, Config.EMBEDDING_DIMENSIONS), dtype=float32)
        return self.vector_handler.batch_search(vector_ids)

    def add_description(
        self,
        description_id: str,
        description: str,
        created_at: date | None = None,
    ) -> None:
        """Store one description for this project, or refresh its text.

        A project holds several, told apart by a caller-supplied id — the table
        is keyed (project_id, project_description_id), so the same id means the
        same description and re-writing it is an update, not a second row.

        The project row must already exist: the foreign key rejects a
        description for a project nothing has written yet. updated_at on
        project_table is left alone; the summary-vector paths own it.
        """
        self.add_descriptions([(description_id, description)], created_at)

    def add_descriptions(
        self,
        descriptions: Sequence[Tuple[str, str]],
        created_at: date | None = None,
    ) -> None:
        """Store several (description_id, description) pairs in one transaction.

        Every pair is validated before any is written, so a bad one partway
        through cannot leave the first half committed. An empty sequence is a
        no-op.
        """
        rows = [
            self.__validate_description(description_id, description)
            for description_id, description in descriptions
        ]
        if not rows:
            return
        identifiers = [description_id for description_id, _ in rows]
        if len(set(identifiers)) != len(identifiers):
            raise MisMatchCount(
                "The batch repeats a description id. Each id may appear at most once."
            )
        self.__write_descriptions(rows, as_timestamp(created_at))

    def get_description(self, description_id: str) -> str | None:
        """One description's text, or None if this project has no such id."""
        return self.__get_description(description_id)

    def get_descriptions(self) -> List[Tuple[str, str, str]]:
        """[(description_id, description, created_at)] for this project, oldest
        first.

        Ordered the same way project_mapping_table is read: datetime() first, so
        a caller-supplied stamp in another format still sorts chronologically;
        then the raw created_at, because datetime() truncates to whole seconds
        and everything written inside one second would otherwise tie; then rowid,
        so identical stamps keep their insertion order.
        """
        return self.__get_descriptions()

    def get_project(
        self,
    ) -> Tuple[str, str, str, str, str, str | None] | None:
        """(project_id, project_name, created_at, updated_at, project_summary,
        user_id), or None.

        Without this project_table would be write-only: every path above updates
        it and nothing would ever read it back.
        """
        return self.__get_project()

    def close(self) -> None:
        if self.__connection is not None:
            self.__connection.close()
            self.__connection = None
        if self.__owns_vector_handler and self.__project_vector_handler is not None:
            self.__project_vector_handler.close()
            self.__project_vector_handler = None

    def __enter__(self) -> "ProjectMetaData":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
