"""The project registry: projects, their descriptions, and their vector ids.

See README.md in this directory.
"""

import operator
import sqlite3
from datetime import date
from pathlib import Path
from typing import List, NamedTuple, Sequence, Tuple

import numpy as np
from numpy import float32, ndarray, uint32
from numpy.typing import NDArray

from config import Config, get_logger
from storage.timestamps import as_timestamp
from data_layer.vector_db_manager.repository.vectorRepository import VectorRepository
from memory.memory_database import MemoryDatabase, Schema
from memory.memory_pool_exceptions import (
    InvalidVectorId,
    MisMatchCount,
    ProjectInAnotherTopic,
    ProjectNotFound,
    TopicNotFound,
)
from memory.topic_pool.topic_pool_repo.topic_pool_meta_handler import (
    SCHEMA as TOPIC_SCHEMA,
    TopicPoolMetaHandler,
)

logger = get_logger(__name__)

MAX_VECTOR_ID = Config.VECTOR_ID_MASK


class TopicProject(NamedTuple):
    """One project under a topic, as the project router lists them."""

    project_id: str
    project_name: str
    project_summary: str


def _create_tables(cursor: sqlite3.Cursor) -> None:
    cursor.execute("""
        create table if not exists project_table(
            project_id text primary key,
            project_name text not null,
            topic_id text not null references topics_mapping_table(topic_id),
            created_at text not null,
            updated_at text not null,
            project_summary text not null,
            user_id text,
            unique (project_id, topic_id)
        )
    """)
    # Both children carry topic_id so the router's topic-wide read needs no
    # join. The pair is the foreign key, so a child cannot disagree with its
    # project's topic, and moving a project cascades to them.
    cursor.execute("""
        create table if not exists project_description_table(
            topic_id text not null,
            project_id text not null,
            project_description_id text not null,
            project_description text not null,
            created_at text not null,
            primary key (project_id, project_description_id),
            foreign key (project_id, topic_id)
                references project_table(project_id, topic_id) on update cascade
        )
    """)
    cursor.execute("""
        create table if not exists project_mapping_table(
            project_id text not null,
            topic_id text not null,
            project_summary_vector_id integer not null,
            created_at text not null,
            primary key (project_id, project_summary_vector_id),
            foreign key (project_id, topic_id)
                references project_table(project_id, topic_id) on update cascade
        )
    """)
    cursor.execute(
        "create index if not exists idx_project_topic on project_table(topic_id)"
    )
    cursor.execute(
        "create index if not exists idx_project_mapping_topic "
        "on project_mapping_table(topic_id)"
    )


SCHEMA = Schema("projects", _create_tables, requires=(TOPIC_SCHEMA,))


def _registry(database: MemoryDatabase | str | Path | None) -> MemoryDatabase:
    registry = MemoryDatabase.of(database)
    registry.ensure(SCHEMA)
    return registry


def project_topic(
    project_id: str, database: MemoryDatabase | str | Path | None = None
) -> str | None:
    """The topic a project is registered under, or None if it is not registered."""
    with _registry(database).reading() as cursor:
        row = cursor.execute(
            "select topic_id from project_table where project_id = ?", (project_id,)
        ).fetchone()
    return row[0] if row is not None else None


def is_registered(
    project_id: str, database: MemoryDatabase | str | Path | None = None
) -> bool:
    """Whether a project row exists — what a refused foreign key usually means."""
    return project_topic(project_id, database) is not None


def list_topic_projects(
    topic_id: str, database: MemoryDatabase | str | Path | None = None
) -> List[TopicProject]:
    """Every project under a topic, oldest first."""
    with _registry(database).reading() as cursor:
        cursor.execute(
            """
            select project_id, project_name, project_summary
            from project_table
            where topic_id = ?
            order by datetime(created_at), created_at, rowid
            """,
            (topic_id,),
        )
        return [TopicProject(*row) for row in cursor.fetchall()]


def list_topic_vector_ids(
    topic_id: str, database: MemoryDatabase | str | Path | None = None
) -> List[Tuple[str, int]]:
    """[(project_id, vector_id)] for every project under a topic."""
    with _registry(database).reading() as cursor:
        cursor.execute(
            """
            select project_id, project_summary_vector_id
            from project_mapping_table
            where topic_id = ?
            order by project_id, datetime(created_at), created_at, rowid
            """,
            (topic_id,),
        )
        return cursor.fetchall()


class ProjectRow(NamedTuple):
    """One project_table row."""

    project_id: str
    project_name: str
    topic_id: str
    created_at: str
    updated_at: str
    project_summary: str
    user_id: str | None






class ProjectMetaData:
    def __init__(
        self,
        project_id: str,
        topic_id: str,
        database: MemoryDatabase | str | Path | None = None,
        vector_repository: VectorRepository | None = None,
    ) -> None:
        self.__project_vector_handler = vector_repository
        self.__owns_vector_handler = vector_repository is None

        self.project_id = project_id
        self.topic_id = self.__validate_topic_id(topic_id)
        self.database = _registry(database)

    def __refused(self, error: sqlite3.IntegrityError) -> Exception:
        """Name the relationship a foreign key refused; SQLite says only that one did."""
        if "FOREIGN KEY" not in str(error):
            return error
        if not TopicPoolMetaHandler(self.database).has_topic_id(self.topic_id):
            return TopicNotFound(self.topic_id)
        project = self.__get_project()
        if project is None:
            return ProjectNotFound(self.project_id)
        if project.topic_id != self.topic_id:
            return ProjectInAnotherTopic(self.project_id, self.topic_id, project.topic_id)
        return error

    @property
    def vector_handler(self) -> VectorRepository:
        """Built on first use — it opens a PostgreSQL connection."""
        if self.__project_vector_handler is None:
            self.__project_vector_handler = VectorRepository(self.project_id)
        return self.__project_vector_handler

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
    def __validate_topic_id(topic_id) -> str:
        """topic_id is NOT NULL in all three tables and has no table of its own"""
        if not isinstance(topic_id, str) or not topic_id.strip():
            raise ValueError("topic_id must be a non-empty string")
        return topic_id

    @staticmethod
    def __validate_summary(project_summary) -> str:
        """project_summary is NOT NULL, so refuse None here rather than at the"""
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
        """Accept a list of vectors or a 2-D array; hand VectorRepository an array."""
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
        """Insert the project row, or refresh it if it already exists."""
        cursor.execute(
            """
            insert into project_table
                (project_id, project_name, topic_id, created_at, updated_at,
                 project_summary, user_id)
            values (?, ?, ?, ?, ?, ?, ?)
            on conflict(project_id) do update set
                    project_name    = excluded.project_name,
                    topic_id        = excluded.topic_id,
                    project_summary = excluded.project_summary,
                    updated_at      = excluded.updated_at
            """,
            (
                self.project_id,
                    project_name,
                self.topic_id,
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
        """Both metadata tables in one transaction."""
        try:
            with self.database.writing() as cursor:
                self.__upsert_project(
                    cursor, project_name, project_summary, created_at, updated_at, user_id
                )
                cursor.executemany(
                    """
                    insert or ignore into project_mapping_table
                        (project_id, topic_id, project_summary_vector_id, created_at)
                    values (?, ?, ?, ?)
                    """,
                    [
                        (self.project_id, self.topic_id, vector_id, updated_at)
                        for vector_id in vector_ids
                    ],
                )
        except sqlite3.IntegrityError as error:
            raise self.__refused(error) from error

    def __compensate(self, vector_ids: Sequence[int]) -> None:
        """Remove vectors whose metadata failed to land."""
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

        with self.database.writing() as cursor:
            cursor.execute(
                "update project_table set updated_at = ? where project_id = ?",
                (updated, self.project_id),
            )

    def __write_descriptions(
        self, rows: Sequence[Tuple[str, str]], created_at: str
    ) -> None:
        """Insert descriptions, or refresh the text of ones already stored."""
        try:
            self.__insert_descriptions(rows, created_at)
        except sqlite3.IntegrityError as error:
            raise self.__refused(error) from error

    def __insert_descriptions(
        self, rows: Sequence[Tuple[str, str]], created_at: str
    ) -> None:
        with self.database.writing() as cursor:
            cursor.executemany(
                """
                insert into project_description_table
                    (project_id, topic_id, project_description_id,
                     project_description, created_at)
                values (?, ?, ?, ?, ?)
                on conflict(project_id, project_description_id) do update set
                    project_description = excluded.project_description
                """,
                [
                    (
                        self.project_id,
                        self.topic_id,
                        description_id,
                        description,
                        created_at,
                    )
                    for description_id, description in rows
                ],
            )

    def __get_description(self, description_id: str) -> str | None:
        with self.database.reading() as cursor:
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
        with self.database.reading() as cursor:
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

    def __set_project_summary(self, project_summary: str, updated_at: date | None) -> None:
        summary = self.__validate_summary(project_summary)
        updated = as_timestamp(updated_at)

        with self.database.writing() as cursor:
            cursor.execute(
                "update project_table set project_summary = ?, updated_at = ? "
                "where project_id = ?",
                (summary, updated, self.project_id),
            )
            # An UPDATE matching nothing is not an error to SQLite.
            if cursor.rowcount == 0:
                raise ValueError(f"No project row for {self.project_id!r}")

    def __get_summary_vector(self, vector_id: uint32) -> NDArray[float32]:
        return self.vector_handler.search(self.__validate_vector_id(vector_id))

    def __get_all_summary_vector_id(self) -> List[int]:
        with self.database.reading() as cursor:
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

    def __get_project(self) -> ProjectRow | None:
        with self.database.reading() as cursor:
            cursor.execute(
                """
                select project_id, project_name, topic_id, created_at, updated_at,
                       project_summary, user_id
                from project_table where project_id = ?
                """,
                (self.project_id,),
            )
            row = cursor.fetchone()
            return ProjectRow(*row) if row is not None else None

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
        """Record one summary vector for this project."""
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
        """Record several summary vectors in one pass. An empty batch is a no-op."""
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
        """Replace the embedding stored under an existing id, in place."""
        self.__update_summary_vector(vector_id, vector, updated_at)

    def get_summary_vector(self, vector_id: uint32) -> NDArray[float32]:
        """One summary vector by id. Raises VectorNotFoundEror if absent."""
        return self.__get_summary_vector(vector_id)

    def get_all_summary_vector_id(self) -> List[int]:
        """Every summary vector id for this project, oldest first."""
        return self.__get_all_summary_vector_id()

    def get_all_summary_vector(self) -> NDArray[float32]:
        """Every summary vector for this project, oldest first; row i matches id i."""
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
        """Store one description for this project, or refresh its text."""
        self.add_descriptions([(description_id, description)], created_at)

    def add_descriptions(
        self,
        descriptions: Sequence[Tuple[str, str]],
        created_at: date | None = None,
    ) -> None:
        """Store several (description_id, description) pairs in one transaction."""
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
        """[(description_id, description, created_at)] for this project, oldest first."""
        return self.__get_descriptions()

    def set_project_summary(
        self, project_summary: str, updated_at: date | None = None
    ) -> None:
        """Replace the summary text without touching any vector. Raises ValueError if the project is absent."""
        self.__set_project_summary(project_summary, updated_at)

    def get_topic_projects(self) -> List[TopicProject]:
        """Every project under this topic, oldest first."""
        return list_topic_projects(self.topic_id, self.database)

    def get_project(self) -> ProjectRow | None:
        """This project's row, or None."""
        return self.__get_project()

    def get_topic_summary_vector_ids(self) -> List[Tuple[str, int]]:
        """[(project_id, vector_id)] for every project under this topic."""
        return list_topic_vector_ids(self.topic_id, self.database)

    def close(self) -> None:
        """Closes the vector store this registry opened; the database is shared."""
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
