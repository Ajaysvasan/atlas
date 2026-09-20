"""vector data table

See README.md in this directory.
"""

import os
from typing import List

import numpy as np
import psycopg
from dotenv import load_dotenv
from numpy import float32, ndarray, uint32
from numpy.typing import NDArray

from config import Config, get_logger
from data_layer.datalayer_exceptions.datalayer_exceptions import (
    DuplicateVectorException,
    InvalidBatchSize,
    InvalidVectorDimension,
    MissingDatabaseConfiguration,
    VectorInsertionError,
    VectorNotFoundEror,
)

logger = get_logger(__name__)


def register_vector_types(conn) -> None:
    """Teach psycopg the pgvector types. Imported late so the module loads without it."""
    from pgvector.psycopg import register_vector

    register_vector(conn)


class VectorRepository:
    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        load_dotenv()
        self.__db_name = os.getenv("DB_NAME")
        self.__user = os.getenv("DB_USER")
        self.__password = os.getenv("DB_PASSWORD")
        self.__host = os.getenv("DB_HOST")
        self.__port = os.getenv("DB_PORT")

        missing = [
            name
            for name, value in (
                ("DB_NAME", self.__db_name),
                ("DB_USER", self.__user),
                ("DB_PASSWORD", self.__password),
                ("DB_HOST", self.__host),
                ("DB_PORT", self.__port),
            )
            if value is None
        ]
        if missing:
            logger.error(
                "PostgreSQL configuration incomplete; missing %s", ", ".join(missing)
            )
            raise MissingDatabaseConfiguration(missing)

        self.conn = psycopg.connect(
            dbname=self.__db_name,
            user=self.__user,
            password=self.__password,
            host=self.__host,
            port=self.__port,
        )
        self.curr = self.conn.cursor()
        self.__create_extension()
        self.conn.commit()
        # Before any statement touches a vector column: psycopg cannot adapt a
        # numpy array on its own, and reads come back as text without this.
        # register_vector looks up the type, so the extension must exist first.
        register_vector_types(self.conn)
        self.__create_table()
        # Host and database only — never the password, never the whole DSN.
        logger.info(
            "Connected to vector store %s@%s:%s (project %s)",
            self.__db_name,
            self.__host,
            self.__port,
            self.project_id,
        )

    def __create_extension(self):
        query = f"create extension if not exists vector;"
        self.curr.execute(query)

    def __create_table(self, embedding_dimension=Config.EMBEDDING_DIMENSIONS):
        query = f"""
        create table if not exists vectors(project_id varchar, vector_id bigint, embedding vector({embedding_dimension}), primary key (project_id, vector_id))
        """
        self.curr.execute(query)
        self.conn.commit()

    def __insert_vector(self, vector: ndarray, vector_id: uint32):
        if len(vector) != Config.EMBEDDING_DIMENSIONS:
            raise InvalidVectorDimension(len(vector), Config.EMBEDDING_DIMENSIONS)
        query = """
        insert into vectors (project_id, vector_id, embedding) values (%s, %s, %s);
        """
        try:
            self.curr.execute(query, (self.project_id, int(vector_id), vector))
            self.conn.commit()
        except psycopg.errors.UniqueViolation as e:
            self.conn.rollback()
            logger.debug(
                "Vector %s already present for project %s", vector_id, self.project_id
            )
            raise DuplicateVectorException(vector_id) from e
        except Exception as e:
            self.conn.rollback()
            logger.error("Insert of vector %s failed: %s", vector_id, e)
            raise VectorInsertionError(vector_id, e) from e

    def __insert_batch_vector(self, vectors: ndarray, vector_ids: List[uint32]):
        if len(vectors) != len(vector_ids):
            raise InvalidBatchSize("The size of the batch does not match")
        for vector in vectors:
            if len(vector) != Config.EMBEDDING_DIMENSIONS:
                raise InvalidVectorDimension(len(vector), Config.EMBEDDING_DIMENSIONS)
        query = """
        insert into vectors (project_id, vector_id, embedding) values (%s, %s, %s) on conflict (project_id, vector_id) do nothing;
        """
        try:
            rows = [
                (self.project_id, int(id), vector)
                for id, vector in zip(vector_ids, vectors)
            ]
            self.curr.executemany(query, rows)
            self.conn.commit()
            logger.debug(
                "Stored %d vector(s) for project %s", len(rows), self.project_id
            )
        except Exception as e:
            self.conn.rollback()
            logger.error("Batch insert of %d vector(s) failed: %s", len(vector_ids), e)
            raise VectorInsertionError(vector_ids, e) from e

    def __update_vector(self, vector: ndarray, vector_id: uint32) -> None:
        if len(vector) != Config.EMBEDDING_DIMENSIONS:
            raise InvalidVectorDimension(len(vector), Config.EMBEDDING_DIMENSIONS)
        query = """
        update vectors set embedding = %s where project_id = %s and vector_id = %s;
        """
        try:
            self.curr.execute(query, (vector, self.project_id, int(vector_id)))
            # An UPDATE matching nothing is not an error to psycopg.
            if self.curr.rowcount == 0:
                self.conn.rollback()
                raise VectorNotFoundEror(vector_id)
            self.conn.commit()
        except VectorNotFoundEror:
            raise
        except Exception as e:
            self.conn.rollback()
            raise VectorInsertionError(vector_id, e) from e

    def __get_vector(self, vector_id: uint32) -> NDArray[float32]:
        query = """
        select embedding from vectors where project_id = %s and vector_id = %s;
        """
        self.curr.execute(query, (self.project_id, int(vector_id)))
        result = self.curr.fetchone()
        if result is None:
            raise VectorNotFoundEror(vector_id)
        return np.asarray(result[0], dtype=float32)

    def __get_vectors(self, vector_ids: List[uint32]) -> NDArray[float32]:
        vectors = []
        for vector_id in vector_ids:
            vectors.append(self.__get_vector(vector_id))

        return np.array(vectors)

    def __delete_vectors(self, vector_ids: List[uint32]) -> None:
        query = """
        delete from vectors where project_id = %s and vector_id = %s;
        """
        try:
            self.curr.executemany(
                query, [(self.project_id, int(vid)) for vid in vector_ids]
            )
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.error("Delete of %d vector(s) failed: %s", len(vector_ids), e)
            raise VectorInsertionError(vector_ids, e) from e

    def insert(self, vector_id: uint32, vector: ndarray) -> None:
        self.__insert_vector(vector, vector_id)

    def update(self, vector_id: uint32, vector: ndarray) -> None:
        """Replace an existing embedding in place."""
        self.__update_vector(vector, vector_id)

    def delete(self, vector_id: uint32) -> None:
        self.__delete_vectors([vector_id])

    def batch_delete(self, vector_ids: List[uint32]) -> None:
        """Used to undo vectors written for a snapshot whose metadata failed."""
        if not vector_ids:
            return
        self.__delete_vectors(vector_ids)

    def batch_insert(self, vector_ids: List[uint32], vectors: ndarray) -> None:
        self.__insert_batch_vector(vectors, vector_ids)

    def search(self, vector_id: uint32) -> NDArray[float32]:
        return self.__get_vector(vector_id)

    def batch_search(self, vector_ids: List[uint32]) -> NDArray[float32]:
        return self.__get_vectors(vector_ids)

    def close(self):
        self.curr.close()
        self.conn.close()
