"""

vector data table
vector_id
vector

"""

import os
from typing import List

import numpy as np
import psycopg
from dotenv import load_dotenv
from numpy import float32, ndarray, uint32
from numpy.typing import NDArray

from config import Config
from data_layer.datalayer_exceptions.datalayer_exceptions import (
    DuplicateVectorException,
    InvalidBatchSize,
    InvalidVectorDimension,
    MissingDatabaseConfiguration,
    VectorInsertionError,
    VectorNotFoundEror,
)


class VectorRepository:
    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        load_dotenv()
        self.__db_name = os.getenv("DBNAME")
        # DB_USER, not USER: every login shell on Linux and macOS exports USER,
        # and load_dotenv() does not override a variable already in the
        # environment — so the USER= line in .env was ignored and the connection
        # was made as whoever happened to run the process. It only looked
        # correct because that name matched a real Postgres role.
        self.__user = os.getenv("DB_USER")
        self.__password = os.getenv("PASSWORD")
        self.__host = os.getenv("HOST")
        self.__port = os.getenv("PORT")

        # psycopg substitutes libpq's defaults for anything passed as None —
        # the OS username among them — which is the same silent misconnection
        # the rename above exists to prevent. Fail here instead, where the
        # missing setting is named, rather than at a confusing "role does not
        # exist" from the server.
        missing = [
            name
            for name, value in (
                ("DBNAME", self.__db_name),
                ("DB_USER", self.__user),
                ("PASSWORD", self.__password),
                ("HOST", self.__host),
                ("PORT", self.__port),
            )
            if value is None
        ]
        if missing:
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
        self.__create_table()

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
            # Reported apart from a failed write: the caller can carry on
            # knowing the vector is stored, rather than compensating for it.
            self.conn.rollback()
            raise DuplicateVectorException(vector_id) from e
        except Exception as e:
            self.conn.rollback()
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
                (self.project_id, int(id), vector.tolist())
                for id, vector in zip(vector_ids, vectors)
            ]
            self.curr.executemany(query, rows)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise VectorInsertionError(vector_ids, e) from e

    def __update_vector(self, vector: ndarray, vector_id: uint32) -> None:
        if len(vector) != Config.EMBEDDING_DIMENSIONS:
            raise InvalidVectorDimension(len(vector), Config.EMBEDDING_DIMENSIONS)
        query = """
        update vectors set embedding = %s where project_id = %s and vector_id = %s;
        """
        try:
            self.curr.execute(query, (vector, self.project_id, int(vector_id)))
            # An UPDATE that matches nothing is not an error to psycopg, so an
            # id that was never inserted would silently succeed and leave the
            # caller believing the new embedding is stored.
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
            raise VectorInsertionError(vector_ids, e) from e

    def insert(self, vector_id: uint32, vector: ndarray) -> None:
        self.__insert_vector(vector, vector_id)

    def update(self, vector_id: uint32, vector: ndarray) -> None:
        """Replace an existing embedding in place.

        A single statement rather than delete-then-insert: the pair is two
        commits, and a failure between them loses the vector entirely.
        """
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
