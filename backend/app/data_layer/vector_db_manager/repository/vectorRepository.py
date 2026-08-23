"""

vector data
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
    InvalidBatchSize,
    InvalidVectorDimension,
    VectorInsertionError,
    VectorNotFoundEror,
)


class VectorRepository:
    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        load_dotenv()
        self.__db_name = os.getenv("DBNAME")
        self.__user = os.getenv("USER")
        self.__password = os.getenv("PASSWORD")
        self.__host = os.getenv("HOST")
        self.__port = os.getenv("PORT")

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
        except Exception as e:
            self.conn.rollback()
            raise VectorInsertionError(e)

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
                (self.project_id, int(id), vector.tolist()) for id, vector in zip(vector_ids, vectors)
            ]
            self.curr.executemany(query, rows)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise VectorInsertionError(e)

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
            raise VectorInsertionError(e)

    def insert(self, vector_id: uint32, vector: ndarray) -> None:
        self.__insert_vector(vector, vector_id)

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
