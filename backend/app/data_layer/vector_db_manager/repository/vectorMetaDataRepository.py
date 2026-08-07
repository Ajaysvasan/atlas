import sqlite3
from typing import List

from data_layer.datalayer_exceptions.datalayer_exceptions import (
    InvalidBatchSize,
    InvalidColumnNameException,
    InvalidVectorID,
)
from numpy import uint32

from config import Config


class VectorMetaDataRepository:
    def __init__(self, db_path: str) -> None:
        self.connection = sqlite3.connect(db_path)
        self.connection.execute("PRAGMA foreign_keys = ON;")
        self.cur = self.connection.cursor()
        self.__create_meta_data_table()
        self.__valid_column_name = (
            "vectorId",
            "chunkId",
            "embeddingModelUsed",
            "dimensions",
        )

    def __create_meta_data_table(self) -> None:
        query = """
        create table if not exists vector_meta_data(
        vectorId int primary key ,
        chunkId text,
        embeddingModelUsed text ,
        dimensions  int,
        foreign key (chunkId) references Chunks(chunkId)
        );
        """
        self.cur.execute(query)
        self.connection.commit()

    def __insert_meta(
        self,
        vectorId: uint32,
        chunkId: str,
        embeddingModelUsed: str,
        dimensions: int = Config.EMBEDDING_DIMENSIONS,
    ) -> None:
        query = """
            insert into vector_meta_data(vectorId , chunkId , embeddingModelUsed , dimensions)
            values (? , ? , ? , ?) on conflict (vectorId) do nothing;
        """
        try:
            self.cur.execute(query, (int(vectorId), chunkId, embeddingModelUsed, dimensions))
            self.connection.commit()

        except Exception as e:
            self.connection.rollback()
            raise e

    def __insert_batch_meta_data(
        self,
        vectorIds: List[uint32],
        chunkIds: List[str],
        embeddingModelUsed: str,
        dimensions: int,
    ) -> None:
        query = """
            insert into vector_meta_data(vectorId , chunkId , embeddingModelUsed , dimensions)
            values (? , ? , ? , ?) on conflict (vectorId) do nothing;
        """
        if len(vectorIds) != len(chunkIds):
            raise InvalidBatchSize(
                "The number of vectorIds and number of chunk ids don't match"
            )
        try:
            rows = [
                (int(vectorId), chunkId, embeddingModelUsed, dimensions)
                for vectorId, chunkId in zip(vectorIds, chunkIds)
            ]
            self.cur.executemany(query, rows)
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            raise e

    def __get_meta_data(self, vectorId: uint32, columnName: str) -> str | int:
        if columnName not in self.__valid_column_name:
            raise InvalidColumnNameException(columnName)
        query = f"""select {columnName} from vector_meta_data where vectorId = ?"""
        self.cur.execute(query, (int(vectorId),))
        result = self.cur.fetchone()
        if result == None:
            raise InvalidVectorID(vectorId)

        return result[0]

    def insert(
        self,
        vectorId: uint32,
        chunkId: str,
        embeddingModelUsed: str,
        dimensions: int = Config.EMBEDDING_DIMENSIONS,
    ):
        self.__insert_meta(vectorId, chunkId, embeddingModelUsed, dimensions)

    def batch_insert(
        self,
        vectorIds: List[uint32],
        chunkIds: List[str],
        embeddingModelUsed: str,
        dimensions: int,
    ):
        self.__insert_batch_meta_data(
            vectorIds, chunkIds, embeddingModelUsed, dimensions
        )

    def get_meta_data(self, vectorId: uint32, columnName: str) -> str | int:
        return self.__get_meta_data(vectorId, columnName)

    def close(self):
        self.cur.close()
        self.connection.close()
