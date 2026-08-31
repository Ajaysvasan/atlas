import os
import sqlite3
from typing import List

from data_layer.datalayer_exceptions.datalayer_exceptions import InsertionError
from data_layer.ingestion.nodes.nodes import Context, Document, HChunk, RChunk, Section


class Manager:
    def __init__(self, db_path: str, is_chunker_type_hierarchical: bool) -> None:
        self.is_chunker_type_hierarchical = is_chunker_type_hierarchical
        parent_dir = os.path.dirname(db_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        self.connection = sqlite3.connect(db_path, check_same_thread=False)
        self.connection.execute("PRAGMA foreign_keys = ON;")
        self.cursor = self.connection.cursor()
        self._create_table()

    def __create_document_htable(self):
        documentTableQuery = """
            CREATE TABLE IF NOT EXISTS Documents (
                documentId TEXT PRIMARY KEY,
                documentName TEXT
            )
        """
        self.cursor.execute(documentTableQuery)

    def __create_section_htable(self):
        sectionTableQuery = """
            CREATE TABLE IF NOT EXISTS Sections(
                sectionId TEXT PRIMARY KEY, 
                documentId TEXT, 
                sectionName TEXT,
                content TEXT NOT NULL,
                contentLength INTEGER NOT NULL,
                startoffset int not null,
                endoffset int not null,
                FOREIGN KEY (documentId) REFERENCES Documents(documentId)
            )
        """
        self.cursor.execute(sectionTableQuery)

    def __create_context_htable(self):
        contextTableQuery = """
            CREATE TABLE IF NOT EXISTS Contexts(
                contextId TEXT PRIMARY KEY,
                sectionId TEXT , 
                context TEXT not null,
                contextLength int not null,
                startoffset int not null,
                endoffset int not null,
                FOREIGN KEY (sectionId) REFERENCES Sections(sectionId)
            )
        """
        self.cursor.execute(contextTableQuery)

    def __create_chunk_htable(self):
        chunkTableQuery = """
            CREATE TABLE IF NOT EXISTS Chunks(
                chunkId TEXT PRIMARY KEY,
                contextId TEXT,
                chunk TEXT not null,
                startoffset int not null,
                endoffset int not null,
                FOREIGN KEY (contextId) REFERENCES Contexts(contextId)
            )
        """
        self.cursor.execute(chunkTableQuery)

    def __create_chunk_rtable(self):
        chunkTableQuery = """
            CREATE TABLE IF NOT EXISTS RecursiveChunks(
                chunkId TEXT PRIMARY KEY,
                documentId TEXT,
                chunk TEXT not null,
                startoffset int not null,
                endoffset int not null,
                FOREIGN KEY (documentId) REFERENCES Documents(documentId)
            )
        """
        self.cursor.execute(chunkTableQuery)

    def _create_table(self):
        try:
            self.__create_document_htable()
            if self.is_chunker_type_hierarchical:
                self.__create_section_htable()
                self.__create_context_htable()
                self.__create_chunk_htable()
            else:
                self.__create_chunk_rtable()
            self.connection.commit()
        except sqlite3.Error as e:
            self.cursor.close()
            self.connection.close()
            raise Exception(f"Database setup failed: {e}") from e

    def __get_hsection(self, sectionId: str):
        getSectionQuery = """
            select * from Sections where sectionId = ?;
        """
        self.cursor.execute(getSectionQuery, (sectionId,))
        rows = self.cursor.fetchall()
        return rows

    def __get_hcontext(self, contextId: str):
        getContextQuery = """select * from Contexts where contextId = ?"""
        self.cursor.execute(getContextQuery, (contextId,))
        rows = self.cursor.fetchall()
        return rows

    def __get_hdocument(self, documentId: str):
        getDocumentQuery = """select * from Documents where documentId = ?"""
        self.cursor.execute(getDocumentQuery, (documentId,))
        rows = self.cursor.fetchall()
        return rows

    def __get_hchunk(self, chunkId: str):
        getChunkQuery = """select * from Chunks where chunkId = ?"""
        self.cursor.execute(getChunkQuery, (chunkId,))
        rows = self.cursor.fetchall()
        return rows

    def __insert_many(self, query: str, rows: list, table: str, ids: list):
        """Insert rows in one statement, ignoring ones already stored.

        Every id here is derived from the document's content plus the position
        of the row within it, so a conflict means the identical row is already
        present — which is what re-ingesting an unchanged folder does, and what
        used to abort the whole run on a primary key violation.
        """
        if not rows:
            return
        try:
            self.cursor.executemany(query, rows)
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            raise InsertionError(e, table, ids[0] if ids else None)

    def get_section_from_context(self, sectionId: str):
        return self.__get_hsection(sectionId)

    def get_context_from_chunk(self, contextId: str):
        return self.__get_hcontext(contextId)

    def get_chunk(self, chunkId: str):
        return self.__get_hchunk(chunkId)

    def get_document_from_section(self, documentId: str):
        return self.__get_hdocument(documentId)

    def insert_chunks(self, Chunks: List[HChunk]):
        query = """insert into Chunks (chunkId , contextId , chunk , startoffset , endoffset)
                   values (? , ? , ? , ? , ?) on conflict(chunkId) do nothing"""
        rows = [
            (c.chunk_id, c.context_id, c.chunk, c.start_off_set, c.end_off_set)
            for c in Chunks
        ]
        self.__insert_many(query, rows, "Chunks", [c.chunk_id for c in Chunks])

    def insert_recursive_chunks(self, Chunks: List[RChunk]):
        query = """insert into RecursiveChunks (chunkId , documentId , chunk , startoffset , endoffset)
                   values (? , ? , ? , ? , ?) on conflict(chunkId) do nothing"""
        rows = [
            (
                c.chunk_id,
                c.meta_data.document_id,
                c.chunk,
                c.start_off_set,
                c.end_off_set,
            )
            for c in Chunks
        ]
        self.__insert_many(
            query, rows, "RecursiveChunks", [c.chunk_id for c in Chunks]
        )

    def insert_contexts(self, Contexts: List[Context]):
        query = """insert into Contexts (contextId , sectionId , context , contextLength , startoffset , endoffset)
                   values (? , ? , ? , ? , ? , ?) on conflict(contextId) do nothing"""
        rows = [
            (
                c.contextId,
                c.sectionId,
                c.context,
                c.contextLen,
                c.startOffSet,
                c.endOffSet,
            )
            for c in Contexts
        ]
        self.__insert_many(query, rows, "Context", [c.contextId for c in Contexts])

    def insert_documents(self, Documents: List[Document]):
        query = """insert into Documents (documentId , documentName) values (? , ?)
                   on conflict(documentId) do nothing"""
        rows = [(d.documentId, d.documentName) for d in Documents]
        self.__insert_many(
            query, rows, "Documents", [d.documentId for d in Documents]
        )

    def insert_sections(self, Sections: List[Section]):
        query = """insert into Sections (sectionId , documentId , sectionName , content , contentLength , startoffset , endoffset)
                   values (? , ? , ? , ? , ? , ? , ?) on conflict(sectionId) do nothing"""
        rows = [
            (
                s.sectionId,
                s.documentId,
                s.sectionName,
                s.content,
                s.contentLength,
                s.startOffSet,
                s.endOffSet,
            )
            for s in Sections
        ]
        self.__insert_many(query, rows, "Section", [s.sectionId for s in Sections])

    def close(self):
        self.cursor.close()
        self.connection.close()
