import os
import sqlite3
from typing import List

from data_layer.datalayer_exceptions.datalayer_exceptions import InsertionError
from data_layer.ingestion.nodes.nodes import Context, Document, HChunk, Section


class Manager:
    def __init__(self, db_path: str, is_chunker_type_hierarchical: bool) -> None:
        self.is_chunker_type_hierarchical = is_chunker_type_hierarchical
        # Bug 1.3 fix: ensure parent directory exists before connecting
        parent_dir = os.path.dirname(db_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        # Bug 1.2 fix: allow cross-thread usage of this connection
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

    def _create_table(self):
        try:
            if self.is_chunker_type_hierarchical:
                self.__create_document_htable()
                self.__create_section_htable()
                self.__create_context_htable()
                self.__create_chunk_htable()
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

    def __insert_hchunk(self, ChunkObj: HChunk):
        chunkId = ChunkObj.chunk_id
        contextId = ChunkObj.context_id
        chunk = ChunkObj.chunk
        startoffset = ChunkObj.start_off_set
        endoffset = ChunkObj.end_off_set
        insertChunkQuery = """insert into Chunks (chunkId , contextId , chunk , startoffset , endoffset) values (? ,? ,?, ?, ? )"""
        self.cursor.execute(
            insertChunkQuery, (chunkId, contextId, chunk, startoffset, endoffset)
        )

    def __insert_hsection(self, SectionObj: Section):
        sectionId = SectionObj.sectionId
        documentId = SectionObj.documentId

        sectionName = SectionObj.sectionName
        content = SectionObj.content
        contentLength = SectionObj.contentLength

        startoffset = SectionObj.startOffSet
        endoffset = SectionObj.endOffSet

        insertSectionQuery = """insert into Sections (sectionId , documentId , sectionName , content , contentLength , startoffset ,endoffset) values (? ,? , ?, ? ,? ,? , ? )"""
        self.cursor.execute(
            insertSectionQuery,
            (
                sectionId,
                documentId,
                sectionName,
                content,
                contentLength,
                startoffset,
                endoffset,
            ),
        )

    def __insert_hdocument(self, DocumentObj: Document):
        documentName = DocumentObj.documentName
        documentId = DocumentObj.documentId
        insertDocumentQuery = (
            """insert into Documents (documentId , documentName) values (? ,  ?)"""
        )
        self.cursor.execute(insertDocumentQuery, (documentId, documentName))

    def __insert_hcontext(self, ContextObj: Context):
        contextId = ContextObj.contextId
        sectionId = ContextObj.sectionId

        context = ContextObj.context
        contextLength = ContextObj.contextLen

        startoffset = ContextObj.startOffSet
        endoffset = ContextObj.endOffSet
        insertContextQuery = """insert into Contexts (contextId , sectionId , context , contextLength , startoffset , endoffset) values(? , ? , ? , ? , ? , ? )"""
        self.cursor.execute(
            insertContextQuery,
            (contextId, sectionId, context, contextLength, startoffset, endoffset),
        )

    def get_section_from_context(self, sectionId: str):
        return self.__get_hsection(sectionId)

    def get_context_from_chunk(self, contextId: str):
        return self.__get_hcontext(contextId)

    def get_chunk(self, chunkId: str):
        return self.__get_hchunk(chunkId)

    def get_document_from_section(self, documentId: str):
        return self.__get_hdocument(documentId)

    def insert_chunks(self, Chunks: List[HChunk]):
        currentChunkId = None
        try:
            for ChunkObj in Chunks:
                currentChunkId = ChunkObj.chunk_id
                self.__insert_hchunk(ChunkObj)
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            raise InsertionError(e, "Chunks", currentChunkId)

    def insert_contexts(self, Contexts: List[Context]):
        currentContextId = None
        try:
            for ContextObj in Contexts:
                currentContextId = ContextObj.contextId
                self.__insert_hcontext(ContextObj)
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            raise InsertionError(e, "Context", currentContextId)

    def insert_documents(self, Documents: List[Document]):
        currentDocId = None
        try:
            for DocumentObj in Documents:
                currentDocId = DocumentObj.documentId
                self.__insert_hdocument(DocumentObj)
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            raise InsertionError(e, "Documents", currentDocId)

    def insert_sections(self, Sections: List[Section]):
        currentSectionId = None
        try:
            for SectionObj in Sections:
                currentSectionId = SectionObj.sectionId
                self.__insert_hsection(SectionObj)
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            raise InsertionError(e, "Section", currentSectionId)

    def close(self):
        self.cursor.close()
        self.connection.close()
