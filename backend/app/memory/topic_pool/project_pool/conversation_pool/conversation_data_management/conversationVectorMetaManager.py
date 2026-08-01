import sqlite3
import threading
from pathlib import Path
from typing import List, Tuple


class ConversationVectorMetaDataManager:
    def __init__(
        self,
        full_conversation_dir: str | Path,
        summary_dir: str | Path,
        project_id: str,
    ) -> None:
        self.project_id = project_id

        # Ensure directories exist
        self.full_conversation_dir = Path(full_conversation_dir)
        self.summary_dir = Path(summary_dir)
        self.full_conversation_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)

        self.full_db_path = (
            self.full_conversation_dir / f"{project_id}_full_conversation.db"
        )
        self.summary_db_path = self.summary_dir / f"{project_id}_summary_metadata.db"

        self._init_db(self.full_db_path)
        self._init_db(self.summary_db_path)

    def _init_db(self, db_path: Path):
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    vector_id INTEGER PRIMARY KEY,
                    chunk_id TEXT NOT NULL,
                    chunk TEXT NOT NULL
                )
            """)
            conn.commit()

    def _insert(self, db_path: Path, vector_id: int, chunk_id: str, chunk: str):
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    INSERT INTO chunks (vector_id, chunk_id, chunk)
                    VALUES (?, ?, ?)
                """,
                    (vector_id, chunk_id, chunk),
                )
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise e

    def _batch_insert(self, db_path: Path, records: List[Tuple[int, str, str]]):
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.executemany(
                    """
                    INSERT INTO chunks (vector_id, chunk_id, chunk)
                    VALUES (?, ?, ?)
                """,
                    records,
                )
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise e

    def insert_full_conversation_chunk(self, vector_id: int, chunk_id: str, chunk: str):
        self._insert(self.full_db_path, vector_id, chunk_id, chunk)

    def insert_summary_chunk(self, vector_id: int, chunk_id: str, chunk: str):
        self._insert(self.summary_db_path, vector_id, chunk_id, chunk)

    def batch_insert_full_conversation_chunks(
        self, records: List[Tuple[int, str, str]]
    ):
        """
        records should be a list of tuples: (vector_id, chunk_id, chunk)
        """
        self._batch_insert(self.full_db_path, records)

    def batch_insert_summary_chunks(self, records: List[Tuple[int, str, str]]):
        """
        records should be a list of tuples: (vector_id, chunk_id, chunk)
        """
        self._batch_insert(self.summary_db_path, records)

    def get_full_conversation_chunk(self, vector_id: int):
        with sqlite3.connect(self.full_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT vector_id, chunk_id, chunk FROM chunks WHERE vector_id = ?",
                (vector_id,),
            )
            return cursor.fetchone()

    def get_summary_chunk(self, vector_id: int):
        with sqlite3.connect(self.summary_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT vector_id, chunk_id, chunk FROM chunks WHERE vector_id = ?",
                (vector_id,),
            )
            return cursor.fetchone()
