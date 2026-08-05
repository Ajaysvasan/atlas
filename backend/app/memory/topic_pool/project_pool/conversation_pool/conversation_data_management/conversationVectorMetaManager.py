"""

Need to have cumulative summary


how do I map the vector_id from the cumulative summary to the summary vector?

have the cumulative summary vector id as the forgein key

summary vectors can be stored in one table

cumulative vectors are stored in another table

have a fact table that maps the cumulative vector and the summary vector


my cumulative vector table should contain the following things
1. cumulative_vector_id : int
2. cumulative_summary : str
3. created_at : date

while retriving it should ordered by created_time

and then I need to get all the summary_vector_ids from the summary table , so I am going to use a fact table

the fact table will contain the following things
1. cumulative_summary_vector_id
2. summary_vector_ids

The project_id is already in the summary table , so I can just use a join to do that

"""

import sqlite3
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
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cumulative_summary_vector (
                    cumulative_summary_vector_id INTEGER PRIMARY KEY,
                    created_at DATETIME NOT NULL,
                    cumulative_summary TEXT NOT NULL
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

    def __insert_cumulative_summary_vector(
        self,
        cumulative_summary_vector_id: int,
        created_at: str,
        cumulative_summary: str,
    ):
        with sqlite3.connect(self.summary_db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "insert into cumulative_summary_vector(cumulative_summary_vector_id , cumulative_summary , created_at) values (? , ? , ?);",
                    (cumulative_summary_vector_id, cumulative_summary, created_at),
                )
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise e

    def __search_cumulative_summary_vector(self, cumulative_vector_id: int):
        with sqlite3.connect(self.summary_db_path) as connect:
            cursor = connect.cursor()
            cursor.execute(
                """select cumulative_summary_vector_id from cumulative_summary_vector where cumulative_summary_vector_id = ?""",
                (cumulative_vector_id,),
            )
            return cursor.fetchone()

    def __get_all_cumulative_summary_vector(self):
        with sqlite3.connect(self.summary_db_path) as connect:
            cursor = connect.cursor()
            cursor.execute(
                """select cumulative_summary_vector_id from cumulative_summary_vector order by created_at""",
            )
            return cursor.fetchall()

    def insert_cumulative_vector_meta_data(
        self,
        cumulative_summary_vector_id: int,
        created_at: str,
        cumulative_summary: str,
    ):
        self.__insert_cumulative_summary_vector(
            cumulative_summary_vector_id, created_at, cumulative_summary
        )

    def search_cumulative_summary_vector_meta_data(self, cumulative_vector_id: int):
        return self.__search_cumulative_summary_vector(cumulative_vector_id)

    def get_all_cumulative_summary(self):
        return self.__get_all_cumulative_summary_vector()
