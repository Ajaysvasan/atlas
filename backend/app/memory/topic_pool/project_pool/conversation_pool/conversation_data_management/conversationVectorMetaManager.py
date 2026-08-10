import sqlite3
from pathlib import Path
from typing import List, Tuple


class ConversationVectorMetaDataRepository:
    def __init__(
        self,
        conversation_path: str | Path,
        project_id: str,
    ) -> None:
        self.project_id = project_id

        # Ensure directories exist
        self.summary_dir = Path(conversation_path)
        self.summary_dir.mkdir(parents=True, exist_ok=True)

        # Store all new schema tables in the summary DB
        self.db_path = self.summary_dir / f"{project_id}_conversation.db"
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS summary_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    chunk TEXT NOT NULL,
                    created_at DATE NOT NULL,
                    chunker_type TEXT NOT NULL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS summary_vector_meta_data (
                    summary_vector_id INTEGER PRIMARY KEY,
                    chunk_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    FOREIGN KEY (chunk_id) REFERENCES summary_chunks(chunk_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cumulative_vector_meta_data (
                    cumulative_vector_id INTEGER PRIMARY KEY,
                    cumulative_summary TEXT NOT NULL,
                    created_at DATE NOT NULL,
                    project_id TEXT NOT NULL,
                    len_of_the_summary TEXT NOT NULL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS summary_snapshot_map (
                    cumulative_vector_id INTEGER NOT NULL,
                    summary_vector_id INTEGER NOT NULL,
                    FOREIGN KEY (cumulative_vector_id) REFERENCES cumulative_vector_meta_data(cumulative_vector_id),
                    FOREIGN KEY (summary_vector_id) REFERENCES summary_vector_meta_data(summary_vector_id)
                )
            """)

            conn.commit()

    def batch_insert_summary_chunks(self, records: List[Tuple[str, str, str, str]]):
        """records: [(chunk_id, chunk, created_at, chunker_type), ...]"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.executemany(
                    "INSERT INTO summary_chunks (chunk_id, chunk, created_at, chunker_type) VALUES (?, ?, ?, ?)",
                    records,
                )
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise e

    def batch_insert_summary_vector_meta_data(
        self, records: List[Tuple[int, str, str]]
    ):
        """records: [(summary_vector_id, chunk_id, project_id), ...]"""
        new_records = [(int(r[0]), r[1], r[2]) for r in records]
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.executemany(
                    "INSERT INTO summary_vector_meta_data (summary_vector_id, chunk_id, project_id) VALUES (?, ?, ?)",
                    new_records,
                )
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise e

    def get_summary_vector_meta_data(self, summary_vector_id: int):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT summary_vector_id, chunk_id, project_id FROM summary_vector_meta_data WHERE summary_vector_id = ?",
                (int(summary_vector_id),),
            )
            return cursor.fetchone()

    def batch_get_summary_vector_meta_data(self, summary_vector_ids: List[int]):
        if not summary_vector_ids:
            return []
        
        int_ids = [int(i) for i in summary_vector_ids]
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            placeholders = ",".join(["?"] * len(int_ids))
            cursor.execute(
                f"SELECT summary_vector_id, chunk_id, project_id FROM summary_vector_meta_data WHERE summary_vector_id IN ({placeholders})",
                tuple(int_ids),
            )
            return cursor.fetchall()

    def insert_cumulative_vector_meta_data(
        self,
        cumulative_vector_id: int,
        cumulative_summary: str,
        created_at: str,
        project_id: str,
        len_of_the_summary: int,
    ):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO cumulative_vector_meta_data (cumulative_vector_id, cumulative_summary, created_at, project_id, len_of_the_summary) VALUES (?, ?, ?, ?, ?)",
                    (
                        int(cumulative_vector_id),
                        cumulative_summary,
                        created_at,
                        project_id,
                        str(len_of_the_summary),
                    ),
                )
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise e

    def batch_insert_cumulative_vector_meta_data(
        self, records: List[Tuple[int, str, str, str, str]]
    ):
        """records: [(cumulative_vector_id, cumulative_summary, created_at, project_id, len_of_the_summary), ...]"""
        new_records = [(int(r[0]), r[1], r[2], r[3], str(r[4])) for r in records]
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.executemany(
                    "INSERT INTO cumulative_vector_meta_data (cumulative_vector_id, cumulative_summary, created_at, project_id, len_of_the_summary) VALUES (?, ?, ?, ?, ?)",
                    new_records,
                )
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise e

    def get_cumulative_vector_meta_data_ids(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT cumulative_vector_id FROM cumulative_vector_meta_data order by created_at;",
            )
            return cursor.fetchall()

    def get_cumulative_vector_meta_data(self, cumulative_vector_id: int):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT cumulative_vector_id, cumulative_summary, created_at, project_id, len_of_the_summary FROM cumulative_vector_meta_data WHERE cumulative_vector_id = ? order by created_at desc",
                (int(cumulative_vector_id),),
            )
            return cursor.fetchone()

    def batch_get_cumulative_vector_meta_data(self, cumulative_vector_ids: List[int]):
        if not cumulative_vector_ids:
            return []
        
        int_ids = [int(i) for i in cumulative_vector_ids]
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            placeholders = ",".join(["?"] * len(int_ids))
            cursor.execute(
                f"SELECT cumulative_vector_id, cumulative_summary, created_at, project_id, len_of_the_summary FROM cumulative_vector_meta_data WHERE cumulative_vector_id IN ({placeholders})",
                tuple(int_ids),
            )
            return cursor.fetchall()

    def insert_map_table(self, cumulative_vector_id: int, summary_vector_id: int):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO summary_snapshot_map (cumulative_vector_id, summary_vector_id) VALUES (?, ?)",
                    (int(cumulative_vector_id), int(summary_vector_id)),
                )
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise e

    def batch_insert_map_table(self, records: List[Tuple[int, int]]):
        """records: [(cumulative_vector_id, summary_vector_id), ...]"""
        new_records = []
        for i in range(len(records)):
            new_records.append((int(records[i][0]), int(records[i][1])))
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.executemany(
                    "INSERT INTO summary_snapshot_map (cumulative_vector_id, summary_vector_id) VALUES (?, ?)",
                    new_records,
                )
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise e

    def get_summary_vector_ids_from_map(self, cumulative_vector_id: int) -> List[int]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT summary_vector_id FROM summary_snapshot_map WHERE cumulative_vector_id = ?",
                (int(cumulative_vector_id),),
            )
            return [row[0] for row in cursor.fetchall()]
