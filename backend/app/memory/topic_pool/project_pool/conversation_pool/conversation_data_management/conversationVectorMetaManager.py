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
        self.conversation_dir = Path(conversation_path)
        self.conversation_dir.mkdir(parents=True, exist_ok=True)

        # Store all new schema tables in the summary DB
        self.db_path = self.conversation_dir / f"{project_id}_conversation.db"
        self._init_db()

    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        cursor = self.conn.cursor()

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
                len_of_the_summary INTEGER NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summary_snapshot_map (
                cumulative_vector_id INTEGER NOT NULL,
                summary_vector_id INTEGER NOT NULL,
                FOREIGN KEY (cumulative_vector_id) REFERENCES cumulative_vector_meta_data(cumulative_vector_id),
                FOREIGN KEY (summary_vector_id) REFERENCES summary_vector_meta_data(summary_vector_id),
                UNIQUE (cumulative_vector_id, summary_vector_id)
            )
        """)

        self.conn.commit()

    def batch_insert_summary_chunks(self, records: List[Tuple[str, str, str, str]]):
        """records: [(chunk_id, chunk, created_at, chunker_type), ...]

        Idempotent. This repository shares its database file — and this table —
        with FullConversationRepository, so the chunks a snapshot covers are
        normally already present, written by append_turn(). Re-inserting them
        raised UNIQUE constraint failed and aborted every snapshot taken over
        stored turns; the existing rows are already correct, so ignore the clash.
        """
        cursor = self.conn.cursor()
        try:
            cursor.executemany(
                "INSERT OR IGNORE INTO summary_chunks (chunk_id, chunk, created_at, chunker_type) VALUES (?, ?, ?, ?)",
                records,
            )
            self.conn.commit()
        except sqlite3.Error as e:
            self.conn.rollback()
            raise e

    def batch_insert_summary_vector_meta_data(
        self, records: List[Tuple[int, str, str]]
    ):
        """records: [(summary_vector_id, chunk_id, project_id), ...]

        Idempotent. summary_vector_id is derived from chunk content, and
        consecutive snapshot windows overlap by design, so the same chunk
        legitimately reappears in a later snapshot with the same id.
        """
        new_records = [(int(r[0]), r[1], r[2]) for r in records]
        cursor = self.conn.cursor()
        try:
            cursor.executemany(
                "INSERT OR IGNORE INTO summary_vector_meta_data (summary_vector_id, chunk_id, project_id) VALUES (?, ?, ?)",
                new_records,
            )
            self.conn.commit()
        except sqlite3.Error as e:
            self.conn.rollback()
            raise e

    def get_summary_vector_meta_data(self, summary_vector_id: int):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT summary_vector_id, chunk_id, project_id FROM summary_vector_meta_data WHERE summary_vector_id = ?",
            (int(summary_vector_id),),
        )
        return cursor.fetchone()

    def batch_get_summary_vector_meta_data(self, summary_vector_ids: List[int]):
        if not summary_vector_ids:
            return []

        int_ids = [int(i) for i in summary_vector_ids]
        cursor = self.conn.cursor()
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
        cursor = self.conn.cursor()
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
            self.conn.commit()
        except sqlite3.Error as e:
            self.conn.rollback()
            raise e

    def batch_insert_cumulative_vector_meta_data(
        self, records: List[Tuple[int, str, str, str, str]]
    ):
        """records: [(cumulative_vector_id, cumulative_summary, created_at, project_id, len_of_the_summary), ...]"""
        new_records = [(int(r[0]), r[1], r[2], r[3], str(r[4])) for r in records]
        cursor = self.conn.cursor()
        try:
            cursor.executemany(
                "INSERT INTO cumulative_vector_meta_data (cumulative_vector_id, cumulative_summary, created_at, project_id, len_of_the_summary) VALUES (?, ?, ?, ?, ?)",
                new_records,
            )
            self.conn.commit()
        except sqlite3.Error as e:
            self.conn.rollback()
            raise e

    def get_cumulative_vector_meta_data_ids(self):
        cursor = self.conn.cursor()
        # datetime() truncates to whole seconds, so snapshots taken in the same
        # second tie and their order becomes arbitrary — which matters because
        # SnapShot's cursors index into this list. The raw TEXT tiebreaker
        # recovers sub-second precision from the ISO-8601 timestamps we write,
        # while datetime() stays the primary key for robustness to older rows
        # stored in looser formats.
        cursor.execute(
            "SELECT cumulative_vector_id FROM cumulative_vector_meta_data ORDER BY datetime(created_at), created_at;",
        )
        return cursor.fetchall()

    def get_cumulative_vector_meta_data(self, cumulative_vector_id: int):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT cumulative_vector_id, cumulative_summary, created_at, project_id, len_of_the_summary FROM cumulative_vector_meta_data WHERE cumulative_vector_id = ? order by created_at desc",
            (int(cumulative_vector_id),),
        )
        return cursor.fetchone()

    def get_latest_summary(self) -> str | None:
        cursor = self.conn.cursor()
        # See get_cumulative_vector_meta_data_ids: datetime() alone truncates to
        # seconds, so two snapshots in the same second would make "latest"
        # arbitrary and the rolling summary could pick up the wrong predecessor.
        cursor.execute("""
        SELECT cumulative_summary
        FROM cumulative_vector_meta_data
        ORDER BY datetime(created_at) DESC, created_at DESC
        LIMIT 1
        """)
        row = cursor.fetchone()
        return row[0] if row is not None else None

    def batch_get_cumulative_vector_meta_data(self, cumulative_vector_ids: List[int]):
        if not cumulative_vector_ids:
            return []

        int_ids = [int(i) for i in cumulative_vector_ids]
        cursor = self.conn.cursor()
        placeholders = ",".join(["?"] * len(int_ids))
        cursor.execute(
            f"SELECT cumulative_vector_id, cumulative_summary, created_at, project_id, len_of_the_summary FROM cumulative_vector_meta_data WHERE cumulative_vector_id IN ({placeholders})",
            tuple(int_ids),
        )
        return cursor.fetchall()

    def insert_map_table(self, cumulative_vector_id: int, summary_vector_id: int):
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO summary_snapshot_map (cumulative_vector_id, summary_vector_id) VALUES (?, ?)",
                (int(cumulative_vector_id), int(summary_vector_id)),
            )
            self.conn.commit()
        except sqlite3.Error as e:
            self.conn.rollback()
            raise e

    def batch_insert_map_table(self, records: List[Tuple[int, int]]):
        """records: [(cumulative_vector_id, summary_vector_id), ...]

        Idempotent against the UNIQUE (cumulative_vector_id, summary_vector_id)
        constraint, so re-mapping an overlapping chunk is a no-op.
        """
        new_records = []
        for i in range(len(records)):
            new_records.append((int(records[i][0]), int(records[i][1])))
        cursor = self.conn.cursor()
        try:
            cursor.executemany(
                "INSERT OR IGNORE INTO summary_snapshot_map (cumulative_vector_id, summary_vector_id) VALUES (?, ?)",
                new_records,
            )
            self.conn.commit()
        except sqlite3.Error as e:
            self.conn.rollback()
            raise e

    def insert_snapshot(
        self,
        chunks: List[Tuple[str, str, str, str]],
        cumulative_row: Tuple[int, str, str, str, int],
        summary_vector_rows: List[Tuple[int, str, str]],
        map_rows: List[Tuple[int, int]],
    ):
        """Write one complete snapshot in a single transaction.

        The four inserts used to be separate public calls, each committing on
        its own. A failure partway through left a snapshot that half-existed —
        chunks and vector metadata committed with no cumulative row to reach
        them by — and every retry added more orphans. Batched here, the whole
        snapshot lands or none of it does.

        Ordering matters within the transaction: summary_chunks is the FK parent
        of summary_vector_meta_data, which is in turn referenced by
        summary_snapshot_map along with cumulative_vector_meta_data.
        """
        cursor = self.conn.cursor()
        try:
            cursor.executemany(
                "INSERT OR IGNORE INTO summary_chunks (chunk_id, chunk, created_at, chunker_type) VALUES (?, ?, ?, ?)",
                chunks,
            )
            cursor.execute(
                "INSERT INTO cumulative_vector_meta_data (cumulative_vector_id, cumulative_summary, created_at, project_id, len_of_the_summary) VALUES (?, ?, ?, ?, ?)",
                (
                    int(cumulative_row[0]),
                    cumulative_row[1],
                    cumulative_row[2],
                    cumulative_row[3],
                    int(cumulative_row[4]),
                ),
            )
            cursor.executemany(
                "INSERT OR IGNORE INTO summary_vector_meta_data (summary_vector_id, chunk_id, project_id) VALUES (?, ?, ?)",
                [(int(r[0]), r[1], r[2]) for r in summary_vector_rows],
            )
            cursor.executemany(
                "INSERT OR IGNORE INTO summary_snapshot_map (cumulative_vector_id, summary_vector_id) VALUES (?, ?)",
                [(int(r[0]), int(r[1])) for r in map_rows],
            )
            self.conn.commit()
        except sqlite3.Error:
            self.conn.rollback()
            raise

    def get_highest_summarised_sequence(self) -> int | None:
        """Highest conversation sequence_number any snapshot has covered.

        No column records a snapshot's range, but summary_vector_meta_data holds
        one row per covered chunk and full_conversation lives in this same
        database file, so the watermark is a single join. Returns None when
        nothing has been summarised yet.

        The table check keeps this usable when the metadata repository is
        constructed on its own, before any conversation rows exist.
        """
        cursor = self.conn.cursor()
        has_conversation = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='full_conversation'"
        ).fetchone()
        if has_conversation is None:
            return None

        cursor.execute(
            """
            SELECT MAX(f.sequence_number)
            FROM summary_vector_meta_data AS s
            JOIN full_conversation AS f ON f.chunk_id = s.chunk_id
            WHERE s.project_id = ?
            """,
            (self.project_id,),
        )
        row = cursor.fetchone()
        return row[0] if row is not None and row[0] is not None else None

    def get_summary_vector_ids_from_map(self, cumulative_vector_id: int) -> List[int]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT summary_vector_id FROM summary_snapshot_map WHERE cumulative_vector_id = ?",
            (int(cumulative_vector_id),),
        )
        return [row[0] for row in cursor.fetchall()]

    def close(self):
        # getattr, not self.conn: if sqlite3.connect failed inside _init_db the
        # attribute never existed, and __del__ would then raise AttributeError
        # and bury the real construction error under "Exception ignored in".
        # Safe to call more than once — the repository may be shared.
        conn = getattr(self, "conn", None)
        if conn is not None:
            conn.close()

    def __del__(self):
        self.close()
