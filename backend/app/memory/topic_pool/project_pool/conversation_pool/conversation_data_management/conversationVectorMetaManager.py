import sqlite3
import threading
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Iterator, List, Tuple

from config import get_logger
from memory.sqlite_setup import (
    connect,
    enable_wal,
)

from memory.topic_pool.project_pool.conversation_pool.schema_migrations import (
    migrate,
)

from memory.identifiers import require_identifier

logger = get_logger(__name__)


class ConversationVectorMetaDataRepository:
    """Snapshot metadata for one project, in SQLite."""

    def __init__(
        self,
        conversation_path: str | Path,
        project_id: str,
        conversation_id: str,
    ) -> None:
        self._lock = threading.RLock()
        self._write_depth = 0
        self.project_id = project_id
        self.conversation_id = require_identifier(conversation_id, "conversation_id")

        self.conversation_dir = Path(conversation_path)
        self.conversation_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.conversation_dir / f"{project_id}_conversation.db"
        self._init_db()

    @contextmanager
    def _reading(self) -> Iterator[sqlite3.Cursor]:
        """A cursor held under the lock for as long as the caller needs it."""
        with self._lock:
            yield self.conn.cursor()

    @contextmanager
    def _writing(self) -> Iterator[sqlite3.Cursor]:
        """A cursor in an immediate transaction, committed or rolled back."""
        # BEGIN IMMEDIATE, not the implicit deferred transaction: _lock is this
        # instance's, and two repositories on one database file have one lock
        # each, so a read-then-write pair (seq allocation) could interleave
        # across them and both claim the same seq. The database's own write lock
        # is what serialises them. The depth counter keeps a nested _writing()
        # inside the outer transaction rather than starting a second one, which
        # SQLite refuses.
        with self._lock:
            cursor = self.conn.cursor()
            outermost = self._write_depth == 0
            if outermost:
                cursor.execute("BEGIN IMMEDIATE;")
            self._write_depth += 1
            try:
                yield cursor
            except BaseException:
                if outermost:
                    cursor.execute("ROLLBACK;")
                    logger.debug("Rolled back a write on %s", self.db_path)
                raise
            else:
                if outermost:
                    cursor.execute("COMMIT;")
            finally:
                self._write_depth -= 1

    def _init_db(self):
        # Before the connection is opened: this database is shared with
        # FullConversationRepository and either class may reach it first, so
        # both run the migration.
        migrate(self.db_path)
        # check_same_thread=False allows the shared instance; _lock makes it correct.
        # isolation_level=None: transactions are issued explicitly by _writing
        # so that BEGIN IMMEDIATE is what opens them.
        self.conn = connect(
            self.db_path, check_same_thread=False, isolation_level=None
        )
        self.journal_mode = enable_wal(self.conn, self.db_path)
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summary_chunks (
                chunk_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
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
                conversation_id TEXT NOT NULL,
                seq INTEGER NOT NULL UNIQUE,
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
        logger.debug(
            "Snapshot metadata store ready at %s (journal=%s)",
            self.db_path,
            self.journal_mode,
        )

    def batch_insert_summary_chunks(self, records: List[Tuple[str, str, str, str]]):
        """records: [(chunk_id, chunk, created_at, chunker_type), ...]"""
        with self._writing() as cursor:
            cursor.executemany(
                "INSERT OR IGNORE INTO summary_chunks (chunk_id, conversation_id, chunk, created_at, chunker_type) VALUES (?, ?, ?, ?, ?)",
                [(r[0], self.conversation_id, r[1], r[2], r[3]) for r in records],
            )

    def batch_insert_summary_vector_meta_data(
        self, records: List[Tuple[int, str, str]]
    ):
        """records: [(summary_vector_id, chunk_id, project_id), ...]"""
        new_records = [(int(r[0]), r[1], r[2]) for r in records]
        with self._writing() as cursor:
            cursor.executemany(
                "INSERT OR IGNORE INTO summary_vector_meta_data (summary_vector_id, chunk_id, project_id) VALUES (?, ?, ?)",
                new_records,
            )

    def get_summary_vector_meta_data(self, summary_vector_id: int):
        with self._reading() as cursor:
            cursor.execute(
                "SELECT summary_vector_id, chunk_id, project_id FROM summary_vector_meta_data WHERE summary_vector_id = ?",
                (int(summary_vector_id),),
            )
            return cursor.fetchone()

    def batch_get_summary_vector_meta_data(self, summary_vector_ids: List[int]):
        if not summary_vector_ids:
            return []

        int_ids = [int(i) for i in summary_vector_ids]
        placeholders = ",".join(["?"] * len(int_ids))
        with self._reading() as cursor:
            cursor.execute(
                f"SELECT summary_vector_id, chunk_id, project_id FROM summary_vector_meta_data WHERE summary_vector_id IN ({placeholders})",
                tuple(int_ids),
            )
            return cursor.fetchall()

    def __next_seq(self, cursor) -> int:
        """The next snapshot sequence for this database.

        Not AUTOINCREMENT: that is only available on an INTEGER PRIMARY KEY, and
        `cumulative_vector_id` already holds that position with a derived hash,
        which is not monotonic. Allocated inside the caller's write transaction,
        under the same lock, so the read and the insert cannot interleave.
        """
        cursor.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM cumulative_vector_meta_data"
        )
        return cursor.fetchone()[0]

    def insert_cumulative_vector_meta_data(
        self,
        cumulative_vector_id: int,
        cumulative_summary: str,
        created_at: str,
        project_id: str,
        len_of_the_summary: int,
    ):
        with self._writing() as cursor:
            cursor.execute(
                "INSERT INTO cumulative_vector_meta_data (cumulative_vector_id, conversation_id, seq, cumulative_summary, created_at, project_id, len_of_the_summary) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    int(cumulative_vector_id),
                    self.conversation_id,
                    self.__next_seq(cursor),
                    cumulative_summary,
                    created_at,
                    project_id,
                    str(len_of_the_summary),
                ),
            )

    def batch_insert_cumulative_vector_meta_data(
        self, records: List[Tuple[int, str, str, str, str]]
    ):
        """records: [(cumulative_vector_id, cumulative_summary, created_at, project_id, len_of_the_summary), ...]"""
        with self._writing() as cursor:
            first = self.__next_seq(cursor)
            cursor.executemany(
                "INSERT INTO cumulative_vector_meta_data (cumulative_vector_id, conversation_id, seq, cumulative_summary, created_at, project_id, len_of_the_summary) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (int(r[0]), self.conversation_id, first + offset,
                     r[1], r[2], r[3], str(r[4]))
                    for offset, r in enumerate(records)
                ],
            )

    def get_cumulative_vector_meta_data_ids(self):
        # Ordered by seq, which is allocated monotonically on write. It replaced
        # `datetime(created_at), created_at`: datetime() truncates to whole
        # seconds, so two snapshots in the same second tied and their order went
        # arbitrary — and SnapShot's cursors index into this list.
        with self._reading() as cursor:
            cursor.execute(
                "SELECT cumulative_vector_id FROM cumulative_vector_meta_data "
                "WHERE conversation_id = ? ORDER BY seq;",
                (self.conversation_id,),
            )
            return cursor.fetchall()

    def get_cumulative_vector_meta_data(self, cumulative_vector_id: int):
        with self._reading() as cursor:
            cursor.execute(
                "SELECT cumulative_vector_id, cumulative_summary, created_at, project_id, len_of_the_summary FROM cumulative_vector_meta_data WHERE cumulative_vector_id = ? order by created_at desc",
                (int(cumulative_vector_id),),
            )
            return cursor.fetchone()

    def get_latest_summary(self) -> str | None:
        with self._reading() as cursor:
            cursor.execute(
                """
                SELECT cumulative_summary
                FROM cumulative_vector_meta_data
                WHERE conversation_id = ?
                ORDER BY seq DESC
                LIMIT 1
                """,
                (self.conversation_id,),
            )
            row = cursor.fetchone()
            return row[0] if row is not None else None

    def batch_get_cumulative_vector_meta_data(self, cumulative_vector_ids: List[int]):
        if not cumulative_vector_ids:
            return []

        int_ids = [int(i) for i in cumulative_vector_ids]
        placeholders = ",".join(["?"] * len(int_ids))
        with self._reading() as cursor:
            cursor.execute(
                f"SELECT cumulative_vector_id, cumulative_summary, created_at, project_id, len_of_the_summary FROM cumulative_vector_meta_data WHERE cumulative_vector_id IN ({placeholders})",
                tuple(int_ids),
            )
            return cursor.fetchall()

    def insert_map_table(self, cumulative_vector_id: int, summary_vector_id: int):
        with self._writing() as cursor:
            cursor.execute(
                "INSERT INTO summary_snapshot_map (cumulative_vector_id, summary_vector_id) VALUES (?, ?)",
                (int(cumulative_vector_id), int(summary_vector_id)),
            )

    def batch_insert_map_table(self, records: List[Tuple[int, int]]):
        """records: [(cumulative_vector_id, summary_vector_id), ...]"""
        new_records = [(int(record[0]), int(record[1])) for record in records]
        with self._writing() as cursor:
            cursor.executemany(
                "INSERT OR IGNORE INTO summary_snapshot_map (cumulative_vector_id, summary_vector_id) VALUES (?, ?)",
                new_records,
            )

    def insert_snapshot(
        self,
        chunks: List[Tuple[str, str, str, str]],
        cumulative_row: Tuple[int, str, str, str, int],
        summary_vector_rows: List[Tuple[int, str, str]],
        map_rows: List[Tuple[int, int]],
    ):
        """Write one complete snapshot in a single transaction."""
        with self._writing() as cursor:
            cursor.executemany(
                "INSERT OR IGNORE INTO summary_chunks (chunk_id, conversation_id, chunk, created_at, chunker_type) VALUES (?, ?, ?, ?, ?)",
                [(c[0], self.conversation_id, c[1], c[2], c[3]) for c in chunks],
            )
            cursor.execute(
                "INSERT INTO cumulative_vector_meta_data (cumulative_vector_id, conversation_id, seq, cumulative_summary, created_at, project_id, len_of_the_summary) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    int(cumulative_row[0]),
                    self.conversation_id,
                    self.__next_seq(cursor),
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
        logger.info(
            "Stored snapshot %s for project %s over %d chunk(s)",
            cumulative_row[0],
            cumulative_row[3],
            len(chunks),
        )

    def get_highest_summarised_sequence(self) -> int | None:
        """Highest conversation sequence_number any snapshot has covered."""
        with self._reading() as cursor:
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

    def get_project_snapshots_since(self, seq: int) -> List[Tuple[int, int, str]]:
        """This project's snapshot summaries after `seq`, across every conversation.

        The project snapshot deliberately ignores conversation_id: it is about
        the project, not about any one conversation.
        """
        with self._reading() as cursor:
            cursor.execute(
                """
                SELECT seq, cumulative_vector_id, cumulative_summary
                FROM cumulative_vector_meta_data
                WHERE project_id = ? AND seq > ?
                ORDER BY seq
                """,
                (self.project_id, int(seq)),
            )
            return [(row[0], row[1], row[2]) for row in cursor.fetchall()]

    def get_highest_snapshot_seq(self) -> int:
        """The latest snapshot sequence in this project, or 0 if there are none."""
        with self._reading() as cursor:
            cursor.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM cumulative_vector_meta_data "
                "WHERE project_id = ?",
                (self.project_id,),
            )
            return cursor.fetchone()[0]

    def get_summary_vector_ids_from_map(self, cumulative_vector_id: int) -> List[int]:
        with self._reading() as cursor:
            cursor.execute(
                "SELECT summary_vector_id FROM summary_snapshot_map WHERE cumulative_vector_id = ?",
                (int(cumulative_vector_id),),
            )
            return [row[0] for row in cursor.fetchall()]

    def close(self):
        # getattr, not attribute access: if sqlite3.connect failed inside
        # _init_db the attribute never existed, and __del__ would then raise
        # AttributeError and bury the real construction error under
        # "Exception ignored in". Safe to call more than once — the repository
        # may be shared. The lock makes it wait for a write in flight on
        # another thread rather than closing the connection out from under it.
        conn = getattr(self, "conn", None)
        if conn is None:
            return
        lock = getattr(self, "_lock", None)
        with lock if lock is not None else nullcontext():
            conn.close()

    def __del__(self):
        self.close()
