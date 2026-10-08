import sqlite3
from pathlib import Path
from typing import List, Tuple

from config import get_logger
from memory.identifiers import require_identifier
from memory.memory_database import MemoryDatabase, Schema
from memory.memory_pool_exceptions import ProjectNotFound
from memory.topic_pool.project_pool.conversation_pool.fullconversation_repository.fullconversation_repository import (
    SCHEMA as TURN_SCHEMA,
)
from memory.topic_pool.project_pool.project_data_repo.project_meta_data import (
    SCHEMA as PROJECT_SCHEMA,
    is_registered,
)

logger = get_logger(__name__)


def _create_tables(cursor: sqlite3.Cursor) -> None:
    cursor.execute("""
        create table if not exists summary_vector_meta_data(
            summary_vector_id integer primary key,
            chunk_id text not null references summary_chunks(chunk_id),
            project_id text not null references project_table(project_id)
        )
    """)
    # The summarised watermark walks a conversation's turns and probes this
    # table by chunk_id for each: 5.3 ms per call without it, 3-5 us with it.
    cursor.execute(
        "create index if not exists idx_summary_vector_chunk "
        "on summary_vector_meta_data(chunk_id)"
    )
    # seq is the project snapshot's watermark, so it is unique per project.
    # Plain UNIQUE held only while every project had a database file of its own.
    cursor.execute("""
        create table if not exists cumulative_vector_meta_data(
            cumulative_vector_id integer primary key,
            conversation_id text not null,
            seq integer not null,
            cumulative_summary text not null,
            created_at text not null,
            project_id text not null references project_table(project_id),
            len_of_the_summary integer not null,
            unique (project_id, seq)
        )
    """)
    cursor.execute("""
        create table if not exists summary_snapshot_map(
            cumulative_vector_id integer not null
                references cumulative_vector_meta_data(cumulative_vector_id),
            summary_vector_id integer not null
                references summary_vector_meta_data(summary_vector_id),
            unique (cumulative_vector_id, summary_vector_id)
        )
    """)


SCHEMA = Schema(
    "conversation_snapshots", _create_tables, requires=(PROJECT_SCHEMA, TURN_SCHEMA)
)


class ConversationVectorMetaDataRepository:
    """Snapshot metadata for one conversation, in the memory database."""

    def __init__(
        self,
        project_id: str,
        conversation_id: str,
        database: MemoryDatabase | str | Path | None = None,
    ) -> None:
        self.project_id = project_id
        self.conversation_id = require_identifier(conversation_id, "conversation_id")
        self.database = MemoryDatabase.of(database)
        self.database.ensure(SCHEMA)

    def __refused(self, error: sqlite3.IntegrityError, project_id: str) -> Exception:
        if "FOREIGN KEY" in str(error) and not is_registered(project_id, self.database):
            return ProjectNotFound(project_id)
        return error

    def batch_insert_summary_chunks(self, records: List[Tuple[str, str, str, str]]):
        """records: [(chunk_id, chunk, created_at, chunker_type), ...]"""
        with self.database.writing() as cursor:
            cursor.executemany(
                "INSERT OR IGNORE INTO summary_chunks (chunk_id, conversation_id, chunk, created_at, chunker_type) VALUES (?, ?, ?, ?, ?)",
                [(r[0], self.conversation_id, r[1], r[2], r[3]) for r in records],
            )

    def batch_insert_summary_vector_meta_data(
        self, records: List[Tuple[int, str, str]]
    ):
        """records: [(summary_vector_id, chunk_id, project_id), ...]"""
        new_records = [(int(r[0]), r[1], r[2]) for r in records]
        with self.database.writing() as cursor:
            cursor.executemany(
                "INSERT OR IGNORE INTO summary_vector_meta_data (summary_vector_id, chunk_id, project_id) VALUES (?, ?, ?)",
                new_records,
            )

    def get_summary_vector_meta_data(self, summary_vector_id: int):
        with self.database.reading() as cursor:
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
        with self.database.reading() as cursor:
            cursor.execute(
                f"SELECT summary_vector_id, chunk_id, project_id FROM summary_vector_meta_data WHERE summary_vector_id IN ({placeholders})",
                tuple(int_ids),
            )
            return cursor.fetchall()

    def __next_seq(self, cursor, project_id: str) -> int:
        """The next snapshot sequence for this project.

        Not AUTOINCREMENT: that is only available on an INTEGER PRIMARY KEY, and
        `cumulative_vector_id` already holds that position with a derived hash,
        which is not monotonic. Allocated inside the caller's write transaction,
        which opens with BEGIN IMMEDIATE, so the read and the insert cannot
        interleave with another writer.
        """
        cursor.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM cumulative_vector_meta_data "
            "WHERE project_id = ?",
            (project_id,),
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
        try:
            with self.database.writing() as cursor:
                cursor.execute(
                    "INSERT INTO cumulative_vector_meta_data (cumulative_vector_id, conversation_id, seq, cumulative_summary, created_at, project_id, len_of_the_summary) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        int(cumulative_vector_id),
                        self.conversation_id,
                        self.__next_seq(cursor, project_id),
                        cumulative_summary,
                        created_at,
                        project_id,
                        str(len_of_the_summary),
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise self.__refused(error, project_id) from error

    def batch_insert_cumulative_vector_meta_data(
        self, records: List[Tuple[int, str, str, str, str]]
    ):
        """records: [(cumulative_vector_id, cumulative_summary, created_at, project_id, len_of_the_summary), ...]"""
        try:
            with self.database.writing() as cursor:
                # One row at a time: each takes the next seq of its own project.
                for r in records:
                    cursor.execute(
                        "INSERT INTO cumulative_vector_meta_data (cumulative_vector_id, conversation_id, seq, cumulative_summary, created_at, project_id, len_of_the_summary) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (int(r[0]), self.conversation_id,
                         self.__next_seq(cursor, r[3]), r[1], r[2], r[3], str(r[4])),
                    )
        except sqlite3.IntegrityError as error:
            raise self.__refused(error, self.project_id) from error

    def get_cumulative_vector_meta_data_ids(self):
        # Ordered by seq, which is allocated monotonically on write. It replaced
        # `datetime(created_at), created_at`: datetime() truncates to whole
        # seconds, so two snapshots in the same second tied and their order went
        # arbitrary — and SnapShot's cursors index into this list.
        with self.database.reading() as cursor:
            cursor.execute(
                "SELECT cumulative_vector_id FROM cumulative_vector_meta_data "
                "WHERE conversation_id = ? ORDER BY seq;",
                (self.conversation_id,),
            )
            return cursor.fetchall()

    def get_cumulative_vector_meta_data(self, cumulative_vector_id: int):
        with self.database.reading() as cursor:
            cursor.execute(
                "SELECT cumulative_vector_id, cumulative_summary, created_at, project_id, len_of_the_summary FROM cumulative_vector_meta_data WHERE cumulative_vector_id = ? order by created_at desc",
                (int(cumulative_vector_id),),
            )
            return cursor.fetchone()

    def get_latest_summary(self) -> str | None:
        with self.database.reading() as cursor:
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
        with self.database.reading() as cursor:
            cursor.execute(
                f"SELECT cumulative_vector_id, cumulative_summary, created_at, project_id, len_of_the_summary FROM cumulative_vector_meta_data WHERE cumulative_vector_id IN ({placeholders})",
                tuple(int_ids),
            )
            return cursor.fetchall()

    def insert_map_table(self, cumulative_vector_id: int, summary_vector_id: int):
        with self.database.writing() as cursor:
            cursor.execute(
                "INSERT INTO summary_snapshot_map (cumulative_vector_id, summary_vector_id) VALUES (?, ?)",
                (int(cumulative_vector_id), int(summary_vector_id)),
            )

    def batch_insert_map_table(self, records: List[Tuple[int, int]]):
        """records: [(cumulative_vector_id, summary_vector_id), ...]"""
        new_records = [(int(record[0]), int(record[1])) for record in records]
        with self.database.writing() as cursor:
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
        project_id = cumulative_row[3]
        try:
            with self.database.writing() as cursor:
                cursor.executemany(
                    "INSERT OR IGNORE INTO summary_chunks (chunk_id, conversation_id, chunk, created_at, chunker_type) VALUES (?, ?, ?, ?, ?)",
                    [(c[0], self.conversation_id, c[1], c[2], c[3]) for c in chunks],
                )
                cursor.execute(
                    "INSERT INTO cumulative_vector_meta_data (cumulative_vector_id, conversation_id, seq, cumulative_summary, created_at, project_id, len_of_the_summary) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        int(cumulative_row[0]),
                        self.conversation_id,
                        self.__next_seq(cursor, project_id),
                        cumulative_row[1],
                        cumulative_row[2],
                        project_id,
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
        except sqlite3.IntegrityError as error:
            raise self.__refused(error, project_id) from error
        logger.info(
            "Stored snapshot %s for project %s over %d chunk(s)",
            cumulative_row[0],
            cumulative_row[3],
            len(chunks),
        )

    def get_highest_summarised_sequence(self) -> int | None:
        """Highest sequence_number in this conversation that a snapshot has covered."""
        with self.database.reading() as cursor:
            cursor.execute(
                """
                SELECT MAX(f.sequence_number)
                FROM summary_vector_meta_data AS s
                JOIN full_conversation AS f ON f.chunk_id = s.chunk_id
                WHERE s.project_id = ? AND f.conversation_id = ?
                """,
                (self.project_id, self.conversation_id),
            )
            row = cursor.fetchone()
            return row[0] if row is not None and row[0] is not None else None

    def get_project_snapshots_since(self, seq: int) -> List[Tuple[int, int, str]]:
        """This project's snapshot summaries after `seq`, across every conversation.

        The project snapshot deliberately ignores conversation_id: it is about
        the project, not about any one conversation.
        """
        with self.database.reading() as cursor:
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
        with self.database.reading() as cursor:
            cursor.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM cumulative_vector_meta_data "
                "WHERE project_id = ?",
                (self.project_id,),
            )
            return cursor.fetchone()[0]

    def get_summary_vector_ids_from_map(self, cumulative_vector_id: int) -> List[int]:
        with self.database.reading() as cursor:
            cursor.execute(
                "SELECT summary_vector_id FROM summary_snapshot_map WHERE cumulative_vector_id = ?",
                (int(cumulative_vector_id),),
            )
            return [row[0] for row in cursor.fetchall()]

    def close(self) -> None:
        """Releases nothing: the database is shared and outlives its owners."""
