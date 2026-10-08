import hashlib
import sqlite3
from pathlib import Path
from typing import List, NamedTuple, Tuple

from config import get_logger
from memory.identifiers import require_identifier
from memory.memory_database import MemoryDatabase, Schema
from memory.memory_pool_exceptions import ProjectNotFound
from memory.topic_pool.project_pool.project_data_repo.project_meta_data import (
    SCHEMA as PROJECT_SCHEMA,
    is_registered,
)
from storage.timestamps import utc_now

logger = get_logger(__name__)

CHUNKER_TYPE_TURN = "turn"

_TURN_QUERY = """
    SELECT f.sequence_number, f.role, s.chunk, f.created_at, f.chunk_id
    FROM full_conversation AS f
    JOIN summary_chunks AS s
      ON s.chunk_id = f.chunk_id AND s.conversation_id = f.conversation_id
    WHERE f.conversation_id = ?
"""


class Turn(NamedTuple):
    """One conversation turn as read back: who said what, and where it sits."""

    sequence_number: int
    role: str
    text: str
    created_at: str
    chunk_id: str

def _create_tables(cursor: sqlite3.Cursor) -> None:
    # This module owns summary_chunks: turns are its rows, and full_conversation
    # points at them. ConversationVectorMetaDataRepository writes snapshot
    # chunks into it too, but no longer creates it (bug 4.65).
    cursor.execute("""
        create table if not exists summary_chunks(
            chunk_id text primary key,
            conversation_id text not null,
            chunk text not null,
            created_at text not null,
            chunker_type text not null,
            unique (chunk_id, conversation_id)
        )
    """)
    # The pair is the foreign key, so a turn and its text cannot disagree about
    # which conversation they belong to.
    cursor.execute("""
        create table if not exists full_conversation(
            project_id text not null references project_table(project_id),
            conversation_id text not null,
            sequence_number integer not null,
            chunk_id text not null,
            role text not null check (role in ('user', 'assistant', 'system')),
            created_at text not null,
            primary key (conversation_id, sequence_number),
            foreign key (chunk_id, conversation_id)
                references summary_chunks(chunk_id, conversation_id)
        )
    """)
    # get_sequence_number() looks a turn up by chunk_id, which the primary key
    # (conversation_id, sequence_number) cannot serve.
    cursor.execute(
        "create index if not exists idx_full_conversation_chunk "
        "on full_conversation(chunk_id)"
    )


SCHEMA = Schema("conversation_turns", _create_tables, requires=(PROJECT_SCHEMA,))


class FullConversationRepository:
    def __init__(
        self,
        project_id: str,
        project_name: str,
        conversation_id: str,
        database: MemoryDatabase | str | Path | None = None,
    ):
        self.project_id = project_id
        self.project_name = project_name
        self.conversation_id = require_identifier(conversation_id, "conversation_id")
        self.database = MemoryDatabase.of(database)
        self.database.ensure(SCHEMA)

    def __refused(self, error: sqlite3.IntegrityError) -> Exception:
        if "FOREIGN KEY" in str(error) and not is_registered(self.project_id, self.database):
            return ProjectNotFound(self.project_id)
        return error

    def __add_chunks(
        self,
        full_conversaton_meta_datas: List[Tuple[str, str, int, str, str]],
        chunks: List[Tuple[str, str , str, str, str]],
    ) -> None:
        try:
            with self.database.writing() as cursor:
                # summary_chunks is the FK parent, so its rows must land first.
                cursor.executemany(
                    "INSERT INTO summary_chunks(chunk_id , conversation_id ,  chunk , created_at , chunker_type) VALUES (? , ? ,? , ? , ?);",
                    chunks,
                )
                cursor.executemany(
                    "INSERT INTO full_conversation(project_id , conversation_id, sequence_number , chunk_id , role , created_at) VALUES (? , ? , ? , ? , ?  , ?);",
                    full_conversaton_meta_datas,
                )
        except sqlite3.IntegrityError as error:
            raise self.__refused(error) from error

    def __make_chunk_id(self, sequence_number: int, text: str) -> str:
        """Deterministic, collision-free id for one conversation turn."""
        payload = (
            f"{self.project_id}\x00{self.conversation_id}\x00"
            f"{sequence_number}\x00{text}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def __next_sequence_number(self, cursor) -> int:
        cursor.execute(
            "SELECT COALESCE(MAX(sequence_number), 0) FROM full_conversation "
            "WHERE conversation_id = ?;",
            (self.conversation_id,),
        )
        return cursor.fetchone()[0] + 1

    def __append_turns(self, turns: List[Tuple[str, str]]) -> List[int]:
        """Append (role, text) turns, allocating sequence numbers atomically."""
        if not turns:
            return []

        created_at = utc_now()
        try:
            # BEGIN IMMEDIATE, through writing(): the MAX() read and the INSERT
            # cannot interleave with another writer and reuse a sequence_number.
            with self.database.writing() as cursor:
                first_sequence = self.__next_sequence_number(cursor)

                chunk_rows = []
                meta_rows = []
                sequences = []
                for offset, (role, text) in enumerate(turns):
                    sequence = first_sequence + offset
                    chunk_id = self.__make_chunk_id(sequence, text)
                    chunk_rows.append(
                        (chunk_id, self.conversation_id, text, created_at,
                         CHUNKER_TYPE_TURN)
                    )
                    meta_rows.append(
                        (self.project_id, self.conversation_id, sequence,
                         chunk_id, role, created_at)
                    )
                    sequences.append(sequence)

                cursor.executemany(
                    "INSERT INTO summary_chunks(chunk_id , conversation_id, chunk , created_at , chunker_type) VALUES (? , ? , ?, ? , ?);",
                    chunk_rows,
                )
                cursor.executemany(
                    "INSERT INTO full_conversation(project_id , conversation_id,  sequence_number , chunk_id , role , created_at) VALUES (? , ? , ? , ? , ? , ?);",
                    meta_rows,
                )
            return sequences
        except sqlite3.IntegrityError as error:
            raise self.__refused(error) from error

    def __get_sequence_number(self, chunk_id: str) -> int | None:
        with self.database.reading() as cursor:
            cursor.execute(
                f"""SELECT sequence_number from full_conversation where chunk_id = ? and conversation_id = ?;""",
                (chunk_id,self.conversation_id),
            )
            row = cursor.fetchone()
            return row[0] if row is not None else None

    def __get_last_n_chunks(self, n: int):
        # A negative n would become LIMIT -1, which SQLite reads as "no limit"
        # and would return the entire conversation.
        if n <= 0:
            return []
        with self.database.reading() as cursor:
            cursor.execute(
                """
                SELECT chunk FROM (
                    SELECT s.chunk AS chunk, f.sequence_number AS sequence_number
                    FROM summary_chunks AS s
                    JOIN full_conversation AS f
                    ON f.chunk_id = s.chunk_id
                    AND f.conversation_id = s.conversation_id
                    WHERE f.conversation_id = ?
                    ORDER BY f.sequence_number DESC
                    LIMIT ?
                )
                ORDER BY sequence_number
            """,
                (self.conversation_id , n),
            )
            return cursor.fetchall()

    def __get_ranged_chunks(self, start: int, end: int):
        with self.database.reading() as cursor:
            cursor.execute(
                """
                select s.chunk
                from summary_chunks as s
                join full_conversation as f
                on s.chunk_id = f.chunk_id
                and s.conversation_id = f.conversation_id
                where f.conversation_id = ?
                and f.sequence_number >= ? and f.sequence_number <= ?
                order by f.sequence_number
            """,
                (self.conversation_id, start, end),
            )

            return cursor.fetchall()

    def __get_ranged_rows(self, start: int, end: int):
        """Full chunk rows for a range, not just the text."""
        with self.database.reading() as cursor:
            cursor.execute(
                """
                select s.chunk_id , s.chunk , s.created_at , s.chunker_type
                from summary_chunks as s
                join full_conversation as f
                on s.chunk_id = f.chunk_id
                and s.conversation_id = f.conversation_id
                where f.conversation_id = ?
                and f.sequence_number >= ? and f.sequence_number <= ?
                order by f.sequence_number
            """,
                (self.conversation_id, start, end),
            )
            return cursor.fetchall()

    def __get_messages_after(self, sequence_number: int):
        with self.database.reading() as cursor:
            cursor.execute(
                """
            select chunk 
            from summary_chunks as s
            join full_conversation as f
            on f.chunk_id = s.chunk_id
            and f.conversation_id = s.conversation_id
            where f.conversation_id = ?
            and f.sequence_number > ?
            order by f.sequence_number;
            """,
                (self.conversation_id, sequence_number),
            )
            return cursor.fetchall()

    def __get_all_from_conversation(self):

        with self.database.reading() as cursor:
            cursor.execute("""
            select chunk
            from summary_chunks as s
            join full_conversation as f
            on f.chunk_id = s.chunk_id
            and f.conversation_id = s.conversation_id
            where f.conversation_id = ?
            order by f.sequence_number
            """, (self.conversation_id,))
            return cursor.fetchall()

    def __select_turns(self, clause: str, params: tuple = ()) -> List[Turn]:
        # _TURN_QUERY already carries `WHERE f.conversation_id = ?`, so its
        # parameter leads and every clause here continues with AND.
        with self.database.reading() as cursor:
            rows = cursor.execute(
                _TURN_QUERY + clause, (self.conversation_id, *params)
            ).fetchall()
        return [Turn(*row) for row in rows]

    def __get_conversation_size(self):
        with self.database.reading() as cursor:
            cursor.execute("""
            select count(chunk) 
            from summary_chunks as s
            join full_conversation as f
            on f.chunk_id = s.chunk_id
            and f.conversation_id = s.conversation_id
            where f.conversation_id = ?
            """, (self.conversation_id,))
            return cursor.fetchone()[0]

    # Public APIs

    def add(
        self,
        full_conversaton_meta_datas: List[Tuple[str , str, int, str, str]],
        chunks: List[Tuple[str, str ,str, str, str]],
    ) -> None:
        """it supports only batch insertion , since we add the chunks only after a conversation , so no individual insertion is needed rather batch insertion is enough"""
        self.__add_chunks(full_conversaton_meta_datas, chunks)

    def append_turns(self, turns: List[Tuple[str, str]]) -> List[int]:
        """Append conversation turns and return their allocated sequence numbers."""
        return self.__append_turns(turns)

    def next_sequence_number(self) -> int:
        """The sequence_number the next appended turn will receive."""
        with self.database.reading() as cursor:
            return self.__next_sequence_number(cursor)

    def get_sequence_number(self, chunk_id: str) -> int:
        return self.__get_sequence_number(chunk_id)

    def get_n_chunks(self, n: int):
        return self.__get_last_n_chunks(n)

    def get_ranged_chunks(self, start: int, end: int):
        return self.__get_ranged_chunks(start, end)

    def get_ranged_rows(self, start: int, end: int):
        """[(chunk_id, chunk, created_at, chunker_type)] for the range."""
        return self.__get_ranged_rows(start, end)

    def get_sequence_after(self, sequence_number: int):
        return self.__get_messages_after(sequence_number)

    def fetch_all(self):
        return self.__get_all_from_conversation()

    def get_turns(self, start: int, end: int) -> List[Turn]:
        """Turns with start <= sequence_number <= end, in conversation order."""
        return self.__select_turns(
            "AND f.sequence_number >= ? AND f.sequence_number <= ? "
            "ORDER BY f.sequence_number",
            (start, end),
        )

    def get_last_n_turns(self, n: int) -> List[Turn]:
        """The newest n turns, returned oldest first."""
        # A negative n would become LIMIT -1, which SQLite reads as "no limit"
        # and would return the entire conversation.
        if n <= 0:
            return []
        turns = self.__select_turns(
            "ORDER BY f.sequence_number DESC LIMIT ?", (n,)
        )
        turns.reverse()
        return turns

    def get_turns_after(self, sequence_number: int) -> List[Turn]:
        """Turns with sequence_number strictly greater than the one given."""
        return self.__select_turns(
            "AND f.sequence_number > ? ORDER BY f.sequence_number",
            (sequence_number,),
        )

    def get_all_turns(self) -> List[Turn]:
        return self.__select_turns("ORDER BY f.sequence_number")

    def get_conversation_size(self):
        return self.__get_conversation_size()
