import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

# A conversation turn is stored whole: one turn -> one chunk. `chunker_type`
# records that provenance so a future splitting strategy can coexist with rows
# written today without a migration.
CHUNKER_TYPE_TURN = "turn"


def utc_now() -> str:
    """Canonical timestamp for every row this repository writes.

    created_at is a caller-supplied TEXT column, which is what made Bug 4.31
    possible. Stamping it here — in one ISO-8601 UTC format, sortable as text —
    removes the caller's ability to get it wrong. Ordering still keys off
    sequence_number; this is for display and auditing.
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class FullConversationRepository:
    def __init__(
        self, conversation_path: str | Path, project_id: str, project_name: str
    ):
        self.project_id = project_id
        self.project_name = project_name
        self.conversation_dir = Path(conversation_path)
        self.conversation_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.conversation_dir / f"{project_id}_conversation.db"
        self.__init_db()

    def __init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            cursor = conn.cursor()
            cursor.execute("""
            create table if not exists summary_chunks (
                chunk_id text primary key,
                chunk text not null,
                created_at date not null,
                chunker_type text not null
            )
            """)
            cursor.execute("""
            create table if not exists full_conversation(
                project_id text not null,
                sequence_number int primary key,
                chunk_id text not null,
                role text not null ,
                created_at date not null ,
                foreign key (chunk_id) references summary_chunks (chunk_id)
            )
            """)
            conn.commit()

    # [ (project_id , sequence_number , chunk_id , role , created_at)] -> full_conversaton_meta_datas
    # [(chunk_id , chunk , created_at , chunker_type)] -> chunks
    def __add_chunks(
        self,
        full_conversaton_meta_datas: List[Tuple[str, int, str, str]],
        chunks: List[Tuple[str, str, str, str]],
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            # full_conversation.chunk_id has an FK onto summary_chunks, so the
            # parent rows must land first. Previously the order was inverted and
            # only "worked" because this connection never enabled foreign_keys
            # (the PRAGMA in __init_db applies to that connection alone) — an
            # orphan meta row was accepted, then silently dropped by every
            # reader's JOIN, consuming its sequence_number forever.
            conn.execute("PRAGMA foreign_keys = ON;")
            try:
                cursor = conn.cursor()
                cursor.executemany(
                    """
                INSERT INTO summary_chunks(chunk_id , chunk , created_at , chunker_type) VALUES (? , ? , ? , ?);
                """,
                    chunks,
                )
                cursor.executemany(
                    """
        INSERT INTO full_conversation(project_id , sequence_number , chunk_id , role , created_at) VALUES (? , ? , ? , ?  , ?);
        """,
                    full_conversaton_meta_datas,
                )

                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise e

    def __make_chunk_id(self, sequence_number: int, text: str) -> str:
        """Deterministic, collision-free id for one conversation turn.

        Content alone is not enough: chunk_id is a PRIMARY KEY and a speaker may
        legitimately repeat themselves ("ok", "yes"), which would abort the whole
        batch on IntegrityError. Binding project_id and sequence_number makes the
        id unique per turn while staying reproducible. NUL separates the fields
        so they cannot run together into a colliding payload.
        """
        payload = f"{self.project_id}\x00{sequence_number}\x00{text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def __next_sequence_number(self, cursor) -> int:
        cursor.execute(
            "SELECT COALESCE(MAX(sequence_number), 0) FROM full_conversation;"
        )
        return cursor.fetchone()[0] + 1

    def __append_turns(self, turns: List[Tuple[str, str]]) -> List[int]:
        """Append (role, text) turns, allocating sequence numbers atomically."""
        if not turns:
            return []

        created_at = utc_now()
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            # BEGIN IMMEDIATE takes the write lock up front, so the MAX() read
            # and the INSERT cannot interleave with another writer and hand out
            # the same sequence_number twice.
            conn.execute("BEGIN IMMEDIATE;")
            try:
                cursor = conn.cursor()
                first_sequence = self.__next_sequence_number(cursor)

                chunk_rows = []
                meta_rows = []
                sequences = []
                for offset, (role, text) in enumerate(turns):
                    sequence = first_sequence + offset
                    chunk_id = self.__make_chunk_id(sequence, text)
                    chunk_rows.append(
                        (chunk_id, text, created_at, CHUNKER_TYPE_TURN)
                    )
                    meta_rows.append(
                        (self.project_id, sequence, chunk_id, role, created_at)
                    )
                    sequences.append(sequence)

                cursor.executemany(
                    "INSERT INTO summary_chunks(chunk_id , chunk , created_at , chunker_type) VALUES (? , ? , ? , ?);",
                    chunk_rows,
                )
                cursor.executemany(
                    "INSERT INTO full_conversation(project_id , sequence_number , chunk_id , role , created_at) VALUES (? , ? , ? , ? , ?);",
                    meta_rows,
                )
                conn.execute("COMMIT;")
                return sequences
            except sqlite3.Error:
                conn.execute("ROLLBACK;")
                raise
        finally:
            conn.close()

    def __get_sequence_number(self, chunk_id: str) -> int | None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""SELECT sequence_number from full_conversation where chunk_id = ?;""",
                (chunk_id,),
            )
            row = cursor.fetchone()
            return row[0] if row is not None else None

    def __get_last_n_chunks(self, n: int):
        # A negative n would become LIMIT -1, which SQLite reads as "no limit"
        # and would return the entire conversation.
        if n <= 0:
            return []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Take the newest n by sequence_number (DESC + LIMIT), then re-sort
            # ascending so the caller receives them in conversation order.
            cursor.execute(
                """
                SELECT chunk FROM (
                    SELECT s.chunk AS chunk, f.sequence_number AS sequence_number
                    FROM summary_chunks AS s
                    JOIN full_conversation AS f
                    ON f.chunk_id = s.chunk_id
                    ORDER BY f.sequence_number DESC
                    LIMIT ?
                )
                ORDER BY sequence_number
            """,
                (n,),
            )
            return cursor.fetchall()

    def __get_ranged_chunks(self, start: int, end: int):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                select s.chunk
                from summary_chunks as s
                join full_conversation as f
                on s.chunk_id = f.chunk_id
                where f.sequence_number >= ? and f.sequence_number <= ?
                order by f.sequence_number
            """,
                (start, end),
            )

            return cursor.fetchall()

    def __get_ranged_rows(self, start: int, end: int):
        """Full chunk rows for a range, not just the text.

        A snapshot has to record which chunks it covers, so it needs chunk_id,
        created_at and chunker_type alongside the text that get_ranged_chunks
        returns on its own.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                select s.chunk_id , s.chunk , s.created_at , s.chunker_type
                from summary_chunks as s
                join full_conversation as f
                on s.chunk_id = f.chunk_id
                where f.sequence_number >= ? and f.sequence_number <= ?
                order by f.sequence_number
            """,
                (start, end),
            )
            return cursor.fetchall()

    def __get_messages_after(self, sequence_number: int):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
            select chunk 
            from summary_chunks as s
            join full_conversation as f
            on f.chunk_id = s.chunk_id
            where f.sequence_number > ?
            order by f.sequence_number;
            """,
                (sequence_number,),
            )
            return cursor.fetchall()

    def __get_all(self):

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            select chunk
            from summary_chunks as s
            join full_conversation as f
            on f.chunk_id = s.chunk_id
            order by f.sequence_number
            """)
            return cursor.fetchall()

    def __get_size(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            select count(chunk) 
            from summary_chunks as s
            join full_conversation as f
            on f.chunk_id = s.chunk_id
            """)
            return cursor.fetchone()[0]

    # Public APIs

    def add(
        self,
        full_conversaton_meta_datas: List[Tuple[str, int, str, str]],
        chunks: List[Tuple[str, str, str, str]],
    ) -> None:
        """it supports only batch insertion , since we add the chunks only after a conversation , so no individual insertion is needed rather batch insertion is enough
        [ (project_id , sequence_number , chunk_id , role , created_at)] -> full_conversaton_meta_datas
         [(chunk_id , chunk , created_at , chunker_type)] -> chunks"""
        self.__add_chunks(full_conversaton_meta_datas, chunks)

    def append_turns(self, turns: List[Tuple[str, str]]) -> List[int]:
        """Append conversation turns and return their allocated sequence numbers.

        `turns` is [(role, text)]. sequence_number, chunk_id and created_at are
        derived here rather than by the caller — they are all things the caller
        has no reliable way to know. Roles are validated one layer up, in
        FullConversation.
        """
        return self.__append_turns(turns)

    def next_sequence_number(self) -> int:
        """The sequence_number the next appended turn will receive.

        Advisory only — read outside the write lock. append_turns() allocates
        its own under BEGIN IMMEDIATE.
        """
        with sqlite3.connect(self.db_path) as conn:
            return self.__next_sequence_number(conn.cursor())

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
        return self.__get_all()

    def get_size(self):
        return self.__get_size()
