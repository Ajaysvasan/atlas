import sqlite3
from pathlib import Path
from typing import List, Tuple


class FullConversationRepository:
    def __init__(
        self, full_conversation_dir: str | Path, project_id: str, project_name: str
    ):
        self.project_id = project_id
        self.project_name = project_name
        self.full_conversation_dir = Path(full_conversation_dir)
        self.full_conversation_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.full_conversation_dir / f"{project_id}_conversation.db"
        self.__init_db()

    def __init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            cursor = conn.cursor()
            cursor.execute("""
            create table if not exists full_conversation(
                project_id text not null, 
                sequence_number int not null, 
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
            try:
                cursor = conn.cursor()
                cursor.executemany(
                    """
        INSERT INTO full_conversation(project_id , sequence_number , chunk_id , role , created_at) VALUES (? , ? , ? , ?  , ?);
        """,
                    full_conversaton_meta_datas,
                )
                cursor.executemany(
                    """
                INSERT INTO summary_chunks(chunk_id , chunk , created_at , chunker_type) VALUES (? , ? , ? , ?);
                """,
                    chunks,
                )

                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise e

    def __get_sequence_number(self, chunk_id: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""SELECT sequence_number from full_conversation where chunk_id = ?;""",
                (chunk_id,),
            )
            return cursor.fetchone()[0]

    def __get_last_n_chunks(self, n: int):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT s.chunk
                FROM summary_chunks as s
                JOIN full_conversation AS f
                ON f.chunk_id = s.chunk_id
                ORDER BY f.created_at DESC
                limit ?
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
                order by f.created_at DESC
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

    def get_sequence_number(self, chunk_id: str) -> int:
        return self.__get_sequence_number(chunk_id)

    def get_n_chunks(self, n: int):
        return self.__get_last_n_chunks(n)

    def get_ranged_chunks(self, start: int, end: int):
        return self.__get_ranged_chunks(start, end)

    def get_sequence_after(self, sequence_number: int):
        return self.__get_messages_after(sequence_number)

    def fetch_all(self):
        return self.__get_all()

    def get_size(self):
        return self.__get_size()
