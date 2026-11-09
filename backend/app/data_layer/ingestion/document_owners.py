"""Who may retrieve an uploaded document: `document_owners` in the chunk store.

A document with no owner rows is public — the built corpus and acquired pages.
A document with any is private to those users. Uploads get their owner row
before a single chunk is written, so a crash part-way through ingestion can
leave a private document unfinished but never public.
"""

import sqlite3
from pathlib import Path
from typing import Iterable, Set

from config import Config, get_logger
from storage.timestamps import utc_now

logger = get_logger(__name__)

_CREATE = """
    create table if not exists document_owners(
        document_id text not null,
        user_id text not null,
        created_at text not null,
        primary key (document_id, user_id)
    )
"""


class DocumentOwners:
    """Reads and writes ownership; one short-lived connection per call, so any thread may use it."""

    def __init__(self, chunk_store_path: str | Path | None = None) -> None:
        self.path = Path(Config.DB_PATH if chunk_store_path is None else chunk_store_path)

    def __connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute(_CREATE)
        return connection

    def add(self, document_id: str, user_id: str) -> bool:
        """Record an owner; True if the row is new."""
        if not document_id or not user_id:
            raise ValueError("document_id and user_id are required")
        connection = self.__connect()
        try:
            with connection:
                cursor = connection.execute(
                    "insert or ignore into document_owners (document_id, user_id, created_at) values (?, ?, ?)",
                    (document_id, user_id, utc_now()),
                )
                return cursor.rowcount == 1
        finally:
            connection.close()

    def remove(self, document_id: str, user_id: str) -> None:
        connection = self.__connect()
        try:
            with connection:
                connection.execute("delete from document_owners where document_id = ? and user_id = ?",
                                   (document_id, user_id))
        finally:
            connection.close()

    def owners(self, document_id: str) -> Set[str]:
        connection = self.__connect()
        try:
            return {row[0] for row in connection.execute(
                "select user_id from document_owners where document_id = ?", (document_id,))}
        finally:
            connection.close()

    def any_private(self) -> bool:
        if not self.path.exists():
            return False
        connection = self.__connect()
        try:
            return connection.execute("select exists(select 1 from document_owners)").fetchone()[0] == 1
        finally:
            connection.close()

    def visible(self, parent_ids: Iterable[str | None], viewer: str | None) -> Set[str]:
        """The parent ids (document ids, or context ids of sectioned documents) `viewer` may see.

        None — no viewer — sees public documents only.
        """
        parents = sorted({p for p in parent_ids if p})
        if not parents:
            return set()
        if not self.path.exists():
            return set(parents)
        connection = self.__connect()
        try:
            tables = {row[0] for row in connection.execute("select name from sqlite_master where type='table'")}
            if not connection.execute("select exists(select 1 from document_owners)").fetchone()[0]:
                return set(parents)
            # A hierarchical chunk's parent is its context; walk up to the document.
            if {"Contexts", "Sections"} <= tables:
                document = ("coalesce((select s.documentId from Contexts c join Sections s "
                            "on s.sectionId = c.sectionId where c.contextId = p.parent_id), p.parent_id)")
            else:
                document = "p.parent_id"
            values = ",".join("(?)" for _ in parents)
            rows = connection.execute(
                f"""
                with p(parent_id) as (values {values}),
                     d as (select p.parent_id, {document} as document_id from p)
                select d.parent_id from d
                where not exists (select 1 from document_owners o where o.document_id = d.document_id)
                   or exists (select 1 from document_owners o
                              where o.document_id = d.document_id and o.user_id = ?)
                """,
                (*parents, viewer if viewer is not None else ""),
            ).fetchall()
            return {row[0] for row in rows}
        finally:
            connection.close()
