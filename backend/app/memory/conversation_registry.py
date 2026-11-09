"""Every conversation a user has — its id, owner, unique name and start — in PostgreSQL beside the exchange log."""

import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, NamedTuple, Tuple

from config import get_logger
from memory.identifiers import require_identifier
from memory.memory_pool_exceptions import InvalidIdentifier

logger = get_logger(__name__)

MAX_NAME_LENGTH = 60
MAX_NAME_ATTEMPTS = 20
_MAX_SUFFIX = 10_000
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_SUFFIX = re.compile(r"^(?P<base>.*) \((?P<n>\d+)\)$")
_COLUMNS = "conversation_id, user_id, conversation_name, created_at"
UNIQUE_NAME = "conversations_user_name_unique"
SQLITE_TABLE = "conversations"
SQLITE_MIGRATED = "conversations_migrated"

# Names are unique per owner, not globally: two users may each have a "Raft
# Leader Election". The constraint, not a prior lookup, is what makes
# concurrent creation safe; it is named so an insert can yield to it alone.
_CREATE = f"""
    create table if not exists conversations(
        conversation_id varchar primary key,
        user_id varchar not null,
        conversation_name varchar not null check (length(btrim(conversation_name)) > 0),
        created_at timestamptz not null,
        constraint {UNIQUE_NAME} unique (user_id, conversation_name)
    )
"""
_INDEX = (
    "create index if not exists idx_conversations_user_created "
    "on conversations(user_id, created_at desc)"
)


class ConversationRecord(NamedTuple):
    conversation_id: str
    user_id: str
    conversation_name: str
    created_at: str


class MigrationReport(NamedTuple):
    found: int
    copied: int
    already_present: int
    renamed: Tuple[Tuple[str, str, str], ...]


def clean_name(text: str | None, max_length: int = MAX_NAME_LENGTH) -> str:
    """One line, no control characters, no wrapping quotes or trailing punctuation, bounded."""
    if not isinstance(text, str):
        return ""
    name = _CONTROL.sub(" ", text)
    name = " ".join(name.split())
    name = name.strip(" \"'`*#_-:.…")
    name = name.rstrip(" !?;,")
    if len(name) > max_length:
        name = name[:max_length].rsplit(" ", 1)[0] or name[:max_length]
    return name.strip()


def name_stem(name: str) -> str:
    """`Topic (3)` -> `Topic`: suffixes are allocated from the stem, so they continue a sequence."""
    match = _SUFFIX.match(name)
    return match.group("base") if match else name


def first_free(stem: str, taken: Iterable[str]) -> str:
    taken = set(taken)
    if stem not in taken:
        return stem
    for n in range(2, _MAX_SUFFIX):
        candidate = f"{stem} ({n})"
        if candidate not in taken:
            return candidate
    raise ValueError(f"no free name left for {stem!r}")


def like_prefix(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _stamp(value) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat(timespec="microseconds")
    return str(value)


def _as_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class ConversationRegistry:
    """The `conversations` table, on the PostgreSQL database that holds the vectors and the exchange log."""

    def __init__(self, connection=None) -> None:
        if connection is None:
            from storage.postgres import connect

            connection = connect()
        self.conn = connection
        # One connection serves every thread; a transaction must not interleave.
        self.__lock = threading.RLock()
        with self.__lock, self.conn.cursor() as cursor:
            cursor.execute(_CREATE)
            cursor.execute(_INDEX)
        self.conn.commit()

    def __run(self, work):
        with self.__lock:
            try:
                with self.conn.cursor() as cursor:
                    result = work(cursor)
                self.conn.commit()
                return result
            except BaseException:
                self.conn.rollback()
                raise

    def create(self, user_id: str, proposed_name: str, conversation_id: str | None = None,
               created_at: str | None = None) -> ConversationRecord:
        """Insert with the first free name: `name`, then `name (2)`, `name (3)`…

        The UNIQUE constraint decides: an insert that loses a race to another
        writer for the same name inserts nothing, and the next free name is
        tried, up to MAX_NAME_ATTEMPTS times.
        """
        user_id = require_identifier(user_id, "user_id")
        stem = name_stem(clean_name(proposed_name))
        if not stem:
            raise ValueError("a conversation name needs at least one visible character")
        conversation_id = uuid.uuid4().hex if conversation_id is None else require_identifier(
            conversation_id, "conversation_id")
        when = datetime.now(timezone.utc) if created_at is None else _as_datetime(created_at)

        def attempt(cursor):
            cursor.execute(
                "select conversation_name from conversations where user_id = %s "
                "and (conversation_name = %s or conversation_name like %s escape '\\')",
                (user_id, stem, like_prefix(stem) + " (%)"),
            )
            name = first_free(stem, (row[0] for row in cursor.fetchall()))
            cursor.execute(
                f"insert into conversations ({_COLUMNS}) values (%s, %s, %s, %s) "
                f"on conflict on constraint {UNIQUE_NAME} do nothing returning {_COLUMNS}",
                (conversation_id, user_id, name, when),
            )
            return cursor.fetchone()

        for _ in range(MAX_NAME_ATTEMPTS):
            row = self.__run(attempt)
            if row is not None:
                record = self.__record(row)
                logger.info("Created conversation %s %r for user %s", conversation_id,
                            record.conversation_name, user_id)
                return record
        raise RuntimeError(f"could not allocate a name for {stem!r} after {MAX_NAME_ATTEMPTS} attempts")

    @staticmethod
    def __record(row) -> ConversationRecord:
        conversation_id, user_id, name, created_at = row
        return ConversationRecord(conversation_id, user_id, name, _stamp(created_at))

    def __select(self, where: str, parameters, suffix: str = "") -> List[ConversationRecord]:
        def query(cursor):
            cursor.execute(f"select {_COLUMNS} from conversations where {where} {suffix}", parameters)
            return cursor.fetchall()

        return [self.__record(row) for row in self.__run(query)]

    def get(self, conversation_id: str, user_id: str) -> ConversationRecord | None:
        """The conversation if this user owns it; another user's conversation is None, as if absent."""
        rows = self.__select("conversation_id = %s and user_id = %s", (conversation_id, user_id))
        return rows[0] if rows else None

    def list(self, user_id: str, limit: int = 50) -> List[ConversationRecord]:
        """This user's conversations, newest first."""
        try:
            user_id = require_identifier(user_id, "user_id")
        except InvalidIdentifier:
            return []
        return self.__select("user_id = %s", (user_id, limit),
                             "order by created_at desc, conversation_id limit %s")

    def find_by_prefix(self, user_id: str, prefix: str) -> List[ConversationRecord]:
        """This user's conversations whose id starts with `prefix`."""
        prefix = (prefix or "").strip()
        if not prefix:
            return []
        return self.__select("user_id = %s and conversation_id like %s escape '\\'",
                             (user_id, like_prefix(prefix) + "%"), "order by created_at desc")

    def delete(self, conversation_id: str, user_id: str) -> bool:
        def remove(cursor):
            cursor.execute("delete from conversations where conversation_id = %s and user_id = %s",
                           (conversation_id, user_id))
            return cursor.rowcount > 0

        return self.__run(remove)

    def count(self) -> int:
        return self.__run(lambda cursor: cursor.execute("select count(*) from conversations").fetchone()[0])

    def import_records(self, records: Iterable[ConversationRecord]) -> Tuple[int, int, List[Tuple[str, str, str]]]:
        """Copy records keeping id, owner and start; returns (copied, already present, renamed).

        One transaction: every record is copied or none is. A name another of
        the owner's conversations already holds here gets the next free suffix,
        and is reported, rather than being dropped.
        """
        renamed: List[Tuple[str, str, str]] = []

        def copy(cursor):
            copied = present = 0
            for record in records:
                cursor.execute("select user_id from conversations where conversation_id = %s",
                               (record.conversation_id,))
                existing = cursor.fetchone()
                if existing is not None:
                    if existing[0] != record.user_id:
                        raise ValueError(f"conversation {record.conversation_id} exists here "
                                         f"under another user")
                    present += 1
                    continue
                stem = name_stem(record.conversation_name) or "Untitled Conversation"
                cursor.execute(
                    "select conversation_name from conversations where user_id = %s "
                    "and (conversation_name = %s or conversation_name like %s escape '\\')",
                    (record.user_id, stem, like_prefix(stem) + " (%)"),
                )
                taken = {row[0] for row in cursor.fetchall()}
                name = record.conversation_name if record.conversation_name not in taken else first_free(stem, taken)
                if name != record.conversation_name:
                    renamed.append((record.conversation_id, record.conversation_name, name))
                cursor.execute(
                    f"insert into conversations ({_COLUMNS}) values (%s, %s, %s, %s)",
                    (record.conversation_id, record.user_id, name, _as_datetime(record.created_at)),
                )
                copied += 1
            return copied, present

        copied, present = self.__run(copy)
        return copied, present, renamed

    def close(self) -> None:
        with self.__lock:
            self.conn.close()


def open_conversation_registry() -> ConversationRegistry:
    """The conversation registry on the PostgreSQL named in .env."""
    return ConversationRegistry()


def migrate_from_sqlite(registry, memory_db: str | Path) -> MigrationReport | None:
    """Move the registry rows an earlier version kept in SQLite into `registry`, losing nothing.

    Copies every row in one transaction, then checks each one is there with the
    same owner and start before renaming the SQLite table to
    `conversations_migrated`, which is kept as a backup and never read again.
    Returns None when there is nothing to migrate; safe to run any number of times.
    """
    from memory.memory_database import MemoryDatabase

    path = Path(memory_db)
    if not path.exists():
        return None
    database = MemoryDatabase.of(path)
    with database.reading() as cursor:
        exists = cursor.execute(
            "select 1 from sqlite_master where type = 'table' and name = ?", (SQLITE_TABLE,)
        ).fetchone()
        if not exists:
            return None
        records = [
            ConversationRecord(*row) for row in cursor.execute(
                f"select {_COLUMNS} from {SQLITE_TABLE} order by created_at, conversation_id")
        ]
    copied, present, renamed = registry.import_records(records)
    missing = [
        record.conversation_id for record in records
        if (stored := registry.get(record.conversation_id, record.user_id)) is None
        or _as_datetime(stored.created_at) != _as_datetime(record.created_at)
    ]
    if missing:
        raise RuntimeError(f"migration incomplete; not verified in PostgreSQL: {missing}")
    with database.writing() as cursor:
        backup, n = SQLITE_MIGRATED, 1
        while cursor.execute("select 1 from sqlite_master where name = ?", (backup,)).fetchone():
            n += 1
            backup = f"{SQLITE_MIGRATED}_{n}"
        cursor.execute(f"alter table {SQLITE_TABLE} rename to {backup}")
    report = MigrationReport(len(records), copied, present, tuple(renamed))
    logger.info("Migrated the conversation registry from %s to PostgreSQL: %s", path, report)
    return report
