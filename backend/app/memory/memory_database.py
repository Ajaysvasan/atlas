"""The single SQLite database every memory-layer table lives in."""

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterator, NamedTuple, Tuple

from config import Config, get_logger
from memory.memory_pool_exceptions import NewerMemorySchema
from storage.sqlite_setup import connect, enable_wal

logger = get_logger(__name__)

SCHEMA_VERSION = 1


class Schema(NamedTuple):
    """One owner's tables, and the schemas whose tables they reference."""

    name: str
    create: Callable[[sqlite3.Cursor], None]
    requires: Tuple["Schema", ...] = ()


class MemoryDatabase:
    """One connection, shared by every owner of a memory-layer table.

    It holds no table's SQL. Each owner module declares its tables as a
    `Schema`, creates them through `ensure`, and keeps its own queries, so a
    table belongs to the same module it did when every owner had its own file.
    """

    _shared: Dict[Path, "MemoryDatabase"] = {}
    _shared_lock = threading.Lock()

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else Path(Config.MEMORY_DB)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._depth = 0
        self._ensured: set = set()
        self.connection: sqlite3.Connection | None = connect(
            self.path, check_same_thread=False, isolation_level=None
        )
        self.__check_version()
        self.journal_mode = enable_wal(self.connection, self.path)

    @classmethod
    def shared(cls, path: str | Path | None = None) -> "MemoryDatabase":
        """The process-wide instance for `path`, `Config.MEMORY_DB` by default."""
        resolved = Path(path if path is not None else Config.MEMORY_DB).resolve()
        with cls._shared_lock:
            database = cls._shared.get(resolved)
            if database is None or database.connection is None:
                database = cls(resolved)
                cls._shared[resolved] = database
            return database

    @classmethod
    def of(cls, database: "MemoryDatabase | str | Path | None") -> "MemoryDatabase":
        """The database an owner was handed: itself, or the shared one for a path."""
        if isinstance(database, MemoryDatabase):
            return database
        return cls.shared(database)

    @classmethod
    def close_shared(cls) -> None:
        with cls._shared_lock:
            for database in cls._shared.values():
                database.close()
            cls._shared.clear()

    def __check_version(self) -> None:
        version = self.connection.execute("pragma user_version;").fetchone()[0]
        if version > SCHEMA_VERSION:
            self.connection.close()
            self.connection = None
            raise NewerMemorySchema(self.path, version, SCHEMA_VERSION)
        if version == 0:
            self.connection.execute(f"pragma user_version = {SCHEMA_VERSION};")

    def __open(self) -> sqlite3.Connection:
        if self.connection is None:
            raise sqlite3.ProgrammingError(f"{self.path} has been closed")
        return self.connection

    def ensure(self, schema: Schema) -> None:
        """Create `schema`'s tables, after every schema they reference."""
        with self._lock:
            if schema.name in self._ensured:
                return
            for parent in schema.requires:
                self.ensure(parent)
            with self.writing() as cursor:
                schema.create(cursor)
            self._ensured.add(schema.name)
            logger.debug("Schema %s ready in %s", schema.name, self.path)

    @contextmanager
    def reading(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            yield self.__open().cursor()

    @contextmanager
    def writing(self) -> Iterator[sqlite3.Cursor]:
        """A cursor inside one transaction, committed on success.

        A nested call runs in a savepoint, so a failure the caller catches
        undoes only the nested part and the outer transaction carries on.
        """
        with self._lock:
            cursor = self.__open().cursor()
            outermost = self._depth == 0
            savepoint = f"memory_write_{self._depth}"
            cursor.execute("begin immediate;" if outermost else f"savepoint {savepoint};")
            self._depth += 1
            try:
                yield cursor
            except BaseException:
                if outermost:
                    cursor.execute("rollback;")
                else:
                    cursor.execute(f"rollback to {savepoint};")
                    cursor.execute(f"release {savepoint};")
                raise
            else:
                if not outermost:
                    cursor.execute(f"release {savepoint};")
                    return
                try:
                    cursor.execute("commit;")
                except BaseException:
                    # A failed COMMIT leaves the transaction open, and every
                    # later BEGIN on this shared connection would then fail.
                    cursor.execute("rollback;")
                    raise
            finally:
                self._depth -= 1

    def close(self) -> None:
        connection = getattr(self, "connection", None)
        if connection is None:
            return
        with self._lock:
            connection.close()
            self.connection = None
