"""The one place ingestion asks how much the corpus occupies and whether it may grow."""

import fcntl
import os
import shutil
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, NamedTuple

from config import Config, get_logger
from data_layer.datalayer_exceptions.datalayer_exceptions import (
    IngestionQuotaExceeded,
    InsufficientDiskSpace,
)

logger = get_logger(__name__)

_process_lock = threading.Lock()


class StorageUsage(NamedTuple):
    files: Dict[str, int]
    postgres_bytes: int | None

    @property
    def total(self) -> int:
        return sum(self.files.values()) + (self.postgres_bytes or 0)


def corpus_paths() -> Dict[str, Path]:
    """What ingestion writes for the document corpus, and nothing else.

    Read at call time, so a redirected Config (tests) is honoured.
    """
    from retrieval_layer import keyword_search

    chunks = Path(Config.DB_PATH)
    return {
        "chunk_store": chunks,
        "chunk_store_wal": chunks.with_name(chunks.name + "-wal"),
        "chunk_store_shm": chunks.with_name(chunks.name + "-shm"),
        "keyword_index": Path(keyword_search.FTS_PATH),
        "vector_index": Path(Config.INDEX_PATH),
        "corpus_files": Path(Config.CORPUS_DIR),
    }


def _bytes_under(path: Path, seen: set) -> int:
    """Allocated bytes of a file or a tree, each inode once, symlinks not followed."""
    total = 0
    try:
        if path.is_symlink():
            return 0
        if path.is_file():
            entries: Iterable[Path] = [path]
        elif path.is_dir():
            entries = (p for p in path.rglob("*") if p.is_file() and not p.is_symlink())
        else:
            return 0
        for entry in entries:
            stat = entry.stat()
            if (stat.st_dev, stat.st_ino) in seen:
                continue
            seen.add((stat.st_dev, stat.st_ino))
            total += max(stat.st_size, getattr(stat, "st_blocks", 0) * 512)
    except FileNotFoundError:
        pass
    return total


def postgres_corpus_bytes() -> int:
    from data_layer.vector_db_manager import stored_vectors

    store = stored_vectors.chunk_vector_store()
    try:
        return int(store.storage_bytes())
    finally:
        store.close()


class Reservation:
    def __init__(self, quota: "IngestionQuota", used: int, reserved: int) -> None:
        self.quota = quota
        self.used_at_start = used
        self.reserved = reserved

    def check(self, more: int) -> StorageUsage:
        """Before writing `more` bytes: still inside the quota and the disk reserve, measured now."""
        usage = self.quota.usage()
        self.quota.require_room(usage.total, more)
        return usage


class IngestionQuota:
    """Counts the corpus's own files and its share of PostgreSQL; serialises every ingestion.

    One ingestion at a time, across threads and processes: the reservation holds
    an exclusive lock on a file beside the data, so two writers cannot each see
    room for themselves and together exceed the limit.
    """

    def __init__(
        self,
        limit_bytes: int | None = ...,
        disk_reserve_bytes: int | None = None,
        postgres_usage: Callable[[], int] | None = postgres_corpus_bytes,
        paths: Callable[[], Dict[str, Path]] = corpus_paths,
        lock_path: Path | None = None,
    ) -> None:
        self.limit_bytes = Config.MAX_INGESTION_STORAGE_BYTES if limit_bytes is ... else limit_bytes
        if self.limit_bytes is not None and (
            isinstance(self.limit_bytes, bool) or not isinstance(self.limit_bytes, int) or self.limit_bytes <= 0
        ):
            raise ValueError(f"limit_bytes must be a positive int or None, got {self.limit_bytes!r}")
        self.disk_reserve_bytes = (
            Config.INGESTION_DISK_RESERVE_BYTES if disk_reserve_bytes is None else disk_reserve_bytes
        )
        self.postgres_usage = postgres_usage
        self.paths = paths
        self.lock_path = Path(Config.DATA_DIR) / ".ingestion.lock" if lock_path is None else lock_path

    def usage(self) -> StorageUsage:
        seen: set = set()
        files = {name: _bytes_under(path, seen) for name, path in self.paths().items()}
        postgres = None
        if self.postgres_usage is not None:
            try:
                postgres = self.postgres_usage()
            except Exception as error:
                # Unknown is not zero: ingestion cannot write vectors without
                # PostgreSQL anyway, and the file part is still enforced.
                logger.warning("PostgreSQL corpus size unavailable: %s", error)
        return StorageUsage(files, postgres)

    def free_disk_bytes(self) -> int:
        target = Path(Config.DATA_DIR)
        while not target.exists():
            target = target.parent
        return shutil.disk_usage(target).free

    def require_room(self, used: int, more: int) -> None:
        if self.limit_bytes is not None and used + more > self.limit_bytes:
            raise IngestionQuotaExceeded(used, more, self.limit_bytes)
        free = self.free_disk_bytes()
        if free - more < self.disk_reserve_bytes:
            raise InsufficientDiskSpace(free, more, self.disk_reserve_bytes)

    @contextmanager
    def reserve(self, estimated_bytes: int) -> Iterator[Reservation]:
        """Hold the ingestion lock and prove `estimated_bytes` fit, for the length of the block."""
        if estimated_bytes < 0:
            raise ValueError("estimated_bytes cannot be negative")
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with _process_lock:
            descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                used = self.usage().total
                self.require_room(used, estimated_bytes)
                yield Reservation(self, used, estimated_bytes)
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
