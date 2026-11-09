"""Saved inference state on disk, and the table that says which conversation each file belongs to."""

import hashlib
import json
import os
import sqlite3
import struct
import time
import uuid
from pathlib import Path
from typing import List, NamedTuple

import numpy as np

from config import Config, get_logger
from inference.inference_exceptions import CheckpointError
from inference.shared_model import CHECKPOINT_FORMAT
from memory.identifiers import require_identifier
from memory.memory_database import MemoryDatabase, Schema
from storage.timestamps import utc_now

logger = get_logger(__name__)

MAGIC = b"ATLASKV\x01"
_HEADER_LENGTH = struct.Struct("<I")
_MAX_HEADER = 4096


def _create_tables(cursor: sqlite3.Cursor) -> None:
    # One live checkpoint per (conversation, user, purpose). The file name is
    # fresh on every save, so a new file is complete and recorded before the
    # old one is deleted, and no row ever names a half-written file.
    cursor.execute("""
        create table if not exists inference_checkpoints(
            cache_key text primary key,
            conversation_id text not null,
            user_id text not null,
            purpose text not null,
            model_fingerprint text not null,
            format_version integer not null,
            n_ctx integer not null,
            n_tokens integer not null,
            prefix_sha256 text not null,
            file_name text not null unique,
            file_sha256 text not null,
            size_bytes integer not null,
            created_at text not null,
            last_used_at text not null,
            unique (conversation_id, user_id, purpose)
        )
    """)


SCHEMA = Schema("inference_checkpoints", _create_tables)

_COLUMNS = (
    "cache_key, conversation_id, user_id, purpose, model_fingerprint, format_version, "
    "n_ctx, n_tokens, prefix_sha256, file_name, file_sha256, size_bytes, created_at, last_used_at"
)


class CheckpointRow(NamedTuple):
    cache_key: str
    conversation_id: str
    user_id: str
    purpose: str
    model_fingerprint: str
    format_version: int
    n_ctx: int
    n_tokens: int
    prefix_sha256: str
    file_name: str
    file_sha256: str
    size_bytes: int
    created_at: str
    last_used_at: str


class Checkpoint(NamedTuple):
    row: CheckpointRow
    tokens: np.ndarray
    state: bytes
    seed: int


class LoadOutcome(NamedTuple):
    status: str
    checkpoint: Checkpoint | None
    reason: str | None = None
    load_ms: float = 0.0


HIT = "hit"
MISSING = "missing"
INVALID = "invalid"


def cache_key(conversation_id: str, user_id: str, purpose: str, fingerprint: str) -> str:
    payload = "\x00".join((conversation_id, user_id, purpose, fingerprint))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def prefix_digest(tokens) -> str:
    return hashlib.sha256(np.asarray(tokens, dtype="<i4").tobytes()).hexdigest()


def encode(tokens, state: bytes, seed: int) -> bytes:
    tokens = np.asarray(tokens, dtype="<i4")
    header = json.dumps(
        {"format": CHECKPOINT_FORMAT, "n_tokens": int(tokens.size),
         "state_size": len(state), "seed": int(seed)},
        sort_keys=True,
    ).encode("ascii")
    return MAGIC + _HEADER_LENGTH.pack(len(header)) + header + tokens.tobytes() + state


def decode(payload: bytes) -> tuple:
    """Parse a checkpoint file without trusting any length it states."""
    if not payload.startswith(MAGIC):
        raise CheckpointError("not a checkpoint file")
    offset = len(MAGIC)
    if len(payload) < offset + _HEADER_LENGTH.size:
        raise CheckpointError("truncated header")
    (length,) = _HEADER_LENGTH.unpack_from(payload, offset)
    offset += _HEADER_LENGTH.size
    if length > _MAX_HEADER or len(payload) < offset + length:
        raise CheckpointError("bad header length")
    try:
        header = json.loads(payload[offset: offset + length].decode("ascii"))
        n_tokens = int(header["n_tokens"])
        state_size = int(header["state_size"])
        seed = int(header["seed"])
        version = int(header["format"])
    except (ValueError, KeyError, TypeError, UnicodeDecodeError) as error:
        raise CheckpointError(f"unreadable header: {error}") from None
    if version != CHECKPOINT_FORMAT:
        raise CheckpointError(f"format {version}, expected {CHECKPOINT_FORMAT}")
    offset += length
    if n_tokens < 1 or state_size < 1 or len(payload) != offset + 4 * n_tokens + state_size:
        raise CheckpointError("sizes do not match the file")
    tokens = np.frombuffer(payload, dtype="<i4", count=n_tokens, offset=offset).astype(np.intc)
    state = payload[offset + 4 * n_tokens:]
    return tokens, state, seed


class CheckpointStore:
    def __init__(
        self,
        directory: str | Path | None = None,
        database: MemoryDatabase | str | Path | None = None,
        max_bytes: int | None = None,
    ) -> None:
        self.directory = Path(Config.INFERENCE_CACHE_DIR if directory is None else directory)
        self.max_bytes = Config.INFERENCE_CACHE_MAX_BYTES if max_bytes is None else max_bytes
        self.database = MemoryDatabase.of(database)
        self.database.ensure(SCHEMA)
        self.directory.mkdir(parents=True, exist_ok=True)
        os.chmod(self.directory, 0o700)

    def __path(self, file_name: str) -> Path:
        # Names come only from this class, but a row is still checked before
        # it is turned into a path.
        if "/" in file_name or "\\" in file_name or file_name.startswith("."):
            raise CheckpointError(f"refusing checkpoint file name {file_name!r}")
        return self.directory / file_name

    def _write_file(self, path: Path, payload: bytes) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def __row(self, where: str, parameters) -> CheckpointRow | None:
        with self.database.reading() as cursor:
            row = cursor.execute(
                f"select {_COLUMNS} from inference_checkpoints where {where}", parameters
            ).fetchone()
        return None if row is None else CheckpointRow(*row)

    def row_for(self, conversation_id: str, user_id: str, purpose: str) -> CheckpointRow | None:
        return self.__row(
            "conversation_id = ? and user_id = ? and purpose = ?",
            (conversation_id, user_id, purpose),
        )

    def save(
        self,
        conversation_id: str,
        user_id: str,
        purpose: str,
        fingerprint: str,
        n_ctx: int,
        tokens,
        state: bytes,
        seed: int,
    ) -> CheckpointRow:
        """Write the file, then record it; the row never names a file that is not complete."""
        conversation_id = require_identifier(conversation_id, "conversation_id")
        user_id = require_identifier(user_id, "user_id")
        payload = encode(tokens, state, seed)
        if self.max_bytes is not None and len(payload) > self.max_bytes:
            raise CheckpointError(
                f"checkpoint of {len(payload)} bytes exceeds the {self.max_bytes}-byte cache"
            )
        key = cache_key(conversation_id, user_id, purpose, fingerprint)
        file_name = f"{key}-{uuid.uuid4().hex[:12]}.kv"
        path = self.__path(file_name)
        try:
            self._write_file(path, payload)
        except OSError as error:
            raise CheckpointError(f"could not write {path.name}: {error}") from error
        now = utc_now()
        row = CheckpointRow(
            key, conversation_id, user_id, purpose, fingerprint, CHECKPOINT_FORMAT, n_ctx,
            int(np.asarray(tokens).size), prefix_digest(tokens), file_name,
            hashlib.sha256(payload).hexdigest(), len(payload), now, now,
        )
        try:
            with self.database.writing() as cursor:
                old = cursor.execute(
                    "select file_name from inference_checkpoints "
                    "where conversation_id = ? and user_id = ? and purpose = ?",
                    (conversation_id, user_id, purpose),
                ).fetchone()
                cursor.execute(
                    "delete from inference_checkpoints "
                    "where conversation_id = ? and user_id = ? and purpose = ?",
                    (conversation_id, user_id, purpose),
                )
                cursor.execute(
                    f"insert into inference_checkpoints ({_COLUMNS}) "
                    f"values ({', '.join('?' * len(row))})",
                    row,
                )
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        if old is not None:
            self.__remove_file(old[0])
        self.evict()
        logger.debug("Saved a %d-token checkpoint (%d bytes) for conversation %s",
                     row.n_tokens, row.size_bytes, conversation_id)
        return row

    def load(
        self,
        conversation_id: str,
        user_id: str,
        purpose: str,
        fingerprint: str,
        n_ctx: int,
    ) -> LoadOutcome:
        """A checkpoint only if every recorded property and the file's digest still match."""
        started = time.monotonic()
        row = self.row_for(conversation_id, user_id, purpose)
        if row is None:
            return LoadOutcome(MISSING, None, "no checkpoint")
        problem = None
        if row.model_fingerprint != fingerprint:
            problem = "model or runtime changed"
        elif row.format_version != CHECKPOINT_FORMAT:
            problem = f"format {row.format_version}"
        elif row.n_ctx != n_ctx:
            problem = f"context window {row.n_ctx}, now {n_ctx}"
        if problem is None:
            try:
                payload = self.__path(row.file_name).read_bytes()
            except OSError as error:
                problem = f"file unreadable ({error.__class__.__name__})"
            else:
                if len(payload) != row.size_bytes or hashlib.sha256(payload).hexdigest() != row.file_sha256:
                    problem = "file does not match its recorded digest"
                else:
                    try:
                        tokens, state, seed = decode(payload)
                    except CheckpointError as error:
                        problem = str(error)
                    else:
                        if tokens.size != row.n_tokens or prefix_digest(tokens) != row.prefix_sha256:
                            problem = "token prefix does not match its record"
        if problem is not None:
            self.invalidate(row, problem)
            return LoadOutcome(INVALID, None, problem, (time.monotonic() - started) * 1000.0)
        with self.database.writing() as cursor:
            cursor.execute(
                "update inference_checkpoints set last_used_at = ? where cache_key = ?",
                (utc_now(), row.cache_key),
            )
        return LoadOutcome(HIT, Checkpoint(row, tokens, state, seed), None,
                           (time.monotonic() - started) * 1000.0)

    def invalidate(self, row: CheckpointRow, reason: str) -> None:
        logger.warning("Discarding checkpoint %s of conversation %s: %s",
                       row.cache_key, row.conversation_id, reason)
        with self.database.writing() as cursor:
            cursor.execute("delete from inference_checkpoints where file_name = ?", (row.file_name,))
        self.__remove_file(row.file_name)

    def delete_conversation(self, conversation_id: str, user_id: str | None = None) -> int:
        """Every checkpoint of a conversation, for when the conversation itself is deleted."""
        where, parameters = "conversation_id = ?", [conversation_id]
        if user_id is not None:
            where += " and user_id = ?"
            parameters.append(user_id)
        with self.database.writing() as cursor:
            names = [r[0] for r in cursor.execute(
                f"select file_name from inference_checkpoints where {where}", parameters)]
            cursor.execute(f"delete from inference_checkpoints where {where}", parameters)
        for name in names:
            self.__remove_file(name)
        return len(names)

    def rows(self) -> List[CheckpointRow]:
        with self.database.reading() as cursor:
            return [CheckpointRow(*r) for r in cursor.execute(
                f"select {_COLUMNS} from inference_checkpoints order by last_used_at")]

    def usage_bytes(self) -> int:
        return sum(row.size_bytes for row in self.rows())

    def evict(self) -> int:
        """Delete least-recently-used checkpoints until the cache fits its cap."""
        if self.max_bytes is None:
            return 0
        rows = self.rows()
        total = sum(row.size_bytes for row in rows)
        evicted = 0
        for row in rows:
            if total <= self.max_bytes:
                break
            self.invalidate(row, "evicted to stay within the cache cap")
            total -= row.size_bytes
            evicted += 1
        return evicted

    def sweep(self) -> int:
        """Remove files no row names and rows whose file is gone; returns how many were removed."""
        rows = self.rows()
        named = {row.file_name for row in rows}
        removed = 0
        for path in self.directory.iterdir():
            if path.is_file() and path.name not in named:
                path.unlink(missing_ok=True)
                removed += 1
        for row in rows:
            if not self.__path(row.file_name).exists():
                self.invalidate(row, "file missing")
                removed += 1
        return removed

    def __remove_file(self, file_name: str) -> None:
        try:
            self.__path(file_name).unlink(missing_ok=True)
        except (OSError, CheckpointError) as error:
            logger.warning("Could not delete checkpoint file %s: %s", file_name, error)
