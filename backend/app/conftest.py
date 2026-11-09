"""Root configuration. Imported before any test module."""

import pytest

# Several test modules install a MagicMock for psycopg, each guarded so it only
# fires when the real module is absent. Importing it here first makes those
# guards see it, so an interpreter that has psycopg tests against the real
# adapters instead of a mock.
try:
    import psycopg  # noqa: F401
except ImportError:
    pass

# The same for llama_cpp: memory_layer_testing stubs it with setdefault, and
# without the real module imported first the real-model tests would collect
# against that stub and "load" a MagicMock.
try:
    import llama_cpp  # noqa: F401
except ImportError:
    pass


@pytest.fixture(scope="session")
def _real_diskannpy():
    """The real diskannpy, imported once, whatever the data layer's tests have
    put in its place."""
    import importlib
    import sys
    from unittest import mock

    current = sys.modules.get("diskannpy")
    if current is not None and not isinstance(current, mock.Mock):
        return current
    sys.modules.pop("diskannpy", None)
    try:
        return importlib.import_module("diskannpy")
    finally:
        if current is not None:
            sys.modules["diskannpy"] = current


@pytest.fixture
def real_diskann(monkeypatch, _real_diskannpy):
    """The real diskannpy for one test, the mock put back after.

    The data layer's tests install a MagicMock as diskannpy at collection time,
    for the whole session; without this every search raises inside the mock.
    Only that one entry is swapped. Rolling back all of sys.modules instead
    evicts whatever the test imported, and torch cannot be imported twice in
    one process — the next test to need it fails, and formatting the failure
    segfaults.
    """
    import sys

    import data_layer.vector_db_manager.vectorDB_diskann as wrapper

    monkeypatch.setitem(sys.modules, "diskannpy", _real_diskannpy)
    monkeypatch.setattr(wrapper, "dann", _real_diskannpy)
    return _real_diskannpy


@pytest.fixture(scope="session")
def _memory_database_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("memory_databases")


@pytest.fixture(autouse=True)
def _no_real_model_by_default(_memory_database_dir, monkeypatch):
    """No test loads the 2 GB draft model or writes the real inference cache unless it asks to.

    The shared runtime is pointed at a file that does not exist, so code that
    reaches for the model fails fast with ModelFileMissing; the real-model
    tests install their own runtime over this one.
    """
    import uuid

    from config import Config
    from inference import runtime
    from inference.shared_model import ModelSpec, SharedModel

    test = uuid.uuid4().hex
    monkeypatch.setattr(Config, "INFERENCE_CACHE_DIR", _memory_database_dir / f"kv-{test}")
    monkeypatch.setattr(Config, "CORPUS_DIR", _memory_database_dir / f"corpus-{test}")
    monkeypatch.setattr(Config, "RUN_DIR", _memory_database_dir / f"run-{test}")
    absent = SharedModel(ModelSpec(_memory_database_dir / "absent-model.gguf"))
    runtime.install_runtime(runtime.InferenceRuntime(absent))
    yield
    runtime.install_runtime(None)


@pytest.fixture(autouse=True)
def _memory_database_per_test(_memory_database_dir, monkeypatch, request):
    """Every test gets its own memory database and none reaches the real one.

    A store built without naming a database uses `MemoryDatabase.shared()`,
    which reads `Config.MEMORY_DB` when it is called — so pointing that at a
    fresh file per test is enough. Before this, tests wrote into the
    developer's own registry (bugs.md 7.5).
    """
    import sys
    import uuid

    from config import Config

    monkeypatch.setattr(
        Config, "MEMORY_DB", _memory_database_dir / f"{uuid.uuid4().hex}.db"
    )
    yield
    module = sys.modules.get("memory.memory_database")
    if module is not None:
        module.MemoryDatabase.close_shared()


@pytest.fixture(autouse=True)
def _chunk_store_per_test(_memory_database_dir, monkeypatch):
    """Every test gets its own chunk store and none reaches the real one.

    The ingestion pipeline's end-to-end test used to write into the developer's
    own data/hierarchical_db on every run (bugs.md 7.5). Everything that defaults
    to Config.DB_PATH reads it when called, so redirecting it here is enough.
    """
    import uuid

    from config import Config

    test = uuid.uuid4().hex
    monkeypatch.setattr(Config, "DB_PATH", str(_memory_database_dir / f"chunks-{test}"))
    monkeypatch.setattr(Config, "INDEX_PATH", str(_memory_database_dir / f"index-{test}"))


class InMemoryVectorStore:
    """Stands in for pgvector's `vectors` table, as the chunk vector store uses it."""

    def __init__(self) -> None:
        self.rows = {}
        self.down = False
        self.closed = False

    def __reachable(self) -> None:
        if self.down:
            raise ConnectionError("the vector store is down")

    def batch_insert(self, vector_ids, vectors) -> None:
        import numpy as np

        self.__reachable()
        for vector_id, vector in zip(vector_ids, vectors):
            self.rows.setdefault(int(vector_id), np.asarray(vector, dtype=np.float32).copy())

    def insert(self, vector_id, vector) -> None:
        self.batch_insert([vector_id], [vector])

    def storage_bytes(self) -> int:
        return sum(4 * len(vector) + 36 for vector in self.rows.values())

    def vectors_for(self, vector_ids):
        self.__reachable()
        return {int(v): self.rows[int(v)] for v in vector_ids if int(v) in self.rows}

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def chunk_vectors(monkeypatch):
    """Document chunk vectors go to memory, never to the real PostgreSQL.

    The suite reads the developer's own .env, so without this the pipeline's
    end-to-end test would write into their database. The live tests reach
    PostgreSQL through VectorRepository directly, which this leaves alone.
    """
    from data_layer.vector_db_manager import stored_vectors

    store = InMemoryVectorStore()
    monkeypatch.setattr(stored_vectors, "chunk_vector_store", lambda: store)
    return store


class InMemoryConversationTable:
    """Stands in for PostgreSQL's `conversation_table`, as ConversationTable uses it."""

    def __init__(self) -> None:
        self.rows = {}
        self.closed = False

    def record_query(self, conversation_id, user_id, query):
        import uuid
        from datetime import datetime, timezone

        from query_decomposition.conversation_table import Exchange

        exchange = Exchange(
            uuid.uuid4().hex, conversation_id, user_id, query,
            datetime.now(timezone.utc), None, None,
        )
        self.rows[exchange.exchange_id] = exchange
        return exchange

    def record_response(self, exchange_id, system_response):
        from datetime import datetime, timezone

        from query_decomposition.query_decomposition_exceptions import (
            ExchangeNotFound,
            ResponseAlreadyRecorded,
        )

        exchange = self.rows.get(exchange_id)
        if exchange is None:
            raise ExchangeNotFound(exchange_id)
        if exchange.system_response is not None:
            raise ResponseAlreadyRecorded(exchange_id)
        exchange = exchange._replace(
            system_response=system_response,
            system_response_time=datetime.now(timezone.utc),
        )
        self.rows[exchange_id] = exchange
        return exchange

    def get(self, exchange_id):
        return self.rows.get(exchange_id)

    def history(self, conversation_id, user_id):
        return sorted(
            (row for row in self.rows.values()
             if row.conversation_id == conversation_id and row.user_id == user_id),
            key=lambda row: (row.query_created_at, row.exchange_id),
        )

    def page(self, conversation_id, user_id, limit, after=None):
        rows = self.history(conversation_id, user_id)
        if after is not None:
            rows = [row for row in rows if (row.query_created_at, row.exchange_id) > tuple(after)]
        return rows[:limit]

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def conversation_rows(monkeypatch):
    """Queries the decomposition layer records go to memory, never to the real PostgreSQL."""
    from query_decomposition import conversation_table

    table = InMemoryConversationTable()
    monkeypatch.setattr(conversation_table, "open_conversation_table", lambda: table)
    return table


# --------------------------------------------------------------------------- #
# No test may leave a file in the app directory. A string meant as an id once
# landed in a `database` parameter, and four SQLite files named after test ids
# — conv_A, conv_B, conv_abc and '   ' — were written here and committed. The
# redirects above guard the paths code defaults to; this guards everything else.
# --------------------------------------------------------------------------- #
import os  # noqa: E402
from pathlib import Path  # noqa: E402

_APP_DIR = Path(__file__).resolve().parent
_NOT_CHECKED = {".venv", ".git", "__pycache__", ".pytest_cache", "log"}


def _app_files() -> set:
    found = set()
    for root, dirs, files in os.walk(_APP_DIR):
        dirs[:] = [d for d in dirs if d not in _NOT_CHECKED]
        found.update(os.path.relpath(os.path.join(root, f), _APP_DIR) for f in files)
    return found


def pytest_sessionstart(session):
    session.config._app_files_before = _app_files()


def pytest_sessionfinish(session, exitstatus):
    before = getattr(session.config, "_app_files_before", None)
    if before is None:
        return
    left = sorted(_app_files() - before)
    if not left:
        return
    session.exitstatus = pytest.ExitCode.TESTS_FAILED
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(
            "Tests left files in the app directory: " + ", ".join(map(repr, left)),
            red=True,
        )


@pytest.fixture(scope="session")
def real_model():
    """The real draft model, loaded once for every real-model test; skips when absent."""
    pytest.importorskip("llama_cpp")
    from inference.shared_model import ModelSpec, SharedModel

    spec = ModelSpec.draft()
    if not spec.path.exists():
        pytest.skip(f"draft model not downloaded ({spec.path})")
    # Smaller than production's window: enough for these prompts, a third of the KV cache.
    model = SharedModel(spec._replace(n_ctx=12_288))
    model.ensure_loaded()
    yield model
    model.close()


class InMemoryConversationRegistry:
    """The PostgreSQL `conversations` table's contract, over a dict: unique names per user, newest first."""

    def __init__(self, exchange_log=None) -> None:
        import threading

        self.rows = {}
        self.lock = threading.Lock()
        # The real summaries() joins conversation_table; the stand-in reads the in-memory one.
        self.exchange_log = exchange_log

    def create(self, user_id, proposed_name, conversation_id=None, created_at=None):
        import uuid

        from memory.conversation_registry import ConversationRecord, clean_name, first_free, name_stem
        from memory.identifiers import require_identifier
        from storage.timestamps import utc_now

        user_id = require_identifier(user_id, "user_id")
        stem = name_stem(clean_name(proposed_name))
        if not stem:
            raise ValueError("a conversation name needs at least one visible character")
        with self.lock:
            taken = {r.conversation_name for r in self.rows.values() if r.user_id == user_id}
            record = ConversationRecord(conversation_id or uuid.uuid4().hex, user_id,
                                        first_free(stem, taken), created_at or utc_now())
            self.rows[record.conversation_id] = record
            return record

    def get(self, conversation_id, user_id):
        record = self.rows.get(conversation_id)
        return record if record is not None and record.user_id == user_id else None

    def list(self, user_id, limit=50):
        mine = [r for r in self.rows.values() if r.user_id == user_id]
        return sorted(mine, key=lambda r: (r.created_at, r.conversation_id), reverse=True)[:limit]

    def summaries(self, user_id, limit=50, after=None):
        from datetime import datetime, timezone

        from memory.conversation_registry import ConversationSummary

        def stamp(value):
            value = datetime.fromisoformat(value) if isinstance(value, str) else value
            value = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)

        out = []
        for record in (r for r in self.rows.values() if r.user_id == user_id):
            exchanges = ([] if self.exchange_log is None
                         else self.exchange_log.history(record.conversation_id, user_id))
            moments = [stamp(record.created_at)]
            moments += [e.query_created_at for e in exchanges]
            moments += [e.system_response_time for e in exchanges if e.system_response_time is not None]
            last = max(moments).isoformat(timespec="microseconds")
            out.append(ConversationSummary(record.conversation_id, record.conversation_name,
                                           stamp(record.created_at).isoformat(timespec="microseconds"),
                                           last, len(exchanges)))
        out.sort(key=lambda row: (stamp(row.last_activity_at), row.conversation_id), reverse=True)
        if after is not None:
            position = (stamp(after[0]), after[1])
            out = [row for row in out if (stamp(row.last_activity_at), row.conversation_id) < position]
        return out[:limit]

    def find_by_prefix(self, user_id, prefix):
        prefix = (prefix or "").strip()
        return [r for r in self.list(user_id, 10_000) if prefix and r.conversation_id.startswith(prefix)]

    def delete(self, conversation_id, user_id):
        return self.rows.pop(conversation_id, None) is not None if self.get(conversation_id, user_id) else False

    def count(self):
        return len(self.rows)

    def import_records(self, records):
        from memory.conversation_registry import first_free, name_stem

        copied = present = 0
        renamed = []
        for record in records:
            if record.conversation_id in self.rows:
                present += 1
                continue
            taken = {r.conversation_name for r in self.rows.values() if r.user_id == record.user_id}
            name = (record.conversation_name if record.conversation_name not in taken
                    else first_free(name_stem(record.conversation_name), taken))
            if name != record.conversation_name:
                renamed.append((record.conversation_id, record.conversation_name, name))
            self.rows[record.conversation_id] = record._replace(conversation_name=name)
            copied += 1
        return copied, present, renamed

    def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _no_real_postgres_outside_live_tests(request, monkeypatch):
    """Only test/live_testing may open the real database through storage.postgres.

    The registry and the exchange log both connect through it; a test that
    reached the real one by importing a factory before it was patched once
    wrote 64 conversations into the developer's PostgreSQL (bugs.md 9.15).
    """
    if request.node.get_closest_marker("live") is not None:
        return
    from storage import postgres

    def refuse(*args, **kwargs):
        raise AssertionError("a non-live test tried to connect to the real PostgreSQL")

    monkeypatch.setattr(postgres, "connect", refuse)


@pytest.fixture(autouse=True)
def conversation_registry(monkeypatch, conversation_rows):
    """Conversations the pipeline and CLI create go to memory, never to the real PostgreSQL."""
    from memory import conversation_registry as module

    registry = InMemoryConversationRegistry(exchange_log=conversation_rows)
    monkeypatch.setattr(module, "open_conversation_registry", lambda **options: registry)
    return registry

