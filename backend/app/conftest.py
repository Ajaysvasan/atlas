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
