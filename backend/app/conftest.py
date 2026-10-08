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

    monkeypatch.setattr(
        Config, "DB_PATH", str(_memory_database_dir / f"chunks-{uuid.uuid4().hex}")
    )


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
