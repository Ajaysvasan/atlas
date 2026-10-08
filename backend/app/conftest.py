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
