"""Reachability probe and fixtures for the tests that use a real PostgreSQL."""

import os
import sys
import uuid
from importlib import metadata
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_APP_DIR = Path(__file__).resolve().parents[2]
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))


def _unavailable() -> str | None:
    """Why these tests cannot run here, or None if they can."""
    # Asked of the installed distributions, not of sys.modules, which by now may
    # hold a stub another test module installed. Otherwise an interpreter simply
    # lacking psycopg reports a mocking conflict and sends the reader hunting one.
    try:
        metadata.version("psycopg")
    except metadata.PackageNotFoundError:
        return "psycopg is not installed in this interpreter (try the conda env)"

    import psycopg

    # Another test module may have put a MagicMock in sys.modules first. A mock
    # connects to nothing and adapts anything, so it would turn these tests into
    # assertions about the mock — the exact blindness they exist to remove.
    if isinstance(psycopg, MagicMock):
        return "psycopg is mocked by another module in this run"

    try:
        from dotenv import load_dotenv
    except ImportError:
        return "python-dotenv is not installed in this interpreter"

    load_dotenv(str(_APP_DIR / ".env"))
    settings = {
        name: os.getenv(name)
        for name in ("DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT")
    }
    missing = [name for name, value in settings.items() if not value]
    if missing:
        return f"missing in .env: {', '.join(missing)}"

    try:
        with psycopg.connect(
            dbname=settings["DB_NAME"],
            user=settings["DB_USER"],
            password=settings["DB_PASSWORD"],
            host=settings["DB_HOST"],
            port=settings["DB_PORT"],
            connect_timeout=5,
        ) as conn:
            if not conn.execute(
                "select 1 from pg_extension where extname = 'vector'"
            ).fetchone():
                return (
                    f"the vector extension is not created in {settings['DB_NAME']!r}"
                )
    except Exception as error:
        return f"cannot reach PostgreSQL: {type(error).__name__}"

    return None


SKIP_REASON = _unavailable()


def pytest_collection_modifyitems(items):
    if SKIP_REASON is None:
        return
    skip = pytest.mark.skip(reason=SKIP_REASON)
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def repository():
    """A VectorRepository on a project id no other run will collide with."""
    from data_layer.vector_db_manager.repository.vectorRepository import (
        VectorRepository,
    )

    repo = VectorRepository(f"__live_test__{uuid.uuid4().hex}")
    try:
        yield repo
    finally:
        repo.curr.execute(
            "delete from vectors where project_id = %s", (repo.project_id,)
        )
        repo.conn.commit()
        repo.close()
