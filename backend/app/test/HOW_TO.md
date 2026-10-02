# Test Execution Guide

This document outlines how to execute the comprehensive test suites for this backend service. The testing architecture is organized by modules and requires specific execution paths to avoid dependency lookup failures.

## Prerequisites
- `uv` installed, and `uv sync --all-extras` run once in `app/`. That builds
  `.venv` from `pyproject.toml` + `uv.lock`, fetching CPython 3.11 if needed.
- Tests must be executed from the root `app/` directory (where this project is situated).
- Nothing to activate and no `PYTHONPATH` to set: `uv run` picks the interpreter,
  and the root `conftest.py` puts the project directory on `sys.path`.

## Directory Structure
The `test/` directory is logically separated:
- **`data_layer_testing/`**: Contains unit and integration tests for chunking, text processing, database insertion, and end-to-end ingestion pipelines.
- **`memory_layer_testing/`**: Contains tests for the conversation pool, sqlite meta managers, and snapshot logic.

---

## How to Run the Tests

Prefix every command with **`uv run`**. It resolves the pinned interpreter and
the locked dependencies, so the tests cannot silently run against whatever Python
happens to be on `PATH`.

### 1. Run the Entire Test Suite
To execute all test files across all layers simultaneously:
```bash
uv run pytest test/ -v
```

### 2. Run Only Data Layer Tests
If you only want to validate changes made to the `data_layer`:
```bash
uv run pytest test/data_layer_testing/ -v
```

### 3. Run Only Memory Layer Tests
If you only want to validate changes made to the `memory` layer:
```bash
uv run pytest test/memory_layer_testing/ -v
```

### 4. Run a Specific Test File
If you are iterating on a single file (for instance, the new snapshot bugs):
```bash
uv run pytest test/memory_layer_testing/test_snapshot_bugs.py -v
```

### 5. Run the Live Tests (real PostgreSQL)
```bash
uv run pytest test/live_testing/ -v
```
These skip themselves unless a real server is reachable, so they are safe to
run anywhere. To see why they skipped, add `-rs`.

---

## Live Tests and Why They Exist

Most of the suite replaces `psycopg` with a `MagicMock`, which is what lets it
run without a database. A mock cursor adapts any object it is handed and returns
whatever it is told to, so two whole classes of defect are invisible to it: which
types the driver can actually adapt, and which types come back from a read. Bugs
5.13 and 5.14 were both of that kind — both passed the mocked suite and both
failed on the first contact with a real server.

`test/live_testing/` runs the same code against a real PostgreSQL. It needs:

| Requirement | Command |
| :--- | :--- |
| `psycopg` on the interpreter | `uv sync --all-extras` (it is a locked dependency) |
| the pgvector extension | `sudo dnf install pgvector` (Fedora's package, not PGDG's `pgvector_18`) |
| the database | `createdb Vectors`, then `CREATE EXTENSION vector;` |

`scripts/smoke.py` checks all of these at once and names whatever is missing.

The root `conftest.py` imports the real `psycopg` before any test module loads.
Every mock in the suite is installed with `setdefault` or an equivalent guard, so
importing the real module first makes those guards stand down and the suite runs
against the real driver where one exists. Without it the live tests skip even on
a machine that has a server, because a test module imported earlier in
collection would have already installed the stub.

---

## Troubleshooting

- **`ModuleNotFoundError` (e.g., `No module named 'memory'`)**: The project directory is not on `sys.path`. Run from `app/` and go through `uv run pytest`, which finds the root `conftest.py` that inserts it. Invoking a bare `pytest` from elsewhere will not.
- **`sqlite3.OperationalError: unable to open database file`**: Ensure that the `data/` directory (or wherever local DBs are initialized) exists on your filesystem.
- **`ModuleNotFoundError: No module named 'psycopg'`**: You are not on the project environment. `psycopg` is a locked dependency, so `uv run` always has it; a bare `python`/`pytest` may not. The mocked suite tolerates its absence by design, which is why this surfaces as skipped live tests rather than an error.
- **8 tests skipped**: the live tests could not reach PostgreSQL. `uv run pytest test/live_testing/ -rs` prints the precondition that failed, and `uv run python scripts/smoke.py` checks the whole setup at once.
