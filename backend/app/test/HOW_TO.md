# Test Execution Guide

This document outlines how to execute the comprehensive test suites for this backend service. The testing architecture is organized by modules and requires specific execution paths to avoid dependency lookup failures.

## Prerequisites
- You must have `pytest` installed (`pip install pytest`).
- Tests must be executed from the root `app/` directory (where this project is situated).

## Directory Structure
The `test/` directory is logically separated:
- **`data_layer_testing/`**: Contains unit and integration tests for chunking, text processing, database insertion, and end-to-end ingestion pipelines.
- **`memory_layer_testing/`**: Contains tests for the conversation pool, sqlite meta managers, and snapshot logic.

---

## How to Run the Tests

To ensure that Python correctly resolves module imports (`data_layer`, `memory`, etc.), **you must prefix your commands with `PYTHONPATH=.`** when running tests from the root `/app` folder.

### 1. Run the Entire Test Suite
To execute all test files across all layers simultaneously:
```bash
PYTHONPATH=. pytest test/ -v
```

### 2. Run Only Data Layer Tests
If you only want to validate changes made to the `data_layer`:
```bash
PYTHONPATH=. pytest test/data_layer_testing/ -v
```

### 3. Run Only Memory Layer Tests
If you only want to validate changes made to the `memory` layer:
```bash
PYTHONPATH=. pytest test/memory_layer_testing/ -v
```

### 4. Run a Specific Test File
If you are iterating on a single file (for instance, the new snapshot bugs):
```bash
PYTHONPATH=. pytest test/memory_layer_testing/test_snapshot_bugs.py -v
```

### 5. Run the Live Tests (real PostgreSQL)
```bash
PYTHONPATH=. pytest test/live_testing/ -v
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
| an interpreter with `psycopg` | the `fyp2` conda env, not `/usr/bin/python3` |
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

- **`ModuleNotFoundError` (e.g., `No module named 'memory'`)**: This happens when Python's import paths aren't correctly resolving the root folder. Make sure you are in the `app/` directory and are prepending `PYTHONPATH=.` before calling `pytest`.
- **`sqlite3.OperationalError: unable to open database file`**: Ensure that the `data/` directory (or wherever local DBs are initialized) exists on your filesystem.
- **`ModuleNotFoundError: No module named 'psycopg'`**: Tests may try to connect to the external PostgreSQL database. The memory tests mock this internally, but if it fails, ensure dependencies in `requirements.txt` are installed and `.env` is populated.
