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

---

## Troubleshooting

- **`ModuleNotFoundError` (e.g., `No module named 'memory'`)**: This happens when Python's import paths aren't correctly resolving the root folder. Make sure you are in the `app/` directory and are prepending `PYTHONPATH=.` before calling `pytest`.
- **`sqlite3.OperationalError: unable to open database file`**: Ensure that the `data/` directory (or wherever local DBs are initialized) exists on your filesystem.
- **`ModuleNotFoundError: No module named 'psycopg'`**: Tests may try to connect to the external PostgreSQL database. The memory tests mock this internally, but if it fails, ensure dependencies in `requirements.txt` are installed and `.env` is populated.
