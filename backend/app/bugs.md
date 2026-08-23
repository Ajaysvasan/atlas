# Comprehensive Project Bug Inventory (`backend/app`)

This document catalogs all logical, architectural, and execution pipeline bugs identified across the project codebase. The bugs are organized in a **section-wise manner** grouped by architectural component, complete with detailed explanations, **Criticality** ratings (`Critical`, `High`, `Medium`, `Low`), and recommended resolution **Priority** levels (`P0`, `P1`, `P2`, `P3`).

---

## Section 1: System Configuration & Path Management (`config.py` & `main.py`)

### Bug 1.1: Hardcoded Log File Directory Misalignment (`config.py` & `main.py`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** `Config.LOG_FILE` defaults to `"log/app.log"`. In `get_logger()`, non-absolute paths are joined against `os.path.dirname(os.path.abspath(__file__))` (`app/log/app.log`). However, execution invocations from different working directories cause log handlers to create disconnected log files across both `app/log/app.log` and `./log/app.log`.

---

## Section 2: CLI & Pipeline Execution (`cli/` & `main.py`)

### Bug 2.1: Placeholder Execution in Single Query Mode (`main.py`)

- **Criticality:** Medium
- **Priority:** P2
- **Explanation:** In `main.py`, when a user specifies the `--query` or `-q` argument, `main()` logs `Executing single query` but outputs `Result for '<query>': [Processing placeholder]`. The single query CLI mode is disconnected from `IngestionPipeline`, vector database search, and RAG retrieval pipelines.

### Bug 2.2: Unimplemented Query Loop in Interactive CLI (`cli/cli_interface.py`)

- **Criticality:** Medium
- **Priority:** P2
- **Explanation:** In `cli_interface()`, the interactive terminal loop accepts user input, logs the query, and prints `Processing query: <query>`, but contains `# some stuff` with no downstream execution logic. Query processing is not connected to search or response generation modules.

---

## Section 3: Core Architecture & Empty Module Directory Shells

### Bug 3.1: Missing Knowledge Acquisition Module (`knowledge_acquisition/`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** The `knowledge_acquisition/` directory is completely empty (0 files). Components responsible for acquiring external knowledge or web scraping are missing from the codebase.

### Bug 3.2: Missing Dataset Storage Directory (`dataset/`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** The `dataset/` directory is completely empty (0 files). Sample corpora or benchmark documents referenced in `Config.DATASET_PATH` are missing from the project workspace.

---

## Section 4: Memory Data Management & Database Layer (`memory/`)

---

### Bug 4.1: Empty Module Directory Shells / Unimplemented Managers (`memory/`)

- **Criticality:** Medium
- **Priority:** P2
- **Explanation:** Three core memory management classes remain completely empty (just containing `pass`), leaving the upper levels of the hierarchy unimplemented: `MemoryManager` (`memory_manager.py`), `ProjectManager` (`project_manager.py`), and `TopicManager` (`topic_manager.py`). `ConversationPoolManager` is now implemented — it owns the `FullConversation` / `ConversationSummary` / `SnapShot` trio for a single conversation, restores snapshot cursors on construction, and applies the `SNAPSHOT_EVERY_N_TURNS` trigger policy. The remaining three need to resolve per-topic and per-project directories and hand a `ConversationPoolManager` back to callers.

---

### Bug 4.2: Inaccurate Docstrings in Vector Manager (`conversationVectorManager.py`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** The docstrings for the `insert` and `batch_insert` methods claim that they insert vectors "by generating the vector_id automatically," but the methods actually require `vector_id` (and `vector_ids`) to be explicitly provided as arguments by the caller. This documentation is highly misleading.

---

### Bug 4.19: `project_name` Stored as Instance Variable but Never Used (`conversationVectorManager.py`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** `ConversationVectorManager.__init__` stores `self.project_name = project_name`, but `project_name` is never referenced anywhere else in the class. `VectorRepository` is initialized using only `self.project_id`. The stored `project_name` is dead code that misleads readers into thinking it participates in repository operations.

---

### Bug 4.20: Useless `ORDER BY` on PRIMARY KEY Lookup in `get_cumulative_vector_meta_data` (`conversationVectorMetaManager.py`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** `get_cumulative_vector_meta_data` queries `WHERE cumulative_vector_id = ?` — a lookup on the PRIMARY KEY which uniquely identifies exactly one row. The appended `ORDER BY created_at DESC` has no effect on a single-row result set and is misleading, suggesting the method is intended for multi-row retrieval when it is not.

---

### Bug 4.21: `add()` / `__add_chunks` Type Annotation Has Wrong Tuple Arity (`fullconversation_repository.py`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** The type annotation for `full_conversaton_meta_datas` in both `__add_chunks` and the public `add()` is `List[Tuple[str, int, str, str]]` — a 4-field tuple. However, the actual SQL `INSERT` statement binds 5 values `(project_id, sequence_number, chunk_id, role, created_at)`, and the docstring correctly lists all 5 fields. Any caller following the type hint and providing a 4-tuple will cause `sqlite3.ProgrammingError: Binding 5 has no name`.

---

### Bug 4.22: `MisMatchCount` Exception Has No `__init__` or `__str__` Override (`memory_pool_exceptions.py`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** Unlike every other exception class in `memory_pool_exceptions.py` (`InvalidCursorException`, `NullPointerException`, `InvalidVectorDimension`, and the newer `InvalidRole` / `EmptyTurnContent` — all of which define structured `__init__` and `__str__`), `MisMatchCount` is a bare `pass` class. It produces no consistent formatted message and is inconsistent with the module's established exception contract.

---

### Bug 4.41: `insert_cumulative_vector_meta_data` Writes `len_of_the_summary` as a String (`conversationVectorMetaManager.py`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** The method binds `str(len_of_the_summary)` into a column declared `INTEGER NOT NULL`, while every other integer binding in the class is wrapped in `int()`. SQLite's INTEGER affinity coerces the value back on the way in (verified: `typeof()` reports `integer`), so nothing breaks today — but the intent is inverted, and `batch_insert_cumulative_vector_meta_data` compounds it by typing the field as `str` in its `List[Tuple[int, str, str, str, str]]` annotation. Any future migration to a stricter backend, or a `STRICT` table, would surface it as a type error.

---

### Bug 4.43: Unused f-string Prefix in `__get_sequence_number` (`fullconversation_repository.py`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** The query is written as `f"""SELECT sequence_number from full_conversation where chunk_id = ?;"""` but contains no interpolation. The `f` prefix is dead, and on a query that takes user-supplied input it reads as though interpolation were intended — the pattern this file must never adopt, since it correctly uses a bound parameter here.

---
