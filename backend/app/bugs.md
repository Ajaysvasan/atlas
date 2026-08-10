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

### Bug 4.1: Empty Module Directory Shells / Unimplemented Managers (`memory/`)

- **Criticality:** Medium
- **Priority:** P2
- **Explanation:** Several core memory management classes are completely empty (just containing `pass`), leaving the architecture unimplemented. These include `MemoryManager` (`memory_manager.py`), `ProjectManager` (`project_manager.py`), `TopicManager` (`topic_manager.py`), `ConversationPoolManager` (`conversation_pool_manager.py`), and `ConversationSummary` (`conversation_summary.py`).

### Bug 4.2: Tuple Return Type Mismatch for Scalar Queries (`fullconversation_repository.py`)

- **Criticality:** Medium
- **Priority:** P2
- **Explanation:** In `fullconversation_repository.py`, methods like `get_sequence_number()` (internally `__get_sequence_numer`) and `get_size()` are type-hinted and expected to return `int`, but they directly return the result of `cursor.fetchone()`, which yields a tuple like `(42,)`. This will cause type mismatch errors downstream for any component expecting an actual integer. Similarly, `__get_last_n_chunks` and other methods return a list of tuples instead of strings.

### Bug 4.3: Typo in Method Name (`fullconversation_repository.py`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** The internal method `__get_sequence_numer` is misspelled and should be `__get_sequence_number`.

### Bug 4.4: Inaccurate Docstrings in Vector Manager (`conversationVectorManager.py`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** The docstrings for the `insert` and `batch_insert` methods claim that they insert vectors "by generating the vector_id automatically," but the methods actually require `vector_id` (and `vector_ids`) to be explicitly provided as arguments by the caller. This documentation is highly misleading.
