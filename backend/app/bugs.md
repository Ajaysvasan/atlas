# Comprehensive Project Bug Inventory (`backend/app`)

This document catalogs all logical, architectural, concurrency, type safety, and data integrity bugs identified across the project codebase. The bugs are organized in a **section-wise manner** grouped by architectural component, complete with detailed explanations, **Criticality** ratings (`Critical`, `High`, `Medium`, `Low`), and recommended resolution **Priority** levels (`P0`, `P1`, `P2`, `P3`).

---

## Section 1: Data Ingestion & Vector Database Layer (`data_layer/`)

### Bug 1.1: Data Type Mismatch Passing Python List and String UUID to DiskANN (`vectorDbManager.py` & `EmbeddingManager.py`)

- **Criticality:** Critical
- **Priority:** P0
- **Explanation:** `EmbeddingManager` outputs embedded chunks (`EmbeddedChunk`) where `vector` is formatted as a standard Python `List[float]` and `chunk_id` inside `EmbeddedChunkMetaData` is formatted as a string UUID (`str`). However, `VectorDbManager.batch_insert` passes these directly to `diskannpy`'s native C++ bindings, which strictly require contiguous `numpy.ndarray` vectors and 32-bit / 64-bit integer vector IDs (`uint32`/`uint64`). Attempting to insert embedded chunks causes an immediate native `TypeError: incompatible function arguments`.

### Bug 1.2: Incorrect C++ Exception Handling in DiskANN Wrapper (`vectorDB_diskann.py`)

- **Criticality:** High
- **Priority:** P1
- **Explanation:** In `VectorDb_diskann.__insert_vector()`, error handling wraps operations in `try ... except (ValueError, RuntimeError): raise VectorInsertionError(vector_id)`. However, `VectorDb_diskann` methods elsewhere attempt to catch `VectorInsertionError` from calls directly into `diskannpy`. Because `diskannpy` raises standard C++ wrapped Python `RuntimeError` or `ValueError` rather than `VectorInsertionError`, custom error handling is bypassed and raw runtime exceptions crash the application unexpectedly.

### Bug 1.3: Markdown Extractor Double Newline Formatting Corruption (`text_extractor.py`)

- **Criticality:** Medium
- **Priority:** P2
- **Explanation:** In `TextExtractor.extract_text_from_file()`, Markdown (`.md`) files are read line-by-line using `file.readlines()`, which preserves existing line-ending `\n` characters. The method then joins the list of lines using `"\n".join(lines)`. This results in doubling every newline in the file (`"Line 1\nLine 2"` becomes `"Line 1\n\nLine 2"`), corrupting text layout formatting and breaking paragraph boundaries during chunking.

### Bug 1.4: DiskANN Index Load State Desynchronization (`vectorDB_diskann.py`)

- **Criticality:** High
- **Priority:** P1
- **Explanation:** In `VectorDb_diskann.load(index_directory)`, the method invokes `self.dynamic_dann.from_file(index_directory)` which instantiates and returns a _new_ DiskANN index object. However, `VectorDb_diskann` fails to update its internal reference `self.dynamic_dann` with this newly created instance. As a result, subsequent `insert()` or `search()` operations continue targeting the initial empty index, silently discarding loaded index data.

### Bug 1.5: `TypeError` on `Path` Object in Document ID Generator (`normalizer.py`)

- **Criticality:** Medium
- **Priority:** P2
- **Explanation:** `TextNormalizer.__generate_document_id(file_name, file_path, normalized_text)` uses `"".join([file_name, file_path, normalized_text])` to compute document hashes. Upstream loaders (such as `FileLoader`) return `pathlib.Path` objects for file paths. Passing a `Path` instance as `file_path` causes `"".join()` to raise `TypeError: expected str instance, PosixPath found`.

---

## Section 2: System Configuration & Path Management (`config.py` & `main.py`)

### Bug 2.1: Invalid Relative Root Base Path Anchor (`config.py`)

- **Criticality:** High
- **Priority:** P1
- **Explanation:** In `Config`, `ABS_PATH` is defined as `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`. Since `config.py` is located inside `backend/app/`, `ABS_PATH` resolves to `backend/` rather than `backend/app/`. Consequently, `DATA_DIR` resolves to `backend/data`, causing databases (`CONVERSATION_SNAPSHOT_DB`, `FULL_CONVERSATION`, `DB_PATH`) and indices (`INDEX_PATH`) to be written outside the standard application directory.

### Bug 2.2: Hardcoded Log File Directory Misalignment (`config.py` & `main.py`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** `Config.LOG_FILE` defaults to `"log/app.log"`. In `get_logger()`, non-absolute paths are joined against `os.path.dirname(os.path.abspath(__file__))` (`app/log/app.log`). However, execution invocations from different working directories cause log handlers to create disconnected log files across both `app/log/app.log` and `./log/app.log`.

---

## Section 3: CLI & Pipeline Execution (`cli/` & `main.py`)

### Bug 3.1: Placeholder Execution in Single Query Mode (`main.py`)

- **Criticality:** Medium
- **Priority:** P2
- **Explanation:** In `main.py`, when a user specifies the `--query` or `-q` argument, `main()` logs `Executing single query` but outputs `Result for '<query>': [Processing placeholder]`. The single query CLI mode is disconnected from `IngestionPipeline`, vector database search, and RAG retrieval pipelines.

### Bug 3.2: Unimplemented Query Loop in Interactive CLI (`cli/cli_interface.py`)

- **Criticality:** Medium
- **Priority:** P2
- **Explanation:** In `cli_interface()`, the interactive terminal loop accepts user input, logs the query, and prints `Processing query: <query>`, but contains `# some stuff` with no downstream execution logic. Query processing is not connected to search or response generation modules.

---

## Section 4: Core Architecture & Empty Module Directory Shells

### Bug 4.1: Missing Knowledge Acquisition Module (`knowledge_acquisition/`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** The `knowledge_acquisition/` directory is completely empty (0 files). Components responsible for acquiring external knowledge or web scraping are missing from the codebase.

### Bug 4.2: Missing Dataset Storage Directory (`dataset/`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** The `dataset/` directory is completely empty (0 files). Sample corpora or benchmark documents referenced in `Config.DATASET_PATH` are missing from the project workspace.
