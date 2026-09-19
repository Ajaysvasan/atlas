# Comprehensive Project Bug Inventory (`backend/app`)

This document catalogs all logical, architectural, and execution pipeline bugs identified across the project codebase. The bugs are organized in a **section-wise manner** grouped by architectural component, complete with detailed explanations, **Criticality** ratings (`Critical`, `High`, `Medium`, `Low`), and recommended resolution **Priority** levels (`P0`, `P1`, `P2`, `P3`).

---

## Section 1: System Configuration & Path Management (`config.py` & `main.py`)

### Bug 1.1: Hardcoded Log File Directory Misalignment (`config.py` & `main.py`) — FIXED

- **Criticality:** Low
- **Priority:** P3
- **Status:** Fixed. Logging moved to `logging_setup.py`, which resolves a relative `LOG_FILE` against the package directory once, in `configure()`. Verified by running `configure()` from `app/` and from `/tmp`: both resolve to `app/log/app.log`.
- **Explanation:** `Config.LOG_FILE` defaults to `"log/app.log"`. In `get_logger()`, non-absolute paths are joined against `os.path.dirname(os.path.abspath(__file__))` (`app/log/app.log`). However, execution invocations from different working directories cause log handlers to create disconnected log files across both `app/log/app.log` and `./log/app.log`.

### Bug 1.2: `Config.DATASET_PATH` Is Resolved Against the Working Directory (`config.py`)

- **Criticality:** Medium
- **Priority:** P2
- **Explanation:** `DATASET_PATH = Path("dataset").resolve().parent / "dataset"` resolves `dataset` against the process's current directory, takes its parent (the current directory) and re-appends `dataset` — so the value is always `<cwd>/dataset`. Every other path constant is anchored to `ABS_PATH`. Verified by importing `config` from two directories: from `app/` it yields `app/dataset`, from `/tmp` it yields `/tmp/dataset`. This is the same class of defect as Bug 1.1, which was fixed for the log path only.

### Bug 1.3: Dead and Duplicated Configuration Constants (`config.py`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** `PROJECT` is assigned twice — `Path("")` at line 46, then the real path at line 59 — so the first assignment is dead. `VECTOR_DIMENSIONS` (a second name for `EMBEDDING_DIMENSIONS`), `MODEL_PATH` and `DATASET_PATH` are referenced nowhere outside `config.py`; `CONVERSATION` survives only in a docstring explaining why it is *not* used. Verified by scanning every non-test module for `Config.<name>`.

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
### Bug 4.44: PostgreSQL Connections Opened by `SnapShot` Are Never Closed (`snapshot.py`, `conversationVectorManager.py`)

- **Criticality:** High
- **Priority:** P1
- **Explanation:** `SnapShot.vector_manager` builds a `ConversationVectorManager` on first use, which opens a `VectorRepository` — a live psycopg connection. `ConversationVectorManager` exposes no `close()`, and `SnapShot.close()` releases only the SQLite metadata repository (`if self._owns_meta_repo: self.meta_repo.close()`). `ConversationSummary.close()` and `ConversationPoolManager.close()` reach the same SQLite repository and nothing else. Every conversation that takes a snapshot or runs a search therefore leaks one PostgreSQL connection for the life of the process, and they scale with the number of conversations opened. Verified with a fake repository: after `with ConversationPoolManager(...) as manager:` exits, the repository's `closed` flag is still `False`. `ProjectMetaData` does close its vector handler — this path simply never grew the equivalent.

### Bug 4.45: `ProjectMetaData` Opens the Shared Registry Without WAL, a Lock, or Thread Safety (`project_meta_data.py`)

- **Criticality:** Medium
- **Priority:** P2
- **Explanation:** `ProjectMetaData.__init__` calls `sqlite3.connect(self.db_path)` directly: no WAL journal, no `RLock`, and `check_same_thread` left at its default, so the connection is bound to the thread that opened it. The file is a single registry shared by every project, so two `ProjectMetaData` instances writing at once contend on a rollback journal where a reader blocks a writer. This is the arrangement already fixed for the conversation database in `sqlite_setup.connect()` (WAL, `synchronous=NORMAL`, `foreign_keys=ON`) and for `ConversationVectorMetaDataRepository` (one connection under a lock). Also tracked in `todo.md` section 2.

### Bug 4.46: `ProjectVectorHandler` and `ProjectMetaData` Disagree on Summary Vectors per Project

- **Criticality:** Medium
- **Priority:** P2
- **Explanation:** `ProjectMetaData` treats a project as holding *many* summary vectors — `project_mapping_table` is keyed `(project_id, project_summary_vector_id)`, `add_project_vector` takes the id from the caller, and `get_all_summary_vector_id()` returns a list. `ProjectVectorHandler` treats a project as holding *one*, at an id it derives from the project id (`summary_vector_id()`). Both write to the same pgvector table and nothing routes between them, so there is no corruption today, but a project's vectors can be written through two different addressing schemes. The model has to be settled before `ProjectManager` is written, because it decides whether `ProjectMetaData` should delegate its vector half to the handler.

---

---

## Section 5: Data Layer — Ingestion, Chunking & Vector Stores (`data_layer/`)

### Bug 5.1: The pgvector Write and Read Paths Cannot Work Against a Real PostgreSQL Server (`vectorRepository.py`)

- **Criticality:** Critical
- **Priority:** P0
- **Explanation:** Nothing registers a pgvector adapter with psycopg — `pgvector` is not in `requirements.txt`, is not installed, and `register_vector()` appears nowhere in the tree. Three consequences, none of which any test can see because `conftest.py` replaces `psycopg` with a `MagicMock`:
  1. `__insert_vector` and `__update_vector` pass a `numpy.ndarray` straight to `cursor.execute`. Verified offline: `psycopg.adapters.get_dumper(numpy.ndarray, ...)` raises `ProgrammingError: cannot adapt type 'ndarray'`. Every single-vector insert and every update fails.
  2. `__insert_batch_vector` calls `.tolist()` first, so psycopg adapts it as a PostgreSQL array and sends `{0.0,1.0}`. pgvector's input syntax is `[0.0,1.0]`, so the server rejects it. (Reasoned from the dumper output, not confirmed against a live server.)
  3. `__get_vector` does `np.asarray(result[0], dtype=float32)` on what comes back. Without the adapter a `vector` column is returned as text; verified that `np.asarray("[0.1,0.2,0.3]", dtype=float32)` raises `ValueError: could not convert string to float`.
  The whole memory-layer vector store — snapshot vectors, cumulative vectors, project summary vectors — depends on this class.

### Bug 5.2: `VectorMetaDataRepository.insert` Always Fails — Its Foreign Key Names a Table in Another Database File (`vectorMetaDataRepository.py`)

- **Criticality:** High
- **Priority:** P1
- **Explanation:** The constructor enables `PRAGMA foreign_keys = ON` and creates `vector_meta_data` with `foreign key (chunkId) references Chunks(chunkId)`. `Chunks` is created by the chunker's `Manager` in a *different* SQLite file (`data/hierarchical_db/`), so the referenced table does not exist in this one. SQLite accepts the `CREATE TABLE` and fails at write time. Verified: a fresh repository contains only `vector_meta_data`, and `insert(1, "chunk_a", "all-MiniLM-L6-v2", 128)` raises `OperationalError: no such table: main.Chunks`. The class has no callers outside a test that only checks it imports, which is why this has gone unnoticed — see Bug 5.3 for why it should have one.

### Bug 5.3: A DiskANN Search Result Cannot Be Resolved Back to Its Chunk (`ingestion_pipeline.py`)

- **Criticality:** High
- **Priority:** P1
- **Explanation:** `EmbeddingManager` derives `vector_id = md5(chunk_id)`, a one-way hash, and `IngestionPipeline` inserts the vector into DiskANN and stops. It never writes a `vector_id -> chunk_id` row: `VectorMetaDataRepository` is never constructed by the pipeline (and is broken anyway, Bug 5.2), and the chunk store's `Chunks` table holds `chunkId, contextId, chunk, startoffset, endoffset` with no vector column. Verified by inspecting the pipeline source and the created schema — no table in the data layer stores a vector id. A search therefore returns ids that nothing can turn back into text, which is the one thing retrieval needs. The pipeline also never calls `VectorDbManager.save()`, so the index is not persisted either (the existing P3 entry).

### Bug 5.4: Section Headings Never Reach a Chunk (`normalizer.py`, `HierarchicalChunker.py`)

- **Criticality:** Medium
- **Priority:** P2
- **Explanation:** The normalizer emits `SectionSpan(name, heading_start, heading_end, content_start, content_end)`, and `HierarchicalChunker.__find_sections` builds each `Section` from `content_start..content_end` — the body only. The heading's own characters lie between `heading_start` and `heading_end` and fall in no section, so no chunk contains them. Verified on a two-heading document: the heading text is present in `NormalizedContent.content` but appears in none of the chunks. The heading is usually the most descriptive line in a section; it survives only as `ChunkMetaData.section_name`, which is never embedded. A query phrased like a heading has nothing to match.

### Bug 5.5: Re-ingesting an Edited Document Leaves the Previous Version Behind Forever (`normalizer.py`, `DB_Manager.py`)

- **Criticality:** High
- **Priority:** P1
- **Explanation:** `document_id` is `sha256(file_name + source_path + normalized_text)`, so editing a file produces a different id and therefore a different `Documents` row. All writes are `on conflict do nothing`, which makes *re-ingesting an unchanged* folder a no-op (as intended) but makes re-ingesting a *changed* file purely additive: the old document, its sections, contexts and chunks stay in SQLite, and their vectors stay in the index, with nothing marking them stale. Verified by chunking two versions of one file into one store: two `Documents` rows and both sets of chunks. Retrieval will keep returning text that no longer exists in the source. There is no delete path anywhere in the data layer to clean it up.

### Bug 5.6: `__generate_document_id` Concatenates Its Fields Without a Separator (`normalizer.py`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** `"".join(str(arg) for arg in args)` over `(file_name, source_path, normalized_text)`. Two different documents whose fields split differently across the same character sequence hash to the same id. Verified: `("report.txt", "/data/2026", body)` and `("report.txt/data", "/2026", body)` produce the same `document_id`. `chunk_id` in the memory layer already separates its fields with `\x00` for exactly this reason; this one does not.

### Bug 5.7: URL and Email Patterns Contain Unintended Character Ranges (`normalizer.py`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** `_replace_urls` uses the class `[$-_@.&+]`, in which `$-_` is a **range** from `\x24` to `\x5F` — digits, uppercase letters, and most punctuation including `.` and `,`. Verified: `"See http://example.com. Next sentence."` normalizes to `"See [URL] Next sentence."`, with the sentence-ending period consumed into the placeholder. The same class also contains `\\(` and `\\)`, which inside a character class means a literal backslash. `_replace_emails` uses `[A-Z|a-z]{2,}` for the TLD, which admits a literal `|`.

### Bug 5.8: `VectorDbManager.load()` Discards What It Loaded (`vectorDbManager.py`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** `idx = self.vector_db.load(load_path)` is assigned under the lock and never read; the method returns `self.vector_db.dynamic_dann` instead. The two happen to be the same object because `VectorDb_diskann.load` assigns `self.dynamic_dann = index` before returning, so the bug is currently invisible — but the dead assignment says the author expected the return value to matter.

### Bug 5.9: `EmbeddedChunkMetaData.modelUsedForChunking` Holds the Embedding Model (`metadata.py`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** `EmbeddingManager.__create_meta_data` passes `self.model_name` — the SentenceTransformer — into a field named for the chunking algorithm. The chunking algorithm is recorded separately, in `ChunkMetaData.chunking_algorithm_used`. Anything later reading this field to learn how a chunk was split gets an embedding model name.

### Bug 5.10: `NormalizedTextMetaData` Has Two Source-Path Fields, One Always `None` (`metadata.py`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** The dataclass declares `source_file_path` (second field) and `source_path` (eighth, defaulting to `None`). The normalizer constructs it with seven positional arguments, so the path lands in `source_file_path` and `source_path` is never set by anything.

### Bug 5.11: Exception Naming and Payload Defects (`datalayer_exceptions.py`, `memory_pool_exceptions.py`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** Three long-standing ones, grouped: `VectorNotFoundEror` is misspelled and is part of the public surface (raised by `VectorRepository`, caught by name in `vectorDbManager` and the memory layer). `InsertionError.message` holds a *table name*, not a message, so `except InsertionError as e: log(e.message)` prints `"Chunks"`. `InvalidVectorDimension` is defined twice — once in `data_layer/datalayer_exceptions` and once in `memory/memory_pool_exceptions` — with the same name and signature but no relationship, so `except` on one silently misses the other (`ProjectVectorHandler` raises the memory one for the same condition `VectorRepository` reports with the data-layer one).

---

## Section 6: Dependencies & Tooling (`requirements.txt`, `download_models/`)

### Bug 6.1: `download_draft_model.py` Fails Against the Pinned `huggingface-hub` (`download_models/download_draft_model.py`)

- **Criticality:** High
- **Priority:** P1
- **Explanation:** The script calls `hf_hub_download(..., local_dir_use_symlinks=False)`. That parameter was deprecated and then removed; `requirements.txt` pins `huggingface-hub==1.21.0`, whose `hf_hub_download` neither accepts it nor takes `**kwargs`. Verified against the installed 1.21.0: `local_dir_use_symlinks` is not in the signature, so the call raises `TypeError` before downloading anything. The draft model cannot be fetched by the documented command, which blocks `ConversationSummary` — the summariser raises `FileNotFoundError` pointing at this same script.

