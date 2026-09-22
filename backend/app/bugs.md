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
- **Explanation:** `MemoryManager` (`memory_manager.py`) is still an empty class containing `pass`, so the top of the hierarchy has no entry point: nothing resolves a topic/project/conversation triple and hands back a `ConversationPoolManager`. The rest is now implemented — `TopicManager` creates, reads and soft-deletes topics; `ProjectManager` routes a query to a project within a topic; `ConversationPoolManager` owns the `FullConversation` / `ConversationSummary` / `SnapShot` trio for one conversation. What is missing besides `MemoryManager` is the wiring *between* the built layers: nothing passes `(topic_id, query)` from the topic layer to the project layer, and `ProjectManager.route()` stops at a `pass` where the thinking layer will go.

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
### Bug 4.44: PostgreSQL Connections Opened by `SnapShot` Are Never Closed (`snapshot.py`, `conversationVectorManager.py`) — FIXED

- **Criticality:** High
- **Priority:** P1
- **Status:** Fixed. `ConversationVectorManager` gained a `close()`; `SnapShot.close()` releases the vector manager it built (and the metadata repository only if it owns it); `ConversationSummary.close()` closes the snapshot as well as the shared metadata repository. `ConversationPoolManager.close()` already reached the summariser, so the whole chain now releases. Tests in `test_conversation_pool_manager.py::TestConnectionsAreReleased`, mutation-checked.
- **Explanation:** `SnapShot.vector_manager` builds a `ConversationVectorManager` on first use, which opens a `VectorRepository` — a live psycopg connection. `ConversationVectorManager` exposes no `close()`, and `SnapShot.close()` releases only the SQLite metadata repository (`if self._owns_meta_repo: self.meta_repo.close()`). `ConversationSummary.close()` and `ConversationPoolManager.close()` reach the same SQLite repository and nothing else. Every conversation that takes a snapshot or runs a search therefore leaks one PostgreSQL connection for the life of the process, and they scale with the number of conversations opened. Verified with a fake repository: after `with ConversationPoolManager(...) as manager:` exits, the repository's `closed` flag is still `False`. `ProjectMetaData` does close its vector handler — this path simply never grew the equivalent.

### Bug 4.45: `ProjectMetaData` Opens the Shared Registry Without WAL, a Lock, or Thread Safety (`project_meta_data.py`) — FIXED

- **Criticality:** Medium
- **Priority:** P2
- **Status:** Fixed. The connection now goes through `memory/sqlite_setup.connect()` (WAL, `synchronous=NORMAL`, `foreign_keys=ON`) with `check_same_thread=False`, and every statement runs under an `RLock` through `_reading()` / `_writing()` — the same arrangement `ConversationVectorMetaDataRepository` uses. `sqlite_setup` moved from the conversation pool to `memory/` because it is now shared by both. Tests in `test_project_meta_data.py::TestConcurrency`, mutation-checked against removing the lock, the WAL call and `check_same_thread`.
- **Explanation:** `ProjectMetaData.__init__` calls `sqlite3.connect(self.db_path)` directly: no WAL journal, no `RLock`, and `check_same_thread` left at its default, so the connection is bound to the thread that opened it. The file is a single registry shared by every project, so two `ProjectMetaData` instances writing at once contend on a rollback journal where a reader blocks a writer. This is the arrangement already fixed for the conversation database in `sqlite_setup.connect()` (WAL, `synchronous=NORMAL`, `foreign_keys=ON`) and for `ConversationVectorMetaDataRepository` (one connection under a lock). Also tracked in `todo.md` section 2.

### Bug 4.46: `ProjectVectorHandler` and `ProjectMetaData` Disagree on Summary Vectors per Project — FIXED

- **Criticality:** Medium
- **Priority:** P2
- **Status:** Fixed. `ProjectVectorHandler` now holds one summary vector **and** one vector per description for a project, which is the count `project_mapping_table` always allowed. Each id is derived from its source — `summary_vector_id(project_id)` and `description_vector_id(project_id, description_id)` — with the source kind and NUL-separated fields in the payload, so a description whose id is `"summary"` cannot collide with the summary. `get_project_vectors(project_id, vector_ids)` reads a batch in the order given, which is the call the query router needs; the ids come from `ProjectMetaData.get_topic_summary_vector_ids()`, because the vector store cannot enumerate. Covered by 65 tests, mutation-checked.
- **Explanation:** `ProjectMetaData` treats a project as holding *many* summary vectors — `project_mapping_table` is keyed `(project_id, project_summary_vector_id)`, `add_project_vector` takes the id from the caller, and `get_all_summary_vector_id()` returns a list. `ProjectVectorHandler` treats a project as holding *one*, at an id it derives from the project id (`summary_vector_id()`). Both write to the same pgvector table and nothing routes between them, so there is no corruption today, but a project's vectors can be written through two different addressing schemes. The model has to be settled before `ProjectManager` is written, because it decides whether `ProjectMetaData` should delegate its vector half to the handler.

---

### Bug 4.47: `ProjectMetaData.close()` Runs Outside the Lock and Segfaults the Interpreter (`project_meta_data.py`)

- **Criticality:** High
- **Priority:** P1
- **Explanation:** Every statement in this class now runs under `_lock` (Bug 4.45), but `close()` does not — it calls `self.__connection.close()` directly. Closing a SQLite connection while another thread is mid-statement on it does not raise; it crashes the process. Verified with a writer thread appending descriptions while the main thread calls `close()`: two of three runs died with `Segmentation fault (core dumped)`, exit code 139. `ConversationVectorMetaDataRepository.close()` already takes its lock for exactly this reason, with a comment recording the same crash. **Introduced by the 4.45 fix in this session** — the lock was added to the statement paths and not to teardown.

### Bug 4.48: One Missing Vector Makes an Entire Topic Unroutable (`project_manager.py`, `project_vector_handler.py`)

- **Criticality:** Medium
- **Priority:** P2
- **Explanation:** `score_projects()` reads each project's vectors through `get_project_vectors()`, which calls `VectorRepository.batch_search()`, which raises `VectorNotFoundEror` on the first id it cannot find. A mapping row in SQLite whose vector is absent from pgvector therefore takes down routing for **every** project in the topic, not just the damaged one. Verified: two healthy projects route fine; deleting one project's vector while leaving its mapping row makes `resolve()` raise, so the undamaged project becomes unreachable too. The two stores cannot share a transaction, so the rows can diverge — that is the premise the compensating-delete design is built on. A router should degrade to "this project scores nothing" rather than fail closed.

### Bug 4.49: A Project Written With a Caller-Supplied Vector Id Cannot Be Updated Through `ProjectManager` (`project_manager.py`)

- **Criticality:** Medium
- **Priority:** P2
- **Explanation:** `ProjectMetaData.add_project_vector()` takes the vector id from the caller; `ProjectManager` derives it with `summary_vector_id(project_id)`. A project created through the repository directly — which its own API invites — stores its summary vector under a different id, and `ProjectManager.update_project_summary()` then raises `VectorNotFoundEror` for a project that plainly exists. Verified. This is the remaining half of Bug 4.46: the two classes now agree on *how many* vectors a project has but not on *who assigns the id*.

### Bug 4.50: Every `resolve()` Reopens the Registry Twice (`project_manager.py`, `project_meta_data.py`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** `resolve()` calls `list_topic_projects()` and `list_topic_vector_ids()`, each of which opens its own SQLite connection, applies two pragmas, runs one query and drops it. Measured: five `resolve()` calls open ten connections. The two reads also hit the same file for related rows and could be one join. On the routing hot path this is the wrong shape. Separately, `ProjectManager.__meta()` builds a fresh `ProjectMetaData` per write, each of which re-runs `__db_init`'s three `CREATE TABLE IF NOT EXISTS` and a `PRAGMA journal_mode = WAL`.

### Bug 4.51: `ProjectMetaData.__del__` Swallows Every Exception (`project_meta_data.py`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** `__del__` wraps `close()` in `except Exception: pass`, so a connection that fails to close reports nothing. `ConversationVectorMetaDataRepository` handles the same problem differently — it uses `getattr(self, "conn", None)` so a half-constructed object has nothing to close, and lets real failures surface. The silent swallow is the pattern this project removed from `SnapShot`'s compensating delete.

### Bug 4.52: Two Threads Can Create the Same Topic Twice (`topic_manager.py`)

- **Criticality:** Medium
- **Priority:** P2
- **Explanation:** `create_new_topic()` checks `__is_topic_exists()` and then inserts, with nothing holding the two together. `topic_name` carries no UNIQUE constraint — it cannot, because soft delete deliberately leaves old rows with the same name — so nothing at the database level catches the second write. Verified by widening the window between the check and the insert: two threads both created `'same'`, leaving **two active rows with the same topic name and different ids**. `get_topic_id()` then returns whichever `LIMIT 1` happens to pick, and every project filed under the other id becomes unreachable. The same check-then-write shape is in `soft_delete()` and, in the project layer, in `ProjectManager.create_project()`.

### Bug 4.53: `TopicManager` Raises Bare `Exception` (`topic_manager.py`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** All three failure paths raise `Exception("The topic doesn't exists")`, `Exception("topic already exists")` and `Exception("the topic doesn't exists")`. A caller cannot distinguish "no such topic" from "already exists" without matching on message text, and `except Exception` around a call swallows programming errors alongside them. `memory/memory_pool_exceptions.py` already holds eight domain exceptions built for exactly this. The messages are also inconsistently capitalised and read "doesn't exists".

### Bug 4.54: Every Topic Operation Runs Its Query Twice (`topic_manager.py`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** Measured with a SQLite trace callback: `get_topic_id()` issues **two** SELECTs — one to check existence, one to fetch the id that the first query already had in reach — and `soft_delete()` issues two SELECTs plus the UPDATE. Beyond the wasted round trips, the gap between the check and the act is the window Bug 4.52 exploits. One query returning the id or `None` answers both questions atomically.

### Bug 4.55: `TopicManager.query` Is Stored and Never Read (`topic_manager.py`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** The constructor validates `query`, rejects it when empty, assigns `self.query` — and nothing ever reads it. It is presumably there for the handoff to `ProjectManager`, which takes `(topic_id, query)`, but that wiring does not exist yet. Same defect as Bug 4.19 (`project_name` on `ConversationVectorManager`): a required constructor argument that forces callers to supply something the class does not use.

### Bug 4.56: Nothing Can List the Topics (`topic_pool_meta_handler.py`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** The handler can test one topic's existence and fetch its id, both by name. There is no way to ask what topics exist. `MemoryManager` — the layer above, still a stub — has to resolve a topic before it can name one, and the CLI will need to show the user what is there. The rows are present; only the reader is missing.

### Bug 4.57: `topic_id` on the Project Tables Has No Referent (`project_meta_data.py`, `topic_pool_meta_handler.py`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** `project_table`, `project_description_table` and `project_mapping_table` all carry `topic_id text not null`, and `topics_mapping_table` now exists with `topic_id` as its primary key — but they live in **different SQLite files** (`project_db/project.sql` and `topic_db/topic.sql`), so no foreign key can join them. A project can name a topic that was never created, or one that has been soft-deleted, and nothing notices. The only guard is the non-empty check in `ProjectMetaData.__validate_topic_id`. Whether the two registries should share one file is part of the on-disk scheme decision in `todo.md` section 2.

### Bug 4.58: `utc_now` Is Defined Four Times (`memory/`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** Byte-identical copies live in `topic_manager.py`, `topic_pool_meta_handler.py`, `project_meta_data.py` and `fullconversation_repository.py`. The docstring on one of them calls it "canonical timestamp for every row this repository writes", which is exactly the thing four copies cannot guarantee — a change to the format in one leaves the other three writing the old one, into columns that are compared as text.

---

## Section 5: Data Layer — Ingestion, Chunking & Vector Stores (`data_layer/`)

### Bug 5.1: The pgvector Write and Read Paths Cannot Work Against a Real PostgreSQL Server (`vectorRepository.py`) — FIXED IN CODE, BLOCKED ON THE SERVER

- **Criticality:** Critical
- **Priority:** P0
- **Status:** The code side is fixed — `pgvector==0.5.0` is pinned in `requirements.txt` and `register_vector_types(self.conn)` runs in `VectorRepository.__init__`, after `CREATE EXTENSION` and before any vector statement. `batch_insert` now passes numpy arrays rather than `.tolist()`, which was being sent as a PostgreSQL array. **Not yet proven end to end:** the pgvector extension is not installed on the development PostgreSQL server (`pg_available_extensions` has no `vector` row) and the `Vectors` database does not exist, so no code has yet written a vector to a real server. `scripts/smoke.py` reports both preconditions.
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

### Bug 6.1: `download_draft_model.py` Fails Against the Pinned `huggingface-hub` (`download_models/download_draft_model.py`) — FIXED

- **Criticality:** High
- **Priority:** P1
- **Status:** Fixed. `local_dir_use_symlinks` removed; every remaining keyword is checked against the installed `hf_hub_download` signature.
- **Explanation:** The script calls `hf_hub_download(..., local_dir_use_symlinks=False)`. That parameter was deprecated and then removed; `requirements.txt` pins `huggingface-hub==1.21.0`, whose `hf_hub_download` neither accepts it nor takes `**kwargs`. Verified against the installed 1.21.0: `local_dir_use_symlinks` is not in the signature, so the call raises `TypeError` before downloading anything. The draft model cannot be fetched by the documented command, which blocks `ConversationSummary` — the summariser raises `FileNotFoundError` pointing at this same script.

### Bug 6.2: `HOST` Collided With conda's Own Environment Variable (`.env`, `vectorRepository.py`) — FIXED

- **Criticality:** High
- **Priority:** P1
- **Status:** Fixed. All five settings are now `DB_`-prefixed: `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`.
- **Explanation:** Found by running `scripts/smoke.py` for the first time. The connection failed with `failed to resolve host 'x86_64-conda-linux-gnu'` — conda exports `HOST` as its compiler triplet, and `load_dotenv()` does not override a variable already in the environment, so the `HOST=localhost` line in `.env` was silently ignored. This is the same defect the project already fixed once for `USER` -> `DB_USER`, and the same class flagged for `PORT`, which PaaS platforms set. Prefixing all five closes the category rather than the instance.

### Bug 5.12: A Failed Adapter Registration Leaks the Connection (`vectorRepository.py`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** `__init__` connects, then calls `__create_extension()`, `register_vector_types()` and `__create_table()`. If any of those raises — and `register_vector_types` will raise on a server without the pgvector extension, which is the current state of the development database — the exception propagates with `self.conn` still open and no `close()` anywhere. The object is never returned, so nothing can close it either.

---

## Section 7: Documentation & Test Guards

### Bug 7.1: Six Docstrings Were Cut Mid-Sentence by the Trimming Pass

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** The pass that shortened every docstring to its summary kept the first *line* rather than the first *sentence*, so any docstring whose opening sentence wrapped now ends mid-clause. Affected: `normalizer._is_mostly_letters` ("fires on anything without a"), `text_extractor._flatten_json` ("stay attached to what"), `project_meta_data.__validate_topic_id` ("has no table of its own"), `project_meta_data.__validate_summary` ("rather than at the"), `conversationVectorManager.batch_delete` ("undo a partially written"), and `conversation_summary.make_summary` ("the current conversation"). **Introduced in this session.** The full text of each is recoverable from the archive taken before the pass.

### Bug 7.2: The "No Raw `sqlite3.connect`" Guard Covers Two Modules Out of Four (`test_conversation_data_management.py`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** `test_every_conversation_connection_goes_through_connect` inspects `fullconversation_repository` and `conversationVectorMetaManager` only. Two more modules now open SQLite through `memory/sqlite_setup.connect()` and depend on the same per-connection pragmas: `project_meta_data.py` (since Bug 4.45) and `topic_pool_meta_handler.py`. Neither is in the guard's list, so a new method in either could silently get `synchronous=FULL` and foreign keys off — the exact defect the guard exists to prevent. Verified by reading the module list the test imports.

### Bug 7.3: `scripts/smoke.py` Cannot Report a Missing `psycopg` (`scripts/smoke.py`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** `preflight()` exists to report setup problems as instructions instead of tracebacks, and it does that for a missing `pgvector`, missing `.env` keys, an unreachable server, an absent extension and an absent database. It imports `psycopg` unconditionally, though, so the one dependency it cannot report is the one it needs to do the reporting. **Introduced in this session.**
