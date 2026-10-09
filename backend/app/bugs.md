# Comprehensive Project Bug Inventory (`backend/app`)

This document catalogs all logical, architectural, and execution pipeline bugs identified across the project codebase. The bugs are organized in a **section-wise manner** grouped by architectural component, complete with detailed explanations, **Criticality** ratings (`Critical`, `High`, `Medium`, `Low`), and recommended resolution **Priority** levels (`P0`, `P1`, `P2`, `P3`).

---

## Section 1: System Configuration & Path Management (`config.py` & `main.py`)

### Bug 1.1: Hardcoded Log File Directory Misalignment (`config.py` & `main.py`) — FIXED

- **Criticality:** Low
- **Priority:** P3
- **Status:** Fixed. Logging moved to `logging_setup.py`, which resolves a relative `LOG_FILE` against the package directory once, in `configure()`. Verified by running `configure()` from `app/` and from `/tmp`: both resolve to `app/log/app.log`.
- **Explanation:** `Config.LOG_FILE` defaults to `"log/app.log"`. In `get_logger()`, non-absolute paths are joined against `os.path.dirname(os.path.abspath(__file__))` (`app/log/app.log`). However, execution invocations from different working directories cause log handlers to create disconnected log files across both `app/log/app.log` and `./log/app.log`.

### Bug 1.2: `Config.DATASET_PATH` Is Resolved Against the Working Directory (`config.py`) - FIXED

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

### Bug 4.52: Two Threads Can Create the Same Topic Twice (`topic_manager.py`) — FIXED

- **Criticality:** Medium
- **Priority:** P2
- **Status:** Fixed at the database rather than in the caller: `create unique index idx_active_topic_name on topics_mapping_table(topic_name) where is_active = 't'`. The index is **partial**, which is what lets it coexist with soft delete — only active rows are constrained, so a name can cycle through create and delete any number of times while never having two live rows. `create_new_topic()` now inserts and converts `IntegrityError` into `TopicAlreadyExists`, so there is no check to race. Verified with a forced interleave: one thread wins, the other raises, one active row.
- **Explanation:** `create_new_topic()` checks `__is_topic_exists()` and then inserts, with nothing holding the two together. `topic_name` carries no UNIQUE constraint — it cannot, because soft delete deliberately leaves old rows with the same name — so nothing at the database level catches the second write. Verified by widening the window between the check and the insert: two threads both created `'same'`, leaving **two active rows with the same topic name and different ids**. `get_topic_id()` then returns whichever `LIMIT 1` happens to pick, and every project filed under the other id becomes unreachable. The same check-then-write shape is in `soft_delete()` and, in the project layer, in `ProjectManager.create_project()`.

### Bug 4.53: `TopicManager` Raises Bare `Exception` (`topic_manager.py`) — FIXED

- **Criticality:** Low
- **Priority:** P3
- **Status:** Fixed. `TopicNotFound` and `TopicAlreadyExists` added to `memory/memory_pool_exceptions.py`, both carrying the topic name and a message that says which case it is.
- **Explanation:** All three failure paths raise `Exception("The topic doesn't exists")`, `Exception("topic already exists")` and `Exception("the topic doesn't exists")`. A caller cannot distinguish "no such topic" from "already exists" without matching on message text, and `except Exception` around a call swallows programming errors alongside them. `memory/memory_pool_exceptions.py` already holds eight domain exceptions built for exactly this. The messages are also inconsistently capitalised and read "doesn't exists".

### Bug 4.54: Every Topic Operation Runs Its Query Twice (`topic_manager.py`) — FIXED

- **Criticality:** Low
- **Priority:** P3
- **Status:** Fixed. `get_topic_id()` is one SELECT, `create_new_topic()` one INSERT, and `soft_delete()` one UPDATE ... RETURNING — down from two, two and three. Asserted with a SQLite trace callback rather than by reading the code.
- **Explanation:** Measured with a SQLite trace callback: `get_topic_id()` issues **two** SELECTs — one to check existence, one to fetch the id that the first query already had in reach — and `soft_delete()` issues two SELECTs plus the UPDATE. Beyond the wasted round trips, the gap between the check and the act is the window Bug 4.52 exploits. One query returning the id or `None` answers both questions atomically.

### Bug 4.55: `TopicManager.query` Is Stored and Never Read (`topic_manager.py`) — FIXED

- **Criticality:** Low
- **Priority:** P3
- **Status:** Partly fixed. `query` is now optional, so callers are no longer forced to supply something nothing reads; it is still held for the topic -> project handoff, which does not exist yet. The entry closes properly when that wiring lands.
- **Explanation:** The constructor validates `query`, rejects it when empty, assigns `self.query` — and nothing ever reads it. It is presumably there for the handoff to `ProjectManager`, which takes `(topic_id, query)`, but that wiring does not exist yet. Same defect as Bug 4.19 (`project_name` on `ConversationVectorManager`): a required constructor argument that forces callers to supply something the class does not use.

### Bug 4.56: Nothing Can List the Topics (`topic_pool_meta_handler.py`) — FIXED

- **Criticality:** Low
- **Priority:** P3
- **Status:** Fixed. `get_all_topics()` on the handler and `list_topics()` on the manager return `Topic(topic_id, topic_name, created_at)` for every active topic, oldest first. Added for enumeration — `MemoryManager` and the CLI both need it — not to save database hits; see `todo.md` for the measurements that ruled that reasoning out.
- **Explanation:** The handler can test one topic's existence and fetch its id, both by name. There is no way to ask what topics exist. `MemoryManager` — the layer above, still a stub — has to resolve a topic before it can name one, and the CLI will need to show the user what is there. The rows are present; only the reader is missing.

### Bug 4.57: `topic_id` on the Project Tables Has No Referent (`project_meta_data.py`, `topic_pool_meta_handler.py`) — FIXED

- **Criticality:** Low
- **Priority:** P3
- **Status:** **Fixed** by the move to one memory database (Section 4d). `project_table.topic_id` references `topics_mapping_table`; a project under an unknown topic raises `TopicNotFound`, and a soft-deleted topic keeps its row, so its projects stay valid.
- **Explanation:** `project_table`, `project_description_table` and `project_mapping_table` all carry `topic_id text not null`, and `topics_mapping_table` now exists with `topic_id` as its primary key — but they live in **different SQLite files** (`project_db/project.sql` and `topic_db/topic.sql`), so no foreign key can join them. A project can name a topic that was never created, or one that has been soft-deleted, and nothing notices. The only guard is the non-empty check in `ProjectMetaData.__validate_topic_id`. Whether the two registries should share one file is part of the on-disk scheme decision in `todo.md` section 2.

### Bug 4.58: `utc_now` Is Defined Four Times (`memory/`) — FIXED

- **Criticality:** Low
- **Priority:** P3
- **Status:** Fixed. `storage/timestamps.py` (originally `memory/timestamps.py`, moved once `knowledge_sufficiency` needed it too) holds `utc_now` and `as_timestamp`; the four modules import from it and re-export, so existing imports keep working. A test asserts none of them defines its own.
- **Explanation:** Byte-identical copies live in `topic_manager.py`, `topic_pool_meta_handler.py`, `project_meta_data.py` and `fullconversation_repository.py`. The docstring on one of them calls it "canonical timestamp for every row this repository writes", which is exactly the thing four copies cannot guarantee — a change to the format in one leaves the other three writing the old one, into columns that are compared as text.

---

## Section 4b: The `conversation_id` Change (`conversation_pool/`)

A `conversation_id` dimension was added so one project's database can hold more
than one conversation. The column reached the schema and most of the SQL; it did
not reach the row builders, three query bodies, the primary key, the sequence
allocator, the `Turn` readers, or the second module that owns the same table.
Every entry was **reproduced, not inferred**, and every fix verified the same way.

**All thirteen are fixed.** Three further pieces were then built on top: the
monotonic `seq` watermark, the project snapshot tables, and `scope=` on
`SnapShot`.

### Bug 4.59: `append_turns` Built Rows With the Old Arity, So Every Turn Write Failed (`fullconversation_repository.py`) — FIXED

- **Criticality:** Critical
- **Priority:** P0
- **Explanation:** `__append_turns` inserted six columns into `full_conversation` and five into `summary_chunks`, but the row builders above it were not updated: `chunk_rows.append((chunk_id, text, created_at, CHUNKER_TYPE_TURN))` was four values and `meta_rows.append((self.project_id, sequence, chunk_id, role, created_at))` was five. `append_turns([("user", "hello")])` raised `sqlite3.ProgrammingError: Incorrect number of bindings supplied. The current statement uses 5, and there are 4 supplied.` This is the primary write path, so nothing could be written at all. Same class as the topic-layer arity bug (4.46).
- **Status:** Fixed — both builders supply `self.conversation_id`. Verified: `append_turns` returns `[1, 2]`.

### Bug 4.60: Three Queries Had `ORDER BY` Before `WHERE`, or Two `WHERE` Clauses (`fullconversation_repository.py`) — FIXED

- **Criticality:** Critical
- **Priority:** P0
- **Explanation:** The `conversation_id` filter was appended after the `ORDER BY` in three bodies. `__get_last_n_chunks` also had a stray comma terminating its `ON` clause (`OperationalError: near "ORDER": syntax error`); `__get_ranged_chunks` had two `where` clauses, the second after the `order by` (`near "where": syntax error`); `__get_all_from_conversation` had `order by` before `where` (same). Unconditional — the statements could not be prepared, so the methods failed on every call regardless of data.
- **Status:** Fixed — all three read `WHERE ... ORDER BY ...`, comma removed, duplicate `where` gone. Verified: `get_n_chunks`, `get_ranged_chunks` and `fetch_all` all return rows.

### Bug 4.61: The `Turn` Readers Were Not Conversation-Scoped and Leaked Across Conversations (`fullconversation_repository.py`) — FIXED

- **Criticality:** Critical
- **Priority:** P0
- **Explanation:** `_TURN_QUERY` was not touched, so `get_turns`, `get_last_n_turns`, `get_turns_after` and `get_all_turns` selected every row in the file. Verified with two repositories on one database, one turn each: `conv_A.get_all_turns()` returned `['A-one', 'B-one']`, and so did `conv_B`'s. These are the readers `CLAUDE.md` tells callers to prefer for prompt assembly, so the failure mode was one conversation's history silently appearing in another's prompt. **It did not raise**, which made it the most dangerous entry here — and `get_conversation_size()` *was* scoped, so size and contents contradicted each other.
- **Status:** Fixed — the filter moved into `_TURN_QUERY` itself (`WHERE f.conversation_id = ?`), its parameter bound by `__select_turns`, and every caller's clause continues with `AND`, so no future reader can forget it. Verified: each conversation sees only its own turns and sizes agree with contents.

### Bug 4.62: `sequence_number` Was Still a Global Primary Key, So a Second Conversation Could Not Start (`fullconversation_repository.py`) — FIXED

- **Criticality:** Critical
- **Priority:** P0
- **Explanation:** `full_conversation` declared `sequence_number int primary key`. Sequence numbers are per-conversation, so two conversations in one database both wanted to start at 1. Verified: seeding `conv_B` at sequence 1 raised `IntegrityError: UNIQUE constraint failed: full_conversation.sequence_number`. This defeated the purpose of the change — the column was added so one database could hold several conversations, and the key still forbade it.
- **Status:** Fixed — `primary key (conversation_id, sequence_number)`. Verified: both conversations start at 1. One consequence, caught by `test_indexes.py`: a lookup by `sequence_number` alone no longer hits the key and scans. Every query the repository issues now pairs it with `conversation_id`, so the test was updated to the shape the code actually uses.

### Bug 4.63: The Sequence Allocator Ignored `conversation_id` (`fullconversation_repository.py`) — FIXED

- **Criticality:** High
- **Priority:** P1
- **Explanation:** `__next_sequence_number` ran `SELECT COALESCE(MAX(sequence_number), 0) FROM full_conversation;` with no `WHERE`, so a new conversation's first turn was numbered after the last turn of every other conversation in the file.
- **Status:** Fixed — filters `WHERE conversation_id = ?`. The `BEGIN IMMEDIATE` around it was already correct and is still needed. Verified: `conv_B`'s second turn is sequence 2, not 3.

### Bug 4.64: `summary_chunks` Was Created by Two Modules With Different Schemas (`fullconversation_repository.py`, `conversationVectorMetaManager.py`) — FIXED

- **Criticality:** Critical
- **Priority:** P0
- **Explanation:** Both classes share one database file and both ran `CREATE TABLE IF NOT EXISTS summary_chunks`. The conversation repository's definition had `conversation_id`; `ConversationVectorMetaDataRepository`'s did not. `IF NOT EXISTS` meant the second to run accepted whatever the first created. Verified by constructing the metadata repository first: `summary_chunks` came out as `['chunk_id', 'chunk', 'created_at', 'chunker_type']` and the next insert raised `OperationalError: table summary_chunks has no column named conversation_id`. Which class initialised first was not controlled anywhere, so this was order-dependent.
- **Status:** Fixed — both definitions agree column-for-column, verified in both initialisation orders. That made the metadata repository's own two 4-column inserts fail, so it now takes a `conversation_id` and supplies it itself rather than widening every caller's tuple; `SnapShot` and `ConversationSummary` forward it.

### Bug 4.65: No Migration, So Every Existing Database Broke (`fullconversation_repository.py`) — FIXED

- **Criticality:** Critical
- **Priority:** P0
- **Explanation:** The schema change was expressed only as `CREATE TABLE IF NOT EXISTS`, which does nothing to a table that already exists. Any database written before the change kept the old columns and its first insert failed as in 4.64. There was no `ALTER TABLE`, no schema version and no detection; deleting `data/memory/` was the only recovery.
- **Status:** Fixed — `conversation_pool/schema_migrations.py`, run from both modules that share the database, keyed on `PRAGMA user_version`. `full_conversation` needs a new primary key and SQLite cannot alter one, so both tables are rebuilt rather than ALTERed, with existing rows assigned `conversation_id = 'legacy'` — not a placeholder, since a project database held exactly one conversation before the column existed. Version 2 then added `conversation_id` and `seq` to `cumulative_vector_meta_data`, backfilling `seq` in the rows' existing chronological order. Verified end to end from both a v0 and a v1 database: rows preserved and readable, schema identical to a fresh one, no dangling foreign keys, and a new conversation starting at sequence 1 beside legacy's 1-3.

### Bug 4.66: `ConversationSummary` Was Passed a `conversation_id` It Did Not Accept (`conversation_pool_manager.py`, `conversation_summary.py`) — FIXED

- **Criticality:** Critical
- **Priority:** P0
- **Explanation:** `ConversationPoolManager.__init__` called `ConversationSummary(..., conversation_id=conversation_id)`, but `ConversationSummary.__init__` took no such parameter, so constructing a `ConversationPoolManager` raised `TypeError`. Its own `FullConversation(...)` call omitted it too. The manager is the public entry point to the conversation layer and could not be constructed at all.
- **Status:** Fixed — `conversation_id` is a required parameter, forwarded to `FullConversation`, the summary repository and the `SnapShot`.

### Bug 4.67: The Join Condition Was Inconsistent Between Readers (`fullconversation_repository.py`) — FIXED

- **Criticality:** Medium
- **Priority:** P2
- **Explanation:** `__get_ranged_chunks` joined on `chunk_id` and `conversation_id`, while four other readers joined on `chunk_id` alone and filtered `conversation_id` in the `WHERE`. Both gave the same answer only while `chunk_id` stayed globally unique, and nothing enforced that the two denormalised copies agreed.
- **Status:** Fixed — every reader joins on both columns and filters `f.conversation_id`.

### Bug 4.68: `idx_full_conversation_chunk` May No Longer Serve the Queries It Was Measured Against (`fullconversation_repository.py`) — FIXED (no change needed)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** The index on `full_conversation(chunk_id)` was added on measurement for the watermark join in `get_highest_summarised_sequence()`. Once rows carried a `conversation_id`, `(conversation_id, chunk_id)` looked like the natural shape, and the conversation-scoped readers filter `conversation_id` while `__get_ranged_chunks` joins on both columns — neither of which this index covers.
- **Status:** **Re-measured, and the current index is correct. Do not change it.** At 5 conversations x 8000 turns, on the watermark join: `(chunk_id)` 3724us, `(conversation_id, chunk_id)` **25960us**, no index 15024us. The composite is 7x slower than the current index and worse than having none, because the join is project-wide and never constrains `conversation_id`, so a `conversation_id`-leading index cannot be seeked — SQLite skip-scans it, which the plan reports as `ANY(conversation_id) AND chunk_id=?`. `(chunk_id, conversation_id)` measured level with the current index (3786us) and is larger for no gain.
  The conversation-scoped readers need nothing: the primary key became `(conversation_id, sequence_number)` in Bug 4.62, which already covers every reader that filters a conversation and walks its sequence numbers. No index variant moved `ranged chunks`, `all turns` or `conversation size` at all.
  Two guards in `test_indexes.py` pin this: one asserts the watermark plan does not skip-scan, the other that the scoped readers are served by the primary key. Both fail if the index is changed to the composite — which is the mistake they exist to catch.
- **Method note:** the first run of this benchmark was invalid and said all four variants were equal. The repository creates `idx_full_conversation_chunk` in `__init_db`, so every "variant" was really that index *plus* the variant, and the "no index" column was not one. Dropping it first is what produced the numbers above. Same class of error as the earlier benchmark that declared `chunk_id unique` and invented a conflict that did not exist.

### Bug 4.69: `conversation_id` Was Accepted but Never Validated — FIXED

- **Criticality:** Medium
- **Priority:** P2
- **Explanation:** `conversation_id` was stored as given and written into `not null` columns with no check. `None` reached SQLite and failed with a constraint error naming the column rather than the caller; `""` was accepted silently and partitioned nothing. `project_meta_data.py` validates its ids; this did not.
- **Status:** Fixed — `memory/identifiers.py::require_identifier` rejects non-strings and blanks and returns the value stripped, raising `InvalidIdentifier`, which names the field and shows the value. It sits beside `memory_manager.py` because `memory/snapshot.py` needs it as well as the conversation pool. Applied where the id is stored or written; the pass-through layers inherit it, so construction still fails immediately. `conversation_id` is now required on the metadata repository and `SnapShot` rather than defaulting to `""`.

### Bug 4.70: Stray Trailing Commas in Two Signatures (`fullconversation_repository.py`) — FIXED

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** `def __get_all_from_conversation(self , ):` and `def __get_conversation_size(self , ):` carried a trailing comma and spacing that suggested a parameter was removed mid-edit. Valid Python, but both read as unfinished.
- **Status:** Fixed — signatures normalised.

### Bug 4.71: `snapshot.py` Was Moved and Four Importers Still Named the Old Path (`memory/snapshot.py`, `conversation_summary.py`) — FIXED

- **Criticality:** Critical
- **Priority:** P0
- **Explanation:** `snapshot.py` moved from `conversation_pool/` to `memory/`. Verified a pure move — the old and new files differed in nothing — so the only damage was stale imports. Four places still named the old path, and **one was production code**: `conversation_summary.py` raised `ModuleNotFoundError`. Because `conversation_pool_manager.py` imports `ConversationSummary`, the entire conversation layer failed at import and pytest aborted with `Interrupted: 4 errors during collection` — **the suite could not even be collected**, so no test anywhere ran.
- **Status:** Fixed — the import is `from memory.snapshot import SnapShot`, and the three test-side references were repointed. The move is recorded as a git rename.

### Bug 4.72: `cumulative_vector_meta_data` Had No Monotonic Ordering Column — FIXED

- **Criticality:** High
- **Priority:** P1
- **Explanation:** Snapshot order came from `ORDER BY datetime(created_at), created_at`. `datetime()` truncates to whole seconds, so two snapshots in the same second tied and their order went arbitrary — and `SnapShot`'s cursors index into that list. `created_at` is also caller-supplied TEXT with no format enforcement, the reason Bug 4.31 was possible, so a caller could perturb the order with an odd timestamp.
- **Status:** Fixed — a `seq` column, allocated monotonically on write. Not `AUTOINCREMENT`: that is only available on an `INTEGER PRIMARY KEY`, and `cumulative_vector_id` already holds that position with a derived hash. **Allocating it exposed a second bug:** `_writing()` took only the instance's `RLock`, and two repositories on one database file have one lock each, so the `MAX(seq)+1` read and the insert interleaved across them and both claimed the same `seq` — `UNIQUE constraint failed`, caught by an existing concurrency stress test. Resolved with `BEGIN IMMEDIATE`, so the database's own write lock serialises them, plus a depth counter so a nested `_writing()` stays in the outer transaction. Four ordering tests changed meaning as a result and were rewritten as the inverse guard: timestamps that disagree with write order must not reorder anything.

---

## Section 4c: `memory_mapping_handler.py` (new, uncommitted)

A mapping from `(conversation_id, user_id)` to the topic, project and latest
project snapshot that conversation belongs to — the lookup `MemoryManager` needs
to resume a conversation without re-routing it. When recorded, **the module did not import and no method worked**. Every entry
below was reproduced before being fixed, and every fix verified the same way.
**All eleven are now fixed**, with 16 regression tests in
`test/memory_layer_testing/test_memory_mapping_handler.py`; reverting any one
fix fails between 1 and 11 of them.

Also note: this introduces `user_id`, a dimension no other table in the project
has. Nothing else is user-scoped, so either this is the first step of a decision
that has to reach the other tables, or it is scope that does not belong yet.

### Bug 4.73: Production Code Imports the Test Suite (`memory_mapping_handler.py`) — FIXED

- **Criticality:** Critical
- **Priority:** P0
- **Status:** **Fixed.** The import is gone. Verified by a production import: `'pytest' in sys.modules` is now `False` afterwards.
- **Explanation:** Line 6 is `from test.memory_layer_testing.test_project_snapshot import project` — a pytest fixture, unused, almost certainly an editor auto-import. It is not a dead line that fails harmlessly: it **resolves**. Verified that importing the module pulls in the test module, and with it `pytest`, into the running process (`'pytest' in sys.modules` is `True` afterwards). So a production import drags the test suite, its mocks and its dependencies along, and would fail outright wherever tests are not shipped.

### Bug 4.74: A Bare Import That Only Resolves Under pytest (`memory_mapping_handler.py`) — FIXED

- **Criticality:** Critical
- **Priority:** P0
- **Status:** **Fixed.** `from memory.sqlite_setup import connect, enable_wal`. Verified outside pytest, with no conftest involved: the module imports and `memory/` is not on `sys.path`.
- **Explanation:** Line 9 is `from sqlite_setup import connect, enable_wal`, not `from memory.sqlite_setup import ...` as every other module in the layer writes it. Under `python`/`main.py` this raises `ModuleNotFoundError: No module named 'sqlite_setup'` — verified. Under pytest it **works**, because `test/memory_layer_testing/conftest.py` puts `memory/` on `sys.path` as a workaround for Bug 4.23. So the module imports in the test environment and fails in production, which is the one combination a test suite cannot warn about. The conftest line was a workaround for one module's bare import; it now silently licenses new ones.

### Bug 4.75: `__db_init` Is Never Called, So the Table Never Exists (`memory_mapping_handler.py`) — FIXED

- **Criticality:** Critical
- **Priority:** P0
- **Status:** **Fixed.** `__init__` calls `__db_init()`. Verified: the table exists immediately after construction, and WAL reports `wal`.
- **Explanation:** `__init__` connects, enables WAL and builds the lock, but never calls `__db_init()`. Verified: immediately after construction `sqlite_master` holds no tables, and every method raises `OperationalError: no such table: memory_mapping_table`. Every other repository in this layer calls its own initialiser from `__init__`.

### Bug 4.76: The Column Is Declared `lastest_` and Updated as `latest_` (`memory_mapping_handler.py`) — FIXED

- **Criticality:** Critical
- **Priority:** P0
- **Status:** **Fixed** by renaming the column to `latest_project_snapshot_id`, since five other spellings in the file already used `latest_`. Both write paths now work.
- **Explanation:** The DDL declares `lastest_project_snapshot_id` (transposed letters). `__search` selects that spelling and works; both write paths — `__insert_into_mapping_table` and `__update_latest_project_snapshot_id` — name `latest_project_snapshot_id` and raise `OperationalError: no such column: latest_project_snapshot_id`. Verified. So the column can be read and never written. Worth fixing by renaming the **column**, since five other spellings in the file already use `latest_`.

### Bug 4.77: No Key on the Lookup Columns, So a Conversation Can Appear Twice (`memory_mapping_handler.py`) — FIXED

- **Criticality:** High
- **Priority:** P1
- **Status:** **Fixed.** `primary key (conversation_id, user_id)`, plus an index on `(project_id, user_id)` for the snapshot-pointer rewrite, which no part of the key covers. Verified: a duplicate `populate_conversation_id` raises `IntegrityError`.
- **Explanation:** `memory_mapping_table` declares no primary key and no unique constraint — verified, `pragma table_info` reports no key column. `populate_conversation_id` is a plain INSERT, so calling it twice for one conversation leaves two rows (verified: count 2). The table exists to answer "which project is this conversation in", and two rows make that unanswerable; `search` then returns whichever row comes first. `(conversation_id, user_id)` is the natural key.

### Bug 4.78: The Routing UPDATE Supplies Four Values for Six Placeholders (`memory_mapping_handler.py`) — FIXED

- **Criticality:** Critical
- **Priority:** P0
- **Status:** **Fixed.** All six values are supplied in order. The private method now returns `cursor.rowcount` and the public one logs a warning when the UPDATE matched nothing — SQLite does not treat that as an error, so the caller would otherwise read it as success. Verified: routing one conversation leaves another's row untouched.
- **Explanation:** `__insert_into_mapping_table`'s statement has six placeholders — four in the `SET`, two in the `WHERE conversation_id = ? and user_id = ?` — and the parameter tuple is `(topic_id, project_id, new_latest_project_snapshot_id, created_at)`, four values. Reproduced once Bug 4.76 is out of the way: `ProgrammingError: Incorrect number of bindings supplied. The current statement uses 6, and there are 4 supplied.` Had the arity matched by accident, the missing `WHERE` values would have made it update **every row in the table**. Same class as Bugs 4.59 and 4.46.

### Bug 4.79: `if not None` Is Always True, So an Empty Search Raises (`memory_mapping_handler.py`) — FIXED

- **Criticality:** High
- **Priority:** P1
- **Status:** **Fixed.** `fetchone()` with `return MemoryMapping(*row) if row is not None else None`, and the result is a `MemoryMapping` NamedTuple rather than a bare tuple, matching `Turn`, `ProjectRow` and `ProjectSnapshotRow`.
- **Explanation:** `__search` ends `return row[0] if not None else None`. The condition tests the literal `None`, not `row`, and `not None` is `True` always, so the guard never fires and an empty result is indexed. Verified: `search()` on an empty table raises `IndexError: list index out of range` where the shape of the line says it should return `None`. It is also `fetchall()[0]` rather than `fetchone()`, so it reads every matching row to use one.

### Bug 4.80: `db_path` Has No Default Although the Body Handles `None` (`memory_mapping_handler.py`) — FIXED

- **Criticality:** Low
- **Priority:** P3
- **Status:** **Fixed.** `db_path: str | Path | None = None`.
- **Explanation:** The signature is `db_path: str | Path | None` with no `= None`, so callers must pass something, while the body carefully falls back to `Config.DATA_DIR / "memory_mapping/memory_mapping.sql"` for a `None` it can only receive if passed explicitly. `ProjectSnapshotRepository` and `ProjectMetaData` both default theirs.

### Bug 4.81: `conversation_id` and `user_id` Are Not Validated (`memory_mapping_handler.py`) — FIXED

- **Criticality:** Medium
- **Priority:** P2
- **Status:** **Fixed.** `require_identifier` on `conversation_id`, `user_id` and `project_id` at every public entry point.
- **Explanation:** Both are written into the table and filtered on, and neither is checked. `""` would be accepted and scope nothing, exactly as in Bug 4.69 — for which `memory/identifiers.py::require_identifier` already exists and is used by four other classes.

### Bug 4.82: No `close()`, and the Unused Constructor Parameter (`memory_mapping_handler.py`) — FIXED

- **Criticality:** Medium
- **Priority:** P2
- **Status:** **Fixed.** `close()` takes `_lock` before closing, is safe to call twice, and is reached from `__del__` via `getattr` so a failed construction does not bury its own error. **Both** dead constructor parameters were dropped, not just `query`: `conversation_id` was never stored either, and every method names the conversation it acts on — so this is a handler for the whole table, which is what the class docstring now says. Unused imports removed.
- **Explanation:** The class opens a connection with `check_same_thread=False` and offers no way to release it — verified, `hasattr(h, "close")` is `False`. Every other repository in this layer has one, and Bug 4.47 is the reminder that it must take `_lock` or closing under an in-flight write segfaults the interpreter rather than raising. Separately, `__init__` accepts `query: str` and never stores or reads it — the same dead parameter as Bug 4.55 in `TopicManager`. The unused imports (`date`, `datetime`, `timezone`, `List`, `NamedTuple`, `Sequence`, `Tuple`) are the same editor noise as Bug 4.73.

### Bug 4.83: The Snapshot Pointer Is Duplicated Per Conversation (`memory_mapping_handler.py`) — FIXED

- **Criticality:** Low
- **Priority:** P3
- **Status:** **Resolved as a documented cache, not a second source of truth.** A comment in `__db_init` records that `latest_project_snapshot_id` is a cache of `ProjectSnapshotRepository.latest()`, which owns the ordered chain in `project_snapshot_mapping`; it exists so resuming a conversation does not have to open the project registry. This follows the precedent of denormalising `topic_id` onto the project tables for the read the router needs.
- **Explanation:** `__update_latest_project_snapshot_id` keys on `(project_id, user_id)`, so it rewrites the pointer on every conversation row belonging to that project — the value is stored once per conversation rather than once per project. It is not wrong, and the comment in `__db_init` says the overwrite is intended, but `project_snapshot_mapping` in the project registry already holds the ordered chain of a project's snapshots, and `ProjectSnapshotRepository.latest()` already answers "the newest one". Worth deciding whether this column is a cache of that or a second source of truth.

---

## Section 4d: One Memory Database (found during the migration)

Every memory table moved into `data/memory/memory_layer/memory_layer.db`, with the relationships between them declared as foreign keys (see `memory/README.md`). Moving them surfaced these.

### Bug 4.84: Two Conversations in One Project Could Not Open With the Same Words (`fullconversation_repository.py`) — FIXED

- **Criticality:** High
- **Priority:** P1
- **Explanation:** A turn's `chunk_id` was `sha256(project_id, sequence_number, text)`. Since one project file held several conversations (4.64), two of them opening with "hello" produced one id, and the second failed with `UNIQUE constraint failed: summary_chunks.chunk_id`. Reproduced directly. The most common opening message in any chat was enough to break a second conversation.
- **Status:** **Fixed.** `chunk_id` binds `conversation_id` too. Existing ids are not rewritten — pgvector stores vectors under ids derived from them. `TestTwoConversationsMaySayTheSameThing`.

### Bug 4.85: A Conversation's Summarised Watermark Was Read Across the Whole Project (`conversationVectorMetaManager.py`) — FIXED

- **Criticality:** High
- **Priority:** P1
- **Explanation:** `get_highest_summarised_sequence()` filtered on `project_id` alone. A conversation with 10 turns and no snapshot reported a watermark of **50**, borrowed from a sibling; `turns_since_last_snapshot()` became `max(0, 10 − 50) = 0`, so the new conversation's snapshot trigger stayed silent until it overtook the old one. Reproduced with two conversations in one project.
- **Status:** **Fixed** by filtering on `conversation_id` as well, with `idx_summary_vector_chunk` added: the per-conversation query drives from the conversation's primary key and probes by chunk, 5.3 ms → 3–5 µs per call, for +75% on a 0.37 µs insert. The plan is pinned in `test_indexes.py`; the regression is `TestTheWatermarkIsPerConversation`.

### Bug 4.86: Tests Wrote Into the Real Project Registry, and One Passed Only Because of It (`test_conversation_pool_manager.py`, `project_snapshot.py`) — FIXED

- **Criticality:** Medium
- **Priority:** P2
- **Explanation:** `ProjectSnapshot` defaulted to the real `data/project_db/project.sql`, and the pool manager's end-to-end test reached it. The three rows in that file — project `proj_pool`, summary "A rolled-up summary." — are that test's output. Worse, `test_a_snapshot_sends_the_model_who_said_what` asserted the model was called **once** per snapshot and passed only because those rows existed: with the watermark already ahead, the project snapshot found nothing pending and skipped its call. Isolated, the roll-forward runs, and the model is called twice, as designed.
- **Status:** **Fixed.** The root `conftest.py` points `Config.MEMORY_DB` at a fresh file per test; `TestTheSuiteNeverTouchesTheRealDatabase` guards the redirect; the test now asserts both calls and what each prompt holds. The stray rows remain in the old file, which nothing reads.

### Bug 4.87: `seq` Was Unique Per File, Which Held Only While Each Project Had Its Own (`conversationVectorMetaManager.py`) — FIXED

- **Criticality:** High (latent)
- **Priority:** P1
- **Explanation:** `seq integer not null unique` and `MAX(seq) + 1` over the whole table. Correct while a file held one project; in a shared database two projects would collide on `seq` 1.
- **Status:** **Fixed** before it could fire: `UNIQUE (project_id, seq)` and allocation per row's project, shared across that project's conversations so the project watermark keeps its order. `TestSeqIsPerProject`.

### Bug 4.88: The Mapping Table Cached the Latest Snapshot by Hand, as Text (`memory_mapping_handler.py`) — FIXED

- **Criticality:** Medium
- **Priority:** P2
- **Explanation:** `latest_project_snapshot_id` copied `ProjectSnapshotRepository.latest()` so resuming would not open a second file, and was stale whenever `update_latest_project_snapshot_id` was not called. It was also `text` against an `integer` key.
- **Status:** **Fixed** by removing it: `search()` reads the chain each time and returns the integer id. `update_latest_project_snapshot_id` and the `new_latest_project_snapshot_id` argument are gone.

### Bug 4.89: Relationships Kept by Code, or Not at All (`project_meta_data.py`, `project_snapshot_repo.py`, `fullconversation_repository.py`, `memory_mapping_handler.py`) — FIXED

- **Criticality:** Medium
- **Priority:** P2
- **Explanation:** Across files nothing could be a foreign key, and some relationships were not one even within a file: `project_snapshot.project_id` had no foreign key though it shared `project.sql` with `project_table`. `ProjectMetaData` kept the child tables' `topic_id` in step with an UPDATE loop that a write skipping the upsert never ran. A turn's two copies of `conversation_id` were never required to agree, and roles were checked only in Python.
- **Status:** **Fixed.** Every relationship in the layer is a foreign key; denormalised copies reference the pair; a project's topic move cascades; `CHECK` holds roles, `is_active` and the routing row's topic/project pairing; refusals are named. Each relationship has a test, and 15 mutations removing them were all caught.

### Bug 7.6: `test_config_paths.py` Left a Different `config` Module Behind for Every Later Test — FIXED

- **Criticality:** Medium
- **Priority:** P2
- **Explanation:** The tests pop `config` from `sys.modules` and re-import it, and never put the original back. Every later test saw a different `Config` class from the one already-imported modules held, so a monkeypatch of one silently missed the other. Found because the memory database redirect missed `MemoryDatabase` in the full suite and a test created the real database file.
- **Status:** **Fixed** with an autouse fixture restoring the original module. Removing it fails the redirect guard.

## Section 5: Data Layer — Ingestion, Chunking & Vector Stores (`data_layer/`)

### Bug 5.1: The pgvector Write and Read Paths Cannot Work Against a Real PostgreSQL Server (`vectorRepository.py`) — FIXED AND PROVEN END TO END

- **Criticality:** Critical
- **Priority:** P0
- **Status:** **Fixed and proven against a real server.** `pgvector==0.5.0` is pinned in `requirements.txt` and `register_vector_types(self.conn)` runs in `VectorRepository.__init__`, after `CREATE EXTENSION` and before any vector statement. `batch_insert` passes numpy arrays rather than `.tolist()`, which was being sent as a PostgreSQL array. The server side is now set up: Fedora's `pgvector-0.8.0-1.fc43`, database `Vectors`, `CREATE EXTENSION vector` applied. `scripts/smoke.py` completes a full round trip — ingest, embed, write, read back — and reports the returned array identical to the one written.
- **What proving it cost:** the "fixed in code" claim above was wrong, and the smoke test found it in one run. Two further defects (Bugs 5.13 and 5.14) sat behind it, both invisible to every existing test because `conftest.py` replaces `psycopg` with a `MagicMock` — a mock cursor adapts anything handed to it and returns whatever you tell it to. No amount of mocked testing could have reached either. This is the argument for `scripts/smoke.py` existing at all.
- **Explanation:** Nothing registers a pgvector adapter with psycopg — `pgvector` is not in `requirements.txt`, is not installed, and `register_vector()` appears nowhere in the tree. Three consequences, none of which any test can see because `conftest.py` replaces `psycopg` with a `MagicMock`:
  1. `__insert_vector` and `__update_vector` pass a `numpy.ndarray` straight to `cursor.execute`. Verified offline: `psycopg.adapters.get_dumper(numpy.ndarray, ...)` raises `ProgrammingError: cannot adapt type 'ndarray'`. Every single-vector insert and every update fails.
  2. `__insert_batch_vector` calls `.tolist()` first, so psycopg adapts it as a PostgreSQL array and sends `{0.0,1.0}`. pgvector's input syntax is `[0.0,1.0]`, so the server rejects it. (Reasoned from the dumper output, not confirmed against a live server.)
  3. `__get_vector` does `np.asarray(result[0], dtype=float32)` on what comes back. Without the adapter a `vector` column is returned as text; verified that `np.asarray("[0.1,0.2,0.3]", dtype=float32)` raises `ValueError: could not convert string to float`.
  The whole memory-layer vector store — snapshot vectors, cumulative vectors, project summary vectors — depends on this class.

### Bug 5.2: `VectorMetaDataRepository.insert` Always Fails — Its Foreign Key Names a Table in Another Database File (`vectorMetaDataRepository.py`) — FIXED

- **Criticality:** High
- **Priority:** P1
- **Status:** **Fixed.** The table moved into the chunk store's own database file, so it sits beside `Chunks` and a search result becomes text through one join. The foreign key was **removed rather than repaired**: a `chunkId` lives in `Chunks` when the document had sections and in `RecursiveChunks` when it did not, and SQLite cannot reference whichever of two tables holds it. The repository now also goes through `storage.sqlite_setup.connect` (WAL, pragmas) and runs every statement under an `RLock`, like the rest of the project's stores.
- **Explanation:** The constructor enables `PRAGMA foreign_keys = ON` and creates `vector_meta_data` with `foreign key (chunkId) references Chunks(chunkId)`. `Chunks` is created by the chunker's `Manager` in a *different* SQLite file (`data/hierarchical_db/`), so the referenced table does not exist in this one. SQLite accepts the `CREATE TABLE` and fails at write time. Verified: a fresh repository contains only `vector_meta_data`, and `insert(1, "chunk_a", "all-MiniLM-L6-v2", 128)` raises `OperationalError: no such table: main.Chunks`. The class has no callers outside a test that only checks it imports, which is why this has gone unnoticed — see Bug 5.3 for why it should have one.

### Bug 5.3: A DiskANN Search Result Cannot Be Resolved Back to Its Chunk (`ingestion_pipeline.py`) — FIXED

- **Criticality:** High
- **Priority:** P1
- **Status:** **Fixed, and verified end to end.** `IngestionPipeline` now allocates a label from `vector_meta_data` for each chunk and indexes the vector under it, so the mapping row exists before the vector does and a hit can never arrive for a chunk the table has not heard of. Verified on a real ingest: a DiskANN search returns labels, `chunk_ids_for` resolves them, and one join returns the text. `persist_index()` was added and calls `VectorDbManager.save()`, which nothing did before (the open P3 entry).
- **Explanation:** `EmbeddingManager` derives `vector_id = md5(chunk_id)`, a one-way hash, and `IngestionPipeline` inserts the vector into DiskANN and stops. It never writes a `vector_id -> chunk_id` row: `VectorMetaDataRepository` is never constructed by the pipeline (and is broken anyway, Bug 5.2), and the chunk store's `Chunks` table holds `chunkId, contextId, chunk, startoffset, endoffset` with no vector column. Verified by inspecting the pipeline source and the created schema — no table in the data layer stores a vector id. A search therefore returns ids that nothing can turn back into text, which is the one thing retrieval needs. The pipeline also never calls `VectorDbManager.save()`, so the index is not persisted either (the existing P3 entry).

### Bug 5.4: Section Headings Never Reach a Chunk (`normalizer.py`, `HierarchicalChunker.py`)

- **Criticality:** Medium
- **Priority:** P2
- **Found again while building the retrieval layer, and it costs more than this entry suggests.** Keyword search indexes the chunk text, so the heading is absent from the lexical index as well as from the embeddings. Measured on a document headed `# Write-ahead logging`: the query `write ahead logging` returns **0 hits**, while `journal file` and `readers writers` — words from the body — each return 1. The heading is usually the most descriptive line and the one a query is most likely to echo, so this is not a fidelity loss at the margin; it removes the best lexical signal in the document. Worth re-rating above P2 before the retrieval layer is relied on.
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

## Section 5b: The DiskANN Write Path (found while fixing 5.2 and 5.3)

Neither of these could have been seen before now: nothing had ever run a vector
through `VectorDbManager` into DiskANN, because 5.3 meant the result was
unusable and so the path was never exercised.

### Bug 5.15: `batch_insert` Passes a Python List Where diskannpy Requires an Array (`vectorDbManager.py`) — FIXED

- **Criticality:** Critical
- **Priority:** P0
- **Explanation:** `batch_insert` converted the vectors with `numpy.array(...)` but passed `vector_ids` through as a list. diskannpy reads `vector_ids.shape`, so every batch insert raised `AttributeError: 'list' object has no attribute 'shape'`. Reproduced on the first real ingest.
- **Status:** **Fixed.** Both are arrays now, and the ids are `uint32` — see 5.16 for why that matters.

### Bug 5.16: Vector Ids Are 63-Bit and DiskANN Indexes uint32 (`EmbeddingManager.py`, `vectorDbManager.py`) — FIXED

- **Criticality:** Critical
- **Priority:** P0
- **Explanation:** `EmbeddingManager` derives `vector_id` masked into the signed 64-bit range (`Config.VECTOR_ID_MASK`, max 9.2e18), which is right for pgvector's `bigint` and impossible for DiskANN: diskannpy labels are `uint32`, max 4.29e9. Verified — `uint32` labels are accepted and `uint64` raises `TypeError: Cannot cast array data from dtype('uint64') to dtype('uint32')`. So no vector could ever be indexed, whatever 5.15 did.
  Masking to 32 bits instead would not work either. At `MAX_VECTORS = 1,000,000` the birthday bound puts expected collisions at **about 116**, and a collision means two chunks sharing a label — a hit resolving to the wrong text.
- **Status:** **Fixed** by letting `vector_meta_data` allocate the label instead of deriving it. Sequential ids cannot collide and use 0.02% of the `uint32` space at `MAX_VECTORS`. This is what the table was for: the id DiskANN returns is a label, and the table is what translates it back. `EmbeddingManager.vector_id` is unchanged and still correct for pgvector, which is where the memory layer uses it.
- **Since 5.22:** the allocated label has its own column, `label`. `vectorId` holds the vector id again — passed in, required, never generated — so the column's name says what it holds.

## Section 5c: The DiskANN Read Path (found while building the retrieval layer)

Like 5b, invisible until something read the index back: every earlier test of the dense path either mocked diskannpy or used an index smaller than the search list, where a broken graph still reaches every point from the start node.

### Bug 5.17: `VectorDbManager.search_vector` Could Not Return More Than `K_NEIGHBORS` (`vectorDbManager.py`) — FIXED

- **Criticality:** Medium
- **Priority:** P1
- **Explanation:** `search_vector(query)` always passed `self.k_neighbors` (9) to DiskANN, although the wrapper beneath it takes `k`. The retrieval layer over-fetches `top_k × 4 = 32` per searcher so reranking and MMR have a pool to choose from; the dense half capped at 9 while bm25 supplied 32, so fusion leaned towards keyword results without anyone choosing that.
- **Status:** **Fixed.** `search_vector` and `batch_search_vectors` take an optional `k_neighbors`, defaulting to the configured value so every other caller is unchanged. DiskANN widens its search list itself when `k > complexity` (verified: `k=120` against `complexity=100` returns 120).

### Bug 5.18: Asking DiskANN for More Neighbours Than It Holds Returns Uninitialised Memory (`vectorDbManager.py`) — FIXED

- **Criticality:** High
- **Priority:** P1
- **Explanation:** diskannpy returns `k` slots whatever the index holds and fills the surplus from uninitialised memory. Measured on an 8-vector index asked for 12: the 8 real neighbours, then labels such as `1067030938` (the bit pattern of the float 1.2), `64`, `22053` and **`1` — a real, allocated label** — at distance `0.0`. Sorted by distance, the junk ranks **first**, and a junk label that happens to be allocated resolves to a real chunk, so the wrong passage leads the result. On a reloaded index every slot is garbage, the first at distance −2.97e28. A new install's corpus is always smaller than the 32-candidate pool, so this is the first thing a user meets.
- **Status:** **Fixed** by never asking for more than the index holds: `k = min(k, count())`, where `count()` reads the native `num_points()` — verified to track inserts (8), deletes (7) and reloads (7), excluding the frozen start point, and `k = num_points` is exactly the boundary (`k=7` of 7 is clean, `k=8` brings in label `0`). The count is read through diskannpy's private `_index`; `test_the_count_follows_inserts_deletes_and_reloads` pins that path so a library change fails there rather than inside a search.

### Bug 5.19: The Dynamic Index Is Built Without `saturate_graph`, So Recall@10 Is 0.09 (`vectorDB_diskann.py`) — FIXED

- **Criticality:** Critical
- **Priority:** P0
- **Explanation:** diskannpy 0.7.0's `DynamicMemoryIndex` defaults `saturate_graph` to `False`, and with that default the insertion path builds an almost empty graph: the index cannot find a vector that was inserted into it. Measured on 5,000 random 128-d vectors, 200 queries, against brute force:

  | Index | recall@10 | finds an inserted vector exactly |
  |---|---|---|
  | `build_memory_index` (static, reference) | 0.958 | — |
  | Dynamic, as `VectorDb_diskann` builds it | **0.086** | no |
  | Dynamic, `num_threads` 1 or 4 | 0.098–0.107 | no |
  | Dynamic, `saturate_graph=True` | **0.965** | yes |

  Raising the search list from 64 to 200 changes nothing, and a 5,000-vector build takes 0.01 s — there is no graph to search. On the 8-point indexes the tests use, the start node links to every point, which is why nothing caught it. At any real corpus size the dense half of retrieval returns about a tenth of the true neighbours.
- **Status:** **Fixed.** `SATURATE_GRAPH = True` in `vectorDB_diskann.py`, passed to both the constructor and `from_file`, which takes the same flag and would otherwise build the sparse graph for every insert after a load. Build time rises to 0.08 s per 5,000 vectors. With it, 99.7–99.8% of 2,000 vectors find themselves as nearest neighbour across three seeds (0% without). `test/data_layer_testing/test_vector_index.py` requires ≥ 98% and recall@10 ≥ 0.9 against brute force, on the real library; without the flag both fail (recall 0.15). **An index built before the change has to be rebuilt** — the graph is fixed at insertion.

### Bug 5.20: A Saved Index Loses Every Label — `ann.tags` Is Written as Zeros (`vectorDB_diskann.py`, diskannpy 0.7.0) — FIXED

- **Criticality:** Critical
- **Priority:** P0
- **Explanation:** `DynamicMemoryIndex.save` writes all-zero tags. Verified: labels `11..18` inserted, `ann.tags` on disk reads `[0, 0, 0, 0, 0, 0, 0, 0, 0]`. A reload in the same process happens to return the right labels; a reload in a **new process** — which is what every restart is — returns pointer-like garbage (`2530163536, 32701`), and the retrieval tests' in-process reload returns `0` for every hit. The vectors and graph survive; the mapping from them to `vector_meta_data` does not. So after any restart the dense half returns labels that resolve to nothing, hydration drops them silently, and retrieval runs on keyword search alone with nothing raised. This is the cause behind the long-standing P3 "DiskANN index not persisted/reloaded correctly between sessions". 0.7.0 is diskannpy's last release (the cp311 ceiling), so no upstream fix is coming.
- **Repairing the files is not possible.** Writing the correct labels into `ann.tags` after `save()` and reloading in a new process still returns the same garbage for every query, so the loader is broken as well as the writer. No reuse of diskannpy's dynamic save/load can work.
- **Status:** **Fixed** by making SQLite the source of truth and the index derived data:
  - `vector_meta_data` gained a `vector` blob, written in the transaction that allocates the label (`allocate` / `allocate_many`), so the label map and the index cannot disagree. Tables written before it gain the column on open, rows kept; those rows are reported as unsearchable until re-embedded.
  - `VectorDbManager.restore(store, after)` builds the index from those rows a page at a time and returns the highest label indexed. Labels only grow, so `Retriever.refresh()` → `VectorSearch.catch_up()` indexes only what was ingested since, instead of rebuilding the graph.
  - `VectorSearch` builds from the chunk store, not from DiskANN files; `IngestionPipeline.persist_index()` is gone. `save()` / `load()` remain, unused.
  - Only one model's vectors at one width are restored: another model's are in a different space.
- **Cost:** a graph build on the first search, growing faster than linearly — 0.9 s for 10k vectors, 6.3 s for 50k, 16.6 s for 100k at `Config`'s settings (complexity 100, degree 120, 4 threads). Reading the vectors from SQLite is under 0.1 s per 100k. A static-index snapshot (`build_memory_index`, row-position ids mapped to labels) would remove it at the price of a full rebuild per ingest; not done.
- **Tests:** `test_vector_index.py::TestSurvivingARestart::test_in_a_new_process` rebuilds in a separate interpreter and requires the exact labels back — every in-process reload had looked correct. Also storage and migration (`test_vector_chunk_mapping.py`), restore and catch-up against the real library, and the pipeline storing what it indexes. The two `xfail(strict=True)` markers became passing tests of the store-backed path. 18 mutations, all caught.
- **Superseded (5.22, 5.23).** The vectors moved to pgvector, where they belong, and the startup rebuild is gone. Ingestion builds a static index *generation* on disk (`index_generations.py`) once `INDEX_REBUILD_AT` vectors wait outside the current one; startup opens it and loads only the vectors added since. diskannpy's static indexes save and load correctly — only the dynamic one is broken. Measured at 100k vectors plus 5k recent, end to end through `VectorSearch`: 0.25 s to start and +130 MB as a memory index, 2.48 s and +81 MB as a disk index, against 5.89 s and 1,236 MB for the rebuild it replaces.

### Bug 5.21: Re-ingesting a Chunk Allocates a New Label Every Time, and Since 5.20 the Copies Persist (`vectorMetaDataRepository.py`, `ingestion_pipeline.py`) — FIXED

- **Criticality:** High
- **Priority:** P1
- **Explanation:** `allocate` / `allocate_many` always insert a new row; nothing checks whether the chunk already has a label. Chunk writes are `on conflict do nothing`, so re-ingesting an unchanged folder adds no chunks — but every chunk still gets a fresh label. Measured in the real `data/hierarchical_db`: **one chunk, 45 labels, 25 of them with a stored vector**, accumulated by test runs (7.5). Before 5.20 the duplicate vectors died with the process; now they are stored, so every rebuild indexes all 25 copies, and a search can fill retrieval's 32-candidate pool with one chunk. Assembly drops repeats, so results stay correct, but recall falls with every re-ingest. **Made permanent by the 5.20 fix**, which is why it is logged with it.
- **Status:** **Fixed.** `unique (chunkId, embeddingModelUsed)` on `vector_meta_data`; `allocate` / `allocate_many` are one `insert … on conflict … do update … returning vectorId`, so a chunk seen before gets its own label back and a label with no vector gains one, atomically. A store written before the constraint is collapsed on open — one row per chunk and model, preferring the row with a stored vector, with a warning naming the count. The pipeline records the labels its own index holds and skips repeats, since DiskANN refuses a label twice (`RuntimeError: … unable to be inserted`). Tests in `test_vector_chunk_mapping.py` and `test_vector_index.py`; 9 mutations, all caught. **The real store was collapsed by a test run:** a mutation that removed 7.5's redirect let the pipeline test open `data/hierarchical_db`, taking it from 55 labels to 1 for its one chunk — the duplicates this bug produced, so nothing distinct was lost.
- **Since 5.22:** the uniqueness is on `chunkId` and on `vectorId` — a vector id belongs to the chunk, so one store holds one model's vectors, and a chunk offered under another model raises `EmbeddingModelMismatch`. `allocate` / `allocate_many` became `insert` / `batch_insert`, which require the vector id. The pipeline no longer holds an index, so it no longer tracks labels.

---

## Section 5d: Where Vectors Live, and an Index That Survives a Restart (found reviewing the retrieval layer)

### Bug 5.22: `vector_meta_data` Stored the Vectors, and Its `vectorId` Was a Generated Label (`vectorMetaDataRepository.py`, `ingestion_pipeline.py`) — FIXED

- **Criticality:** High
- **Priority:** P1
- **Explanation:** Found in review. The table exists to short-circuit a DiskANN hit to its chunk; vectors live in pgvector. Three things had drifted:
  - **It held every vector.** The 5.20 fix added a `float32` blob per row, because the index had to be rebuilt from somewhere and the document vectors were stored nowhere else: ingestion had never written them to PostgreSQL. `VectorRepository` was used only by the memory layer.
  - **`vectorId` was not the vector id.** The 5.16 fix made it `integer primary key autoincrement`, so the column named for the vector id held the DiskANN label, and a write that omitted it was given one silently.
  - **The original column was nullable.** `vectorId int primary key` accepts `NULL`: SQLite allows it in any primary key not spelled exactly `INTEGER PRIMARY KEY`.
  Verified with SQLite 3.50.4: the original schema stores `NULL` for an omitted or explicit-null id; `integer primary key autoincrement` assigns 1, then 2; adding `not null` to it still assigns. Only an ordinary column refuses a missing value.
- **Status:** **Fixed.**
  - Schema: `label integer primary key autoincrement check (label <= 4294967295)`, `vectorId integer not null unique check (typeof(vectorId) = 'integer' and vectorId >= 0)`, `chunkId text not null unique`, no vector. The label is the only number generated; a label past `uint32` is refused rather than wrapped.
  - `checked_vector_id` refuses `None` (`MissingVectorId`) and a bool, non-integer or out-of-range value (`MalformedVectorId`) before any write, and the database refuses them too. A chunk under another vector id, or a vector id under another chunk, raises `VectorIdConflict`.
  - Ingestion writes the vectors to pgvector under the project id `Config.GLOBAL_VECTOR_SCOPE` (`"global"`; real project ids are uuid4 hex) **before** allocating labels, so a label never exists for a vector that was not stored. A store that cannot be opened raises `VectorStoreUnavailable`.
  - An old table is rebuilt on open: labels kept, vector ids derived from the chunk ids by the embedder's own function (`vector_ids.py`), duplicates collapsed, blobs dropped, the sequence kept above every old label. The one row in the real store was removed rather than migrated, as asked.
  - **Tests:** `test_vector_chunk_mapping.py`, the pipeline tests in `test_vector_index.py`, and `test_vector_repository_live.py::TestTheBulkRead` against a real server. 11 mutations, all caught.

### Bug 5.23: The In-Memory Index Reserved 1.2 GB However Little It Held, and Was Rebuilt at Every Start (`vectorDbManager.py`, `ingestion_pipeline.py`, `vector_search.py`) — FIXED

- **Criticality:** High
- **Priority:** P1
- **Explanation:** `DynamicMemoryIndex` allocates every slot at construction, and both `IngestionPipeline` and `VectorSearch` sized theirs to `MAX_VECTORS`. Measured: **1,214 MB held with 5,000 vectors**, 1,236 MB with 100,000; with room for 20,000 it holds 34 MB. On top of that every startup rebuilt the graph from every stored vector (5.20's cost): 5.89 s at 100k, and superlinear.
- **Status:** **Fixed** with a hybrid index under one RAM budget, `VECTOR_INDEX_RAM_MB` (512):
  - Ingestion builds a **generation** on disk — a static memory index while it fits the budget, a static disk index with a RAM-budgeted node cache past it — in a separate process, and swaps `CURRENT` atomically. Startup opens it; only vectors added since are loaded, into a recent index capped at `RECENT_VECTOR_CAPACITY` (20k). Search merges both by distance.
  - Measured at 100k on the NVMe drive (the earlier tmpfs numbers understated the disk index fourfold): memory index 0.50 ms per search in 90 MB; disk index 4.33 ms uncached in 17 MB, 1.33 ms with half its nodes cached in 77 MB. A fully cached disk index uses more RAM than the memory index and is still slower, which is why the budget chooses rather than always caching.
  - **Every allocation is admitted before it is made**, against `MemAvailable` and the tightest cgroup v2 limit, less `MEMORY_HEADROOM_MB`. Linux overcommits, so a native allocation that cannot be backed does not fail — the OOM killer acts later, on whichever process it picks. Fallbacks: a disk index opens with a smaller cache, down to none; a generation that does not fit, is corrupt or was built for another model is skipped for the previous one; with no index the dense half returns nothing and keyword search carries on; refresh tries again. A build that does not fit is planned for disk or skipped; the builder sets its own `oom_score_adj` to 1000 so the kernel kills it, not the application; a killed, failed or hung build (stopped after `max(900 s, 5 ms × vectors)`) leaves the current generation in use. PostgreSQL down at startup costs only the recent vectors.
  - **Tests:** `test_vector_index.py` and `test_index_fallbacks.py` — each worst case made real: a builder killed by SIGKILL, a hung builder, memory and cgroup limits, a full disk, corrupt and foreign generations, PostgreSQL down, a build already running. 21 mutations, all caught.

### Bug 5.24: diskannpy 0.7.0 Traps Met While Persisting the Index (`index_generations.py`, `index_build.py`) — GUARDED

- **Criticality:** High
- **Priority:** P1
- **Explanation:** Each verified in a fresh process; none is fixable without rebuilding diskannpy, whose last release this is.
  - **The dynamic index cannot be saved, re-confirmed.** All 2,001 tags written as zero; recall 0.0 in a new process under three loader settings.
  - **A tagged static build loads with recall 0.156.** `build_memory_index(tags=…)` then `DynamicMemoryIndex.from_file` is the documented way to persist a mutable index, but the compiled builder hard-codes `saturate_graph(false)` — 5.19 again, out of reach.
  - **A disk index searched with `num_threads=1` never returns.** Two threads or more never hang; with four, sixteen concurrent callers got answers identical to serial ones.
  - **A disk index asked for more neighbours than it holds pads with position 0** — a real vector, so the padding becomes a wrong but valid label.
  - **A memory index of one vector cannot be built** ("r = 0 is zero").
  - **`RLIMIT_AS` cannot bound a builder:** importing diskannpy reserves 1.4 GB of address space while using 62 MB, so any limit low enough to matter kills it at import.
- **Status:** **Guarded.** Static indexes only; searches use at least two threads (`search_threads()`); `BuiltIndex.search` asks for no more than it holds and drops positions out of range; builds need two vectors; the builder is bounded by admission and the OOM score instead of a limit. The builder is `python -m data_layer.vector_db_manager.index_build`, not `multiprocessing`: spawn re-imports the caller's main script, and a script without a `__main__` guard re-ran itself inside the builder — found when one did.

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

### Bug 5.13: The Cursor Was Opened Before the Vector Types Were Registered, So Every Write Failed (`vectorRepository.py`) — FIXED

- **Criticality:** Critical
- **Priority:** P0
- **Explanation:** `__init__` did `self.curr = self.conn.cursor()`, then `__create_extension()`, then `register_vector_types(self.conn)`. A psycopg cursor binds the connection's adapter map **at creation**, so `self.curr` — opened before the registration — never saw the ndarray dumper, and every insert raised `ProgrammingError: cannot adapt type 'ndarray' using placeholder '%s' (format: AUTO)`. The registration itself was correct and `conn.adapters` did hold the dumper; only the cursor was stale. Confirmed by a direct A/B: identical code with the cursor opened *after* `register_vector` inserts fine, and with it opened *before* fails. **Fixed** by reopening `self.curr` after `register_vector_types`.
- **Why no test caught it:** the mocked `psycopg` adapts anything, so ordering is unobservable. Only a real connection distinguishes the two.
- **Regression test:** `test/live_testing/test_vector_repository_live.py::TestTheWritePath`. Mutation-checked — reverting the fix fails all 8 live tests, both alone and inside the full suite.

### Bug 5.14: A Vector Read Back From pgvector Cannot Be Coerced by numpy (`vectorRepository.py`) — FIXED

- **Criticality:** Critical
- **Priority:** P0
- **Explanation:** `__get_vector` ended in `np.asarray(result[0], dtype=float32)`. With the adapter correctly registered, pgvector 0.5.0 returns its own `pgvector.Vector` object rather than a list or array, and numpy cannot coerce it: `TypeError: float() argument must be a string or a real number, not 'Vector'`. Note this is the *opposite* failure from the one Bug 5.1 predicted — 5.1 reasoned the column would come back as **text** without an adapter; it comes back as a `Vector` **with** one. **Fixed** by calling `.to_numpy()` when the returned object offers it, which keeps older versions that already return an array working. `__get_vector` is the only read site, so `__get_vectors`, `search` and `batch_search` are all covered by the one change.
- **Regression test:** `test/live_testing/test_vector_repository_live.py::TestTheReadPath`. Mutation-checked — reverting the fix fails 5 of the 8 live tests, leaving exactly the three that never read a vector back.

---

## Section 7: Documentation & Test Guards

### Bug 7.1: Six Docstrings Were Cut Mid-Sentence by the Trimming Pass

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** The pass that shortened every docstring to its summary kept the first *line* rather than the first *sentence*, so any docstring whose opening sentence wrapped now ends mid-clause. Affected: `normalizer._is_mostly_letters` ("fires on anything without a"), `text_extractor._flatten_json` ("stay attached to what"), `project_meta_data.__validate_topic_id` ("has no table of its own"), `project_meta_data.__validate_summary` ("rather than at the"), `conversationVectorManager.batch_delete` ("undo a partially written"), and `conversation_summary.make_summary` ("the current conversation"). **Introduced in this session.** The full text of each is recoverable from the archive taken before the pass.

### Bug 7.2: The "No Raw `sqlite3.connect`" Guard Covers Two Modules Out of Four (`test_conversation_data_management.py`) — FIXED

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** `test_every_conversation_connection_goes_through_connect` inspects `fullconversation_repository` and `conversationVectorMetaManager` only. Two more modules now open SQLite through `memory/sqlite_setup.connect()` and depend on the same per-connection pragmas: `project_meta_data.py` (since Bug 4.45) and `topic_pool_meta_handler.py`. Neither is in the guard's list, so a new method in either could silently get `synchronous=FULL` and foreign keys off — the exact defect the guard exists to prevent. Verified by reading the module list the test imports.
- **Status:** **Fixed.** `test_no_memory_module_opens_its_own_connection` reads every module under `memory/` and fails on any `connect(` outside `memory_database.py`, which is now the only place a memory connection is opened.

### Bug 7.3: `scripts/smoke.py` Cannot Report a Missing `psycopg` (`scripts/smoke.py`)

- **Criticality:** Low
- **Priority:** P3
- **Explanation:** `preflight()` exists to report setup problems as instructions instead of tracebacks, and it does that for a missing `pgvector`, missing `.env` keys, an unreachable server, an absent extension and an absent database. It imports `psycopg` unconditionally, though, so the one dependency it cannot report is the one it needs to do the reporting. **Introduced in this session.**

### Bug 7.4: Three Data-Layer Test Files Replace `diskannpy` With a MagicMock for the Whole Session

- **Criticality:** Medium
- **Priority:** P2
- **Explanation:** `test_data_layer_production.py`, `test_data_layer_bugs.py` and `test_non_memory.py` assign `sys.modules["diskannpy"] = mock.MagicMock()` unconditionally at import, and pytest imports every file before running any test, so every later test in the session gets the mock. The retrieval end-to-end tests built a "real" index, every search raised inside the mock, `VectorSearch` returned `[]`, and the tests **passed on keyword search alone**; only an assertion that the dense half produced results noticed. The other files use `setdefault`, which stands down when the real module is already imported — the arrangement the root `conftest.py` makes for `psycopg`. Switching these three to `setdefault` is not enough on its own: `test_data_layer_production.py` also assigns `DynamicMemoryIndex = MockDiskANN` on the module, which would then patch the real library for the rest of the session, so it needs `monkeypatch` instead.
- **Workaround:** the root `conftest.py` provides `real_diskann`, which imports the real diskannpy once per session and swaps only the `diskannpy` entry in `sys.modules` (and the wrapper's binding) for one test, via `monkeypatch`. Mutation-checked: without the wrapper binding, 7 retrieval tests fail.
- **The first version of that fixture was itself a bug. Introduced and fixed in this session.** It wrapped the test in `mock.patch.dict(sys.modules)`, which on teardown evicts *every* module the test imported, not just diskannpy. A test that imported `ingestion_pipeline` pulled in torch; teardown evicted it; the next test re-imported torch in the same process, which torch does not support, and formatting that failure segfaulted pytest. It surfaced once two such tests ran in one session.

---

## Section 8: Retrieval Layer (`retrieval_layer/`)

### Bug 8.1: Keyword Search Split Queries on ASCII, So Accented and Non-Latin Words Never Matched (`keyword_search.py`) — FIXED

- **Criticality:** Medium
- **Priority:** P1
- **Explanation:** `match_expression` extracted terms with `[A-Za-z0-9_]+`, so `naïve` was searched as `"na" OR "ve"` and `café` as `"caf"` — **0 hits** each against a chunk containing the word, verified on a real FTS5 table. FTS5's `unicode61` tokenizer handles these fine; the pattern in front of it broke them first. **Introduced in this session** (retrieval Phase 2).
- **Status:** **Fixed** with `\w+`. Both now match, and because `unicode61` folds diacritics, `cafe` also finds `café`. Tests include a Cyrillic query.

### Bug 8.2: MMR Re-derived Relevance From Cosine, Undoing the Reranker (`diversity.py`) — FIXED

- **Criticality:** High
- **Priority:** P1
- **Explanation:** `maximal_marginal_relevance` computed relevance as cosine to the query embedding and ignored the scores of whatever ranked the passages before it. It runs after the cross-encoder, so the most expensive and most accurate stage in the pipeline changed nothing about the result: the bi-encoder's ordering came back. Without reranking it equally discarded fusion's judgement — a passage both searchers found lost to one that was merely closer in embedding space. **Introduced in this session** (retrieval Phase 3).
- **Status:** **Fixed.** MMR takes an optional `relevance`, min-max normalised across the pool so a logit, a bm25 value or an RRF score can be traded against cosine redundancy; the retriever always passes the previous stage's scores. Min-max is relative to the pool — the weakest candidate scores 0 whatever its raw score — which is why MMR is given the whole candidate pool, not a shortlist; `test_relevance_is_relative_to_the_pool` pins that property.

### Bug 7.5: `test_data_layer_production.py` Writes Into the Real Chunk Store — FIXED

- **Criticality:** Medium
- **Priority:** P2
- **Explanation:** The end-to-end pipeline test builds a real `IngestionPipeline`, whose chunker and label repository open `Config.DB_PATH` — the developer's own `data/hierarchical_db` — and the test's own comment says so ("It will write to DB_PATH"). Verified: running that file alone moves the real store from 44 labels to 45. Every full-suite run adds rows, and since 5.20 a stored vector with each, so the suite's results and the developer's data are entangled: the real store carries test chunks, and it migrated to the new schema the first time the suite ran rather than when the app did. The fix is to point the pipeline's paths at `tmp_path` for that test.
- **Status:** **Fixed** for every test, not just that one. `Chunker` and the retrieval constructors bound `Config.DB_PATH` as a default argument, which Python evaluates at import, so no redirect could reach them; they now read it when called. The root `conftest.py` points `Config.DB_PATH` at a per-test path, as it does `Config.MEMORY_DB`. Verified: a full run leaves the real store's label count and modification time unchanged. `TestTheSuiteNeverTouchesTheRealChunkStore` guards both halves.
- **Extended with 5.22:** `Config.INDEX_PATH` is redirected per test as well, and the chunk vector store is replaced by an in-memory one (the `chunk_vectors` fixture), because ingestion now writes vectors to PostgreSQL and the suite reads the developer's `.env`. Only `test/live_testing/` reaches the real server, under throwaway project ids it deletes. `TestTheSuiteNeverTouchesTheRealStores` guards it.

### Bug 7.7: A Test's `monkeypatch.undo()` Reverted conftest's Redirects and Reached the Real PostgreSQL — FIXED

- **Criticality:** Medium
- **Priority:** P2
- **Explanation:** Introduced and caught while fixing 5.23. A fallback test ended a temporary patch with `monkeypatch.undo()`. The `monkeypatch` fixture is one instance per test, shared with the root `conftest.py`'s autouse fixtures, so `undo()` also reverted the chunk vector store and the path redirects, and the rest of the test opened `VectorRepository("global")` against the developer's PostgreSQL. What it did there: `create extension if not exists` and `create table if not exists`, both no-ops on a database that has them, then a read that found nothing. Nothing was written. Verified afterwards: the real chunk store unchanged at 0 labels, no index directory created.
- **Status:** **Fixed.** Temporary patches use `pytest.MonkeyPatch.context()`, which reverts only its own. `test_no_test_undoes_every_patch` fails the suite if any test calls `monkeypatch.undo()`.

