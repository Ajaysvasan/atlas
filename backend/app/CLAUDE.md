# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python RAG (Retrieval-Augmented Generation) backend system that ingests multi-format documents, generates embeddings, stores them in a DiskANN vector index, and manages conversation memory with snapshot history. Currently CLI-only — no FastAPI routes exist yet.

## Running the Application

```bash
# From the app/ directory. uv resolves the interpreter and deps from
# pyproject.toml + uv.lock, so there is nothing to activate.
uv run python main.py                    # Interactive CLI mode
uv run python main.py --verbose          # Debug logging, console as well as file
uv run python main.py --log-json         # One JSON object per line instead of text
uv run python main.py --log-file PATH    # Write somewhere other than log/app.log
```

First checkout: `uv sync --all-extras` builds `.venv` from the lock, downloading
CPython 3.11 if needed. **3.11 is a hard ceiling, not a preference** —
`diskannpy` has never published a wheel past `cp311` in its entire release
history, so a newer Python has no on-disk ANN index without a source build.
`llama-cpp-python` is deliberately not a dependency and is installed separately
by `uv run python download_models/install_llama_cpp.py`, which detects the GPU
and sets the `CMAKE_ARGS` a plain install would get wrong.

`LOG_LEVEL`, `LOG_FILE`, `LOG_CONSOLE_LEVEL`, `LOG_FORMAT=json`, `LOG_MAX_BYTES`
and `LOG_BACKUP_COUNT` override the flags, so logging can be retuned without a
code change.

## Running Tests

Tests run from the `app/` directory. No `PYTHONPATH` and no activation — the
root `conftest.py` puts the project directory on `sys.path`, and `uv run`
supplies the interpreter:

```bash
uv run pytest test/ -v                            # All tests
uv run pytest test/data_layer_testing/ -v         # Data layer tests
uv run pytest test/memory_layer_testing/ -v       # Memory layer tests
uv run pytest test/stress_testing/ -v             # Stress tests
uv run pytest test/logging_testing/ -v            # Logging tests
uv run pytest test/live_testing/ -v               # Real PostgreSQL; self-skipping
uv run pytest test/data_layer_testing/test_data_layer_production.py::TestClass::test_name -v
```

**1589 passed** is the expected result. If 8 of them skip, the live tests could
not reach a server; `-rs` prints which precondition failed, and
`scripts/smoke.py` checks the whole setup and names what is missing.

Most of the suite swaps `psycopg` for a `MagicMock`, which is what lets it run
with no database — but a mock adapts anything and returns anything, so it cannot
see which types the driver accepts or returns (Bugs 5.13 and 5.14 were both of
that kind, and both passed the mocked suite). `test/live_testing/` runs the same
code against a real server and skips itself when there is none. The root
`conftest.py` imports the real `psycopg` first so the `setdefault` guards on
every mock stand down where the real driver exists; without it the live tests
skip even on a machine that has a server.

## Architecture

### Data Ingestion Pipeline

`FileLoader → TextExtractor → TextNormalizer → Chunker → EmbeddingManager → VectorDbManager`

All orchestrated by `data_layer/ingestion/ingestion_pipeline.py`.

- **FileLoader** (`TextFileProcessor/file_loader.py`): Recursively scans directories, returns `Dict[extension, List[Path]]`. The policy is a denylist, not an allowlist: anything that is not a known binary format (`NON_DOCUMENT_EXTENSIONS`) is offered to the extractor. Skips VCS/build directories, dotfiles, empty files and files over `max_file_size` (64 MB), and resolves symlinked directories against a visited set so a link to an ancestor cannot loop.
- **TextExtractor** (`TextFileProcessor/text_extractor.py`): Dedicated readers for `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.html`/`.xml`, `.json`, `.ipynb`, and the textract-backed binaries (`.doc`, `.odt`, `.rtf`, `.epub`, …); **every other extension falls back to decoded text**, so source code, logs, config, TeX and unknown formats all ingest. Encoding is BOM → utf-8 → chardet → latin-1; content with NUL bytes raises `InvalidFileType`. Word heading styles, HTML `<h1>`–`<h6>` and PowerPoint slide titles are emitted as markdown `#` headings so formats with no heading syntax still produce sections.
- **TextNormalizer** (`normalizer/normalizer.py`): Cleans text **line by line**, preserving paragraph structure, and returns `NormalizedContent` with a `sections` tuple of `SectionSpan` offsets into the normalized content. Headings are detected on the raw lines — before lowercasing or punctuation stripping — in four shapes: markdown ATX, setext underline, numbered (`1.`, `2.3`), and ALL CAPS (guarded by a letter-ratio test so table rows are not mistaken for headings). Code fences are skipped.
- **Chunker** (`Chunker/chunker.py`): Routes to `HierarchicalChunker` (documents with sections) or `RecursiveChunker` (flat docs). chunk_size=256, overlap=20. Both chunkers window text through `Chunker/windowing.py::sliding_windows`, which breaks on word boundaries and guarantees `len(chunk) <= chunk_size`. Chunk, context and section ids all bind position as well as content, so repeated text does not collide; all writes are `on conflict do nothing`, so re-ingesting an unchanged folder is a no-op rather than a `UNIQUE` failure.
- **EmbeddingManager** (`embedding/EmbeddingManager.py`): SentenceTransformer `all-MiniLM-L6-v2`, 128-dim float32, batch size 64, MD5-based vector IDs
- **VectorDbManager** (`vector_db_manager/vectorDbManager.py`): Thread-safe DiskANN wrapper, k_neighbors=9, l2 distance, up to 1M vectors. Held in memory and rebuilt from the vectors stored in `vector_meta_data` (`restore`), because diskannpy 0.7.0 cannot load an index it saved (bug 5.20); `save()`/`load()` exist but nothing may rely on them. `search_vector` never asks for more than `count()` (5.18), and the graph is built with `saturate_graph=True` (5.19)

### Memory Layer

Hierarchical organization: Topic → Project → Conversation → Snapshot

- **FullConversation / ConversationPoolManager**: turns are written with `append_turn(role, text)` and read back as `Turn(sequence_number, role, text, created_at, chunk_id)` — `history()`, `recent(n)`, `context(start, end)`, `since(seq)` on the manager. Always order by `sequence_number`, never `created_at`. The older text-only readers on the repository and bucket drop the speaker; do not build prompts from them. See `docs/memory_layer_docs/conversation_turns.md`
- **ConversationSummary**: feeds the draft model a speaker-labelled transcript (`render_transcript`), batched between turns so no batch opens mid-turn without its speaker
- **ProjectMetaData** (`project_data_repo/project_meta_data.py`): the project registry, scoped to one project *and its topic* — `ProjectMetaData(project_id, topic_id, ...)`. `add_project_vector(vector, vector_id, project_name, project_summary)` writes the embedding to pgvector, then the project row and mapping row in one SQLite transaction, with a compensating delete if the metadata fails. Descriptions are several per project, keyed `(project_id, project_description_id)`. Both child tables carry a denormalised `topic_id`, kept in step by the upsert, so `get_topic_summary_vector_ids()` answers "every summary vector in this topic, and whose it is" without a join — the read the query router needs. `get_project()` returns a `ProjectRow`. See `docs/memory_layer_docs/project_meta_data.md`
- **ProjectVectorHandler** (`project_data_repo/project_vector_handler.py`): a project's vectors — one for its summary, one per description. Ids are derived from the source (`summary_vector_id(project_id)`, `description_vector_id(project_id, description_id)`), never passed, so add/update/get/delete agree without a lookup. `get_project_vectors(project_id, ids)` reads a batch in the order given; the ids come from `ProjectMetaData.get_topic_summary_vector_ids()` because the vector store cannot enumerate. Keeps one `VectorRepository` per project rather than one per call
- **TopicManager** (`topic_pool/topic_manager.py`): create, read and soft-delete a topic, over `TopicPoolMetaHandler` (`topic_pool_repo/`). Reads filter on `is_active = 't'`, so a soft-deleted topic is invisible while its row stays. The connection is opened `check_same_thread=False` and every statement — including `close()` — runs under an `RLock`
- **ProjectManager** (`topic_pool/project_pool/project_manager.py`): routes `(topic_id, query)` to an existing project. `route()` -> `project_id` or `None`; `resolve()` -> `ProjectMatch` with score, margin, `ambiguous` and ranked candidates. A project scores as its best-matching vector, not its average. The no branch is a `pass` pending the thinking layer. Architecture lives in `memory/topic_pool/project_pool/README.md` — **module READMEs hold architecture, data flow and design rationale; `docs/` stays the API reference; code comments are only for logic that reads wrong without one**
- **SnapShot** (`memory/snapshot.py`): Bidirectional cursor traversal and cosine similarity (torch) over a snapshot history, for **either** scope. `scope="conversation"` (the default) walks this conversation's cumulative summaries and needs a `conversation_id`; `scope="project"` walks the project registry and does not. The cursors, `advance`/`prev` and `search` are the same code for both, because each scope stores its vector under the id its listing returns. `add()` is conversation-only and `add_project_snapshot()` project-only; the wrong one raises `WrongSnapshotScope`
- **ProjectSnapshot** (`topic_pool/project_pool/project_snapshot.py`): a project's rolling description — what it is about and what has been done. Incremental: each snapshot summarises the previous one plus the conversation summaries written since, so cost tracks what changed rather than the project's size. It reads across every conversation (`project_id`, ignoring `conversation_id`) because the result describes the project, not a chat. The watermark is `seq` on `cumulative_vector_meta_data`, allocated monotonically on write under `BEGIN IMMEDIATE`. An empty summary from the draft model does **not** advance the watermark, or those summaries would be dropped from every future snapshot. Triggered on every conversation snapshot, from inside the window where the draft model is already resident — loading it is the expensive part, so a second load is avoided
- **MemoryMappingHandler** (`memory/memory_mapping_handler.py`): `(conversation_id, user_id)` -> topic, project, latest project snapshot — the lookup `MemoryManager` needs to resume a conversation without routing it again, which is why it sits beside `memory_manager.py` rather than in the conversation pool. Takes no `conversation_id`: every method names the conversation it acts on, so one handler serves the whole table. The write is two steps because the information is — a row is opened when a conversation starts and routing fills it in later, so `search` returning `None` ("never seen") and `MemoryMapping(None, None, None)` ("seen, not yet routed") are deliberately different answers. `latest_project_snapshot_id` is a cache of `ProjectSnapshotRepository.latest()`, not the history. `user_id` is the only user-scoped column in the layer — see `bugs.md` 4.83
- **ProjectSnapshotRepository** (`project_data_repo/project_snapshot_repo.py`): `project_snapshot` (summary, length, `last_seq_included`, timestamp) and `project_snapshot_mapping`, append-only on `(project_id, project_snapshot_id)` so it holds the ordered chain rather than a pointer at the newest. `project_snapshot_id` is also the pgvector `vector_id`, so a row needs no lookup to reach its embedding
- **ConversationVectorMetaDataRepository** (`conversation_data_management/conversationVectorMetaManager.py`): SQLite-based metadata with tables: `summary_chunks`, `summary_vector_meta_data`, `cumulative_vector_meta_data` (carrying `conversation_id` and the monotonic `seq`), `summary_snapshot_map`. Thread-safe: one connection opened with `check_same_thread=False`, every statement (including its commit or rollback, and `close()`) under an `RLock`. `insert_snapshot()` writes a whole snapshot in one transaction.

### Storage

| Store | Path | Purpose |
|-------|------|---------|
| DiskANN index | in memory | Approximate nearest neighbor search, rebuilt from `vector_meta_data` on first search. `data/disk_ann_index/` is no longer written (bug 5.20) |
| SQLite (chunker) | `data/hierarchical_db/` | Chunk metadata: `Documents`, `Sections`, `Contexts`, `Chunks` for the hierarchical path and `Documents`, `RecursiveChunks` for the flat one; plus `vector_meta_data` — each DiskANN label, its chunk, and the vector itself (`vectorMetaDataRepository.py`) |
| SQLite (memory) | `data/memory/topic_pool/.../conversation_pool/{project_id}_conversation.db` | Conversation turns **and** snapshot metadata, in one file shared by `FullConversationRepository` and `ConversationVectorMetaDataRepository`. Every open goes through `sqlite_setup.connect()`: WAL journal (so `.db-wal` and `.db-shm` sit alongside it), `synchronous=NORMAL`, `foreign_keys=ON` |
| SQLite (topics) | `data/topic_db/topic.sql` | Every topic: `topics_mapping_table` (`topic_pool_meta_handler.py`). Soft delete flips `is_active`; rows are never removed |
| SQLite (mapping) | `data/memory_mapping/memory_mapping.sql` | `memory_mapping_table`: which topic, project and project snapshot a `(conversation_id, user_id)` belongs to (`memory_mapping_handler.py`). The lookup that resumes a conversation without re-routing it |
| SQLite (projects) | `data/project_db/project.sql` | Shared registry of every project: `project_table` (incl. `project_summary` text), `project_description_table`, `project_mapping_table` (`project_meta_data.py`), plus `project_snapshot` and the append-only `project_snapshot_mapping` (`project_snapshot_repo.py`) |
| PostgreSQL | localhost:5432, DB `Vectors` | Vector repository via pgvector (`vectorRepository.py`) |

PostgreSQL credentials are in `.env` (not committed; see `.env.example`): `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`. Every one is `DB_`-prefixed on purpose — `load_dotenv()` does not override a variable already in the environment, and the unprefixed names are ones other things set: login shells export `USER`, conda exports `HOST` as its compiler triplet, PaaS platforms set `PORT` (bug 6.2).

### Key Data Models

- `NormalizedContent`: normalized text, `has_section` flag, `sections` (`Tuple[SectionSpan, ...]`), metadata
- `SectionSpan`: a heading plus its body, as absolute offsets into `NormalizedContent.content`
- `HChunk` / `RChunk`: hierarchical vs recursive chunks. Offsets on both are absolute into the normalized document, so `content[start_off_set:end_off_set] == chunk`
- `EmbeddedChunk`: numpy float32 vector, MD5 vector_id, chunk metadata
- `Document`, `Section`, `Context`: intermediate pipeline models

### Logging (`logging_setup.py`)

One module owns logging. Every other module calls `get_logger(__name__)` and
nothing else; `main.py` calls `configure_logging()` once. Handlers are installed
on the **root** logger and module loggers reach them by propagation, which is
what makes `--verbose` raise the level everywhere at once — under the previous
arrangement (handlers per module, `propagate = False`) it reached only the one
logger it was applied to, and `get_logger` created `log/` and opened a file as a
side effect of import.

- `get_logger(name)` — a bare logger, no handlers, no side effects. Safe at import.
- `configure_logging(...)` — installs handlers. Once, from the entry point. Idempotent.
- `log_context(**fields)` — binds fields (a query id, a node name) onto every
  record logged inside the block, so one request can be pulled out of an
  interleaved log. A contextvar: follows async tasks, **not** threads.
- `log_timing(logger, name)` — logs a duration on success and on failure, plus a
  structured `duration_ms`. `IngestionPipeline` wraps its five expensive stages.

The file rotates at 10 MB keeping 5 backups; the console defaults to `WARNING`
so it does not talk over the CLI. Noisy third-party loggers
(`sentence_transformers`, `transformers`, `torch`, …) are pinned to `WARNING`.
An unwritable log path or a bad `LOG_LEVEL` degrades rather than raising.

`config.py` re-exports these so `from config import get_logger` keeps working;
the implementation cannot live there because `logging_setup` must not import
`config` (import cycle). Full contract in `docs/core_docs/logging.md`.

**Conventions enforced by tests** in `test/logging_testing/`: loggers are named
`__name__`; no module calls `basicConfig`/`addHandler`/`setLevel`; log calls use
lazy `%s` formatting, never f-strings; and no log call builds its message by
calling a method (a log line must not do work the caller did not ask for).

### Configuration (`config.py`)

Central config class with constants:
- `EMBEDDING_MODEL`: `sentence-transformers/all-MiniLM-L6-v2`
- `EMBEDDING_DIMENSIONS`: 128
- `K_NEIGHBORS`: 9
- `MAX_VECTORS`: 1,000,000
- `INDEX_PATH`, `DB_PATH`, `CONVERSATION` paths
- `LOG_FILE`, `DEBUG`: defaults `main.py` hands to `configure_logging()`

### Exception Hierarchy

- Data layer: `InvalidFileType`, `VectorInsertionError`, `DuplicateVectorException`, `InvalidEmbeddingArgument`, etc. in `data_layer/datalayer_exceptions/`
- Memory layer: `InvalidCursorException`, `NullPointerException`, `MisMatchCount` in `memory/memory_pool_exceptions.py`

## Known Bugs (see `bugs.md` and `production_impact_report.md`)

**P2 — Unimplemented:**
- Bug 2.1/2.2: `cli/cli_interface.py` query loop is a placeholder (`# some stuff`) — no downstream pipeline integration
- Bug 4.1: `MemoryManager` is still an empty stub class. `TopicManager`, `ProjectManager`, `ConversationPoolManager` and `ConversationSummary` are implemented; nothing yet wires the topic layer to the project layer.

**Fixed since this list was written** (each has a regression test):
- Bug 4.3: `search()` no longer moves the instance cursors — `__find_best_snapshot` scans with local ones
- Bug 4.4: a failed similarity search returns `None`, not `[-1]`
- Vector ids are ints masked into the signed 64-bit range (`Config.VECTOR_ID_MASK`), derived from `chunk_id` rather than chunk text
- The DiskANN index survives a restart: it is rebuilt from the vectors stored in `vector_meta_data` (bug 5.20, the old P3), and its graph is saturated so it can be searched at all (5.19)

## Docs

Two kinds, and they do not overlap:

- **Module READMEs** (`<module>/README.md`, one per module and submodule) — what the module does, how data flows through it, and **why it was designed that way**. Start here to understand a module.
- **`docs/`** — the API reference: classes, methods, parameters, schemas.

**Code comments are only for logic that reads wrong without one** (a guard against an infinite loop, an evaluation-order trap, a silent SQLite behaviour). Architecture, rationale and "what this does" belong in the README, not in the file. Docstrings are one line.
