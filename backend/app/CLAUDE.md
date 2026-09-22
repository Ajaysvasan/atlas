# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python RAG (Retrieval-Augmented Generation) backend system that ingests multi-format documents, generates embeddings, stores them in a DiskANN vector index, and manages conversation memory with snapshot history. Currently CLI-only — no FastAPI routes exist yet.

## Running the Application

```bash
# From the app/ directory
python main.py                    # Interactive CLI mode
python main.py --verbose          # Debug logging, on the console as well as the file
python main.py --log-json         # One JSON object per line instead of text
python main.py --log-file PATH    # Write somewhere other than log/app.log
```

`LOG_LEVEL`, `LOG_FILE`, `LOG_CONSOLE_LEVEL`, `LOG_FORMAT=json`, `LOG_MAX_BYTES`
and `LOG_BACKUP_COUNT` override the flags, so logging can be retuned without a
code change.

## Running Tests

Tests must be run from the `app/` directory with `PYTHONPATH` set:

```bash
PYTHONPATH=. pytest test/ -v                            # All tests
PYTHONPATH=. pytest test/data_layer_testing/ -v         # Data layer tests
PYTHONPATH=. pytest test/memory_layer_testing/ -v       # Memory layer tests
PYTHONPATH=. pytest test/stress_testing/ -v             # Stress tests
PYTHONPATH=. pytest test/logging_testing/ -v            # Logging tests
PYTHONPATH=. pytest test/data_layer_testing/test_data_layer_production.py::TestClass::test_name -v  # Single test
```

## Architecture

### Data Ingestion Pipeline

`FileLoader → TextExtractor → TextNormalizer → Chunker → EmbeddingManager → VectorDbManager`

All orchestrated by `data_layer/ingestion/ingestion_pipeline.py`.

- **FileLoader** (`TextFileProcessor/file_loader.py`): Recursively scans directories, returns `Dict[extension, List[Path]]`. The policy is a denylist, not an allowlist: anything that is not a known binary format (`NON_DOCUMENT_EXTENSIONS`) is offered to the extractor. Skips VCS/build directories, dotfiles, empty files and files over `max_file_size` (64 MB), and resolves symlinked directories against a visited set so a link to an ancestor cannot loop.
- **TextExtractor** (`TextFileProcessor/text_extractor.py`): Dedicated readers for `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.html`/`.xml`, `.json`, `.ipynb`, and the textract-backed binaries (`.doc`, `.odt`, `.rtf`, `.epub`, …); **every other extension falls back to decoded text**, so source code, logs, config, TeX and unknown formats all ingest. Encoding is BOM → utf-8 → chardet → latin-1; content with NUL bytes raises `InvalidFileType`. Word heading styles, HTML `<h1>`–`<h6>` and PowerPoint slide titles are emitted as markdown `#` headings so formats with no heading syntax still produce sections.
- **TextNormalizer** (`normalizer/normalizer.py`): Cleans text **line by line**, preserving paragraph structure, and returns `NormalizedContent` with a `sections` tuple of `SectionSpan` offsets into the normalized content. Headings are detected on the raw lines — before lowercasing or punctuation stripping — in four shapes: markdown ATX, setext underline, numbered (`1.`, `2.3`), and ALL CAPS (guarded by a letter-ratio test so table rows are not mistaken for headings). Code fences are skipped.
- **Chunker** (`Chunker/chunker.py`): Routes to `HierarchicalChunker` (documents with sections) or `RecursiveChunker` (flat docs). chunk_size=256, overlap=20. Both chunkers window text through `Chunker/windowing.py::sliding_windows`, which breaks on word boundaries and guarantees `len(chunk) <= chunk_size`. Chunk, context and section ids all bind position as well as content, so repeated text does not collide; all writes are `on conflict do nothing`, so re-ingesting an unchanged folder is a no-op rather than a `UNIQUE` failure.
- **EmbeddingManager** (`embedding/EmbeddingManager.py`): SentenceTransformer `all-MiniLM-L6-v2`, 128-dim float32, batch size 64, MD5-based vector IDs
- **VectorDbManager** (`vector_db_manager/vectorDbManager.py`): Thread-safe DiskANN wrapper, k_neighbors=9, l2 distance, up to 1M vectors

### Memory Layer

Hierarchical organization: Topic → Project → Conversation → Snapshot

- **FullConversation / ConversationPoolManager**: turns are written with `append_turn(role, text)` and read back as `Turn(sequence_number, role, text, created_at, chunk_id)` — `history()`, `recent(n)`, `context(start, end)`, `since(seq)` on the manager. Always order by `sequence_number`, never `created_at`. The older text-only readers on the repository and bucket drop the speaker; do not build prompts from them. See `docs/memory_layer_docs/conversation_turns.md`
- **ConversationSummary**: feeds the draft model a speaker-labelled transcript (`render_transcript`), batched between turns so no batch opens mid-turn without its speaker
- **ProjectMetaData** (`project_data_repo/project_meta_data.py`): the project registry, scoped to one project *and its topic* — `ProjectMetaData(project_id, topic_id, ...)`. `add_project_vector(vector, vector_id, project_name, project_summary)` writes the embedding to pgvector, then the project row and mapping row in one SQLite transaction, with a compensating delete if the metadata fails. Descriptions are several per project, keyed `(project_id, project_description_id)`. Both child tables carry a denormalised `topic_id`, kept in step by the upsert, so `get_topic_summary_vector_ids()` answers "every summary vector in this topic, and whose it is" without a join — the read the query router needs. `get_project()` returns a `ProjectRow`. See `docs/memory_layer_docs/project_meta_data.md`
- **ProjectVectorHandler** (`project_data_repo/project_vector_handler.py`): a project's vectors — one for its summary, one per description. Ids are derived from the source (`summary_vector_id(project_id)`, `description_vector_id(project_id, description_id)`), never passed, so add/update/get/delete agree without a lookup. `get_project_vectors(project_id, ids)` reads a batch in the order given; the ids come from `ProjectMetaData.get_topic_summary_vector_ids()` because the vector store cannot enumerate. Keeps one `VectorRepository` per project rather than one per call
- **TopicManager** (`topic_pool/topic_manager.py`): create, read and soft-delete a topic, over `TopicPoolMetaHandler` (`topic_pool_repo/`). Reads filter on `is_active = 't'`, so a soft-deleted topic is invisible while its row stays. The connection is opened `check_same_thread=False` and every statement — including `close()` — runs under an `RLock`
- **ProjectManager** (`topic_pool/project_pool/project_manager.py`): routes `(topic_id, query)` to an existing project. `route()` -> `project_id` or `None`; `resolve()` -> `ProjectMatch` with score, margin, `ambiguous` and ranked candidates. A project scores as its best-matching vector, not its average. The no branch is a `pass` pending the thinking layer. Architecture lives in `memory/topic_pool/project_pool/README.md` — **module READMEs hold architecture, data flow and design rationale; `docs/` stays the API reference; code comments are only for logic that reads wrong without one**
- **Snapshot** (`memory/.../snapshot.py`): Bidirectional cursor traversal of conversation history; uses cosine similarity (torch) to find similar snapshots; stores summary and cumulative vectors
- **ConversationVectorMetaDataRepository** (`conversation_data_management/conversationVectorMetaManager.py`): SQLite-based metadata with tables: `summary_chunks`, `summary_vector_meta_data`, `cumulative_vector_meta_data`, `summary_snapshot_map`. Thread-safe: one connection opened with `check_same_thread=False`, every statement (including its commit or rollback, and `close()`) under an `RLock`. `insert_snapshot()` writes a whole snapshot in one transaction.

### Storage

| Store | Path | Purpose |
|-------|------|---------|
| DiskANN index | `data/disk_ann_index/` | Approximate nearest neighbor vector search |
| SQLite (chunker) | `data/hierarchical_db/` | Chunk metadata: `Documents`, `Sections`, `Contexts`, `Chunks` for the hierarchical path and `Documents`, `RecursiveChunks` for the flat one |
| SQLite (memory) | `data/memory/topic_pool/.../conversation_pool/{project_id}_conversation.db` | Conversation turns **and** snapshot metadata, in one file shared by `FullConversationRepository` and `ConversationVectorMetaDataRepository`. Every open goes through `sqlite_setup.connect()`: WAL journal (so `.db-wal` and `.db-shm` sit alongside it), `synchronous=NORMAL`, `foreign_keys=ON` |
| SQLite (topics) | `data/topic_db/topic.sql` | Every topic: `topics_mapping_table` (`topic_pool_meta_handler.py`). Soft delete flips `is_active`; rows are never removed |
| SQLite (projects) | `data/project_db/project.sql` | Shared registry of every project: `project_table` (incl. `project_summary` text), `project_description_table`, `project_mapping_table` (`project_meta_data.py`) |
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

**P3:**
- DiskANN index not persisted/reloaded correctly between sessions

**Fixed since this list was written** (each has a regression test):
- Bug 4.3: `search()` no longer moves the instance cursors — `__find_best_snapshot` scans with local ones
- Bug 4.4: a failed similarity search returns `None`, not `[-1]`
- Vector ids are ints masked into the signed 64-bit range (`Config.VECTOR_ID_MASK`), derived from `chunk_id` rather than chunk text

## Docs

Two kinds, and they do not overlap:

- **Module READMEs** (`<module>/README.md`, one per module and submodule) — what the module does, how data flows through it, and **why it was designed that way**. Start here to understand a module.
- **`docs/`** — the API reference: classes, methods, parameters, schemas.

**Code comments are only for logic that reads wrong without one** (a guard against an infinite loop, an evaluation-order trap, a silent SQLite behaviour). Architecture, rationale and "what this does" belong in the README, not in the file. Docstrings are one line.
