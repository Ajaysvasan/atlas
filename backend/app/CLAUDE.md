# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Atlas: a Python RAG backend that ingests documents into a DiskANN index, keeps hierarchical conversation memory, and answers a query end to end — decomposition → memory → retrieval/KSV/acquisition → thinking (plan, jobs, critique) → generation → persistence — on one shared local Qwen2.5-3B model. It is driven by a development CLI; there is no HTTP surface yet (a Java/Spring backend over gRPC is the next phase).

## Running the Application

```bash
# From the app/ directory. uv resolves the interpreter and deps from
# pyproject.toml + uv.lock, so there is nothing to activate.
uv run python main.py start              # Interactive CLI (also the default with no command)
uv run python main.py start --user NAME  # Whose conversations (default $ATLAS_USER or login name)
uv run python main.py --no-acquire start # Never fetch from trusted sources
uv run python main.py -q "question"      # One query in a new conversation, then exit
uv run python main.py --verbose          # Debug logging, console as well as file
uv run python main.py --log-json         # One JSON object per line instead of text
uv run python main.py --log-file PATH    # Write somewhere other than log/app.log
```

At the `Atlas >` prompt anything is a query except a line that is exactly a
command: `switch`, `switch <n|id-prefix>`, `new`, `status`, `status --time`,
`status --retries`, `help`, `exit` (each also with a leading `/`). `start` loads
the model, embedder, reranker and index before the first prompt.

The corpus is built with `uv run python scripts/build_corpus.py` (58 named pages
from postgresql.org, sqlite.org, cassandra.apache.org, ingested inside the
quota, provenance in `data/corpus/manifest.jsonl`) and checked with
`scripts/validate_corpus.py`. `bash scripts/cli_smoke.sh` drives the real CLI
through stdin across a restart.

First checkout: `uv sync --all-extras` builds `.venv` from the lock, downloading
CPython 3.11 if needed. **3.11 is a hard ceiling, not a preference** —
`diskannpy` has never published a wheel past `cp311` in its entire release
history, so a newer Python has no on-disk ANN index without a source build.
`llama-cpp-python` is deliberately not a dependency, so **`uv sync` removes it**.
Reinstall with `uv pip install llama-cpp-python` (CPU; add `CMAKE_ARGS=-DGGML_CUDA=on`
for an NVIDIA GPU). `download_models/install_llama_cpp.py` detects the GPU but
calls `python -m pip`, which the uv venv lacks (bug 9.9). The draft model is
fetched by `uv run python download_models/download_draft_model.py`; set
`HF_HUB_DISABLE_XET=1` if the Xet transfer stalls.

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
uv run pytest test/ -m "not real_model"           # Everything but the real-model tests (~30 s)
uv run pytest test/ -m real_model                 # Real draft model; self-skipping (~5.5 min CPU)
uv run pytest test/data_layer_testing/test_data_layer_production.py::TestClass::test_name -v
```

**2210 passed** is the expected full result: 2196 without the model and 14
`real_model` tests (KV checkpoints against llama.cpp, and the pipeline with the
real model). The real-model tests skip themselves when `llama_cpp` or the model
file is missing; the live tests when PostgreSQL is unreachable. `-rs` prints
which precondition failed, and `scripts/smoke.py` checks the whole setup.

Most of the suite swaps `psycopg` for a `MagicMock`, which is what lets it run
with no database — but a mock adapts anything and returns anything, so it cannot
see which types the driver accepts or returns (Bugs 5.13 and 5.14 were both of
that kind, and both passed the mocked suite). `test/live_testing/` runs the same
code against a real server and skips itself when there is none. The root
`conftest.py` imports the real `psycopg` first so the `setdefault` guards on
every mock stand down where the real driver exists; without it the live tests
skip even on a machine that has a server.

The suite reads the real `.env`, so the root `conftest.py` also keeps it off the
developer's data: per-test `Config.DB_PATH`, `Config.INDEX_PATH` and
`Config.MEMORY_DB`, `Config.INFERENCE_CACHE_DIR` and `Config.CORPUS_DIR`, an
in-memory chunk vector store (the `chunk_vectors` fixture) in place of
PostgreSQL, and a shared model runtime pointed at a file that does not exist,
so nothing loads the 2 GB model unless it asks for the `real_model` fixture.
The conversation registry is an in-memory stand-in (the `conversation_registry`
fixture), and `storage.postgres.connect` raises in any test not marked `live`
(bug 9.15) — reach the registry through the fixture or the module attribute,
never by importing `open_conversation_registry` by name. It
imports the real `llama_cpp` first for the same reason as `psycopg` (bug 7.8). **Never call `monkeypatch.undo()`** — the
fixture instance is shared with those redirects, so it reverts them and the rest
of the test reaches the real database (bug 7.7); scope a temporary patch with
`pytest.MonkeyPatch.context()`. A test enforces this.

## Architecture

### Data Ingestion Pipeline

`FileLoader → TextExtractor → TextNormalizer → Chunker → EmbeddingManager → pgvector + vector_meta_data → IndexGenerations`

All orchestrated by `data_layer/ingestion/ingestion_pipeline.py`.

- **FileLoader** (`TextFileProcessor/file_loader.py`): Recursively scans directories, returns `Dict[extension, List[Path]]`. The policy is a denylist, not an allowlist: anything that is not a known binary format (`NON_DOCUMENT_EXTENSIONS`) is offered to the extractor. Skips VCS/build directories, dotfiles, empty files and files over `max_file_size` (64 MB), and resolves symlinked directories against a visited set so a link to an ancestor cannot loop.
- **TextExtractor** (`TextFileProcessor/text_extractor.py`): Dedicated readers for `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.html`/`.xml`, `.json`, `.ipynb`, and the textract-backed binaries (`.doc`, `.odt`, `.rtf`, `.epub`, …); **every other extension falls back to decoded text**, so source code, logs, config, TeX and unknown formats all ingest. Encoding is BOM → utf-8 → chardet → latin-1; content with NUL bytes raises `InvalidFileType`. Word heading styles, HTML `<h1>`–`<h6>` and PowerPoint slide titles are emitted as markdown `#` headings so formats with no heading syntax still produce sections.
- **TextNormalizer** (`normalizer/normalizer.py`): Cleans text **line by line**, preserving paragraph structure, and returns `NormalizedContent` with a `sections` tuple of `SectionSpan` offsets into the normalized content. Headings are detected on the raw lines — before lowercasing or punctuation stripping — in four shapes: markdown ATX, setext underline, numbered (`1.`, `2.3`), and ALL CAPS (guarded by a letter-ratio test so table rows are not mistaken for headings). Code fences are skipped.
- **Chunker** (`Chunker/chunker.py`): Routes to `HierarchicalChunker` (documents with sections) or `RecursiveChunker` (flat docs). chunk_size=256, overlap=20. Both chunkers window text through `Chunker/windowing.py::sliding_windows`, which breaks on word boundaries and guarantees `len(chunk) <= chunk_size`. Chunk, context and section ids all bind position as well as content, so repeated text does not collide; all writes are `on conflict do nothing`, so re-ingesting an unchanged folder is a no-op rather than a `UNIQUE` failure — for labels too: `vector_meta_data` is unique on `chunkId` and on `vectorId`, and inserting a known chunk hands back its existing label (bug 5.21).
- **EmbeddingManager** (`embedding/EmbeddingManager.py`): SentenceTransformer `all-MiniLM-L6-v2`, 128-dim float32, batch size 64. Vector ids come from `embedding/vector_ids.py::vector_id_for(chunk_id)` — MD5 of the chunk id, 63 bits
- **Vectors and labels** (`IngestionPipeline.batch_insert_vectors`): the vector goes to pgvector under its vector id (project id `Config.GLOBAL_VECTOR_SCOPE`, `"global"`) **before** `vector_meta_data` allocates its DiskANN label, so a label never exists without a stored vector. The vector id is **required** — `MissingVectorId` / `MalformedVectorId`, and the column is `NOT NULL` with a `CHECK` — and `label` is the only number generated (bug 5.22). `vector_meta_data` holds no vectors
- **The DiskANN index** (`vector_db_manager/`): a **built generation** on disk (`index_generations.py`) — a static memory index while it fits `VECTOR_INDEX_RAM_MB` (512), a static disk index with a RAM-budgeted node cache past it — opened, not rebuilt, at startup; plus the vectors ingested since, in a dynamic index capped at `RECENT_VECTOR_CAPACITY` (20k). `VectorDbManager` searches both and merges by distance. Ingestion builds a new generation every `INDEX_REBUILD_AT` (10k) vectors, in a separate process, and swaps `CURRENT` atomically. diskannpy 0.7.0 cannot save its dynamic index at all (5.20). **Every allocation is admitted against free memory first** (`memory_guard.py`; Linux overcommits, so a `MemoryError` never comes): short of memory, disk, a usable generation or PostgreSQL, the index degrades — smaller cache, previous generation, built index only, keyword search alone — and never takes the app down; the builder sets its own `oom_score_adj` to 1000. No search asks a part for more than it holds (5.18, 5.24), a disk index is never searched with one thread (it hangs, 5.24), and the recent index is built with `saturate_graph=True` (5.19). See `vector_db_manager/README.md`

### Query Pipeline (`pipeline/`)

`AtlasPipeline.ask(conversation_id, user_id, query)` in order, each stage through its owner's public interface:

1. **Ownership** — `ConversationRegistry.get(conversation_id, user_id)`; another user's conversation is "not found".
2. **Decomposition** — `Decompose.decompose()` records the exchange in PostgreSQL and returns `Decomposition` (mode, topics, sub-queries).
3. **Memory** — `MemoryManager(query, …, topic=primary_topic or "General")`: route, `conversation_context()`, `conversation_retrieval()` (KSV inside).
4. **Global retrieval + KSV** — skipped for a `DOUBT` that memory answers `FULL`. Otherwise `Retriever` over the working query and each sub-query; KSV keeps only supporting passages as evidence.
5. **Acquisition** — on a `none`/empty verdict, `Acquirer` (below), then one refresh and re-search.
6. **Thinking** → 7. **Generation** → 8. **Persistence** — `MemoryManager.update` or `store_new_topic`/`store_new_project` (project named by the draft model), then `Decompose.record_response`. A failed write is a warning on the response, never hidden.

`AtlasResponse` carries the text, status, `persisted` and a `RequestRecord` (`observability/telemetry.py`): monotonic stage timings, retries, model calls by purpose, cache and retrieval outcomes, never prompt text. `status --time`/`--retries` read it. `pipeline/bootstrap.py::Components` builds everything once per process; `observability/health.py` runs 29 bounded checks (no text generation) for `status`. Evidence is tagged `M1…` (memory) / `G1…` (global) with provenance URLs from `corpus_ingestion.source_urls`.

### Thinking Layer (`thinking_layer/`)

Depth comes from decomposition, with no new classification: `DOUBT` → **DIRECT** (one answer job + job critique), `DEVELOPMENT` → **PLANNED** (model planner, ≤3 jobs, plan critique, 1 revision), `DEVELOPMENT` with sub-queries → **EXPANDED** (one worker job per sub-query + a stitch job). `ThinkingOrchestrator.think(ThinkingRequest)` coordinates `Planner`, `JobPool` (validated DAG, enforced state machine, claim-once), `Scheduler` (dependency order, failed prerequisites cancel dependents), `WorkerPool` (bounded threads — logical concurrency only; inference is serialised by the model lock), `Accumulator` (accepted results only; model output is never filed as evidence), `JobCritic` (rule checks, e.g. invented citations, before the model) and `PlanCritic` (reopens only the named jobs and their dependents). Budgets in `settings.py`/Config: job reasoning retries, runtime retries (separate), plan revisions, a request deadline. The original query always reaches every prompt; a condensed one is shown beside it, never instead. Returns `ThinkingResult` (status, jobs, accepted, cited evidence, critiques, limitations, `RetryStats`, timings).

### Inference (`inference/`)

- **One model per process** (`SharedModel`, `runtime.shared_runtime()`): loaded on first use at `Config.SHARED_MODEL_CONTEXT_WINDOW` (32768), every call under one `RLock`. Query decomposition's `DraftModel()` and the summariser borrow it; `ReasoningModel` sends worker/stitch jobs to the main model (the draft unless `Config.MAIN_MODEL_PATH` is set). Callers use `infer(system, user, schema=, max_tokens=, purpose=)` (`model_port.TextModel`).
- **KV checkpoints** (`KVSessionManager`, `CheckpointStore`): main generation formats and tokenises the prompt with the model's own template, restores the conversation's saved state when its tokens are a longer exact prefix than what is resident, generates, then truncates to the evidence-free prefix (system + history) and saves it. Files in `data/inference_cache/` (0600, atomic, no pickle) are loaded only if model fingerprint, context window, format version and SHA-256 match the `inference_checkpoints` row; anything else is discarded and recomputed. The generation history window moves in half-window steps so the saved prefix stays reusable. llama.cpp's results depend on batch partitioning, so a cold recompute is not bit-identical to a restored run — compare like with like.

### Generation Layer (`generation_layer/`)

`assemble()` builds a deterministic, budgeted context — prefix: system prompt + rolling summary + history; turn: request verbatim, accepted results (the stitch when there is one), evidence cited first, knowledge status, limitations — and `Generator` writes the answer through the session manager. It does no retrieval, planning or memory writes.

### Knowledge Acquisition (`pipeline/acquisition.py`, `knowledge_sufficiency/`)

Discovery asks a source through its search (sqlite.org, IETF) or, when it has none or the search found nothing, through its published sitemap (postgresql.org's current manual, Cassandra's latest docs; `sitemaps.py`, configured by `sitemap`/`sitemap_scope` in `sources.toml`). Search results drop menu/header/footer links and are ranked by the target's terms (bug 9.1); sitemap URLs are ranked by path words weighted by rarity (9.13). Everything obeys robots.txt, evaluated per RFC 9309 (`robots.py`, 9.16). The acquirer chunks readable text, not markup, and stores through `ingest_files` with two limits: `MAX_ACQUISITION_DOWNLOAD_BYTES` per acquisition and `MAX_ACQUISITION_STORAGE_BYTES` for acquired material in all, checked and recorded inside the ingestion lock.

### Ingestion Quota (`data_layer/ingestion/quota.py`, `corpus_ingestion.py`)

`ingest_files()` ingests one document at a time inside `IngestionQuota.reserve()` (a process and file lock, so ingestions are serialised): it estimates growth (24x text + 128 KiB + 8 MiB WAL headroom), refuses before writing, re-measures after, and stops a run that passes the limit. Counted: the chunk store and its WAL, the keyword index, the DiskANN directory, `data/corpus/`, and the corpus's share of PostgreSQL (`VectorRepository.storage_bytes()`). `IngestionPipeline.batch_insert_vectors` checks the quota too, as a backstop. `None` disables the quota, never the disk reserve.

### Memory Layer

Hierarchical organization: Topic → Project → Conversation → Snapshot

- **MemoryDatabase** (`memory/memory_database.py`): every memory table lives in one SQLite file, `Config.MEMORY_DB` (`data/memory/memory_layer/memory_layer.db`), on one connection per process (`MemoryDatabase.shared()`). It holds **no table SQL**: each owner module declares a `Schema(name, create, requires)` and keeps its own queries, so ownership is what it was with five files. Relationships are foreign keys, not code — a child carrying a copy of its parent's column references the pair, and a project's topic move cascades. Owners take `database: MemoryDatabase | str | Path | None`; their `close()` releases nothing. Refused foreign keys surface as `TopicNotFound` / `ProjectNotFound` / `ProjectInAnotherTopic`. `writing()` opens `BEGIN IMMEDIATE`; nested writes are savepoints. Tests never reach the real file — the root `conftest.py` points `Config.MEMORY_DB` at a per-test path; seed parents with the `seed_topics` / `seed_projects` fixtures. See `memory/README.md`

- **FullConversation / ConversationPoolManager**: turns are written with `append_turn(role, text)` and read back as `Turn(sequence_number, role, text, created_at, chunk_id)` — `history()`, `recent(n)`, `context(start, end)`, `since(seq)` on the manager. Always order by `sequence_number`, never `created_at`. The older text-only readers on the repository and bucket drop the speaker; do not build prompts from them. See `docs/memory_layer_docs/conversation_turns.md`
- **ConversationSummary**: feeds the draft model a speaker-labelled transcript (`render_transcript`), batched between turns so no batch opens mid-turn without its speaker
- **ProjectMetaData** (`project_data_repo/project_meta_data.py`): the project registry, scoped to one project *and its topic* — `ProjectMetaData(project_id, topic_id, ...)`. `add_project_vector(vector, vector_id, project_name, project_summary)` writes the embedding to pgvector, then the project row and mapping row in one SQLite transaction, with a compensating delete if the metadata fails. Descriptions are several per project, keyed `(project_id, project_description_id)`. Both child tables carry a denormalised `topic_id`, kept in step by the upsert, so `get_topic_summary_vector_ids()` answers "every summary vector in this topic, and whose it is" without a join — the read the query router needs. `get_project()` returns a `ProjectRow`. See `docs/memory_layer_docs/project_meta_data.md`
- **ProjectVectorHandler** (`project_data_repo/project_vector_handler.py`): a project's vectors — one for its summary, one per description. Ids are derived from the source (`summary_vector_id(project_id)`, `description_vector_id(project_id, description_id)`), never passed, so add/update/get/delete agree without a lookup. `get_project_vectors(project_id, ids)` reads a batch in the order given; the ids come from `ProjectMetaData.get_topic_summary_vector_ids()` because the vector store cannot enumerate. Keeps one `VectorRepository` per project rather than one per call
- **TopicManager** (`topic_pool/topic_manager.py`): create, read and soft-delete a topic, over `TopicPoolMetaHandler` (`topic_pool_repo/`). Reads filter on `is_active = 't'`, so a soft-deleted topic is invisible while its row stays, and still satisfies the projects' foreign key
- **ProjectManager** (`topic_pool/project_pool/project_manager.py`): routes `(topic_id, query)` to an existing project. `route()` -> `project_id` or `None`; `resolve()` -> `ProjectMatch` with score, margin, `ambiguous` and ranked candidates. A project scores as its best-matching vector, not its average. The no branch is a `pass` pending the thinking layer. Architecture lives in `memory/topic_pool/project_pool/README.md` — **module READMEs hold architecture, data flow and design rationale; `docs/` stays the API reference; code comments are only for logic that reads wrong without one**
- **SnapShot** (`memory/snapshot.py`): Bidirectional cursor traversal and cosine similarity (torch) over a snapshot history, for **either** scope. `scope="conversation"` (the default) walks this conversation's cumulative summaries and needs a `conversation_id`; `scope="project"` walks the project registry and does not. The cursors, `advance`/`prev` and `search` are the same code for both, because each scope stores its vector under the id its listing returns. `add()` is conversation-only and `add_project_snapshot()` project-only; the wrong one raises `WrongSnapshotScope`
- **ProjectSnapshot** (`topic_pool/project_pool/project_snapshot.py`): a project's rolling description — what it is about and what has been done. Incremental: each snapshot summarises the previous one plus the conversation summaries written since, so cost tracks what changed rather than the project's size. It reads across every conversation (`project_id`, ignoring `conversation_id`) because the result describes the project, not a chat. The watermark is `seq` on `cumulative_vector_meta_data`, allocated monotonically on write under `BEGIN IMMEDIATE`. An empty summary from the draft model does **not** advance the watermark, or those summaries would be dropped from every future snapshot. Triggered on every conversation snapshot, from inside the window where the draft model is already resident — loading it is the expensive part, so a second load is avoided
- **ConversationRegistry** (`memory/conversation_registry.py`): `conversations(conversation_id, user_id, conversation_name, created_at timestamptz)` **in PostgreSQL**, beside `conversation_table`, through `storage/postgres.py`; obtain it with `open_conversation_registry()`. Names unique per user by the constraint `conversations_user_name_unique`; a clashing name gets ` (n)` via `ON CONFLICT ... DO NOTHING` and a bounded retry. `migrate_from_sqlite` moved the rows an earlier version kept in `memory_layer.db` (that table is now the backup `conversations_migrated`; bug 9.14). Names are proposed by the draft model from the first query (`pipeline/naming.py`), cleaned and bounded, with a deterministic fallback
- **MemoryMappingHandler** (`memory/memory_mapping_handler.py`): `(conversation_id, user_id)` -> topic, project, latest project snapshot — the lookup `MemoryManager` needs to resume a conversation without routing it again, which is why it sits beside `memory_manager.py` rather than in the conversation pool. Takes no `conversation_id`: every method names the conversation it acts on, so one handler serves the whole table. The write is two steps because the information is — a row is opened when a conversation starts and routing fills it in later, so `search` returning `None` ("never seen") and `MemoryMapping(None, None, None)` ("seen, not yet routed") are deliberately different answers. The latest project snapshot is read from the chain on every `search`, not cached — routing stores only topic and project, which reference `project_table` as a pair. `user_id` is the only user-scoped column in the layer — see `bugs.md` 4.83
- **ProjectSnapshotRepository** (`project_data_repo/project_snapshot_repo.py`): `project_snapshot` (summary, length, `last_seq_included`, timestamp) and `project_snapshot_mapping`, append-only on `(project_id, project_snapshot_id)` so it holds the ordered chain rather than a pointer at the newest. `project_snapshot_id` is also the pgvector `vector_id`, so a row needs no lookup to reach its embedding
- **ConversationVectorMetaDataRepository** (`conversation_data_management/conversationVectorMetaManager.py`): SQLite metadata: `summary_vector_meta_data`, `cumulative_vector_meta_data` (carrying `conversation_id` and `seq`, unique **per project**), `summary_snapshot_map`; it writes snapshot chunks into `summary_chunks`, which `FullConversationRepository` owns. `get_highest_summarised_sequence()` is per **conversation**. `insert_snapshot()` writes a whole snapshot in one transaction.

### Storage

| Store | Path | Purpose |
|-------|------|---------|
| DiskANN index | `data/disk_ann_index/` | `gen-NNNNNN/` built generations (the current one and the one before, as a fallback) and `CURRENT` naming the one in use; plus an in-memory index of what was ingested since |
| SQLite (chunker) | `data/hierarchical_db/` | Chunk metadata: `Documents`, `Sections`, `Contexts`, `Chunks` for the hierarchical path and `Documents`, `RecursiveChunks` for the flat one; plus `vector_meta_data` — each DiskANN label, its vector id and its chunk, no vectors (`vectorMetaDataRepository.py`) |
| SQLite (memory) | `data/memory/memory_layer/memory_layer.db` | **Every** memory table, one file (`memory_database.py`): topics, the project registry and its snapshot chain, conversation turns and snapshot metadata, the conversation mapping and `inference_checkpoints`. (`conversations_migrated` is the pre-9.14 registry, kept as a backup and never read.) Foreign keys enforced across all of them. WAL journal (so `.db-wal` and `.db-shm` sit alongside it), `synchronous=NORMAL`. The old `data/topic_db/`, `data/project_db/` and `data/memory_mapping/` files are no longer read |
| PostgreSQL | localhost:5432, DB `Vectors` | Every vector, via pgvector (`vectorRepository.py`): document chunks under project id `"global"`, the memory layer's under theirs; `conversation_table`, the exchange log (query, response); and `conversations`, the registry (id, owner, unique name, start) |
| Corpus files | `data/corpus/` | Downloaded sources (`sources/`), acquired text (`acquired/`), `manifest.jsonl` (url, topic, fetched_at, sha256, document_id, stored bytes) |
| KV checkpoints | `data/inference_cache/` | One llama.cpp state per conversation; disposable — deleting it costs only recomputation |
| Models | `models/draft_model/` | `qwen2.5-3b-instruct-q4_k_m.gguf` (~2 GB) |

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
- `MAX_VECTORS`: 1,000,000 — the corpus size designed for; no index is sized to it
- `VECTOR_INDEX_RAM_MB` (512), `RECENT_VECTOR_CAPACITY` (20,000), `INDEX_REBUILD_AT` (10,000), `MEMORY_HEADROOM_MB` (256), `GLOBAL_VECTOR_SCOPE` (`"global"`)
- `INDEX_PATH`, `DB_PATH`, `CONVERSATION` paths
- `SHARED_MODEL_CONTEXT_WINDOW` (32768), `MAIN_MODEL_PATH` (None = the draft answers), `MODEL_THREADS`
- `THINKING_*` budgets (workers 2, job retries 1, runtime retries 1, plan revisions 1, planned jobs 3, deadline 900 s); `GENERATION_MAX_TOKENS` (1200), `GENERATION_CONTEXT_TOKENS` (12000), `GENERATION_HISTORY_TURNS` (6)
- `INFERENCE_CACHE_DIR`, `INFERENCE_CACHE_MAX_BYTES` (2 GiB, LRU)
- `MAX_INGESTION_STORAGE_BYTES` (1 GiB), `MAX_ACQUISITION_STORAGE_BYTES` (256 MiB), `MAX_ACQUISITION_DOWNLOAD_BYTES` (16 MiB): bytes or `None`; the environment overrides each (`null`/`none` = unlimited, anything non-positive is refused at import). `INGESTION_DISK_RESERVE_BYTES` (512 MiB), `CORPUS_DIR`
- `LOG_FILE`, `DEBUG`: defaults `main.py` hands to `configure_logging()`

### Exception Hierarchy

- Data layer: `InvalidFileType`, `VectorInsertionError`, `DuplicateVectorException`, `InvalidEmbeddingArgument`, `MissingVectorId`, `MalformedVectorId`, `VectorIdConflict`, `VectorStoreUnavailable`, etc. in `data_layer/datalayer_exceptions/`
- Memory layer: `InvalidCursorException`, `NullPointerException`, `MisMatchCount` in `memory/memory_pool_exceptions.py`

## Known Bugs (see `bugs.md` and `production_impact_report.md`)

**Open:**
- Python and MDN docs are reachable only by explicit URL: no usable search, and their sitemaps list version roots / gzipped children
- Bug 9.9 (P2): `uv sync` removes llama-cpp-python and `install_llama_cpp.py` cannot reinstall it (no pip in the venv)
- Bug 9.8 (P3): hydration's `Passage.document_id` is a context id for hierarchical chunks (worked around in provenance)
- No corpus or conversation deletion yet; `CheckpointStore.delete_conversation` is ready for it

**Fixed since this list was written** (each has a regression test):
- Bugs 2.1/2.2/4.1: the CLI and `-q` run the whole pipeline; `MemoryManager` is implemented and wired
- Bugs 9.13–9.16: sitemap discovery, the registry moved to PostgreSQL with its SQLite rows migrated, the test leak into PostgreSQL, RFC 9309 robots.txt
- Bugs 9.1–9.7, 9.10–9.12, 7.8: discovery, robots.txt, source search URLs, acquisition storage and limits, streaming fetch, one shared model, ingestion estimate, critic scope, naming, the `llama_cpp` stub
- Bug 4.3: `search()` no longer moves the instance cursors — `__find_best_snapshot` scans with local ones
- Bug 4.4: a failed similarity search returns `None`, not `[-1]`
- Vector ids are ints masked into the signed 64-bit range (`Config.VECTOR_ID_MASK`), derived from `chunk_id` rather than chunk text
- The DiskANN index survives a restart: built generations are opened from disk, and only vectors ingested since are loaded (bugs 5.20, 5.23, the old P3); the recent index's graph is saturated so it can be searched at all (5.19)

## Docs

Two kinds, and they do not overlap:

- **Module READMEs** (`<module>/README.md`, one per module and submodule) — what the module does, how data flows through it, and **why it was designed that way**. Start here to understand a module.
- **`docs/`** — the API reference: classes, methods, parameters, schemas.

**Code comments are only for logic that reads wrong without one** (a guard against an infinite loop, an evaluation-order trap, a silent SQLite behaviour). Architecture, rationale and "what this does" belong in the README, not in the file. Docstrings are one line.
