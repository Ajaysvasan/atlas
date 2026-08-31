# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python RAG (Retrieval-Augmented Generation) backend system that ingests multi-format documents, generates embeddings, stores them in a DiskANN vector index, and manages conversation memory with snapshot history. Currently CLI-only — no FastAPI routes exist yet.

## Running the Application

```bash
# From the app/ directory
python main.py                    # Interactive CLI mode
python main.py --verbose          # With debug logging
```

## Running Tests

Tests must be run from the `app/` directory with `PYTHONPATH` set:

```bash
PYTHONPATH=. pytest test/ -v                            # All tests
PYTHONPATH=. pytest test/data_layer_testing/ -v         # Data layer tests
PYTHONPATH=. pytest test/memory_layer_testing/ -v       # Memory layer tests
PYTHONPATH=. pytest test/stress_testing/ -v             # Stress tests
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

- **Snapshot** (`memory/.../snapshot.py`): Bidirectional cursor traversal of conversation history; uses cosine similarity (torch) to find similar snapshots; stores summary and cumulative vectors
- **ConversationVectorMetaDataRepository** (`conversation_data_management/conversationVectorMetaManager.py`): SQLite-based metadata with tables: `summary_chunks`, `summary_vector_meta_data`, `cumulative_vector_meta_data`, `summary_snapshot_map`

### Storage

| Store | Path | Purpose |
|-------|------|---------|
| DiskANN index | `data/disk_ann_index/` | Approximate nearest neighbor vector search |
| SQLite (chunker) | `data/hierarchical_db/` | Chunk metadata: `Documents`, `Sections`, `Contexts`, `Chunks` for the hierarchical path and `Documents`, `RecursiveChunks` for the flat one |
| SQLite (memory) | `data/memory/topic_pool/.../conversation_pool/{project_id}_conversation.db` | Conversation snapshot metadata |
| PostgreSQL | localhost:5432, DB `Vectors` | Vector repository via pgvector (`vectorRepository.py`) |

PostgreSQL credentials are in `.env` (not committed; see `.env.example`): `DBNAME`, `DB_USER`, `PASSWORD`, `HOST`, `PORT`. `DB_USER` is deliberately not `USER` — login shells export `USER`, and `load_dotenv()` will not override it.

### Key Data Models

- `NormalizedContent`: normalized text, `has_section` flag, `sections` (`Tuple[SectionSpan, ...]`), metadata
- `SectionSpan`: a heading plus its body, as absolute offsets into `NormalizedContent.content`
- `HChunk` / `RChunk`: hierarchical vs recursive chunks. Offsets on both are absolute into the normalized document, so `content[start_off_set:end_off_set] == chunk`
- `EmbeddedChunk`: numpy float32 vector, MD5 vector_id, chunk metadata
- `Document`, `Section`, `Context`: intermediate pipeline models

### Configuration (`config.py`)

Central config class with constants:
- `EMBEDDING_MODEL`: `sentence-transformers/all-MiniLM-L6-v2`
- `EMBEDDING_DIMENSIONS`: 128
- `K_NEIGHBORS`: 9
- `MAX_VECTORS`: 1,000,000
- `INDEX_PATH`, `DB_PATH`, `CONVERSATION` paths

### Exception Hierarchy

- Data layer: `InvalidFileType`, `VectorInsertionError`, `DuplicateVectorException`, `InvalidEmbeddingArgument`, etc. in `data_layer/datalayer_exceptions/`
- Memory layer: `InvalidCursorException`, `NullPointerException`, `MisMatchCount` in `memory/memory_pool_exceptions.py`

## Known Bugs (see `bugs.md` and `production_impact_report.md`)

**P1 — snapshot.py:**
- Bug 4.3: `get_snapshots()` destructively resets cursor positions to 0/len-1 on every call
- Bug 4.4: Failed similarity search returns `[-1]` index instead of `None`/empty

**P2 — Unimplemented:**
- Bug 2.1/2.2: `cli/cli_interface.py` query loop is a placeholder (`# some stuff`) — no downstream pipeline integration
- Bug 4.1: `MemoryManager`, `ProjectManager`, `TopicManager`, `ConversationPoolManager`, `ConversationSummary` are empty stub classes

**P3 — Data type mismatches:**
- Vector IDs passed as strings instead of unsigned integers
- DiskANN index not persisted/reloaded correctly between sessions

## Docs

Internal documentation lives in `docs/core_docs/` including architecture diagrams (Mermaid) and per-module API docs.
