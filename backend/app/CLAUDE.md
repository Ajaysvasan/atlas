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

- **FileLoader** (`TextFileProcessor/file_loader.py`): Recursively scans directories, returns `Dict[extension, List[Path]]`
- **TextExtractor** (`TextFileProcessor/text_extractor.py`): Extracts text from `.txt`, `.pdf`, `.docx`, `.doc`, `.md`, `.html`, `.xml`, `.csv`
- **TextNormalizer** (`normalizer/normalizer.py`): Cleans text, detects sections via regex `^(?=.*[A-Z])[A-Z\s]+$`, generates SHA256 content IDs, returns `NormalizedContent`
- **Chunker** (`Chunker/chunker.py`): Routes to `HierarchicalChunker` (documents with sections) or `RecursiveChunker` (flat docs). chunk_size=256, overlap=20
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
| SQLite (chunker) | `data/hierarchical_db/` | Hierarchical chunk metadata |
| SQLite (memory) | `data/memory/topic_pool/.../conversation_pool/{project_id}_conversation.db` | Conversation snapshot metadata |
| PostgreSQL | localhost:5432, DB `Vectors` | Vector repository via pgvector (`vectorRepository.py`) |

PostgreSQL credentials are in `.env` (not committed; see `.env.example`): `DBNAME`, `DB_USER`, `PASSWORD`, `HOST`, `PORT`. `DB_USER` is deliberately not `USER` — login shells export `USER`, and `load_dotenv()` will not override it.

### Key Data Models

- `NormalizedContent`: normalized text, `has_section` flag, metadata
- `HChunk` / `RChunk`: hierarchical vs recursive chunks with offsets and metadata
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

**P3 — Data type mismatches (causing 100% test failure):**
- `TextExtractor` returns `PosixPath` objects; `TextNormalizer` expects strings
- `EmbeddingManager` returns lists instead of numpy arrays
- Markdown extractor doubles newlines
- Vector IDs passed as strings instead of unsigned integers
- DiskANN index not persisted/reloaded correctly between sessions

## Docs

Internal documentation lives in `docs/core_docs/` including architecture diagrams (Mermaid) and per-module API docs.
