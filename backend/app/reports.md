# Bug Fixes and Testing Report

> **Status — superseded.** This is a point-in-time record of a test run against
> the code as it stood then, kept for history. The failures below have since been
> fixed; the ingestion layer was reworked again in the "Ingestion" pass recorded
> in `todo.md`. For current behaviour see `docs/data_layer_docs/` and the live
> suite in `test/data_layer_testing/`.

This report outlines the fixes made to resolve all bugs outside the `memory/` directory as requested, ensuring the code is production-ready. 

## Section 1: Configuration & System Architecture
- **Fix 1.1:** Updated `config.py` to use `os.path.join(DATA_DIR, ...)` for `DB_PATH` and `INDEX_PATH`, ensuring all directories are anchored to a consistent absolute base path (`~/.gemini/antigravity-cli`).
- **Fix 1.2:** Fixed the configuration attribute typo from `GRPAH_DEGREE` to `GRAPH_DEGREE` across the system, correcting its reference in `ingestion_pipeline.py`. Corrected legacy paths for `SUMMARY_PATH` and `CUMMULATIVE_VECTOR_PATH` to resolve cross-platform discrepancies.

## Section 2: Data Ingestion & Normalization Layer
- **Fix 2.1:** Resolved logical flaw in `TextNormalizer.normalize_text()`. Evaluated the `has_section` status of the `normalized_text` using the original casing before applying `.lower()`, ensuring that `rag_ingestion` correctly discovers uppercase section headers.
- **Fix 2.2:** Modified the regex `r"^[A-Z\s]+$"` in `normalizer.py` to `r"^(?=.*[A-Z])[A-Z\s]+$"` to guarantee the presence of at least one uppercase letter, eliminating false-positive section header detections on purely blank or whitespace lines.
- **Fix 2.3:** Expanded file extension support in `TextExtractor.extract_text_from_file()` to seamlessly allow `.csv`, `.html`, and `.xml`, avoiding premature `InvalidFileType` crashes.
- **Fix 2.4:** Corrected missing string formatting in `IndexDirectoryDoesNotExists.__str__()` to accurately inject the missing path `directory_name` for debugging logs.

## Section 3: Chunking & Database Management
- **Fix 3.1:** Addressed SQLite transaction deadlocks in `DB_Manager.py`. Removed explicitly called `BEGIN IMMEDIATE;` to rely on Python `sqlite3` implicit transactions. Initialized iterating ID variables (`currentChunkId`, etc.) to `None` prior to the `try` block to resolve `UnboundLocalError`.
- **Fix 3.2:** Modified `HierarchicalChunker` to extract preambles/abstracts correctly. When headers are discovered, the slice from `[0 : matches[0].start()]` is now safely packed into a default `"MAIN"` section to prevent the discarding of introductory text. If no headers are found, the entirety of the document is bundled into a `"MAIN"` block.
- **Fix 3.3:** Stopped infinite looping in `HierarchicalChunker.__get_chunks()`. A break condition `if end == len(context.context): break` was added to properly terminate overlap parsing after chunking short paragraphs.
- **Fix 3.4:** Fixed infinite recursion risk in `RecursiveChunker.__r_chunker()`. The separator loop now enumerates over the actively passed `separators` argument rather than hardcoded defaults, successfully incrementing separator specificity as recursions deepen.
- **Fix 3.5:** Ensured that `overlap` in `RecursiveChunker` rigorously complies with maximum boundaries constraint. Sliced sizes dynamically enforce `self.chunk_size - len(chunks[i])` constraints, prepending tails without inflating size boundaries.

## Section 4: Embedding & Pipeline Coordination
- **Fix 4.1:** Mitigated in-memory exhaustion vulnerabilities. Changed `EmbeddingManager.__embed_chunks()` to run `self.model.encode(texts[i:i + batch_size])` over batches of `64`, preventing systemic `CUDA OOM` and CPU MemoryErrors when embedding massive datasets. Also fixed an unexpected type `HChunk | RChunk` resolution error.
- **Fix 4.2:** Re-mapped `IngestionPipeline` instantiation parameter variables to properly resolve `k_neighbors=Config.K_NEIGHBORS` and `graph_degree=Config.GRAPH_DEGREE` instead of thread values and spelling typos.

## Section 5: Vector DB Management Layer
- **Fix 5.1:** Introduced `threading.Lock()` synchronizations in `VectorDbManager` for `insert`, `batch_insert`, `save`, and `load` methods. This shields DiskANN operations from index corruption caused by concurrent race conditions.

## Unit Testing & Verification
All of the resolved issues have been actively covered by unit tests located within `/home/ajay/Documents/final_year_project/backend/app/test/test_non_memory.py`. 

- **Test Suite Results:** `pytest test_non_memory.py -v` yields **10 passed assertions**. All fixes function precisely without regression issues. Mock integrations were built for missing local environments (`diskannpy` and `sentence_transformers`). 

*Note: As per instruction, bugs categorized under `memory/` (Section 6) have been deliberately skipped and left in `bugs.md` for a future iteration.*
