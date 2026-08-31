# Production Data Layer Bug Impact Report

> **Status — superseded.** This is a point-in-time record of a test run against
> the code as it stood then, kept for history. The failures below have since been
> fixed; the ingestion layer was reworked again in the "Ingestion" pass recorded
> in `todo.md`. For current behaviour see `docs/data_layer_docs/` and the live
> suite in `test/data_layer_testing/`.

This report summarizes the results of running a clean "production-simulation" unit test suite on the current `data_layer` codebase. The goal of this suite was to test the layer assuming it *should* work flawlessly in a production environment, effectively revealing exactly how many native workflows crash due to the underlying bugs.

## Overall Test Execution Results
- **Total Tests Run**: 6
- **Tests Passed**: 0
- **Tests Failed**: 6 (100% Failure Rate)

Every single core component of the `data_layer` currently crashes or corrupts data under standard operational load.

---

## Detailed Breakdown of Crashes

### 1. `test_full_ingestion_pipeline_end_to_end`
- **What it tested:** Running the entire `IngestionPipeline` end-to-end, mimicking what happens when a user drops a file into the app (Load -> Extract -> Normalize -> Chunk -> Embed -> Ingest).
- **Result:** **FAILED**. 
- **The Bug:** `TypeError: sequence item 1: expected str instance, PosixPath found`. 
- **Impact:** The pipeline dies at the Normalization phase because `TextExtractor` returns `pathlib.Path` objects as keys, which the normalizer blindly attempts to concatenate as strings to generate document IDs.

### 2. `test_normalizer_batch_processing`
- **What it tested:** Feeding the output of the `TextExtractor` directly into the `Normalizer` API.
- **Result:** **FAILED**. 
- **The Bug:** `TypeError: expected str instance, PosixPath found`.
- **Impact:** Any attempt to batch process files fails completely due to the `pathlib.Path` datatype mismatch, blocking all file ingestion.

### 3. `test_text_extractor_markdown_formatting`
- **What it tested:** Extracting a standard `.md` file to ensure the structure remains intact.
- **Result:** **FAILED**. 
- **The Bug:** `AssertionError: 'Line 1\n\nLine 2\n\nLine 3\n' != 'Line 1\nLine 2\nLine 3\n'`
- **Impact:** The extractor automatically doubles every single newline in the document, completely corrupting Markdown spacing and layout across the entire chunking pipeline.

### 4. `test_embedding_manager_output_types`
- **What it tested:** Checking if the `EmbeddingManager` produces arrays formatted for DiskANN vector databases.
- **Result:** **FAILED**. 
- **The Bug:** `AssertionError: <class 'list'> is not an instance of <class 'numpy.ndarray'>`
- **Impact:** The `EmbeddingManager` stores vector lists natively in Python, which strictly violates DiskANN's requirement for contiguous numpy arrays.

### 5. `test_vector_db_manager_batch_insert`
- **What it tested:** Using `VectorDbManager` to batch insert production-ready chunks into the vector database.
- **Result:** **FAILED**. 
- **The Bug:** `TypeError: DiskANN vectors must be numpy.ndarray`
- **Impact:** Because the EmbeddingManager passes lists, and `chunk_id`s are passed as strings instead of unsigned integers, DiskANN's pybind11 C++ backend immediately crashes when attempting to parse the Python list payloads.

### 6. `test_vector_db_manager_index_load_persistence`
- **What it tested:** Loading an existing DiskANN index from the disk and attempting to resume operations on it.
- **Result:** **FAILED**.
- **The Bug:** `AssertionError: The internal DiskANN instance was not updated after load()`
- **Impact:** The index loads from the disk, but the manager completely discards the reference to it, meaning the application silently operates on an empty index while users assume their database was successfully restored.

---

## Conclusion
The current `data_layer` has a **0% passing rate** in production simulation. Every single core module (Extraction, Normalization, Embedding, Vector Database Management, and Pipeline Integration) suffers from a critical data type mismatch or architectural oversight. 

To achieve production readiness, we must patch these 6 critical failures documented in `bugs.md` Section 7.
