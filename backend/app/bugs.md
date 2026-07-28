# Comprehensive Project Bug Inventory (`backend/app`)

This document catalogs all logical, architectural, concurrency, and data integrity bugs identified across the project. The bugs are organized in a **section-wise manner** grouped by architectural layer, complete with detailed explanations, **Criticality** ratings (`Critical`, `High`, `Medium`, `Low`), and recommended resolution **Priority** levels (`P0`, `P1`, `P2`, `P3`).

---

## Section 1: Memory & Conversation Pool Layer (`memory/`)

### Bug 1.1: Hardcoded Empty Tensors Cause Immediate Runtime Crash in `SnapShot.__find_best_snapshot` (`snapshot.py`)
- **Criticality:** Critical
- **Priority:** P0
- **Explanation:** In `__find_best_snapshot(self, query: List)`, `right_snap_vector_cumulative` and `left_snap_vector_cumulative` are initialized as empty 0D/1D PyTorch tensors `tensor([])`. Immediately afterwards, `torch.cosine_similarity(left_snap_vector_cumulative, tensor(query))` is executed. Because `tensor([])` has size `0` while `tensor(query)` has dimension `D`, `cosine_similarity` immediately raises `RuntimeError: The size of tensor a (0) must match the size of tensor b (D) at non-singleton dimension 0`. The method never loads or compares actual snapshot vectors.

### Bug 1.2: Cursor Reset Clobbering & Duplicate Comparison in `SnapShot` (`snapshot.py`)
- **Criticality:** Medium
- **Priority:** P2
- **Explanation:** In `SnapShot.add()`, whenever a new snapshot node is appended, `self.__left_cursor` is reset to `0` and `self.__right_cursor` to `len - 1` by default (`reset_right_pointer=True, reset_left_pointer=True`), clobbering any active iteration cursors. Additionally, in `__find_best_snapshot`, the while loop condition is `while self.__left_cursor <= self.__right_cursor:`. When `self.__left_cursor == self.__right_cursor` (the exact midpoint of an odd-length snapshot list), both `left_snap` and `right_snap` point to the same node, computing and comparing `cosine_similarity` twice for the exact same object before terminating.

### Bug 1.3: Swapped Path Definitions & Unimplemented Method in Legacy `ConversationVectorManager` (`conversatoinVectorManager.py`)
- **Criticality:** Critical
- **Priority:** P0
- **Explanation:** In `conversation_data_management/conversatoinVectorManager.py` (which has a typo in its filename), `add_cummulative_summary_vector` is completely unimplemented (`pass`). Furthermore, `add_summary_vectors` stores binary vectors inside `self.cummulative_vector_path`, while `get_summary_vector` reads from `self.cummulative_vector_path` and `get_cummulative_summary_vector` reads from `self.summary_path`. The directory paths and vector roles are completely inverted.

### Bug 1.4: SQLite Connection Thread Incompatibility & Missing Transaction Rollbacks (`conversationVectorMetaManager.py`)
- **Criticality:** High
- **Priority:** P1
- **Explanation:** `ConversationVectorMetaDataManager` maintains a persistent SQLite connection `self.conn` (`sqlite3.connect()`) across instance lifetime. If instance methods (`insert_snapshot`, `load_snap_shot_objects`) are invoked from different background worker threads or async tasks, `sqlite3.ProgrammingError: SQLite objects created in a thread can only be used in that same thread` is thrown. Additionally, multi-row operations (`__insert_vector_ids`, `__insert_snapshot_metadata`) lack `try ... except sqlite3.Error: self.conn.rollback()` handling, leaving connections in aborted transaction states if a primary/foreign key constraint fails.

### Bug 1.5: Empty & Placeholder Modules In Memory Pool 
- **Criticality:** Medium
- **Priority:** P2
- **Explanation:** Several core memory modules contain unwritten placeholder files or typos: `conversaton_summary.py` (typo `conversaton`) contains only `class ConversationSummary: pass`, `full_conversation_bucket.py` contains only `class FullConversationBucket: pass`, and `conversation_pool_manager.py`, `memory_manager.py`, `topic_manager.py`, `project_manager.py` are completely empty (0 bytes). This indicates that the core components of the memory architecture are fundamentally missing.

---

## Section 7: Data Layer Architecture & Integrity (New Discoveries)

### Bug 7.1: Invalid String UUID Passed to DiskANN Vector ID (`vectorDbManager.py`)
- **Criticality:** Critical
- **Priority:** P0
- **Explanation:** `VectorDbManager` extracts `vector_id = embedded_chunk_obj.meta_data.chunk_id` (which is a SHA256 string hash generated in the Chunker) and passes it directly to DiskANN. `diskannpy` strictly requires vector IDs to be unsigned integers (`np.uint32` or `np.uint64`). Passing a string will trigger an immediate `TypeError` from pybind11 during `insert()` or `batch_insert()`.

### Bug 7.2: Python List Passed to Numpy Array in Batch Insertion (`EmbeddingManager.py` / `vectorDB_diskann.py`)
- **Criticality:** Critical
- **Priority:** P0
- **Explanation:** `EmbeddingManager` stores embeddings as a Python `List[float]` via `.tolist()`. `VectorDbManager` directly accumulates these lists and passes them into DiskANN's `batch_insert(vectors, vector_ids)`. DiskANN explicitly requires a contiguous 2D `numpy.ndarray` (specifically `np.float32` or `np.int8`). Supplying a nested `List[List[float]]` will cause a fatal type incompatibility crash.

### Bug 7.3: Invalid Custom Exception Catching from Native Extension (`vectorDB_diskann.py`)
- **Criticality:** High
- **Priority:** P1
- **Explanation:** In `VectorDb_diskann.__insert_vector()`, the insertion call is wrapped in a `try...except VectorInsertionError:`. However, `VectorInsertionError` is a custom Python exception defined in our `datalayer_exceptions.py`. The native C++ wrapper (`diskannpy`) has absolutely no awareness of this custom exception and will never raise it. Any actual insertion failures will throw standard errors (like `RuntimeError` or `ValueError`), bypassing our exception handler entirely.

### Bug 7.4: Markdown Extractor Double-Newline Corruption (`text_extractor.py`)
- **Criticality:** Medium
- **Priority:** P2
- **Explanation:** The private `__extract_text_from_md()` method reads Markdown files using `file.readlines()`, which retains the trailing `\n` character on every line. It then merges them using `"\n".join(text)`. This effectively appends an extra, unintended newline between every single line of text, deeply corrupting the spacing, formatting, and semantic parsing of the Markdown document.

### Bug 7.5: Index Load State Desynchronization (`vectorDB_diskann.py` / `vectorDbManager.py`)
- **Criticality:** Critical
- **Priority:** P0
- **Explanation:** `diskannpy.DynamicMemoryIndex.from_file()` returns a *brand new* initialized index instance instead of updating the current object. However, `VectorDb_diskann.load()` simply returns this instance without overriding `self.dynamic_dann`. Likewise, `VectorDbManager.load()` returns the new index to the caller but fails to update its own `self.vector_db`. Thus, after a `.load()`, any subsequent calls to `.insert()` or `.search_vector()` will silently route into the original, empty index instance, completely ignoring the loaded disk data.
