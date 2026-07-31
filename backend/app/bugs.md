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

