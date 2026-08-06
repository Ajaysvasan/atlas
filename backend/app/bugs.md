# Comprehensive Project Bug Inventory (`backend/app`)

This document catalogs all logical, architectural, concurrency, and data integrity bugs identified across the project. The bugs are organized in a **section-wise manner** grouped by architectural layer, complete with detailed explanations, **Criticality** ratings (`Critical`, `High`, `Medium`, `Low`), and recommended resolution **Priority** levels (`P0`, `P1`, `P2`, `P3`).

---

## Section 1: Memory & Conversation Pool Layer (`memory/`)

### Bug 1.1: Cursor Reset Clobbering & Duplicate Comparison in `SnapShot` (`snapshot.py`)

- **Criticality:** Medium
- **Priority:** P2
- **Explanation:** In `SnapShot.add()`, whenever a new snapshot node is appended, `self.__left_cursor` is reset to `0` and `self.__right_cursor` to `len - 1` by default (`reset_right_pointer=True, reset_left_pointer=True`), clobbering any active iteration cursors. Additionally, in `__find_best_snapshot`, the while loop condition is `while self.__left_cursor <= self.__right_cursor:`. When `self.__left_cursor == self.__right_cursor` (the exact midpoint of an odd-length snapshot list), both `left_snap` and `right_snap` point to the same node, computing and comparing `cosine_similarity` twice for the exact same object before terminating.

### Bug 1.2: Empty & Placeholder Modules In Memory Pool

- **Criticality:** Medium
- **Priority:** P2
- **Explanation:** Several core memory modules contain unwritten placeholder files or typos: `conversation_summary.py` (typo `conversation`) contains only `class ConversationSummary: pass`, `full_conversation_bucket.py` contains only `class FullConversationBucket: pass`, and `conversation_pool_manager.py`, `memory_manager.py`, `topic_manager.py`, `project_manager.py` are completely empty (0 bytes). This indicates that the core components of the memory architecture are fundamentally missing.

### Bug 1.3: Broken Module Import Path in `snapshot.py`

- **Criticality:** High
- **Priority:** P1
- **Explanation:** In `snapshot.py`, the import statements for `ConversationVectorManager` and `ConversationVectorMetaDataRepository` incorrectly assume `conversation_data_management` is in the `sys.path` root. It uses `from conversation_data_management.conversationVectorManager import ...` instead of the correct absolute path `from memory.topic_pool.project_pool.conversation_pool.conversation_data_management...`. This raises a `ModuleNotFoundError` when the application runs from the standard `app/` backend root directory.

### Bug 1.4: Unhandled Exception in Batch Size Validation (`snapshot.py`)

- **Criticality:** High
- **Priority:** P1
- **Explanation:** Inside `__add_snap_shot` in `snapshot.py`, there is a validation check `if len(chunk_ids) != len(summary_vector_ids): raise`. Raising a bare `raise` statement outside of an `except` block immediately crashes the program with a `RuntimeError: No active exception to reraise`, completely halting execution without providing a descriptive error.

### Bug 1.5: Missing `dim` Argument in PyTorch `cosine_similarity` (`snapshot.py`)

- **Criticality:** Critical
- **Priority:** P0
- **Explanation:** In `__find_best_snapshot`, `cosine_similarity(left_snap_vector_cumulative, tensor(query))` is called without specifying the `dim` parameter. Since the vectors are 1-Dimensional (from the numpy arrays), `cosine_similarity` defaults to computing along `dim=1`, which does not exist. This triggers an `IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)`. The calls must explicitly include `dim=0`.

### Bug 1.6: Global Vector Table Collision in Multi-Tenancy Architecture (`ConversationVectorManager` & `VectorRepository`)

- **Criticality:** Critical
- **Priority:** P0
- **Explanation:** `ConversationVectorManager` accepts `project_name` and `project_id` in its constructor, but does absolutely nothing with them, merely instantiating `self.repository = VectorRepository()`. In `VectorRepository`, the SQL schema defines a single, global `vectors` table (`create table if not exists vectors`). This means vector IDs are not scoped by project. If two different projects generate identical vector IDs, the `batch_insert` conflict clause (`on conflict (vector_id) do nothing`) will silently drop the new project's vector, causing critical data loss and severe cross-tenant data pollution.
