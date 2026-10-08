# Conversation Vector Metadata Management

## Overview & Purpose
The `conversation_data_management` sub-module holds the two classes that stand between the snapshot layer and its two storage backends:

1. `conversationVectorMetaManager.py` defines `ConversationVectorMetaDataRepository` — the SQLite metadata for one conversation's snapshots.
2. `conversationVectorManager.py` defines `ConversationVectorManager` — a thin proxy onto the PostgreSQL/pgvector `VectorRepository`.

---

## `class ConversationVectorMetaDataRepository`

Snapshot metadata for one conversation, in the memory database alongside the
turns it covers and the project it belongs to.

### Constructor
```python
ConversationVectorMetaDataRepository(project_id: str, conversation_id: str,
                                     database: MemoryDatabase | str | Path | None = None)
```
`conversation_id` must be a non-blank string (`InvalidIdentifier`). `database` is
a `MemoryDatabase` or a path to one; `None` uses the shared memory database.
Creates its three tables if they are missing, after the project and turn tables
they reference. Constructing it twice is a no-op.

### Thread safety
**Safe to share across threads.** It holds no connection: every statement —
execute, fetch, and commit or rollback together — runs on the memory database's
connection under that database's lock. `SnapShot` holds one for its lifetime
(and accepts one through `meta_repo` so it does not build a second), and
`ConversationSummary` builds its own on the same database. Connection setup,
WAL and the durability trade are documented in `memory_database.md`.

### Schema
| Table | Key | Notes |
|-------|-----|-------|
| `summary_chunks` | `chunk_id TEXT PK` | Owned and created by `FullConversationRepository`; this repository inserts snapshot chunks into it |
| `summary_vector_meta_data` | `summary_vector_id INTEGER PK` | FK → `summary_chunks.chunk_id`, FK → `project_table`; indexed on `chunk_id` |
| `cumulative_vector_meta_data` | `cumulative_vector_id INTEGER PK`, `UNIQUE (project_id, seq)` | FK → `project_table`; carries `conversation_id` and the monotonic `seq` |
| `summary_snapshot_map` | `UNIQUE (cumulative_vector_id, summary_vector_id)` | FK onto both tables above |

### Public methods

**Writing a snapshot**
- `insert_snapshot(chunks, cumulative_row, summary_vector_rows, map_rows)`: all four inserts in **one** transaction, ordered so FK parents land first. The whole snapshot commits or none of it does. Raises `ProjectNotFound` if `cumulative_row`'s project was never registered, with nothing written. This is the path `SnapShot.add()` uses; the calls below predate it and each commit on their own.
- `batch_insert_summary_chunks(records)`: `[(chunk_id, chunk, created_at, chunker_type)]`. `INSERT OR IGNORE` — the turns a snapshot covers are normally already in the table, written by `append_turns()`.
- `batch_insert_summary_vector_meta_data(records)`: `[(summary_vector_id, chunk_id, project_id)]`. `INSERT OR IGNORE` — snapshot windows overlap by design, so a chunk legitimately reappears with the same id. `OR IGNORE` does not cover foreign keys: an unknown chunk or project still raises.
- `insert_cumulative_vector_meta_data(cumulative_vector_id, cumulative_summary, created_at, project_id, len_of_the_summary)` and `batch_insert_cumulative_vector_meta_data(records)`: each row takes the next `seq` **of its own project**, allocated inside the write. A duplicate id raises `sqlite3.IntegrityError`; an unregistered project raises `ProjectNotFound`.
- `insert_map_table(cumulative_vector_id, summary_vector_id)` / `batch_insert_map_table(records)`: the batch form is `INSERT OR IGNORE` against the UNIQUE constraint; the single form is not.

**Reading**
- `get_cumulative_vector_meta_data_ids()`: this conversation's snapshot ids, `ORDER BY seq`. `SnapShot`'s cursors index into this list, so its order must be total; `seq` cannot tie the way whole-second timestamps did.
- `get_latest_summary() -> str | None`: this conversation's most recent `cumulative_summary`, by `seq`. Returns the string, not a row.
- `get_project_snapshots_since(seq) -> [(seq, cumulative_vector_id, cumulative_summary)]`: the **project's** snapshot summaries after `seq`, across every conversation — what `ProjectSnapshot` folds in.
- `get_highest_snapshot_seq() -> int`: the project's highest `seq`, `0` when it has none.
- `get_highest_summarised_sequence() -> int | None`: the highest `sequence_number` **in this conversation** that a snapshot covers, `None` when none does. It was project-wide, so a new conversation inherited a sibling's progress and its snapshot trigger stayed silent.
- `get_cumulative_vector_meta_data(id)` / `batch_get_cumulative_vector_meta_data(ids)`
- `get_summary_vector_meta_data(id)` / `batch_get_summary_vector_meta_data(ids)`
- `get_summary_vector_ids_from_map(cumulative_vector_id) -> List[int]`: the chunks one snapshot covers.

**Lifecycle**
- `close()`: releases nothing — the database is shared and outlives its owners.

Empty-list arguments are no-ops on every batch method; the `batch_get_*` methods return `[]` rather than issuing a query with no placeholders.

---

## `class ConversationVectorManager`
A pass-through to the PostgreSQL `VectorRepository` for one project. It holds no state beyond the project identifiers and does **not** derive vector ids — callers pass ids that `EmbeddingManager` has already derived and masked into the signed 64-bit range.

### Constructor
```python
def __init__(self, project_name: str, project_id: str)
```
Builds a `VectorRepository(project_id)`, which opens the PostgreSQL connection. `SnapShot` therefore builds this lazily, on first use, since cursor navigation never needs it.

### Public methods
- `insert(vector_id, vector) -> np.uint32`: single insert; returns the id it was given. Raises `DuplicateVectorException` if the project already stores that id.
- `batch_insert(vector_ids, vectors) -> List[np.uint32]`: tolerates duplicates (`on conflict do nothing`).
- `batch_delete(vector_ids) -> None`: used as the compensating delete when a snapshot's metadata transaction fails after its vectors were written.
- `get_vector(vector_id) -> np.ndarray`
- `get_vectors(vector_ids) -> np.ndarray`

---

## Historical Design Decisions (Legacy Documentation)
> **Note:** The following documentation describes the *legacy* architecture prior to the current SQLite transaction refactor and Postgres integration. It is preserved here for historical context and design rationale tracking.

### Legacy `ConversationVectorMetaDataManager`
Persistent SQLite metadata and vector ID mapping engine for conversational topic turns and snapshots. Maintained a persistent connection across instance lifetimes (which caused thread incompatibility bugs).

#### Legacy Constructor
`__init__(self, db_path: Optional[str] = None) -> None`
Initialized the SQLite database connection (`sqlite3.connect`) and enforced foreign keys for `conversation_snapshots`, `snapshot_vector_ids`, and `cumulative_summary_offsets`.

#### Legacy Relational Schema
- **`conversation_snapshots`**: `(row_id INTEGER PK AUTOINCREMENT, snapshot_id TEXT UNIQUE, project_id TEXT, topic_id TEXT, conversation_id TEXT, timestamp TEXT, size_of_the_summary INTEGER, len_of_the_summary INTEGER, cumulative_summary_vector_id INTEGER)`
- **`snapshot_vector_ids`**: `(snapshot_id TEXT FK, vector_id INTEGER, vector_position INTEGER, PRIMARY KEY (snapshot_id, vector_position))`
- **`cumulative_summary_offsets`**: `(snapshot_id TEXT PRIMARY KEY FK, file_offset INTEGER NOT NULL)`

#### Legacy Methods
- `insert_snapshot(self, snapshot_node: SnapShotNode, topic_id: str, project_id: str) -> str`: Inserted a `SnapShotNode`.
- `load_snap_shot_objects(self, conversation_id: str) -> List[SnapShotNode]`: Reconstructed snapshots using chronological `LEFT JOIN` queries.
- `get_snapshot_metadata(self, conversation_id: str) -> List[tuple]`: Retrieved raw tuples directly from `conversation_snapshots`.
- `insert_vector_ids(self, snapshot_id: str, vector_ids: List[np.uint32]) -> None`: Mapped vector IDs preserving position.
- `insert_cumulative_summary_offset(self, snapshot_id: str, file_offset: int) -> None`: Updated byte offsets.
- `get_latest_cumulative_summary_vector_id(self, conversation_id: str) -> Optional[int]`: Retrieved the most recent cumulative summary vector ID.
- `get_cumulative_vector_id(self, snapshot_id: str) -> Optional[int]`: Direct query for specific snapshot.
- `get_file_offset(self, snapshot_id: str) -> Optional[int]`: Direct query for specific file offset.
- `close(self) -> None`: Handled connection closures.

### Legacy `ConversationVectorManager`
Legacy memory-mapped binary vector storage manager for appending and slicing vector arrays from raw `.bin` disk files.

#### Legacy Methods
- `add_summary_vectors(project_id: str, vectors: np.ndarray) -> Tuple[int, int]`: Appended 2D `np.float32` vector arrays directly to `<cumulative_vector_path>/<project_id>.bin` using `open(file_path, "ab")`.
- `get_cumulative_summary_vector(start_idx: int, end_idx: int, project_id: str) -> np.ndarray`: Memory-mapped and sliced vectors.
- `get_summary_vector(start_idx: int, end_idx: int, project_id: str) -> np.ndarray`: Memory-mapped and sliced vectors from inverted directories due to an architectural bug.
