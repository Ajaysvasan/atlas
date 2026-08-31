# Conversation Vector Metadata Management

## Overview & Purpose
The `conversation_data_management` sub-module holds the two classes that stand between the snapshot layer and its two storage backends:

1. `conversationVectorMetaManager.py` defines `ConversationVectorMetaDataRepository` — the SQLite metadata for one project's snapshots.
2. `conversationVectorManager.py` defines `ConversationVectorManager` — a thin proxy onto the PostgreSQL/pgvector `VectorRepository`.

---

## `class ConversationVectorMetaDataRepository`

One project, one database file: `<conversation_dir>/<project_id>_conversation.db`. `FullConversationRepository` writes the conversation turns into that **same** file, which is what lets `get_highest_summarised_sequence()` join snapshot metadata against `full_conversation` without a second connection.

### Constructor
```python
def __init__(self, conversation_path: str | Path, project_id: str) -> None:
```
Creates the directory if it does not exist, opens the connection, enables `PRAGMA foreign_keys`, and creates the four tables if they are missing. Constructing it twice against the same path is a no-op.

### Thread safety
**Safe to share across threads.** The connection is opened with `check_same_thread=False` and every statement runs under a `threading.RLock` held for the whole operation — execute, fetch, and commit or rollback together.

Both halves are needed, and the lock is the important one:

- Without `check_same_thread=False`, any use from a second thread raises `sqlite3.ProgrammingError: SQLite objects created in a thread can only be used in that same thread`. This is the failure a caller sees first, and it is the harmless one.
- Without the lock, `commit()` and `rollback()` are the hazard. They act on the **connection**, not on a cursor, so a second thread's `commit()` lands in the middle of the first thread's transaction and publishes its half-written rows — and its `rollback()` discards them. Under a 12-thread write load this produces orphaned `summary_chunks` rows from snapshots that reported failure, plus spurious `DatabaseError: not an error` and `no more rows available` from cursors being stepped on mid-fetch.
- `close()` takes the same lock. Closing a connection while another thread is mid-statement does not raise — it segfaults the interpreter.

The lock covers this repository's own connection only. `FullConversationRepository` opens short-lived connections to the same file; those are serialised by SQLite's file locking, and by WAL — see below.

Two callers rely on the shared instance: `SnapShot` holds one for its lifetime (and accepts one through its `meta_repo` argument so it does not open a redundant second connection), and `ConversationSummary` builds its own.

### Connection setup
Every open of the conversation database goes through `sqlite_setup.connect()`, and its initialisers additionally call `sqlite_setup.enable_wal()`. The split is not cosmetic — the two kinds of pragma behave differently:

| Pragma | Value | Scope | Set by |
|---|---|---|---|
| `journal_mode` | `WAL` | Written into the **file**; survives every later open, across processes | `enable_wal()`, at initialisation |
| `synchronous` | `NORMAL` | Per **connection**; resets to `FULL` on every open | `connect()`, every time |
| `foreign_keys` | `ON` | Per **connection**; resets to off on every open | `connect()`, every time |

`FullConversationRepository` opens a connection per call, so the per-connection pragmas are not one-time setup that can be checked at construction. A new method with a raw `sqlite3.connect()` would silently get `FULL` and no foreign keys — which has happened before, in `__add_chunks`, where it let orphan rows into `full_conversation`. Routing every open through `connect()` is what stops it recurring, and a test asserts neither module contains a raw `sqlite3.connect()`.

`enable_wal()` cannot convert the file while another connection holds a write transaction. Losing that race returns the current mode instead of raising: the database is still perfectly correct in a rollback journal, just slower, and the next open tries again. `journal_mode` on either repository reports the mode actually in force.

### Why WAL and NORMAL
Two repositories share this file. Under the default rollback journal a reader holding a transaction open blocks **every** writer with `database is locked` — and a reader with a transaction open is precisely what `SnapShot.search()` is while `append_turns()` wants to write. `synchronous=FULL` then fsyncs the log on every single commit on top of that.

Measured through the real classes on an NVMe/btrfs filesystem — one writer appending turns against six readers, plus a separate metadata commit loop on the long-lived connection:

| journal | synchronous | appends/s | reads/s | metadata commits/s |
|---|---|---|---|---|
| `delete` | `FULL` | ~330 | ~5 | ~390 |
| `wal` | `FULL` | ~850 | ~6500 | ~1000 |
| `wal` | `NORMAL` | ~2200 | ~5300 | ~20000–100000 |

WAL is what unblocks reads — three orders of magnitude, because under a rollback journal they spend nearly all their time waiting behind the writer. `NORMAL` is what unblocks commits, most visibly on the metadata repository's long-lived connection. Reads dip slightly in the last row: the writer is no longer the bottleneck, so it competes for more of the time. Zero `database is locked` errors in every configuration.

**Measure these on real storage, not `/tmp`.** It is `tmpfs` on most Linux systems, where `fsync` is free and both settings look like they do nothing.

### The durability trade
`synchronous = NORMAL` is a deliberate weakening, chosen with the risk understood:

- **Process crash — still safe.** The write-ahead log is on disk and intact; the next open recovers every committed transaction.
- **OS crash or power loss — the last few committed transactions can be rolled back.** They were written but not fsynced.
- **Corruption — not a risk either way.** WAL keeps the database consistent regardless; the exposure is bounded to losing recent commits, never to a broken file.

For conversation turns and snapshot metadata that is an acceptable trade: the cost of losing the last few turns after a power cut is a re-ask, and the alternative was paying an fsync on every commit forever. Set `SYNCHRONOUS = "FULL"` in `sqlite_setup.py` to reverse it — one constant, and it applies everywhere because every open goes through `connect()`.

### Operational notes
- **The database is no longer one file.** SQLite keeps `<name>.db-wal` and `<name>.db-shm` alongside it. Copying or backing up the `.db` on its own can lose recently committed transactions.
- **WAL needs shared memory**, so it does not work over most network filesystems. On one, `enable_wal()` falls back to the existing mode rather than failing.
- **Existing databases convert in place** on the next open, with their rows intact. No migration step.

### Schema
| Table | Key | Notes |
|-------|-----|-------|
| `summary_chunks` | `chunk_id TEXT PK` | Shared with `FullConversationRepository` |
| `summary_vector_meta_data` | `summary_vector_id INTEGER PK` | FK → `summary_chunks.chunk_id` |
| `cumulative_vector_meta_data` | `cumulative_vector_id INTEGER PK` | One row per snapshot |
| `summary_snapshot_map` | `UNIQUE (cumulative_vector_id, summary_vector_id)` | FK onto both tables above |

### Public methods

**Writing a snapshot**
- `insert_snapshot(chunks, cumulative_row, summary_vector_rows, map_rows)`: all four inserts in **one** transaction, ordered so FK parents land first. The whole snapshot commits or none of it does. This is the path `SnapShot.add()` uses; the four calls below predate it and each commit on their own.
- `batch_insert_summary_chunks(records)`: `[(chunk_id, chunk, created_at, chunker_type)]`. `INSERT OR IGNORE` — the turns a snapshot covers are normally already in the table, written by `append_turns()`.
- `batch_insert_summary_vector_meta_data(records)`: `[(summary_vector_id, chunk_id, project_id)]`. `INSERT OR IGNORE` — snapshot windows overlap by design, so a chunk legitimately reappears with the same id.
- `insert_cumulative_vector_meta_data(cumulative_vector_id, cumulative_summary, created_at, project_id, len_of_the_summary)` and `batch_insert_cumulative_vector_meta_data(records)`: plain inserts, so a duplicate id raises `sqlite3.IntegrityError`.
- `insert_map_table(cumulative_vector_id, summary_vector_id)` / `batch_insert_map_table(records)`: the batch form is `INSERT OR IGNORE` against the UNIQUE constraint; the single form is not.

**Reading**
- `get_cumulative_vector_meta_data_ids()`: every snapshot id, `ORDER BY datetime(created_at), created_at`. The raw-TEXT tiebreaker recovers sub-second precision that `datetime()` truncates away — `SnapShot`'s cursors index into this list, so ties would make its ordering arbitrary.
- `get_latest_summary() -> str | None`: the most recent `cumulative_summary`, ordered the same way. Returns the string, not a row.
- `get_cumulative_vector_meta_data(id)` / `batch_get_cumulative_vector_meta_data(ids)`
- `get_summary_vector_meta_data(id)` / `batch_get_summary_vector_meta_data(ids)`
- `get_summary_vector_ids_from_map(cumulative_vector_id) -> List[int]`: the chunks one snapshot covers.
- `get_highest_summarised_sequence() -> int | None`: the highest `full_conversation.sequence_number` any snapshot has covered. Returns `None` when the `full_conversation` table does not exist yet, so the repository stays usable when constructed on its own.

**Lifecycle**
- `close()`: idempotent, and called from `__del__`. Both use `getattr` rather than attribute access, so a failure inside the constructor surfaces as itself instead of as `AttributeError` inside `Exception ignored in`.

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
