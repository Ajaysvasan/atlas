# Conversation Vector Metadata Management

## Overview & Purpose
The `conversation_data_management` sub-module now consists of two modernized files for robust vector metadata tracking and storage integration:
1. `conversationVectorMetaManager.py` defines `ConversationVectorMetaDataManager`. It creates and manages isolated SQLite databases for each project to store chunk and vector ID metadata (`<project_id>_full_conversation.db` and `<project_id>_summary_metadata.db`).
2. `conversatoinVectorManager.py` defines `ConversationVectorManager`. It generates deterministic `uint32` vector IDs using MD5 hashes of the project and chunk identifiers, and acts as a direct integration proxy for the PostgreSQL `VectorRepository`.

---

## Current Architecture & Public APIs

### `class ConversationVectorMetaDataManager` (`conversationVectorMetaManager.py`)
Responsible for managing SQLite chunk metadata. By design, each project owns its own chunk tables across two separate directories: `full_conversation` and `summary`. It relies on strictly scoped context managers (`with sqlite3.connect()`) for every transaction to ensure thread safety.

#### Constructor
```python
def __init__(self, full_conversation_dir: str | Path, summary_dir: str | Path, project_id: str) -> None:
```
Creates the required directories if they don't exist, and initializes `chunks` tables within `<project_id>_full_conversation.db` and `<project_id>_summary_metadata.db`.

#### Public Methods
- `insert_full_conversation_chunk(vector_id: int, chunk_id: str, chunk: str)`: Inserts a single metadata row into the project's full conversation database.
- `insert_summary_chunk(vector_id: int, chunk_id: str, chunk: str)`: Inserts a single metadata row into the project's summary database.
- `batch_insert_full_conversation_chunks(records: List[Tuple[int, str, str]])`: Uses `executemany` for efficient batch insertion into the full conversation database.
- `batch_insert_summary_chunks(records: List[Tuple[int, str, str]])`: Uses `executemany` for efficient batch insertion into the summary database.
- `get_full_conversation_chunk(vector_id: int)`: Queries the chunk metadata by vector ID from the full conversation database.
- `get_summary_chunk(vector_id: int)`: Queries the chunk metadata by vector ID from the summary database.

---

### `class ConversationVectorManager` (`conversatoinVectorManager.py`)
Provides deterministic vector ID generation and serves as the integration bridge to the external `VectorRepository`.

#### Constructor
```python
def __init__(self, project_name: str, project_id: str)
```
Initializes the manager and instantiates a `VectorRepository` connection.

#### Public Methods
- `generate_vector_id(chunk_id: str) -> int`: Hashes `"{project_name}_{project_id}_{chunk_id}"` via MD5 and takes the first 4 bytes to generate an unsigned 32-bit integer, satisfying the PostgreSQL vector extension constraints.
- `insert(chunk_id: str, vector: np.ndarray) -> int`: Generates the vector ID and inserts a single vector via `VectorRepository`.
- `batch_insert(chunk_ids: List[str], vectors: np.ndarray) -> List[int]`: Generates vector IDs and inserts a batch of vectors.
- `get_vector(vector_id: int) -> np.ndarray`: Retrieves a vector from PostgreSQL.
- `get_vectors(vector_ids: List[int]) -> np.ndarray`: Retrieves a batch of vectors.

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
- `add_summary_vectors(project_id: str, vectors: np.ndarray) -> Tuple[int, int]`: Appended 2D `np.float32` vector arrays directly to `<cummulative_vector_path>/<project_id>.bin` using `open(file_path, "ab")`.
- `get_cummulative_summary_vector(start_idx: int, end_idx: int, project_id: str) -> np.ndarray`: Memory-mapped and sliced vectors.
- `get_summary_vector(start_idx: int, end_idx: int, project_id: str) -> np.ndarray`: Memory-mapped and sliced vectors from inverted directories due to an architectural bug.
