# Vector Database Management Layer (`vector_db_manager/`)

## Overview & Purpose
The `vector_db_manager` submodule holds the project's two vector stores:

- **DiskANN** (`vectorDbManager.py` → `vectorDB_diskann.py`) — the data layer's approximate nearest neighbour index over ingested document chunks, built on `diskannpy.DynamicMemoryIndex`. It is held **in memory** and rebuilt from the vectors stored in `vector_meta_data` (`restore`); diskannpy 0.7.0 cannot load a dynamic index it saved (bug 5.20).
- **PostgreSQL / pgvector** (`repository/vectorRepository.py`) — the memory layer's exact store for conversation snapshot vectors, keyed by `(project_id, vector_id)`.

`repository/vectorMetaDataRepository.py` holds the SQLite table that allocates DiskANN labels, maps each back to its chunk, and stores each vector — the source of truth the in-memory index is rebuilt from.

The two stores are independent and are not kept in sync with each other; which one a caller wants depends on whether it is searching documents or conversation history.

---

## `class VectorDbManager` (`vectorDbManager.py`)
Thread-safe wrapper over the DiskANN driver. Every mutating call is taken under a single `threading.Lock` held on the instance (`self.lock`), because `diskannpy`'s dynamic index is not safe for concurrent writes.

#### Constructor: `__init__(self, distance_metrics, vector_dtype, dimensions, max_vectors, complexity, graph_degree, num_threads, k_neighbors) -> None`

**All eight parameters are required** — none has a default. `IngestionPipeline` supplies them from `Config` explicitly.

| Parameter | Type | Supplied by `IngestionPipeline` as | Description |
| :--- | :--- | :--- | :--- |
| `distance_metrics` | `str` | `Config.DISTANCE_METRIC` (`"l2"`) | Passed straight through to `diskannpy` as a string. |
| `vector_dtype` | `Type[np.float32 \| np.int8 \| np.uint8]` | `Config.VECTOR_DTYPE` (`np.float32`) | Element type of stored vectors. |
| `dimensions` | `int` | `Config.EMBEDDING_DIMENSIONS` (`128`) | Vector dimensionality. |
| `max_vectors` | `int` | `Config.MAX_VECTORS` (`1_000_000`) | Capacity of the index graph. |
| `complexity` | `int` | `Config.COMPLEXITY` (`100`) | Search beam width (`L`), used at build and query time. |
| `graph_degree` | `int` | `Config.GRAPH_DEGREE` (`120`) | Maximum out-degree (`R`) of Vamana graph nodes. |
| `num_threads` | `int` | `Config.NUM_THREADS` (`4`) | Worker threads for search. |
| `k_neighbors` | `int` | `Config.K_NEIGHBORS` (`9`) | Neighbours returned by `search_vector`. Held on the manager only; the driver takes it per call. |

#### Methods

| Method | Signature | Behaviour |
| :--- | :--- | :--- |
| `insert` | `(embedded_chunk_obj: EmbeddedChunk, vector_id=None) -> None` | Takes the lock and inserts `embedded_chunk_obj.vector` under `vector_id` (the allocated label) as `uint32`, falling back to `embedded_chunk_obj.vector_id`. |
| `batch_insert` | `(embedded_chunk_objs: List[EmbeddedChunk], vector_ids=None)` | Collects vectors into a single `np.float32` 2-D array and the ids into a `uint32` array — diskannpy requires both as arrays (bug 5.15). |
| `count` | `() -> int` | Vectors in the index now, read from diskannpy's native `num_points()`: tracks inserts, deletes and restores, and excludes the frozen start point. |
| `search_vector` | `(query, k_neighbors=None)` | `k_neighbors` defaults to the instance's (`9`) and is **capped at `count()`**: asked for more than it holds, diskannpy fills the surplus from uninitialised memory — labels that can be real ones, at distance `0.0`, sorted ahead of every true result (bug 5.18). An empty index returns two empty arrays without searching. |
| `batch_search_vectors` | `(queries, k_neighbors=None)` | Batch form of the above; an empty index returns arrays of shape `(len(queries), 0)`. |
| `restore` | `(store, after: int = 0) -> int` | Inserts every vector `store.vectors(after=after)` yields, a page at a time under the lock, and returns the highest label now indexed (`after` if none). Labels only grow, so passing that back indexes just what was ingested since. Logs a warning naming how many labels have no usable stored vector. |
| `delete_vector` | `(vector_id) -> None` | Under the lock. |
| `delete_vectors` | `(vector_ids) -> None` | Under the lock. |
| `save` | `(save_path=Config.INDEX_PATH)` | Under the lock. **Not used by the project** — see below. |
| `load` | `(load_path=Config.INDEX_PATH)` | Under the lock. Returns the reloaded `dynamic_dann`, or **`None`** if the directory is absent — `IndexDirectoryDoesNotExists` is caught here and logged as a warning rather than propagated. **Not used by the project** — see below. |

> **`save` and `load` cannot persist an index.** diskannpy 0.7.0's `DynamicMemoryIndex.save` writes every label as `0`, and `from_file` returns garbage in a new process even when the labels on disk are correct (bug 5.20). Nothing in the project calls either; the index is rebuilt with `restore` instead.

> **Note.** `insert` keys the vector on `EmbeddedChunk.vector_id`, an `int`, not on `meta_data.chunk_id`. DiskANN tags are unsigned integers; passing the SHA-256 `chunk_id` string was a historical bug.

---

## `class VectorDb_diskann` (`vectorDB_diskann.py`)
Thin driver over `diskannpy.DynamicMemoryIndex`, exposed as `self.dynamic_dann`.

#### Constructor: `__init__(self, distance_metrics, vector_dtype, dimensions, max_vectors, complexity, graph_degree, num_threads) -> None`
Stores the parameters and constructs `dann.DynamicMemoryIndex(...)` directly, with `saturate_graph=SATURATE_GRAPH` (`True`). It performs **no** metric-string conversion and **no** parameter validation — `distance_metrics` is handed to `diskannpy` as the string it was given, and an invalid value surfaces as a `diskannpy` error.

> **`SATURATE_GRAPH` is not a tuning knob.** diskannpy defaults it to `False`, and the dynamic index then builds a graph too sparse to search: recall@10 of 0.09 on 5,000 vectors against 0.96 with it on, and an inserted vector cannot find itself (bug 5.19).

#### Methods

| Method | Signature | Behaviour |
| :--- | :--- | :--- |
| `insert` | `(vector, vector_id)` | `ValueError` and `RuntimeError` are re-raised as `VectorInsertionError(vector_id, cause)`, chained with `from`. |
| `batch_insert` | `(vectors, vector_ids)` | Same wrapping, with the id list as `vector_id`. |
| `search_vector` | `(query, k_neighbors, complexity)` | `dynamic_dann.search(...)`, returning `(tags, distances)`. |
| `batch_search_vector` | `(queries, k_neighbors, complexity)` | `dynamic_dann.batch_search(..., self.num_threads)`. |
| `count` | `() -> int` | `dynamic_dann._index.num_points()`. diskannpy exposes the count only on its native object, hence the private attribute; `test_the_count_follows_inserts_deletes_and_restores` pins it. |
| `delete_vector` | `(id)` | `mark_deleted` then `consolidate_delete`. |
| `delete_vectors` | `(ids)` | Marks each, then a single `consolidate_delete`. |
| `save` | `(save_path=Config.INDEX_PATH)` | Creates the directory if absent, then `dynamic_dann.save(save_path)`. The path is used as given — no filename is appended. |
| `load` | `(load_path=Config.INDEX_PATH)` | Checks the **directory** exists (not the individual index files), calls `DynamicMemoryIndex.from_file(..., saturate_graph=SATURATE_GRAPH)`, and **reassigns `self.dynamic_dann`** to the result so later inserts reach the loaded index. Raises `IndexDirectoryDoesNotExists` when the directory is missing. The loaded index cannot name its neighbours in a new process (bug 5.20). |

> Both insert paths wrap driver failures identically. `batch_insert` used to wrap nothing, so `except VectorInsertionError` around it caught no failure at all.

---

## `class VectorRepository` (`repository/vectorRepository.py`)
The memory layer's pgvector store. One row per `(project_id, vector_id)` in a `vectors` table; `psycopg` 3 connection held for the object's lifetime.

Connection settings come from `.env` via `python-dotenv`: `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`. Any missing key raises `MissingDatabaseConfiguration` at construction, naming the absent keys. The `DB_` prefix is load-bearing: `load_dotenv()` will not override a variable the environment already has, and `USER`, `HOST` and `PORT` are all set by something (bug 6.2).

`register_vector_types(self.conn)` runs in the constructor, after `CREATE EXTENSION` and before any statement touches a vector column. Without it psycopg cannot adapt a numpy array at all, and a `vector` column reads back as text (bug 5.1).

> **On `DB_USER`.** The key is deliberately not `USER`. Every login shell exports `USER`, and `load_dotenv()` does not override a variable already in the environment, so the `.env` value was ignored and the connection was made as whoever ran the process.

| Method | Signature | Raises |
| :--- | :--- | :--- |
| `insert` | `(vector_id, vector) -> None` | `InvalidVectorDimension`; `DuplicateVectorException` when the id is already stored; `VectorInsertionError` for any other failure |
| `batch_insert` | `(vector_ids, vectors) -> None` | `InvalidBatchSize`, `InvalidVectorDimension`, `VectorInsertionError`. Uses `on conflict … do nothing`. |
| `update` | `(vector_id, vector) -> None` | `VectorNotFoundEror` when no row matches, `InvalidVectorDimension`, `VectorInsertionError`. A single `UPDATE` rather than delete-then-insert, which is two commits and loses the vector if the second fails. |
| `delete` | `(vector_id) -> None` | `VectorInsertionError` |
| `batch_delete` | `(vector_ids) -> None` | As above; a no-op on an empty list. Used to undo vectors whose metadata write failed. |
| `search` | `(vector_id) -> NDArray[float32]` | `VectorNotFoundEror` |
| `batch_search` | `(vector_ids) -> NDArray[float32]` | `VectorNotFoundEror` |
| `close` | `()` | — |

---

## `class VectorMetaDataRepository` (`repository/vectorMetaDataRepository.py`)

#### Constructor: `__init__(self, db_path: str | None = None) -> None`
Opens `db_path` (default `Config.DB_PATH`, the chunk store) through `storage.sqlite_setup.connect` with `check_same_thread=False`, enables WAL, and creates the table. Every statement runs under an `RLock`.

#### Schema
```sql
vector_meta_data(
    vectorId           INTEGER PRIMARY KEY AUTOINCREMENT,  -- the DiskANN label
    chunkId            TEXT NOT NULL,
    embeddingModelUsed TEXT NOT NULL,
    dimensions         INTEGER NOT NULL,
    vector             BLOB                                -- float32 bytes, dimensions × 4
)
-- idx_vector_meta_chunk on (chunkId)
```
No foreign key: a chunk id lives in `Chunks` or `RecursiveChunks`, and SQLite cannot reference whichever of two tables holds it (bug 5.2). A table created before the `vector` column existed gains it on open, with its rows kept; those rows have no vector and are reported by `missing_vectors()`.

#### Methods

| Method | Signature | Notes |
| :--- | :--- | :--- |
| `allocate` | `(chunkId, vector=None, embeddingModelUsed=Config.EMBEDDING_MODEL, dimensions=Config.EMBEDDING_DIMENSIONS) -> int` | Takes the next label for the chunk and stores `vector` (cast to `float32`) in the same row. Raises `InvalidVectorDimension` when its shape is not `(dimensions,)`. |
| `allocate_many` | `(chunkIds, vectors=None, embeddingModelUsed=..., dimensions=...) -> List[int]` | Labels for a batch in the order given, one transaction. `InvalidBatchSize` when `vectors` and `chunkIds` differ in length; `InvalidVectorDimension` on any wrong-shaped vector — both before anything is written. |
| `vectors` | `(batch_size=RESTORE_BATCH, after=0, embeddingModelUsed=..., dimensions=...) -> Iterator[Tuple[ndarray, ndarray]]` | Stored `(labels: uint32[n], vectors: float32[n, dimensions])` with labels above `after`, in label order, `batch_size` (`50_000`) rows a page. Keyset-paginated. Only rows from this model at this width, with a vector of the right length: another model's vectors are in a different space. Each page is read under the lock and yielded outside it. |
| `missing_vectors` | `(embeddingModelUsed=..., dimensions=...) -> int` | Rows `vectors()` skips: no vector, another model, or the wrong width. |
| `chunk_ids_for` | `(vectorIds) -> Dict[int, str]` | Every label's chunk in one query; unknown labels are absent rather than raising. |
| `vector_ids_for` | `(chunkIds) -> Dict[str, int]` | The reverse, in one query. |
| `count` | `() -> int` | Rows in the table. |
| `insert` | `(vectorId, chunkId, embeddingModelUsed, dimensions=Config.EMBEDDING_DIMENSIONS)` | Explicit label, no vector. `on conflict (vectorId) do nothing`. |
| `batch_insert` | `(vectorIds, chunkIds, embeddingModelUsed, dimensions)` | Raises `InvalidBatchSize` when the id lists differ in length. |
| `get_meta_data` | `(vectorId, columnName) -> str \| int` | `columnName` is checked against `("vectorId", "chunkId", "embeddingModelUsed", "dimensions")` and raises `InvalidColumnNameException` otherwise — the column is interpolated into the SQL, so this allowlist is what keeps the query safe. Raises `InvalidVectorID` when no row matches. |
| `close` | `()` | Waits for a write in flight; safe if construction failed. |
