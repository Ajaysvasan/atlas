# Vector Database Management Layer (`vector_db_manager/`)

## Overview & Purpose
The `vector_db_manager` submodule holds the project's two vector stores:

- **DiskANN** (`vectorDbManager.py` → `vectorDB_diskann.py`) — the data layer's approximate nearest neighbour index over ingested document chunks, built on `diskannpy.DynamicMemoryIndex`.
- **PostgreSQL / pgvector** (`repository/vectorRepository.py`) — the memory layer's exact store for conversation snapshot vectors, keyed by `(project_id, vector_id)`.

`repository/vectorMetaDataRepository.py` holds a SQLite sidecar table mapping vector ids back to the chunk they came from.

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
| `insert` | `(embedded_chunk_obj: EmbeddedChunk) -> None` | Takes the lock and inserts `embedded_chunk_obj.vector` under `embedded_chunk_obj.vector_id`. |
| `batch_insert` | `(embedded_chunk_objs: List[EmbeddedChunk])` | Collects vectors into a single `np.float32` 2-D array — this is the only place the conversion happens — and passes the ids as a plain list. |
| `search_vector` | `(query)` | Delegates with the instance's `k_neighbors` and `complexity`. There is **no per-call `k` override**. |
| `batch_search_vectors` | `(queries)` | Batch form of the above. |
| `delete_vector` | `(vector_id) -> None` | Under the lock. |
| `delete_vectors` | `(vector_ids) -> None` | Under the lock. |
| `save` | `(save_path=Config.INDEX_PATH)` | Under the lock. |
| `load` | `(load_path=Config.INDEX_PATH)` | Under the lock. Returns the reloaded `dynamic_dann`, or **`None`** if the directory is absent — `IndexDirectoryDoesNotExists` is caught here and logged as a warning rather than propagated. |

> **Note.** `insert` keys the vector on `EmbeddedChunk.vector_id`, an `int`, not on `meta_data.chunk_id`. DiskANN tags are unsigned integers; passing the SHA-256 `chunk_id` string was a historical bug.

---

## `class VectorDb_diskann` (`vectorDB_diskann.py`)
Thin driver over `diskannpy.DynamicMemoryIndex`, exposed as `self.dynamic_dann`.

#### Constructor: `__init__(self, distance_metrics, vector_dtype, dimensions, max_vectors, complexity, graph_degree, num_threads) -> None`
Stores the parameters and constructs `dann.DynamicMemoryIndex(...)` directly. It performs **no** metric-string conversion and **no** parameter validation — `distance_metrics` is handed to `diskannpy` as the string it was given, and an invalid value surfaces as a `diskannpy` error.

#### Methods

| Method | Signature | Behaviour |
| :--- | :--- | :--- |
| `insert` | `(vector, vector_id)` | `ValueError` and `RuntimeError` are re-raised as `VectorInsertionError(vector_id, cause)`, chained with `from`. |
| `batch_insert` | `(vectors, vector_ids)` | Same wrapping, with the id list as `vector_id`. |
| `search_vector` | `(query, k_neighbors, complexity)` | `dynamic_dann.search(...)`, returning `(tags, distances)`. |
| `batch_search_vector` | `(queries, k_neighbors, complexity)` | `dynamic_dann.batch_search(..., self.num_threads)`. |
| `delete_vector` | `(id)` | `mark_deleted` then `consolidate_delete`. |
| `delete_vectors` | `(ids)` | Marks each, then a single `consolidate_delete`. |
| `save` | `(save_path=Config.INDEX_PATH)` | Creates the directory if absent, then `dynamic_dann.save(save_path)`. The path is used as given — no filename is appended. |
| `load` | `(load_path=Config.INDEX_PATH)` | Checks the **directory** exists (not the individual index files), calls `DynamicMemoryIndex.from_file(...)`, and **reassigns `self.dynamic_dann`** to the result so later inserts reach the loaded index. Raises `IndexDirectoryDoesNotExists` when the directory is missing. |

> Both insert paths wrap driver failures identically. `batch_insert` used to wrap nothing, so `except VectorInsertionError` around it caught no failure at all.

---

## `class VectorRepository` (`repository/vectorRepository.py`)
The memory layer's pgvector store. One row per `(project_id, vector_id)` in a `vectors` table; `psycopg` 3 connection held for the object's lifetime.

Connection settings come from `.env` via `python-dotenv`: `DBNAME`, `DB_USER`, `PASSWORD`, `HOST`, `PORT`. Any missing key raises `MissingDatabaseConfiguration` at construction, naming the absent keys.

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
SQLite table `vector_meta_data(vectorId INT PRIMARY KEY, chunkId TEXT FK → Chunks(chunkId), embeddingModelUsed TEXT, dimensions INT)`, recording which chunk and model produced a vector.

| Method | Signature | Notes |
| :--- | :--- | :--- |
| `insert` | `(vectorId, chunkId, embeddingModelUsed, dimensions=Config.EMBEDDING_DIMENSIONS)` | `on conflict (vectorId) do nothing`. |
| `batch_insert` | `(vectorIds, chunkIds, embeddingModelUsed, dimensions)` | Raises `InvalidBatchSize` when the id lists differ in length. |
| `get_meta_data` | `(vectorId, columnName) -> str \| int` | `columnName` is checked against `("vectorId", "chunkId", "embeddingModelUsed", "dimensions")` and raises `InvalidColumnNameException` otherwise — the column is interpolated into the SQL, so this allowlist is what keeps the query safe. Raises `InvalidVectorID` when no row matches. |
| `close` | `()` | — |
