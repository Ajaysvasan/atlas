# Vector Database Management Layer (`vector_db_manager/`)

## Overview & Purpose
The `vector_db_manager` submodule holds the document chunk index and the stores behind it:

- **PostgreSQL / pgvector** (`repository/vectorRepository.py`) — where every vector lives, document chunks under the project id `Config.GLOBAL_VECTOR_SCOPE` (`"global"`), the memory layer's under their own project ids. Keyed by `(project_id, vector_id)`.
- **`vector_meta_data`** (`repository/vectorMetaDataRepository.py`) — SQLite, in the chunk store: each DiskANN label, the vector id it stands for, and its chunk. No vectors.
- **The DiskANN index** (`vectorDbManager.py`) — a *built generation* on disk (`index_generations.py`), opened rather than rebuilt at startup, plus an in-memory index of the vectors ingested since (`vectorDB_diskann.py`). Searched together and merged by distance.
- **`memory_guard.py`** — every allocation the index makes is admitted against free memory first, and degraded rather than made when it does not fit.

Rationale and measurements are in `data_layer/vector_db_manager/README.md`.

---

## `class VectorDbManager` (`vectorDbManager.py`)
The index retrieval searches: an optional built generation (`base`) and the recent vectors in a `diskannpy.DynamicMemoryIndex`. Mutations of the recent index run under `self.lock`.

#### Constructor: `__init__(self, distance_metrics, vector_dtype, dimensions, max_vectors, complexity, graph_degree, num_threads, k_neighbors) -> None`

All eight parameters are required. `VectorSearch` supplies them from `Config`.

| Parameter | Type | Supplied as | Description |
| :--- | :--- | :--- | :--- |
| `distance_metrics` | `str` | `Config.DISTANCE_METRIC` (`"l2"`) | Passed straight through to `diskannpy`. |
| `vector_dtype` | `Type[np.float32 \| np.int8 \| np.uint8]` | `Config.VECTOR_DTYPE` | Element type of stored vectors. |
| `dimensions` | `int` | `Config.EMBEDDING_DIMENSIONS` (`128`) | Vector dimensionality. |
| `max_vectors` | `int` | `Config.RECENT_VECTOR_CAPACITY` (`20_000`) | Capacity of the **recent** index. diskannpy allocates every slot up front (1.7 KB each), which is why this is not `MAX_VECTORS`: at 1M it reserved 1.2 GB however little it held. |
| `complexity` | `int` | `Config.COMPLEXITY` (`100`) | Search list size (`L`). |
| `graph_degree` | `int` | `Config.GRAPH_DEGREE` (`120`) | Maximum out-degree (`R`). |
| `num_threads` | `int` | `Config.NUM_THREADS` (`4`) | Worker threads. |
| `k_neighbors` | `int` | `Config.K_NEIGHBORS` (`9`) | Neighbours returned when a search names no `k`. |

#### Methods

| Method | Signature | Behaviour |
| :--- | :--- | :--- |
| `use_base` | `(base: BuiltIndex \| None) -> None` | Searches a built generation alongside the recent vectors. |
| `through` | property `-> int` | The highest label the built generation holds; `0` without one. The recent index is filled from above it. |
| `insert` | `(embedded_chunk_obj, vector_id=None) -> None` | Into the recent index, under `vector_id` as `uint32`, falling back to `embedded_chunk_obj.vector_id`. |
| `batch_insert` | `(embedded_chunk_objs, vector_ids=None)` | The same as one `float32` 2-D array and one `uint32` array — diskannpy requires both as arrays (bug 5.15). |
| `count` | `() -> int` | Built plus recent. |
| `recent_count` | `() -> int` | The recent index alone, from diskannpy's native `num_points()`. |
| `restore` | `(source: StoredVectors, after: int = 0) -> int` | Inserts the source's vectors labelled above `after` into the recent index, a page at a time, **until it is full**; returns the last label taken. What does not fit is logged with its count and waits for the next build, still found by keyword search. Labels without a stored vector are skipped and reported. |
| `search_vector` | `(query, k_neighbors=None) -> (labels, distances)` | Searches the built generation and the recent index, each for no more than it holds, and merges by distance (both are exact squared L2). The recent search is capped at `recent_count()`: asked for more, diskannpy fills the surplus from uninitialised memory (bug 5.18). |
| `batch_search_vectors` | `(queries, k_neighbors=None)` | One `search_vector` per query, as 2-D arrays. |
| `delete_vector` / `delete_vectors` | `(vector_id)` / `(vector_ids)` | The recent index only; a built generation is immutable until the next build. Nothing calls either (bug 5.5). |

> There is no `save` or `load`. diskannpy 0.7.0 writes every label of a dynamic index as `0` and cannot load one back in a new process (bug 5.20); the built generations replace them.

---

## `class VectorDb_diskann` (`vectorDB_diskann.py`)
Thin driver over `diskannpy.DynamicMemoryIndex` (`self.dynamic_dann`), constructed with `saturate_graph=SATURATE_GRAPH` (`True`). No parameter validation: an invalid metric surfaces as a `diskannpy` error.

> **`SATURATE_GRAPH` is not a tuning knob.** diskannpy defaults it to `False`, and the dynamic index then builds a graph too sparse to search: recall@10 of 0.09 against 0.96 (bug 5.19).

| Method | Signature | Behaviour |
| :--- | :--- | :--- |
| `insert` / `batch_insert` | `(vector, vector_id)` / `(vectors, vector_ids)` | `ValueError` and `RuntimeError` are re-raised as `VectorInsertionError(vector_id, cause)`. |
| `search_vector` | `(query, k_neighbors, complexity)` | `(tags, distances)`. |
| `batch_search_vector` | `(queries, k_neighbors, complexity)` | `dynamic_dann.batch_search(..., self.num_threads)`. |
| `count` | `() -> int` | `dynamic_dann._index.num_points()`; the count lives only on the native object. |
| `delete_vector` / `delete_vectors` | `(id)` / `(ids)` | `mark_deleted`, then one `consolidate_delete`. |

---

## Built generations (`index_generations.py`)

```
data/disk_ann_index/
  CURRENT                 the generation in use, replaced atomically
  .build.lock             held by the one build that may run
  gen-000007/             the generation before (kept as the fallback)
  gen-000008/
    manifest.json         kind, count, through, model, dimensions, metric, degree, previous, files {name: size}
    labels.npy            uint32, position -> label
    ann ann.data          a memory index, or
    ann_disk.index ann_pq_*.bin ann_sample_*.bin    a disk index
```

#### `class IndexGenerations(root=None)`
`root` defaults to `Config.INDEX_PATH`, read when called.

| Method | Signature | Behaviour |
| :--- | :--- | :--- |
| `current` | `() -> str \| None` | The name in `CURRENT`, if it is a generation name. |
| `manifest` | `(name) -> dict \| None` | `None` when missing or unreadable. |
| `through` | `() -> int` | The highest label the current generation was built through; `0` without one. |
| `open` | `(allowance: int) -> BuiltIndex \| None` | The current generation, else its `previous`, within `allowance` bytes of RAM. A generation is refused — logged with the reason, never raised — when its manifest is unreadable, it was built for another model, dimension or metric, a file is missing or the wrong size, its labels do not match, it needs more than `allowance`, or more than `memory_guard.spare_memory()`. A **disk** generation needs only its compressed vectors; its node cache is sized to whatever the allowance and free memory leave, down to none. |
| `build` | `(source: StoredVectors) -> BuildOutcome` | Builds a generation from every stored vector and makes it current. **Never raises.** See below. |

**`build`, step by step**, each failure returning a `BuildOutcome` with the reason and leaving the current generation in use:

1. Take `.build.lock` without waiting; another build running → skipped.
2. Fewer than two vectors → nothing to build (DiskANN cannot make a graph of one).
3. Plan: a **memory** index when it fits the budget (`VECTOR_INDEX_RAM_MB` less the recent index) *and* building it fits free memory; else a **disk** index, its compressed vectors given a quarter of that allowance and its build bounded by `memory_guard.disk_build_budget()`; neither → skipped.
4. Free disk below `memory_guard.index_disk_bytes()` → skipped.
5. Sweep `*.building` leftovers and generations older than the previous one.
6. Write `vectors.bin` (DiskANN's format) and `labels.npy` a page at a time into `gen-N.building/`.
7. Run `builder_command(spec)` — `python -m data_layer.vector_db_manager.index_build SPEC` — with its output in `build.log`, stopped after `max(900 s, 5 ms × vectors)`. Killed by a signal, exited non-zero, timed out → skipped, with the signal or the log's tail.
8. Delete what search does not need (`vectors.bin`, `ann_mem.index.data`, `build.log`), write the manifest, fsync every file, rename to `gen-N`, replace `CURRENT` atomically, sweep.

#### `class BuildOutcome(NamedTuple)`
`built: bool`, `reason: str`, `generation: str | None`, `kind: "memory" | "disk" | None`, `count: int`.

#### `class BuiltIndex`
`name`, `kind`, `labels`, `through`, `count`. `search(query, k, complexity) -> (labels, distances)` asks for no more than `count` — a disk index pads a short answer with position `0`, a real vector — and maps positions to labels. A disk index is searched with beam width 4 and at least two threads: with one, it never returns.

#### `builder_command(spec) -> List[str]` · `search_threads() -> int`
The build process's command line, swapped by tests to kill or fail one; `max(2, Config.NUM_THREADS)`.

---

## `index_build.py`
Run as `python -m data_layer.vector_db_manager.index_build SPEC`, never imported by the application. It sets its own `oom_score_adj` to `1000` and lowers its priority, then calls `diskannpy.build_memory_index` or `build_disk_index` on the vector file. A separate interpreter rather than `multiprocessing`: spawn re-imports the caller's main script, re-running any entry point without a `__main__` guard.

---

## `memory_guard.py`

| Function | Returns |
| :--- | :--- |
| `available_memory()` | The lower of `/proc/meminfo`'s `MemAvailable` and the room under the tightest cgroup v2 `memory.max` above this process; `None` when neither can be read. |
| `spare_memory()` | `available_memory()` less `Config.MEMORY_HEADROOM_MB`; `None` if unknown. |
| `can_hold(size)` | Whether `size` bytes fit in `spare_memory()`. Unknown memory admits it, on the budget alone. |
| `free_disk(path)` | Free bytes where `path` is or would be created. |
| `budget()` | `Config.VECTOR_INDEX_RAM_MB` in bytes. |
| `recent_index_bytes(capacity, dims, degree)` · `memory_index_bytes(count, dims, degree)` · `cache_node_bytes(dims, degree)` · `memory_build_bytes(count, dims, degree)` · `index_disk_bytes(count, dims, degree)` | Estimates, with measured margins. |
| `disk_build_budget()` | Up to 4 GB of what is spare less 128 MB; `None` below 256 MB. |

---

## `stored_vectors.py`

| Name | Behaviour |
| :--- | :--- |
| `chunk_vector_store()` | `VectorRepository(Config.GLOBAL_VECTOR_SCOPE)`. Looked up as a module attribute, which is how the test suite replaces it with an in-memory store. |
| `class StoredVectors(mapping, store)` | `pages(after=0, batch_size=50_000)` yields `Page(labels, vectors, through)` in label order: one page of labels from SQLite, their vectors from PostgreSQL in one query. Labels with no stored vector are left out and counted in `missing`; `through` is the last label looked at, so they are not looked for again. `pending(after)` counts this model's labels above `after`. |

---

## `class VectorRepository` (`repository/vectorRepository.py`)
The pgvector store. One row per `(project_id, vector_id)` in `vectors`; a `psycopg` 3 connection for the object's lifetime.

Connection settings come from `.env`: `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`. Any missing key raises `MissingDatabaseConfiguration`, naming it. The `DB_` prefix is load-bearing: `load_dotenv()` does not override a variable the environment already has (bug 6.2). `register_vector_types(self.conn)` runs before any statement touches a vector column (bug 5.1).

| Method | Signature | Raises |
| :--- | :--- | :--- |
| `insert` | `(vector_id, vector) -> None` | `InvalidVectorDimension`; `DuplicateVectorException`; `VectorInsertionError` |
| `batch_insert` | `(vector_ids, vectors) -> None` | `InvalidBatchSize`, `InvalidVectorDimension`, `VectorInsertionError`. `on conflict … do nothing`, so re-ingesting is a no-op. |
| `update` | `(vector_id, vector) -> None` | `VectorNotFoundEror`, `InvalidVectorDimension`, `VectorInsertionError` |
| `delete` · `batch_delete` | `(vector_id)` · `(vector_ids)` | `VectorInsertionError` |
| `search` · `batch_search` | `(vector_id)` · `(vector_ids)` | `VectorNotFoundEror`. `batch_search` is one query per id. |
| `vectors_for` | `(vector_ids) -> Dict[int, NDArray[float32]]` | One query, `vector_id = any(%s::bigint[])`; ids not stored are absent. The cast keeps 63-bit ids `bigint`, which psycopg would otherwise send as `numeric`. |
| `close` | `()` | — |

`as_array(embedding)` converts what pgvector returns — its own `Vector` from 0.4 on — to `float32`.

---

## `class VectorMetaDataRepository` (`repository/vectorMetaDataRepository.py`)

#### Constructor: `__init__(self, db_path: str | None = None) -> None`
Opens `db_path` (default `Config.DB_PATH`, the chunk store) with `check_same_thread=False`, enables WAL, and creates the table. Every statement runs under an `RLock`.

#### Schema
```sql
vector_meta_data(
    label              INTEGER PRIMARY KEY AUTOINCREMENT CHECK (label <= 4294967295),  -- the DiskANN label
    vectorId           INTEGER NOT NULL UNIQUE
                       CHECK (typeof(vectorId) = 'integer' AND vectorId >= 0),          -- passed in, never generated
    chunkId            TEXT NOT NULL UNIQUE,
    embeddingModelUsed TEXT NOT NULL,
    dimensions         INTEGER NOT NULL
)
```

The vector id is required by the database as well as the code: a write that omits it, or passes `NULL`, text or a negative number, is refused. `label` is the only number the table generates, never reused, and a label past `uint32` is refused rather than wrapped. No foreign key on `chunkId`: a chunk lives in `Chunks` or `RecursiveChunks` (bug 5.2).

A table from before the vector id was required — `vectorId` holding the label, possibly a `vector` blob — is rebuilt on open (bug 5.22): labels kept, vector ids derived from the chunk ids, one row per chunk preferring the configured model's oldest, labels that do not fit `uint32` dropped, blobs dropped, and the sequence kept above every old label. A warning gives the counts.

#### `checked_vector_id(vector_id, chunk_id) -> int`
The id as an `int`. `MissingVectorId` for `None`; `MalformedVectorId` for a `bool`, a non-integer, or a value outside `0 … 2**63 - 1`. NumPy integers are accepted.

#### Methods

| Method | Signature | Notes |
| :--- | :--- | :--- |
| `insert` | `(vectorId, chunkId, embeddingModelUsed=Config.EMBEDDING_MODEL, dimensions=Config.EMBEDDING_DIMENSIONS) -> int` | The chunk's label: the one it has, or the next. `VectorIdConflict` when the chunk is stored under another vector id or the vector id under another chunk; `EmbeddingModelMismatch` when the chunk was stored by another model. |
| `batch_insert` | `(vectorIds, chunkIds, embeddingModelUsed=..., dimensions=...) -> List[int]` | The same per row, in order, in one transaction — every id is checked before anything is written, and a refused row rolls the batch back. `InvalidBatchSize` when the lists differ in length. |
| `labels` | `(after=0, batch_size=50_000, embeddingModelUsed=..., dimensions=...) -> Iterator[(uint32[n], int64[n])]` | `(labels, vector ids)` above `after`, in label order, keyset-paginated; this model and width only. |
| `pending` | `(after=0, embeddingModelUsed=..., dimensions=...) -> int` | This model's labels above `after`. |
| `chunk_ids_for` | `(labels) -> Dict[int, str]` | One query; unknown labels are absent. |
| `labels_for` | `(chunkIds) -> Dict[str, int]` | The reverse, for keyword hits. |
| `get_meta_data` | `(vectorId, columnName) -> str \| int` | Keyed by the vector id. `columnName` is checked against `("label", "vectorId", "chunkId", "embeddingModelUsed", "dimensions")` — it is interpolated into the SQL. `InvalidVectorID` when no row matches. |
| `count` | `() -> int` | Rows in the table. |
| `close` | `()` | Waits for a write in flight; safe if construction failed. |
