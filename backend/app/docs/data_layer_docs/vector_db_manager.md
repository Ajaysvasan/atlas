# Vector Database Management Layer (`vector_db_manager/`)

## Overview & Purpose
The `vector_db_manager` submodule provides high-performance graph-based Approximate Nearest Neighbor (ANN) vector indexing and retrieval. It delegates low-level graph construction (`Vamana` graph) to the `DiskANN` library (`diskannpy`), wrapping it in a domain-specific interface (`VectorDbManager` $\rightarrow$ `VectorDb_diskann`) that manages dynamic vector insertions, batching, disk persistence (`save`/`load`), and $k$-NN cosine/L2 similarity queries.

---

## Classes & Public APIs

### `class VectorDbManager` (`vectorDbManager.py`)
High-level public wrapper over the DiskANN index driver that handles parameter validation, node formatting, and exception propagation.

#### Constructor: `__init__(self, ...)`
Initializes the vector index driver using application configuration parameters.

##### Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `distance_metrics` | `str` | `Config.DISTANCE_METRIC` | Distance calculation metric (`"l2"`, `"cosine"`, or `"inner_product"`). |
| `vector_dtype` | `type` | `Config.VECTOR_DTYPE` (`np.float32`) | Element data type for vector arrays. |
| `dimensions` | `int` | `Config.DIMENSIONS` | Dimensionality of vectors stored in the index. |
| `max_vectors` | `int` | `Config.MAX_VECTORS` | Maximum capacity limit of the index graph. |
| `complexity` | `int` | `Config.COMPLEXITY` | Search beam width (`L`) during index build and lookups. |
| `graph_degree` | `int` | `Config.GRPAH_DEGREE` | Maximum out-degree (`R`) of nodes in the Vamana graph. |
| `num_threads` | `int` | `Config.NUM_THREADS` | Number of concurrent worker threads. |
| `k_neighbors` | `int` | `Config.NUM_THREADS` (mapped to `Config.K_NEIGHBORS`) | Default number of nearest neighbors (`k`) returned during searches. |

---

#### Methods

##### `insert(self, embedded_chunk: EmbeddedChunk) -> None`
Inserts a single `EmbeddedChunk` into the active DiskANN graph index (`self.db.insert`).

###### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `embedded_chunk` | `EmbeddedChunk` | Domain node containing `vector: List[float]` and `meta_data.chunk_id: str`. |

---

##### `batch_insert(self, embedded_chunks: List[EmbeddedChunk]) -> None`
Inserts a list of `EmbeddedChunk` nodes into the graph index (`self.db.batch_insert`).

###### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `embedded_chunks` | `List[EmbeddedChunk]` | List of embedded chunk domain nodes. |

---

##### `search_vector(self, query_vector: Union[List[float], np.ndarray], k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]`
Queries the DiskANN graph for the $k$ nearest neighbor vectors matching the provided query array.

###### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `query_vector` | `Union[List[float], np.ndarray]` | Target query vector of dimension `dimensions`. |
| `k` | `Optional[int]` | Optional override for number of neighbors to return (defaults to `self.k_neighbors`). |

###### Return Value
- **Type:** `Tuple[np.ndarray, np.ndarray]`
- **Description:** 2-tuple `(neighbor_tags, distances)` where `neighbor_tags` contains the `np.uint32` vector IDs/tags of the matched neighbors and `distances` contains their corresponding float distance scores (`np.float32`).

---

##### `save_index(self, directory: Optional[str] = None) -> None`
Persists the dynamic in-memory graph index to local disk files (`self.db.save`).

###### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `directory` | `Optional[str]` | Optional custom directory path (`None` defaults to `Config.INDEX_PATH`). |

---

##### `load_index(self, directory: Optional[str] = None) -> None`
Loads previously saved DiskANN graph index files from local disk into memory (`self.db.load`).

###### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `directory` | `Optional[str]` | Optional target directory path (`None` defaults to `Config.INDEX_PATH`). |

---

### `class VectorDb_diskann` (`vectorDB_diskann.py`)
Low-level driver interacting directly with `diskannpy.DynamicMemoryIndex`.

#### Constructor: `__init__(self, ...)`
Converts string metrics to `diskannpy.L2` / `diskannpy.COSINE`, verifies parameters, and instantiates `diskannpy.DynamicMemoryIndex(...)`.

#### Methods

##### `insert(self, vector: Union[List[float], np.ndarray], tag: Union[int, str, np.uint32]) -> None`
Converts `vector` to 1D `np.float32` array and `tag` to integer/SHA hash `np.uint32`, executing `self.index.insert(vector, tag)`.

##### `batch_insert(self, vectors: Union[List[List[float]], np.ndarray], tags: Union[List[int], List[str], np.ndarray]) -> None`
Formats multi-row 2D `np.float32` arrays and 1D `np.uint32` tags, executing `self.index.batch_insert(vectors, tags)`.

##### `search(self, query: Union[List[float], np.ndarray], k_neighbors: int) -> Tuple[np.ndarray, np.ndarray]`
Executes `self.index.search(query_array, k_neighbors, complexity=self.complexity)`, returning `(tags, distances)`.

##### `save(self, directory: Optional[str] = None) -> None`
Checks directory existence (creating if missing) and calls `self.index.save(os.path.join(dir, "diskann_dynamic"))`.

##### `load(self, directory: Optional[str] = None) -> None`
Checks whether directory and index files exist (raising `IndexDirectoryDoesNotExists` if missing) and executes `self.index.load(...)`.
