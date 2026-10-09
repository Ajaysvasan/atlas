# Unified Ingestion Pipeline Module (`ingestion_pipeline.py`)

## Overview & Purpose
The `ingestion_pipeline.py` module defines the `IngestionPipeline` facade class. It provides a single, simplified, stateless orchestration interface connecting all ingestion submodules: file discovery (`FileLoader`), multi-format text extraction (`TextExtractor`), regex cleaning (`TextNormalizer`), hierarchical/recursive chunking (`Chunker`), vector embedding (`EmbeddingManager`), vector storage (pgvector, through `VectorRepository`), label allocation (`VectorMetaDataRepository`), and building the DiskANN index generation (`IndexGenerations`).

---

## Classes & Public APIs

### `class IngestionPipeline`
A facade wrapping internal component instances to expose clean public methods for step-by-step or pipeline document ingestion.

#### Constructor: `__init__(self, vector_store=None, index_path=None) -> None`
Initializes internal component instances using defaults from `Config`.
- `self.f_loader = FileLoader()`
- `self.t_extractor = TextExtractor()`
- `self.t_normalizer = NormalizationProfiles.rag_ingestion()`
- `self.chunker = Chunker()`
- `self.embedder = EmbeddingManager()`
- `self.vector_meta = VectorMetaDataRepository(self.chunker.db_path)` — allocates each vector's DiskANN label beside its vector id, in the chunk store's database file. It holds no vectors.
- `self.vector_store` — where the vectors go: `vector_store` if given, else `stored_vectors.chunk_vector_store()` (`VectorRepository("global")`), opened on the first ingest. One handed in is not closed by `close()`.
- `self.index_path` — where generations are built; `None` means `Config.INDEX_PATH`, read when used.

The pipeline holds no index of its own. It used to build one sized for `MAX_VECTORS`, which reserved 1.2 GB however little it ingested (bug 5.23).

---

#### Methods

##### `load_file(self, folder_path: Union[str, Path]) -> Dict[str, List[Path]]`
Scans a target folder for candidate documents, categorized by extension. Anything that is not a known binary format is a candidate.

###### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `folder_path` | `Union[str, Path]` | Target directory path to scan for input files. |

###### Return Value
- **Type:** `Dict[str, List[Path]]`
- **Description:** Dictionary mapping extension strings (`"pdf"`, `"docx"`, `"rs"`, `"tex"`, …, or `"noext"`) to lists of `Path` objects. Only non-empty categories appear.

---

##### `extract_text_from_file(self, file_path: str) -> Tuple[str, str]`
Extracts raw text content from a single file path.

###### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `file_path` | `str` | Path to the file to extract. |

###### Return Value
- **Type:** `Tuple[str, str]`
- **Description:** 2-tuple containing `(file_path, extracted_raw_text)`.

---

##### `extract_text_from_files(self, loaded_files: Dict[str, List[Path]]) -> Dict[str, str]`
Extracts raw text content from a dictionary of categorized file lists.

###### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `loaded_files` | `Dict[str, List[Path]]` | Output returned by `load_file()`. |

###### Return Value
- **Type:** `Dict[str, str]`
- **Description:** Dictionary mapping absolute file paths (`str`) to raw extracted text strings (`str`).

---

##### `normalize_doc(self, file_path: Union[str, Path], text: str) -> NormalizedContent`
Sanitizes a single document text string using the default RAG normalizer profile (`rag_ingestion`).

###### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `file_path` | `Union[str, Path]` | Original path of the file (used for ID generation). |
| `text` | `str` | Raw text string to normalize. |

###### Return Value
- **Type:** `NormalizedContent`
- **Description:** Sanitized output wrapper containing string `content`, the `has_section` flag, and the `sections` tuple of `SectionSpan` offsets.

---

##### `normalize_docs(self, extracted_texts: Dict[str, str]) -> List[NormalizedContent]`
Sanitizes all extracted texts contained within a path-to-text dictionary.

###### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `extracted_texts` | `Dict[str, str]` | Output dictionary returned by `extract_text_from_files()`. |

###### Return Value
- **Type:** `List[NormalizedContent]`
- **Description:** List of `NormalizedContent` objects ready for chunking.

---

##### `chunk_text(self, normalized_content: NormalizedContent) -> Union[List[HChunk], List[RChunk]]`
Chunks a single document, routed the same way a batch would be. Delegates to `Chunker.chunk_document()`.

###### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `normalized_content` | `NormalizedContent` | A single normalized document object. |

###### Return Value
- **Type:** `Union[List[HChunk], List[RChunk]]`
- **Description:** Hierarchical chunks when the document has sections, recursive chunks otherwise.

---

##### `chunk_texts(self, normalized_contents: List[NormalizedContent]) -> Tuple[List[HChunk], List[RChunk]]`
Partitions normalized documents into hierarchical and recursive chunks.

###### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `normalized_contents` | `List[NormalizedContent]` | List of normalized document objects. |

###### Return Value
- **Type:** `Tuple[List[HChunk], List[RChunk]]`
- **Description:** 2-tuple `(h_chunks, r_chunks)` containing generated hierarchical and recursive chunks.

---

##### `embed(self, arg: Union[HChunk, RChunk, List[Union[HChunk, RChunk]]]) -> Union[EmbeddedChunk, List[EmbeddedChunk]]`
Converts single chunks or chunk lists into dense vector embeddings.

###### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `arg` | `Union[HChunk, RChunk, List[Union[HChunk, RChunk]]]` | Chunk object or list of chunk objects. |

###### Return Value
- **Type:** `Union[EmbeddedChunk, List[EmbeddedChunk]]`
- **Description:** Single `EmbeddedChunk` or list of `EmbeddedChunk` instances.

---

##### `ingest_vector(self, embedded_value: EmbeddedChunk) -> int`
`batch_insert_vectors([embedded_value])[0]`: the chunk's label.

---

##### `batch_insert_vectors(self, embedded_objs: List[EmbeddedChunk]) -> List[int]`
Stores the vectors, then allocates their labels, then lets `maintain_index()` decide whether to build. An empty list does nothing.

1. Every `EmbeddedChunk.vector_id` is checked by `checked_vector_id` before anything is written: `MissingVectorId` for `None`, `MalformedVectorId` for anything that is not an integer in `0 … 2**63 - 1`.
2. The vectors go to pgvector under their vector ids, `on conflict do nothing`. A store that cannot be opened raises `VectorStoreUnavailable`; a failed write raises `VectorInsertionError`. Either way no label is written — a label never exists for a vector that was not stored.
3. `VectorMetaDataRepository.batch_insert(vector_ids, chunk_ids)` allocates the labels in one transaction. A chunk seen before gets its own label back (bug 5.21); `VectorIdConflict` and `EmbeddingModelMismatch` refuse the batch.

###### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `embedded_objs` | `List[EmbeddedChunk]` | `vector`, `vector_id` (from the embedder; required) and `meta_data.chunk_id`. |

###### Return Value
- **Type**: `List[int]`
- **Description**: The labels, in the order of `embedded_objs`. The label is what DiskANN returns; the vector id is what pgvector is keyed by (bug 5.16).

---

##### `maintain_index(self, force: bool = False) -> BuildOutcome | None`
Builds a new index generation from every stored vector once `Config.INDEX_REBUILD_AT` (`10_000`) labels wait above the current generation's `through`, or whenever any wait if `force`. Returns `None` when nothing was due, else the `BuildOutcome` — which never raises: a build that does not fit in memory or on disk, is already running, fails, is killed or hangs leaves the current generation in use and says why. The one exception is a vector store that cannot be opened to read from, `VectorStoreUnavailable` — which cannot happen when it runs after `batch_insert_vectors`, which opened it. See `vector_db_manager.md`.

---

##### `close(self) -> None`
Closes the label repository's connection, and the vector store if the pipeline opened it.
