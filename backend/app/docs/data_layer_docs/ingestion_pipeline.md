# Unified Ingestion Pipeline Module (`ingestion_pipeline.py`)

## Overview & Purpose
The `ingestion_pipeline.py` module defines the `IngestionPipeline` facade class. It provides a single, simplified, stateless orchestration interface connecting all ingestion submodules: file discovery (`FileLoader`), multi-format text extraction (`TextExtractor`), regex cleaning (`TextNormalizer`), hierarchical/recursive chunking (`Chunker`), vector embedding (`EmbeddingManager`), and DiskANN vector indexing (`VectorDbManager`).

---

## Classes & Public APIs

### `class IngestionPipeline`
A facade wrapping internal component instances to expose clean public methods for step-by-step or pipeline document ingestion.

#### Constructor: `__init__(self) -> None`
Initializes internal component instances using defaults from `Config`.
- `self.f_loader = FileLoader()`
- `self.t_extractor = TextExtractor()`
- `self.t_normalizer = NormalizationProfiles.rag_ingestion()`
- `self.chunker = Chunker()`
- `self.embedder = EmbeddingManager()`
- `self.vector_db = VectorDbManager(...)` configured with `Config` metric, dimensions, max vectors, complexity, and thread parameters. This index lives only as long as the pipeline; it is not what makes vectors survive a restart.
- `self.vector_meta = VectorMetaDataRepository(self.chunker.db_path)` — allocates each vector's DiskANN label and **stores the vector itself**, in the chunk store's database file. These rows are what the index is rebuilt from (bug 5.20).

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

##### `ingest_vector(self, embedded_value: EmbeddedChunk) -> None`
Allocates a label for the chunk and stores its vector in `vector_meta_data` (one transaction), then inserts the vector into the pipeline's DiskANN index under that label.

###### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `embedded_value` | `EmbeddedChunk` | Embedded chunk node containing `vector` and `meta_data.chunk_id`. |

---

##### `batch_insert_vectors(self, embedded_objs: List[EmbeddedChunk]) -> List[int]`
Allocates one label per chunk and stores every vector beside it in a single transaction (`VectorMetaDataRepository.allocate_many`), then batch-inserts the vectors into the pipeline's DiskANN index under those labels. An empty list does nothing.

###### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `embedded_objs` | `List[EmbeddedChunk]` | List of embedded chunk instances to insert. |

Re-ingesting a chunk returns the label it already has (bug 5.21). The pipeline
records which labels its own index holds and inserts only new ones — DiskANN
refuses a label it already has — so the same chunks ingested twice in one
session, or twice in one batch, are indexed once. `ingest_vector` does the same.

###### Return Value
- **Type**: `List[int]`
- **Description**: The labels, in the order of `embedded_objs` — existing ones for chunks seen before. The label, not `EmbeddedChunk.vector_id`, is what DiskANN indexes: that id is 63-bit for pgvector and DiskANN labels are `uint32` (bug 5.16).

> There is no `persist_index()`. It wrote DiskANN's own index files, which diskannpy 0.7.0 cannot load back (bug 5.20); the stored vectors replace it.

---

##### `close(self) -> None`
Closes the label repository's connection.
