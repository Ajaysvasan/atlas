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
- `self.vector_db = VectorDbManager(...)` configured with `Config` metric, dimensions, max vectors, complexity, and thread parameters.

---

#### Methods

##### `load_file(self, folder_path: Union[str, Path]) -> Dict[str, List[Path]]`
Scans a target folder for supported document files categorized by their extension.

###### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `folder_path` | `Union[str, Path]` | Target directory path to scan for input files. |

###### Return Value
- **Type:** `Dict[str, List[Path]]`
- **Description:** Dictionary mapping extension strings (`"pdf"`, `"docx"`, etc.) to lists of `Path` objects.

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
- **Description:** Sanitized output wrapper containing string `content` and `has_section` boolean flag.

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
Inserts a single embedded chunk into the active DiskANN vector database index (`self.vector_db.insert`).

###### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `embedded_value` | `EmbeddedChunk` | Embedded chunk node containing `vector` and `meta_data.chunk_id`. |

---

##### `batch_insert_vectors(self, embedded_objs: List[EmbeddedChunk]) -> None`
Batch-inserts a list of embedded chunk nodes into the active DiskANN index (`self.vector_db.batch_insert`).

###### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `embedded_objs` | `List[EmbeddedChunk]` | List of embedded chunk instances to insert. |
