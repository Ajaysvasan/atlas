# Nodes & Metadata Structures (`nodes.py` & `metadata.py`)

## Overview & Purpose
The `nodes.py` and `metadata.py` modules define the core domain objects used across the ingestion pipeline. Modeled as immutable Python frozen `@dataclass` structures (`frozen=True`), these classes guarantee thread safety and prevent accidental modification during chunking, embedding, database insertion, and retrieval.

---

## Metadata Dataclasses (`metadata.py`)

### `NormalizedTextMetaData`
Stores lineage and tracking information for a sanitized document returned by `TextNormalizer`.

| Field | Type | Description |
| :--- | :--- | :--- |
| `document_id` | `str` | SHA-256 hash identifying the unique document. |
| `source_file_path` | `Optional[str]` | Absolute or relative filesystem path of the input file. |
| `file_name` | `str` | Base filename including suffix (e.g. `"report.pdf"`). |
| `file_type` | `str` | Lowercase document file extension (e.g. `".pdf"`). |
| `ingestion_time` | `str` | UTC timestamp string indicating when normalization occurred (`ISO 8601`). |
| `normalizer_version` | `str` | Version tag of the normalizer pipeline (e.g. `"rag_v1"`). |
| `content_hash` | `str` | SHA-256 hash of the normalized text content itself. |
| `source_path` | `Optional[str]` | Alias path attribute (`None` by default). |

---

### `ChunkMetaData`
Stores descriptive context identifying which document and algorithm generated a specific chunk.

| Field | Type | Description |
| :--- | :--- | :--- |
| `document_name` | `str` | Base filename of the parent document. |
| `document_id` | `str` | SHA-256 identifier of the parent document. |
| `chunking_algorithm_used` | `str` | Name of the chunking algorithm (`"hierarchical"` or `"recursive"`). |

---

### `EmbeddedChunkMetaData`
Stores vector embedding provenance tied to a specific chunk and model.

| Field | Type | Description |
| :--- | :--- | :--- |
| `chunk_id` | `str` | SHA-256 unique identifier of the parent chunk (`HChunk` or `RChunk`). |
| `chunk` | `str` | Raw string text of the chunk that was embedded. |
| `modelUsedForChunking` | `str` | HuggingFace embedding model name (e.g. `"sentence-transformers/all-MiniLM-L6-v2"`). |

---

## Node Dataclasses (`nodes.py`)

### `Document`
Represents the top-level normalized document unit processed by `HierarchicalChunker`.

| Field | Type | Description |
| :--- | :--- | :--- |
| `documentId` | `str` | Unique document identifier. |
| `documentName` | `str` | Filename of the document. |
| `normalizedText` | `str` | Complete sanitized text string of the document. |

---

### `Section`
Represents a major structural heading or chapter identified within a document by `HierarchicalChunker`.

| Field | Type | Description |
| :--- | :--- | :--- |
| `sectionId` | `str` | SHA-256 identifier generated from `(sectionName, documentId)`. |
| `documentId` | `str` | Foreign key referencing the parent `Document`. |
| `sectionName` | `str` | Extracted header title string (e.g. `"INTRODUCTION"`). |
| `content` | `str` | Text body underneath the section heading up to the next section heading. |
| `contentLength` | `int` | Character count of `content`. |
| `startOffSet` | `int` | Absolute character start index within the parent document. |
| `endOffSet` | `int` | Absolute character end index within the parent document. |

---

### `Context`
Represents a logical paragraph or contextual block segmented from a `Section`.

| Field | Type | Description |
| :--- | :--- | :--- |
| `contextId` | `str` | SHA-256 identifier generated from `(context_text, sectionId)`. |
| `sectionId` | `str` | Foreign key referencing the parent `Section`. |
| `context` | `str` | Text string of the paragraph block (`re.finditer(r"\n\s*\n")`). |
| `contextLen` | `int` | Character count of `context`. |
| `startOffSet` | `int` | Relative character start index within the parent section. |
| `endOffSet` | `int` | Relative character end index within the parent section. |

---

### `HChunk` (Hierarchical Chunk)
Represents a fixed-window text slice generated from a `Context` node by `HierarchicalChunker`.

| Field | Type | Description |
| :--- | :--- | :--- |
| `chunk_id` | `str` | SHA-256 identifier generated from `(chunk_text, contextId)`. |
| `context_id` | `str` | Foreign key referencing the parent `Context`. |
| `chunk` | `str` | Sliced text segment (`context[start:end]`). |
| `start_off_set` | `int` | Character start index within the parent context. |
| `end_off_set` | `int` | Character end index within the parent context. |
| `meta_data` | `ChunkMetaData` | Attached metadata object (`chunking_algorithm_used="hierarchical"`). |

---

### `RChunk` (Recursive Chunk)
Represents a text segment produced by `RecursiveChunker` splitting on natural punctuation boundaries (`\n\n`, `\n`, `.`, ` `).

| Field | Type | Description |
| :--- | :--- | :--- |
| `chunk` | `str` | Sliced text string. |
| `meta_data` | `ChunkMetaData` | Attached metadata object (`chunking_algorithm_used="recursive"`). |
| `chunk_id` | `str` | SHA-256 identifier generated from `(chunk_text, documentName, documentId)`. |

---

### `NormalizedContent`
The standardized wrapper output produced by `TextNormalizer.normalize_text()`.

| Field | Type | Description |
| :--- | :--- | :--- |
| `content` | `str` | Cleaned and normalized text string. |
| `has_section` | `bool` | Flag indicating whether the text contains uppercase section headers. |
| `meta_data` | `NormalizedTextMetaData` | Lineage metadata tracking file name, path, hash, and timestamp. |

---

### `EmbeddedChunk`
Represents a fully processed chunk whose textual content has been transformed into a dense numeric vector embedding.

| Field | Type | Description |
| :--- | :--- | :--- |
| `vector` | `List[float]` | Dense floating-point vector (`np.float32`) produced by `EmbeddingManager`. |
| `meta_data` | `EmbeddedChunkMetaData` | Provenance metadata tracking `chunk_id`, text content, and embedding model name. |
