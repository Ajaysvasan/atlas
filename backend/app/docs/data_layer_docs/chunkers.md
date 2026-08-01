# Chunking Algorithms & Storage Layer (`Chunker/`)

## Overview & Purpose
The `Chunker` submodule partitions long normalized document texts into smaller, semantically coherent text segments suitable for vector embedding and approximate nearest neighbor search. It routes documents based on internal structure (`has_section`), utilizing two distinct chunking strategies:
1. **Hierarchical Chunking (`HierarchicalChunker`)**: Segments structured documents by headings (`Section`), paragraphs (`Context`), and sliding character windows (`HChunk`), persisting the full tree relationships inside a relational SQLite database (`DB_Manager`).
2. **Recursive Chunking (`RecursiveChunker`)**: Segments unstructured documents by recursively splitting text along hierarchical separator lists (`["\n\n", "\n", ".", " ", ""]`) up to `chunk_size` character limits with sliding overlap (`RChunk`).

---

## Classes & Public APIs

### `class Chunker` (`chunker.py`)
The high-level routing controller that analyzes `NormalizedContent` lists and delegates items to either hierarchical or recursive processing.

#### Constructor: `__init__(self, chunk_size: int = 256, overlap: int = 20, db_path: str = Config.DB_PATH) -> None`
Initializes window sizes and sets up the SQLite database path for hierarchical relationship persistence.

#### Methods

##### `chunk_per_document(self, normalised_content: List[NormalizedContent]) -> Tuple[List[HChunk], List[RChunk]]`
Splits normalized documents by structural criteria (`has_section`) and invokes their respective chunking engines.

###### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `normalised_content` | `List[NormalizedContent]` | List of sanitized document objects returned by `TextNormalizer`. |

###### Return Value
- **Type:** `Tuple[List[HChunk], List[RChunk]]`
- **Description:** A 2-tuple `(h_chunks, r_chunks)` containing the lists of generated hierarchical `HChunk` objects and recursive `RChunk` objects.

###### How It Works
1. Iterates through `normalised_content`. If `content.has_section is True`, appends to `hierarchical_chunker_list`; otherwise appends to `recursive_chunker_list`.
2. Passes `hierarchical_chunker_list` to `HierarchicalChunker(overlap, chunk_size, db_path, ...).process_doc()`.
3. Passes `recursive_chunker_list` to `RecursiveChunker(..., chunk_size, overlap).recursive_chunker()`.
4. Returns the combined `(h_chunks, r_chunks)` tuple.

---

### `class HierarchicalChunker` (`HierarchicalChunker.py`)
Executes three-tier hierarchical document decomposition (`Document` $\rightarrow$ `Section` $\rightarrow$ `Context` $\rightarrow$ `HChunk`) and stores structural records in SQLite.

#### Constructor: `__init__(self, chunkOverlap: int, chunkSize: int, db_path: str, normalizedDocumentsContents: List[NormalizedContent]) -> None`
Initializes window settings, target database path, and document lists.

#### Methods

##### `process_doc(self) -> List[HChunk]`
Orchestrates document structuring, database table creation/insertion, and chunk slicing across all assigned documents.

###### Parameters
*None.*

###### Return Value
- **Type:** `List[HChunk]`
- **Description:** Complete list of `HChunk` instances across all hierarchical documents.

###### How It Works
1. Converts `NormalizedContent` items into `Document` node dataclasses (`__make_document_objs()`).
2. Opens `Manager(self.db_path, is_chunker_type_hierarchical=True)` to initialize `Documents`, `Sections`, `Contexts`, and `Chunks` tables.
3. For each document:
   - Finds uppercase section headers (`re.compile(r"^[A-Z\s]+$", re.MULTILINE)`) and constructs `Section` nodes (`__find_sections(doc)`).
   - Inserts `Section` nodes into SQLite via `h_manager.insert_sections(...)`.
   - For each section, splits double-newline boundaries (`r"\n\s*\n"`) into paragraph `Context` nodes (`__find_contexts(sections)`).
   - Inserts `Context` nodes into SQLite via `h_manager.insert_contexts(...)`.
   - For each `Context`, applies sliding character window slicing (`chunkSize`, `chunkOverlap`) to generate `HChunk` nodes (`__get_chunks(contexts, doc)`).
   - Inserts `HChunk` nodes into SQLite (`h_manager.insert_chunks(...)`).
4. Closes database connection (`h_manager.close()`) and returns all compiled `HChunk` objects.

---

### `class RecursiveChunker` (`RecursiveChunker.py`)
Performs top-down separator splitting on unstructured text strings until segments fit within `chunk_size` bounds.

#### Constructor: `__init__(self, normalized_documents_contents: List[NormalizedContent], chunk_size: int, overlap: int = 20, separator: Optional[List[str]] = None) -> None`
Initializes maximum character limits (`chunk_size`), sliding window overlap (`overlap`), and separator hierarchy (default: `["\n\n", "\n", ".", " ", ""]`).

#### Methods

##### `recursive_chunker(self) -> List[RChunk]`
Iterates across all input documents, invokes recursive separator splitting, and wraps outputs in `RChunk` objects.

###### Parameters
*None.*

###### Return Value
- **Type:** `List[RChunk]`
- **Description:** Complete list of recursive chunks generated across all assigned documents.

###### How It Works
1. For each `NormalizedContent` document string, calls internal helper `__r_chunker(content, self.default_separators)`.
2. Inside `__r_chunker`:
   - If string length $\le$ `chunk_size`, returns the string as a single-element list.
   - Finds the largest separator (`\n\n`, `\n`, etc.) present in the text.
   - Splits text using that separator. If merged segments exceed `chunk_size`, recursively invokes `__r_chunker(piece, remaining_separator)` on oversized pieces.
   - Applies character overlap (`prev_tail + chunks[i]`) across adjacent chunk slices (`self.overlap > 0`).
3. Wraps resulting text strings in `RChunk(chunk, ChunkMetaData(...), chunk_id)` and returns them.

---

### `class Manager` (`DB_Manager.py`)
Relational SQLite database manager that creates tables and handles transactional batch insertions for hierarchical nodes (`Document`, `Section`, `Context`, `HChunk`).

#### Constructor: `__init__(self, db_path: str, is_chunker_type_hierarchical: bool) -> None`
Opens an SQLite connection (`sqlite3.connect(db_path)`), enables foreign key enforcement (`PRAGMA foreign_keys = ON;`), and creates relational tables if `is_chunker_type_hierarchical=True`.

#### Relational Schema
- **`Documents`:** `(documentId TEXT PRIMARY KEY, documentName TEXT)`
- **`Sections`:** `(sectionId TEXT PRIMARY KEY, documentId TEXT FK, sectionName TEXT, content TEXT, contentLength INT, startoffset INT, endoffset INT)`
- **`Contexts`:** `(contextId TEXT PRIMARY KEY, sectionId TEXT FK, context TEXT, contextLength INT, startoffset INT, endoffset INT)`
- **`Chunks`:** `(chunkId TEXT PRIMARY KEY, contextId TEXT FK, chunk TEXT, startoffset INT, endoffset INT)`

#### Public Methods
| Method | Parameters | Return / Behavior | Description |
| :--- | :--- | :--- | :--- |
| `insert_documents` | `Documents: List[Document]` | `None` | Executes transactional batch insertion (`BEGIN IMMEDIATE`) into `Documents` table. Raises `InsertionError` and rolls back on failure. |
| `insert_sections` | `Sections: List[Section]` | `None` | Executes transactional batch insertion into `Sections` table. |
| `insert_contexts` | `Contexts: List[Context]` | `None` | Executes transactional batch insertion into `Contexts` table. |
| `insert_chunks` | `Chunks: List[HChunk]` | `None` | Executes transactional batch insertion into `Chunks` table. |
| `get_section_from_context` | `sectionId: str` | `List[tuple]` | Queries all columns from `Sections` where `sectionId = ?`. |
| `get_context_from_chunk` | `contextId: str` | `List[tuple]` | Queries all columns from `Contexts` where `contextId = ?`. |
| `get_chunk` | `chunkId: str` | `List[tuple]` | Queries all columns from `Chunks` where `chunkId = ?`. |
| `get_document_from_section` | `documentId: str` | `List[tuple]` | Queries all columns from `Documents` where `documentId = ?`. |
| `close` | *None* | `None` | Closes `self.cursor` and `self.connection`. |
