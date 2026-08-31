# Chunking Algorithms & Storage Layer (`Chunker/`)

## Overview & Purpose
The `Chunker` submodule partitions normalized document text into smaller segments suitable for vector embedding and approximate nearest neighbour search. It routes documents on `has_section`:

1. **Hierarchical Chunking (`HierarchicalChunker`)** — structured documents, decomposed `Document → Section → Context → HChunk`, with the tree persisted in SQLite (`DB_Manager`).
2. **Recursive Chunking (`RecursiveChunker`)** — flat documents, split along a separator hierarchy down to `chunk_size`, with overlap.

Both share `windowing.sliding_windows()` for the final character-level split.

---

## Invariants

These hold for every chunk either chunker emits, and are covered by
`test/data_layer_testing/test_ingestion.py`.

| Invariant | Why it matters |
| :--- | :--- |
| `len(chunk) <= chunk_size` | The embedding model has a fixed input budget. |
| `content[start_off_set:end_off_set] == chunk` | Offsets are **absolute into the normalized document**, so a retrieved chunk can be traced back to its source. They were previously relative to the parent context or section, which made them unresolvable. |
| Chunk ids are unique within a document | Ids bind position as well as content. Hashing content alone gave one id to every repeated paragraph. |
| Words are not cut in half | `sliding_windows` retreats to the last whitespace in the window. |
| Re-ingesting unchanged input is a no-op | All inserts are `on conflict … do nothing`. |

---

## `windowing.py`

### `sliding_windows(text: str, size: int, overlap: int) -> List[Tuple[int, int]]`
Returns the offsets of successive windows over `text`, none longer than `size`.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `text` | `str` | The text to window. |
| `size` | `int` | Maximum window length in characters. Must be positive. |
| `overlap` | `int` | Characters each window extends back into the previous one. Must satisfy `0 <= overlap < size`. |

Raises `ValueError` for impossible geometry rather than looping or emitting oversized windows.

#### How It Works
1. Skips leading whitespace so no window starts mid-gap.
2. Takes `end = min(start + size, len(text))`, then retreats to the last whitespace character in `[start, end)` so words survive. A run with no whitespace in it — a base64 blob, a minified line — has nothing to retreat to and is split at exactly `size`.
3. Advances to `max(end - overlap, start + 1)`. The `max()` is load-bearing: a window pulled back to a word boundary can be *shorter* than the overlap, and `end - overlap` would then step backwards forever.

---

## Classes & Public APIs

### `class Chunker` (`chunker.py`)
Routing controller that splits a `NormalizedContent` list by structure and delegates.

#### Constructor: `__init__(self, chunk_size: int = 256, overlap: int = 20, db_path: str = Config.DB_PATH) -> None`

#### Methods

##### `chunk_per_document(self, normalised_content: List[NormalizedContent]) -> Tuple[List[HChunk], List[RChunk]]`

###### Return Value
- **Type:** `Tuple[List[HChunk], List[RChunk]]`

###### How It Works
1. Partitions on `content.has_section`.
2. Invokes each chunker **only when its list is non-empty**, so an all-flat batch never opens a SQLite connection it has nothing to write to.
3. Returns `(h_chunks, r_chunks)`.

##### `chunk_document(self, normalised_content: NormalizedContent) -> List[HChunk] | List[RChunk]`
Single-document convenience wrapper, routed identically. Backs `IngestionPipeline.chunk_text()`.

---

### `class HierarchicalChunker` (`HierarchicalChunker.py`)

#### Constructor: `__init__(self, chunkOverlap: int, chunkSize: int, db_path: str, normalizedDocumentsContents: List[NormalizedContent]) -> None`
Validates the window geometry up front — `chunkSize > 0` and `0 <= chunkOverlap < chunkSize` — raising `ValueError` rather than deferring the check to slicing time.

#### Methods

##### `process_doc(self) -> List[HChunk]`

###### Return Value
- **Type:** `List[HChunk]`

###### How It Works
1. Returns `[]` immediately for an empty document list, without creating a database.
2. Converts `NormalizedContent` items into `Document` nodes.
3. Opens `Manager(db_path, is_chunker_type_hierarchical=True)` inside a `try` whose `finally` closes it — the connection previously leaked whenever any step raised.
4. For each document:
   - `__find_sections(doc, spans)` converts the normalizer's `SectionSpan` list into `Section` nodes. **No heading regex runs here.** Text before the first heading becomes a `PREAMBLE` section; a document with no spans becomes a single `MAIN` section.
   - `__find_contexts(sections)` splits each section on `r"\n\s*\n"` into paragraph `Context` nodes, converting section-relative indices to document-absolute offsets.
   - `__get_chunks(...)` windows each context through `sliding_windows` and attaches the owning section name to `ChunkMetaData.section_name`.
5. Returns every `HChunk`.

###### Identifier construction
| Node | Id derived from | Reason |
| :--- | :--- | :--- |
| `Section` | `(documentId, ordinal, name)` | A document with two `NOTES` headings previously hashed both to one `sectionId` and aborted the run on the primary key. |
| `Context` | `(sectionId, ordinal, text)` | Same failure for a repeated paragraph. |
| `HChunk` | `(contextId, ordinal, chunk)` | Same failure for a repeated window. |

---

### `class RecursiveChunker` (`RecursiveChunker.py`)

#### Constructor: `__init__(self, normalized_documents_contents: List[NormalizedContent], chunk_size: int, overlap: int = 20, separator: Optional[List[str]] = None, db_path: Optional[str] = None) -> None`
Validates window geometry as above. `db_path` is optional; when supplied, chunks are persisted to the `RecursiveChunks` table.

Default separators: `["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]`. The sentence separators carry their trailing space so a split lands between sentences rather than inside an abbreviation, and the empty-string terminator is gone — `sliding_windows` is the base case now.

#### Methods

##### `recursive_chunker(self) -> List[RChunk]`

###### Return Value
- **Type:** `List[RChunk]`

###### How It Works
1. For each document, `__split(content, 0, separators)` returns **absolute `(start, end)` spans**, not strings, so offsets stay exact.
2. Inside `__split`:
   - Text at or under `chunk_size` is returned as one span.
   - The first separator present in the text is chosen; `__separator_pieces` yields offsets in which each piece keeps its *own* separator and the pieces tile the text exactly. The previous implementation rebuilt pieces as `part + separator`, which appended a separator the document never had to the final piece.
   - Pieces are accumulated up to `chunk_size`; any single piece exceeding it recurses on the remaining separators. Exhausting the separators falls through to `sliding_windows`.
3. `__apply_overlap()` runs **once**, over the finished spans. It previously ran inside the recursion, applying again at every level and copying text near a boundary into a chunk several times over. Each chunk extends back into the *original* previous span, so overlap does not compound.
4. Spans are materialised into `RChunk` objects whose `chunk_id` binds `(document_id, ordinal, chunk)`.
5. When `db_path` is set, the parent `Document` rows are written before the chunks, satisfying the foreign key.

---

### `class Manager` (`DB_Manager.py`)
SQLite manager that creates tables and performs batch insertion for both chunker paths.

#### Constructor: `__init__(self, db_path: str, is_chunker_type_hierarchical: bool) -> None`
Creates the parent directory, connects with `check_same_thread=False`, enables `PRAGMA foreign_keys = ON`, and creates the tables for the selected path. `Documents` is created in **both** modes; the `False` branch previously created no tables at all, so every insert against it failed with "no such table".

#### Relational Schema
Hierarchical path:
- **`Documents`:** `(documentId TEXT PRIMARY KEY, documentName TEXT)`
- **`Sections`:** `(sectionId TEXT PRIMARY KEY, documentId TEXT FK, sectionName TEXT, content TEXT, contentLength INT, startoffset INT, endoffset INT)`
- **`Contexts`:** `(contextId TEXT PRIMARY KEY, sectionId TEXT FK, context TEXT, contextLength INT, startoffset INT, endoffset INT)`
- **`Chunks`:** `(chunkId TEXT PRIMARY KEY, contextId TEXT FK, chunk TEXT, startoffset INT, endoffset INT)`

Recursive path:
- **`Documents`:** as above
- **`RecursiveChunks`:** `(chunkId TEXT PRIMARY KEY, documentId TEXT FK, chunk TEXT, startoffset INT, endoffset INT)`

Both sets coexist in one file, which is what `Chunker` produces when a batch contains documents of both kinds.

#### Insertion semantics
All four `insert_*` methods issue a single `executemany` with `on conflict(<pk>) do nothing`, commit, and wrap any failure in `InsertionError` after a rollback. Ids bind content *and* position, so a conflict means the identical row is already stored — which is exactly what re-ingesting an unchanged folder produces. A plain `INSERT` previously aborted the second run on `UNIQUE constraint failed: Documents.documentId`.

#### Public Methods
| Method | Parameters | Description |
| :--- | :--- | :--- |
| `insert_documents` | `Documents: List[Document]` | Batch insert into `Documents`. |
| `insert_sections` | `Sections: List[Section]` | Batch insert into `Sections`. |
| `insert_contexts` | `Contexts: List[Context]` | Batch insert into `Contexts`. |
| `insert_chunks` | `Chunks: List[HChunk]` | Batch insert into `Chunks`. |
| `insert_recursive_chunks` | `Chunks: List[RChunk]` | Batch insert into `RecursiveChunks`. |
| `get_section_from_context` | `sectionId: str` | Rows from `Sections` where `sectionId = ?`. |
| `get_context_from_chunk` | `contextId: str` | Rows from `Contexts` where `contextId = ?`. |
| `get_chunk` | `chunkId: str` | Rows from `Chunks` where `chunkId = ?`. |
| `get_document_from_section` | `documentId: str` | Rows from `Documents` where `documentId = ?`. |
| `close` | *None* | Closes `self.cursor` and `self.connection`. |
