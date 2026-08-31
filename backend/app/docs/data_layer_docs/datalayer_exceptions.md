# Data Layer Exceptions Module (`datalayer_exceptions.py`)

## Overview & Purpose
Twelve domain exceptions covering file loading, text extraction, chunk persistence, embedding arguments, and both vector stores.

> **Naming.** Some names here are misleading (`VectorNotFoundEror`, `InsertionError.message` holding a *table name*). They are documented as they are, because renaming them is a breaking change for every `except` clause in the codebase. See the note at the end.

---

## Ingestion

### `class InvalidFileType(Exception)`
Raised by `TextExtractor` when a file's **bytes are not text** — a NUL byte in the first 8 KB, or a decode that no encoding in the ladder can complete.

An unrecognised *extension* no longer raises this: unknown extensions fall through to the plain-text reader, so `.rs`, `.tex`, `.log` and anything else are read normally. `.csv`, `.html` and `.xml`, which older versions of this document listed as unsupported, all have working readers.

| Constructor | `__init__(self, file_extention) -> None` |
| :--- | :--- |
| `file_extention` | The file's suffix, or the literal `"binary content"` for an extensionless file. Stored as `self.file_extention`. |

**`__str__`** → `Unsupported file type : {file_extention}`

---

### `class InvalidEmbeddingArgument(Exception)`
Raised by `EmbeddingManager` when `embed()` receives something that is not an `HChunk`, `RChunk` or list of them; when `embed_text()` receives a non-string or blank string; and when `embed_texts()` receives `chunk_ids` that do not align with `texts`.

| Constructor | `__init__(self, error_message) -> None` |
| :--- | :--- |
| `error_message` | Message naming the received type against the expected ones. |

**`__str__`** → the message verbatim.

---

## Chunk persistence

### `class InsertionError(Exception)`
Raised by `DB_Manager` when a batch insert into `Documents`, `Sections`, `Contexts`, `Chunks` or `RecursiveChunks` fails. The connection is rolled back before it is raised.

| Constructor | `__init__(self, error, tableName, id) -> None` |
| :--- | :--- |
| `error` | The underlying `sqlite3` error. Stored as `self.error`. |
| `tableName` | Table the insert targeted. **Stored as `self.message`**, not `self.table_name`. |
| `id` | Primary key of the first row in the failing batch. Stored as `self.id`. Since the failure aborts the whole `executemany`, this identifies the batch, not necessarily the offending row. |

**`__str__`** → `{error}:Error occured while inserting values in the following table {message} , for the id : {id}`

---

## DiskANN

### `class VectorInsertionError(Exception)`
Raised when a vector write fails, in either store.

| Constructor | `__init__(self, vector_id, cause: BaseException \| None = None) -> None` |
| :--- | :--- |
| `vector_id` | The id, or the list of ids for a batch. Stored as `self.vector_id`. |
| `cause` | The underlying driver exception. Stored as `self.cause`; also chained with `raise … from`, so `__cause__` and the original traceback are reachable. |

**`__str__`** → `An Error occured while inserting the vector : {ids}`, plus `. Caused by {type}: {message}` when a cause is present. A batch of more than `MAX_IDS_SHOWN` (5) ids renders as `1, 2, 3, 4, 5, ... (4000 ids)` rather than listing every one.

Raised by `VectorDb_diskann.insert` and `batch_insert`, and by `VectorRepository`'s insert, batch-insert, update and delete paths. Every site passes the id (or ids) as `vector_id` and the driver error as `cause`.

> **Previously**, `vector_id` held the id when raised from DiskANN and the psycopg exception object when raised from the repository, so anything reading the attribute got one or the other. And `VectorDb_diskann.batch_insert` wrapped nothing, so `except VectorInsertionError` around a batch insert caught no failure at all. Both are fixed; the tests are in `test/data_layer_testing/test_vector_store_errors.py`.

---

### `class IndexDirectoryDoesNotExists(Exception)`
Raised by `VectorDb_diskann.load()` when the index **directory** does not exist. The individual DiskANN files inside it are not checked. `VectorDbManager.load()` catches this and returns `None`.

| Constructor | `__init__(self, directory_name) -> None` |
| :--- | :--- |

**`__str__`** → `The directory with the following name doesn't exist: {directory_name}`

---

## pgvector store

### `class MissingDatabaseConfiguration(Exception)`
Raised by `VectorRepository.__init__` when any of `DBNAME`, `DB_USER`, `PASSWORD`, `HOST`, `PORT` is absent from the environment.

| Constructor | `__init__(self, missing_keys) -> None` |
| :--- | :--- |
| `missing_keys` | Iterable of the absent key names; stored as a `list`. |

**`__str__`** names the missing keys and explains that they are not optional: `psycopg` substitutes libpq's defaults for anything left unset — including the OS username — so an unset key silently connects somewhere unintended rather than failing.

---

### `class VectorNotFoundEror(Exception)`
*(Spelling is the class's own.)* Raised by `VectorRepository.search`, `batch_search`, and by `update` when the `UPDATE` matches no row — an update that matches nothing is not an error to `psycopg`, so without this check a write to a missing id would report success.

| Constructor | `__init__(self, vector_id: uint32) -> None` |
| :--- | :--- |

**`__str__`** → `No vectors found for the vector Id : {vector_id}`

---

### `class InvalidVectorDimension(Exception)`
Raised by `VectorRepository` when a vector's length differs from `Config.EMBEDDING_DIMENSIONS`.

| Constructor | `__init__(self, passed_dimension: int, expected_dimension: int) -> None` |
| :--- | :--- |

**`__str__`** → `Expected dimension {expected_dimension} , got {passed_dimension}`

> A **different** `InvalidVectorDimension` exists in `memory/memory_pool_exceptions.py`. They are unrelated classes with the same name; `except` clauses must import from the right module.

---

### `class InvalidBatchSize(Exception)`
Raised when parallel lists do not line up — `vectors` against `vector_ids` in `VectorRepository.batch_insert`, and `vectorIds` against `chunkIds` in `VectorMetaDataRepository.batch_insert`.

| Constructor | `__init__(self, error_message) -> None` |
| :--- | :--- |

**`__str__`** → the message verbatim.

---

### `class DuplicateVectorException(Exception)`
Raised by `VectorRepository.insert` when the project already stores that id — a `psycopg.errors.UniqueViolation` on the `(project_id, vector_id)` primary key.

| Constructor | `__init__(self, vector_id: uint32) -> None` |
| :--- | :--- |

**`__str__`** → `The vector with vector id {vector_id} , already exists`

It is deliberately **not** a `VectorInsertionError`: the vectors-first snapshot path needs to tell "already written" (carry on) from "the write failed" (compensate), and folding both into one exception made that undecidable. `batch_insert` never raises it — that path is `on conflict … do nothing` by design and tolerates duplicates silently.

---

## Vector metadata sidecar

### `class InvalidColumnNameException(Exception)`
Raised by `VectorMetaDataRepository.get_meta_data` when `columnName` is not one of `("vectorId", "chunkId", "embeddingModelUsed", "dimensions")`. The column name is interpolated into the SQL string, so this allowlist is what keeps that query safe — the check is a security control, not a convenience.

| Constructor | `__init__(self, columnName: str)` |
| :--- | :--- |

**`__str__`** → `Got invalid column name : {columnName}`

---

### `class InvalidVectorID(Exception)`
Raised by `VectorMetaDataRepository.get_meta_data` when no metadata row matches the requested `vectorId`. Despite the name, it signals a **missing** row, not a malformed id.

| Constructor | `__init__(self, vectorID) -> None` |
| :--- | :--- |
| `vectorID` | Stored as `self.vectorId` — note the differing capitalisation between parameter and attribute. |

**`__str__`** → `No vector meta data found for the vector id : {vectorId}`

> Not to be confused with `memory/memory_pool_exceptions.py::InvalidVectorId`, which is a range check on ids and does have a descriptive message.

---

## Known naming problems

Recorded rather than fixed, because each rename breaks existing `except` clauses:

| Symbol | Problem |
| :--- | :--- |
| `VectorNotFoundEror` | Misspelled (`Eror`). |
| `InvalidVectorID` | Name says invalid id; the condition is a missing row. The message now states the condition, but the class name still does not. |
| `InsertionError.message` | Attribute named `message` holds the table name. |
| `InvalidVectorDimension` | Name collides with an unrelated class in `memory/memory_pool_exceptions.py`. |
