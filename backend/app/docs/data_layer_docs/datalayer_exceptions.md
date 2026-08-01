# Data Layer Exceptions Module (`datalayer_exceptions.py`)

## Overview & Purpose
The `datalayer_exceptions.py` module defines a suite of custom domain-specific exception classes that provide expressive error handling across file loading, text extraction, normalizer type checks, vector database operations, and SQLite relational insertions.

---

## Classes & Public APIs

### `class InvalidFileType(Exception)`
Raised when `TextExtractor` encounters a file suffix that is unsupported by its internal extraction engines (`.csv`, `.html`, `.xml`, or unknown extensions).

#### Constructor: `__init__(self, file_type: str) -> None`
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `file_type` | `str` | The invalid file extension string (e.g. `".csv"`). |

#### String Representation (`__str__`)
Returns: `"The given file type: '{self.file_type}' is not supported"`

---

### `class VectorInsertionError(Exception)`
Raised when an error occurs during single (`insert`) or batch (`batch_insert`) vector insertions inside `VectorDb_diskann`.

#### Constructor: `__init__(self, exception_type: Exception, is_batch: bool) -> None`
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `exception_type` | `Exception` | The original underlying exception raised by `diskannpy`. |
| `is_batch` | `bool` | `True` if the failure occurred during `batch_insert()`, `False` if during `insert()`. |

#### String Representation (`__str__`)
Returns: `"Error during {'Batch ' if self.is_batch else ''}Insertion of vector due to type {type(self.exception_type).__name__}. Exception Details: {self.exception_type}"`

---

### `class IndexDirectoryDoesNotExists(Exception)`
Raised by `VectorDb_diskann.load()` when the target directory or required DiskANN binary files (`diskann_dynamic.data`, `diskann_dynamic.search_graph`, etc.) do not exist on disk.

#### Constructor: `__init__(self, directory_name: str) -> None`
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `directory_name` | `str` | Path to the missing index directory. |

#### String Representation (`__str__`)
Returns: `"The directory with the following name doesn't exists"`

---

### `class InsertionError(Exception)`
Raised when an SQLite relational insertion query (`INSERT INTO Documents/Sections/Contexts/Chunks`) fails within `DB_Manager`.

#### Constructor: `__init__(self, exception_type: Exception, table_name: str, primary_key: str) -> None`
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `exception_type` | `Exception` | The underlying SQLite error (`sqlite3.OperationalError`, `sqlite3.IntegrityError`, etc.). |
| `table_name` | `str` | Name of the table where insertion failed (`"Documents"`, `"Sections"`, `"Contexts"`, or `"Chunks"`). |
| `primary_key` | `str` | The ID value (`documentId`, `chunkId`, etc.) of the row being inserted when failure occurred. |

#### String Representation (`__str__`)
Returns: `"Insertion into Table {self.table_name} failed for primary key: {self.primary_key}. Exception Details: {type(self.exception_type).__name__}: {self.exception_type}"`

---

### `class InvalidEmbeddingArgument(Exception)`
Raised by `EmbeddingManager.embed()` when the passed argument is neither an `HChunk`, an `RChunk`, nor a `List` of chunks.

#### Constructor: `__init__(self, error_message: str) -> None`
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `error_message` | `str` | Detailed error message describing the received invalid type vs expected types. |

#### String Representation (`__str__`)
Returns: `{self.error_message}`
