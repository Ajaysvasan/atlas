# Text File Processing Layer (`FileLoader` & `TextExtractor`)

## Overview & Purpose
The `TextFileProcessor` submodule is responsible for scanning local file directories, categorizing documents by extension, and converting binary/structured document formats (`.pdf`, `.docx`, `.doc`, `.txt`, `.md`) into raw Unicode text strings ready for downstream normalization and chunking.

---

## Classes & Public APIs

### `class FileLoader` (`file_loader.py`)
Scans target directories on the filesystem and returns categorized dictionary maps of absolute `Path` objects segregated by their file extensions.

#### Constructor: `__init__(self) -> None`
Initializes the allowed extensions set (`{".doc", ".docx", ".txt", ".pdf", ".csv", ".md", ".html", ".xml"}`) and prepares internal directory iterators.

#### Methods

##### `load_files(self, folder_path: Union[str, Path]) -> Dict[str, List[Path]]`
Recursively traverses a directory structure (`os.walk`) and collects all supported files grouped by their extension.

###### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `folder_path` | `Union[str, Path]` | The absolute or relative directory path to scan for input files. |

###### Return Value
- **Type:** `Dict[str, List[Path]]`
- **Description:** A dictionary keyed by file extension string without leading dot (`"pdf"`, `"docx"`, `"txt"`, `"md"`, `"doc"`, `"csv"`, etc.), where each value is a list of resolved `Path` objects pointing to matched files.

###### How It Works
1. Validates that `folder_path` exists and is a directory (`path_obj.exists() and path_obj.is_dir()`). Raises `NotADirectoryError` or `FileNotFoundError` if invalid.
2. Initializes an empty collection dictionary: `loaded_files = {ext[1:]: [] for ext in self.allowed_extensions}`.
3. Walks every subdirectory and checks file suffixes against `self.allowed_extensions`.
4. Appends matching absolute `Path` objects to their respective extension buckets and logs total count summaries (`logger.info(...)`).

---

### `class TextExtractor` (`text_extractor.py`)
Converts raw files identified by `FileLoader` into plain string contents using format-specific parsing libraries (`pypdf`, `python-docx`, `antiword`).

#### Constructor: `__init__(self) -> None`
Initializes extraction engine settings and logger instances (`get_logger(__name__)`).

#### Methods

##### `extract_text_from_file(self, file_path: Union[str, Path]) -> Tuple[str, str]`
Inspects a single file's extension and routes it to the appropriate extraction method.

###### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `file_path` | `Union[str, Path]` | Path to the specific document file to extract text from. |

###### Return Value
- **Type:** `Tuple[str, str]`
- **Description:** A 2-tuple `(file_path_str, extracted_text_str)` containing the absolute string path and the extracted raw text string.

###### How It Works
1. Resolves `file_path` and checks `suffix.lower()`.
2. Routes to internal handlers based on extension:
   - **`.txt` / `.md`:** Reads directly using standard Python file input/output (`open(..., "r", encoding="utf-8", errors="ignore")`).
   - **`.docx`:** Uses `docx.Document(file_path)` and joins paragraph texts with newline characters (`"\n".join([p.text for p in doc.paragraphs])`).
   - **`.pdf`:** Uses `pypdf.PdfReader` to iterate through all pages and extract text blocks (`page.extract_text()`).
   - **`.doc`:** Executes external Linux utility `antiword` via `subprocess.run(["antiword", str(file_path)], capture_output=True)` to decode legacy binary Word documents.
3. Raises custom `InvalidFileType` exception if the file suffix is unsupported (`.csv`, `.html`, `.xml`, etc.).

---

##### `extract_all(self, loaded_files: Dict[str, List[Path]]) -> Dict[str, str]`
Batch-processes an entire dictionary of categorized `Path` lists returned by `FileLoader.load_files()`.

###### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `loaded_files` | `Dict[str, List[Path]]` | Dictionary mapping extensions to lists of document paths. |

###### Return Value
- **Type:** `Dict[str, str]`
- **Description:** A flat dictionary mapping each absolute file path (`str`) to its complete extracted raw text string (`str`).

###### How It Works
1. Iterates through all `(extension, path_list)` pairs in `loaded_files`.
2. Calls `extract_text_from_file(path)` for each file inside the list.
3. Catches individual file extraction exceptions, logs warnings (`logger.warning(...)`), and continues processing remaining files without terminating the entire batch job.
4. Returns the compiled dictionary of successfully extracted texts.
