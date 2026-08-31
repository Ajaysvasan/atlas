# Text File Processing Layer (`FileLoader` & `TextExtractor`)

## Overview & Purpose
The `TextFileProcessor` submodule scans local directories and converts whatever it finds into plain Unicode text for downstream normalization and chunking.

The design goal is **coverage**: a document should not be dropped because nobody added its extension to a list. Both classes therefore work by exclusion rather than inclusion — `FileLoader` skips known binary formats and offers everything else to `TextExtractor`, which has dedicated readers for the formats worth parsing structurally and decodes the rest as text.

---

## Classes & Public APIs

### `class FileLoader` (`file_loader.py`)
Scans a directory tree and returns `Path` objects grouped by extension.

#### Constructor: `__init__(self, allowed_extensions=None, excluded_extensions=None, max_file_size=DEFAULT_MAX_FILE_SIZE) -> None`

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `allowed_extensions` | `Optional[Iterable[str]]` | `None` | Escape hatch. When supplied, **only** these extensions are loaded, restoring allowlist behaviour. |
| `excluded_extensions` | `Optional[Iterable[str]]` | `NON_DOCUMENT_EXTENSIONS` | Denylist of formats that hold no extractable text — images, audio, video, archives, executables, fonts, model weights, pickles, compiled artefacts. |
| `max_file_size` | `int` | `64 * 1024 * 1024` | Files larger than this are logged and skipped, so a stray multi-gigabyte log cannot exhaust memory. |

`IGNORED_DIRECTORIES` additionally skips VCS metadata, `node_modules`, `__pycache__`, virtualenvs, caches and build output.

#### Methods

##### `load_files(self, folder_path: Union[str, Path]) -> Dict[str, List[Path]]`
Recursively traverses a directory and collects candidate documents.

###### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `folder_path` | `Union[str, Path]` | Directory to scan. |

###### Return Value
- **Type:** `Dict[str, List[Path]]`
- **Description:** Dictionary keyed by extension without the leading dot (`"pdf"`, `"rs"`, `"tex"`, …), or `"noext"` for extensionless files such as `Makefile`. **Only non-empty categories are present** — the map was previously pre-seeded with an empty list for every known extension.

###### Raises
- `ValueError` if `folder_path` does not exist or is not a directory.

###### How It Works
1. Validates the path.
2. `__scan_directory` recurses, maintaining a set of `os.path.realpath()` values already visited. A directory whose resolved path has been seen is skipped.
3. For each file: rejects dotfiles, empty files, oversized files, and extensions in the denylist; everything else is appended to its category.
4. Logs a total and returns.

> **On the visited set.** A symlink pointing at an ancestor previously recursed until Python's stack limit. The resulting `RecursionError` was caught by a broad `except Exception` and logged as though the folder were merely unreadable — so the scan returned a *partial* tree and reported success. The exception handler is now narrowed to `PermissionError` and `OSError`, which cannot mask a control-flow bug this way.

---

### `class TextExtractor` (`text_extractor.py`)
Converts a file of any type into plain text.

#### Constructor: `__init__(self) -> None`
Builds `self._handlers`, the extension-to-method registry. All third-party imports are performed lazily inside their handler, so the module imports cleanly on a machine missing any of them.

#### Format coverage

| Extension | Handler | Notes |
| :--- | :--- | :--- |
| `.pdf` | `_extract_from_pdf` | PyPDF2. `page.extract_text()` returns `None` for image-only pages — coerced to `""`, which the `"\n".join` previously raised `TypeError` on. |
| `.docx` | `_extract_from_docx` | python-docx. Emits paragraphs **and table rows** (`" \| "`-joined); tables were previously dropped entirely. Word heading styles become markdown `#`. |
| `.pptx` | `_extract_from_pptx` | python-pptx. Slide titles become `# ` headings. |
| `.xlsx`, `.xlsm` | `_extract_from_xlsx` | openpyxl if installed, otherwise falls through to textract. |
| `.html`, `.htm`, `.xhtml`, `.xml` | `_extract_from_markup` | BeautifulSoup. `<script>`, `<style>`, `<noscript>` removed; `<h1>`–`<h6>` become markdown headings. |
| `.json` | `_extract_from_json` | Flattened to `a.b[0]: value` lines so keys stay attached to the values they label after chunking. Invalid JSON falls back to plain text. |
| `.ipynb` | `_extract_from_notebook` | Markdown and code cell sources, blank line separated. |
| `.doc`, `.odt`, `.odp`, `.ods`, `.rtf`, `.epub`, `.xls` | `_extract_from_binary_document` | textract. |
| **everything else** | `_extract_from_txt` | Source code, logs, configuration, TeX, CSV, Markdown, subtitles, and any format not listed above. |

> **On heading injection.** `.docx`, `.pptx` and HTML carry structure that has no textual syntax of its own. Rendering it as markdown `#` is what lets `TextNormalizer` detect sections in those formats; without it every Word document and web page routed to the flat chunker.

#### Methods

##### `extract_text_from_file(self, file_path: Union[str, Path]) -> Tuple[str, str]`

###### Return Value
- **Type:** `Tuple[str, str]`
- **Description:** `(file_path_str, extracted_text)`. The path is always a `str` — it was previously whatever was passed in, so a `PosixPath` from `FileLoader` propagated into the normalizer.

###### Raises
- `FileNotFoundError` if the path does not exist.
- `InvalidFileType` if the bytes are not text (see below). An unrecognised *extension* is no longer an error.

##### `_decode(self, raw: bytes, file_path: str) -> str`
The encoding ladder shared by every text-based handler.

1. **Byte order mark** — if present, decode with the encoding it names (`utf-8-sig`, `utf-16`, `utf-32`).
2. **NUL sniff** — a `\x00` in the first 8 KB means binary; raise `InvalidFileType`.
3. **`utf-8`**.
4. **chardet**, accepted only at confidence ≥ 0.7.
5. **`latin-1`** with `errors="replace"`, which cannot fail.

> **Why the BOM gate.** `utf-16` accepts almost any even-length byte string. Attempting it unconditionally decoded the Latin-1 bytes for `café naïve` into `慣渠敶` and raised nothing at all.

##### `extract_all(self, loaded_files: Dict[str, List[Path]]) -> Dict[str, str]`

###### Return Value
- **Type:** `Dict[str, str]`
- **Description:** Maps each file path (`str`) to its extracted text. Files that yield no text, and files that raise `InvalidFileType`, `FileNotFoundError` or `OSError`, are logged and skipped rather than aborting the batch.
