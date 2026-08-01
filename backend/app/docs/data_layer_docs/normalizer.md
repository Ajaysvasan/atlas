# Text Normalizer Module (`normalizer.py`)

## Overview & Purpose
The `normalizer.py` module cleans, standardizes, and structures raw document strings extracted by `TextExtractor`. It applies configurable regex transformation pipelines to strip extraneous noise (such as URLs, email addresses, excessive whitespace, or broken line breaks) and determines whether documents exhibit hierarchical section structuring (`has_section`).

---

## Classes & Public APIs

### `class TextNormalizer`
A configurable text sanitization pipeline engine that processes strings and wraps them inside immutable `NormalizedContent` objects.

#### Constructor: `__init__(self, ...)`
Configures boolean flag switches governing text transformations.

##### Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `lowercase` | `bool` | `True` | Converts the entire text string to lower case (`text.lower()`). |
| `remove_extra_whitespace` | `bool` | `True` | Collapses consecutive tabs and spaces into a single space character (`re.sub(r"\s+", " ", text)`). |
| `remove_special_chars` | `bool` | `False` | Removes all non-alphanumeric symbols except punctuation (`.,!?;:-'`). |
| `remove_numbers` | `bool` | `False` | Strips all numeric digits (`\d+`) from the text. |
| `remove_punctuation` | `bool` | `False` | Strips all standard punctuation symbols from the string. |
| `remove_urls` | `bool` | `False` | Replaces HTTP/HTTPS and `www.` web URLs with the string placeholder `[URL]`. |
| `remove_emails` | `bool` | `False` | Replaces standard email addresses with the string placeholder `[EMAIL]`. |
| `remove_newlines` | `bool` | `False` | Normalizes line breaks (`re.sub(r"\n\s*\n+", "\n\n", text)`) to preserve paragraph separation while removing intra-paragraph line breaks. |
| `strip_whitespace` | `bool` | `True` | Trims leading and trailing whitespace from the final processed text (`text.strip()`). |

---

#### Methods

##### `normalize_text(self, file_path: Union[str, Path], text: str) -> NormalizedContent`
Applies all active text transformations to a single document string and packages the output with tracking metadata.

###### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `file_path` | `Union[str, Path]` | Original path of the document file (used to generate deterministic document IDs and extract file types). |
| `text` | `str` | Raw text string extracted by `TextExtractor`. |

###### Return Value
- **Type:** `NormalizedContent`
- **Description:** A dataclass wrapping the sanitized string (`content`), boolean section flag (`has_section`), and `NormalizedTextMetaData`.

###### How It Works
1. Computes `file_name` and `file_type` (`Path(file_path).suffix.lower()`).
2. Runs `__process_text(text)`, sequentially executing regex replacements according to the boolean flags initialized in `__init__`.
3. Computes `document_id` via SHA-256 hash of `(file_name, file_path, normalized_text)`.
4. Computes `content_id` via SHA-256 hash of `normalized_text`.
5. Evaluates `has_section = len(self.__find_section_names(normalized_text)) != 0`, where `__find_section_names` searches for all-caps section header lines using regex `r"^[A-Z\s]+$"`.
6. Returns the immutable `NormalizedContent` dataclass.

---

##### `normalize_all(self, extracted_texts: Dict[str, str]) -> List[NormalizedContent]`
Applies `normalize_text()` across an entire dictionary mapping file paths to raw text strings.

###### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `extracted_texts` | `Dict[str, str]` | Output dictionary returned by `TextExtractor.extract_all()`. |

###### Return Value
- **Type:** `List[NormalizedContent]`
- **Description:** A list of normalized document objects ready for chunking.

---

### `class NormalizationProfiles`
A static factory class providing pre-configured `TextNormalizer` instances tailored for specific ingestion tasks.

#### Methods

##### `static rag_ingestion() -> TextNormalizer`
Returns a normalizer profile optimized for Retrieval-Augmented Generation (RAG) chunking and embedding pipelines.

###### Profile Settings
- `lowercase`: `False` (preserves case sensitivity for named entities and section headers).
- `remove_extra_whitespace`: `True`
- `remove_urls`: `True` (replaces URLs with `[URL]`)
- `remove_emails`: `True` (replaces emails with `[EMAIL]`)
- `remove_newlines`: `True` (structures paragraph boundaries into clean `\n\n` splits)
- `remove_special_chars`, `remove_numbers`, `remove_punctuation`: `False`
- `strip_whitespace`: `True`

---

##### `static minimal() -> TextNormalizer`
Returns a lightweight profile that performs minimal alteration, only collapsing extra whitespace and stripping edge padding (`lowercase=False, remove_urls=False`).
