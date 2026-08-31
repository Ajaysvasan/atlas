# Text Normalizer Module (`normalizer.py`)

## Overview & Purpose
The `normalizer.py` module cleans, standardizes, and structures raw document strings extracted by `TextExtractor`. It applies configurable transformation pipelines to strip extraneous noise (URLs, email addresses, redundant spacing) and — critically — it is the stage that **locates the document's sections**, publishing them as `SectionSpan` offsets on the returned `NormalizedContent`.

### Why section detection lives here
The normalizer is the last stage that still sees the document's original line structure. Once it has run, headings can no longer be recovered: a profile that lowercases destroys `ALL CAPS` headings, and one that strips punctuation destroys markdown `#`. Detection therefore runs **per line, before any lexical transform**, and the resulting spans are handed downstream.

`HierarchicalChunker` consumes those spans rather than re-deriving them. Previously both stages ran their own copy of the heading regex over different text, and disagreed: `has_section` was computed on the *raw* string while the chunker searched the *normalized* one. Because the whitespace collapse flattened every document to a single line, the chunker's regex matched nothing, and every "hierarchical" document silently degenerated into one `MAIN` section holding one context.

---

## Module-Level Helpers

### `_heading_name(line: str) -> str | None`
Returns the heading a line announces, or `None` if the line is body text. Recognises three shapes; setext underlines need a lookahead line and are handled by the caller.

| Shape | Example | Name returned |
| :--- | :--- | :--- |
| Markdown ATX | `## Installation` | `"Installation"` |
| Numbered | `1. Introduction`, `2.3 Related Work` | the whole line |
| ALL CAPS | `BACKGROUND` | the whole line |
| Setext (caller) | `Overview` above `========` | `"Overview"` |

#### Guards against false positives
| Constant | Value | Rejects |
| :--- | :--- | :--- |
| `HEADING_MAX_LENGTH` | `80` | Long paragraphs that happen to lack lowercase letters. |
| `HEADING_MAX_WORDS` | `12` | Shouted sentences. |
| `HEADING_MIN_LETTERS` | `3` | `"A"`, `"I/O"`. |
| `HEADING_MIN_LETTER_RATIO` | `0.5` | Spreadsheet and table rows such as `"Q1 \| 1.2M"`, which contain no lowercase letters and so match the ALL CAPS pattern on their own. |

Numbered headings are additionally rejected when the line ends in sentence punctuation (`. , ; : ! ?`), and lines inside a fenced code block (```` ``` ````/`~~~`) are never headings.

---

## Classes & Public APIs

### `class TextNormalizer`
A configurable text sanitization pipeline that processes strings and wraps them in immutable `NormalizedContent` objects.

#### Constructor: `__init__(self, ...)`

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `lowercase` | `bool` | `True` | Converts text to lower case. Applied **after** heading detection, so it cannot destroy an `ALL CAPS` heading. |
| `remove_extra_whitespace` | `bool` | `True` | Collapses runs of spaces and tabs (`re.sub(r"[^\S\n]+", " ", text)`). **Horizontal only** — line breaks are preserved. |
| `remove_special_chars` | `bool` | `False` | Removes non-alphanumeric symbols except `.,!?;:-'`. |
| `remove_numbers` | `bool` | `False` | Strips numeric digits (`\d+`). |
| `remove_punctuation` | `bool` | `False` | Strips standard punctuation. |
| `remove_urls` | `bool` | `False` | Replaces HTTP/HTTPS and `www.` URLs with `[URL]`. |
| `remove_emails` | `bool` | `False` | Replaces email addresses with `[EMAIL]`. |
| `remove_newlines` | `bool` | `False` | Joins the lines *within* a paragraph with a space. Paragraph breaks (`\n\n`) and heading lines always survive. |
| `strip_whitespace` | `bool` | `True` | Trims each line and the final text. |

> **Note on `remove_extra_whitespace`.** This flag previously used `re.sub(r"\s+", " ", text)`, which collapsed newlines along with spaces. That single regex was the root cause of the chunking failures described above: it removed the paragraph boundaries `HierarchicalChunker` splits on and the `"\n\n"` / `"\n"` separators `RecursiveChunker` prefers, so both chunkers fell through to naive character slicing.

---

#### Methods

##### `normalize_text(self, file_path: Union[str, Path], text: str) -> NormalizedContent`
Applies all active transformations to a single document string and packages the output with tracking metadata and section spans.

###### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `file_path` | `Union[str, Path]` | Original path of the document file. Coerced with `str()`, so a `Path` is accepted. |
| `text` | `str` | Raw text string extracted by `TextExtractor`. |

###### Return Value
- **Type:** `NormalizedContent`
- **Description:** A frozen dataclass wrapping the sanitized string (`content`), the `has_section` flag, `NormalizedTextMetaData`, and the `sections` tuple of `SectionSpan` offsets into `content`.

###### How It Works
1. Normalizes line endings (`\r\n`, `\r` → `\n`) and collapses blank-line runs to a single `\n\n`.
2. `__group_lines()` walks the lines and produces `(is_heading, lines)` blocks. Heading detection runs here, on untransformed lines. Setext underlines are dropped; code-fence contents are skipped.
3. `__build_blocks()` applies `__process_line()` to every line — the lexical transforms, each of which is line-local and never merges or splits lines — then joins each block's lines with `" "` when `remove_newlines` is set, otherwise `"\n"`, and joins blocks with `"\n\n"`.
4. Offsets are accumulated while the blocks are joined, producing one `SectionSpan` per heading with `heading_start`, `heading_end`, `content_start` and `content_end` **exact against the final `content` string**.
5. `document_id` = SHA-256 of `(file_name, file_path, normalized_text)`; `content_hash` = SHA-256 of `normalized_text`.
6. `has_section = len(sections) > 0` — derived, so the flag and the spans cannot disagree.

---

##### `normalize_all(self, extracted_texts: Dict[str, str]) -> List[NormalizedContent]`
Applies `normalize_text()` across a dictionary mapping file paths to raw text. Shares one implementation with `normalize_text()` rather than duplicating it.

###### Return Value
- **Type:** `List[NormalizedContent]`

---

### `class NormalizationProfiles`
Static factory providing pre-configured `TextNormalizer` instances.

#### `static rag_ingestion() -> TextNormalizer`
The profile used by `IngestionPipeline`.

- `lowercase`: `False` (preserves case for named entities and headers)
- `remove_extra_whitespace`: `True`
- `remove_urls`: `True` — replaces with `[URL]`
- `remove_emails`: `True` — replaces with `[EMAIL]`
- `remove_newlines`: `True` — joins hard-wrapped lines, keeps paragraph and heading structure
- `remove_special_chars`, `remove_numbers`, `remove_punctuation`: `False`
- `strip_whitespace`: `True`

#### `static minimal() -> TextNormalizer`
A lightweight profile that only collapses horizontal whitespace and strips edge padding (`lowercase=False`, `remove_urls=False`).

---

## Versioning
`NORMALIZATION_VERSION` is `"rag_v2"`, recorded on every `NormalizedTextMetaData.normalizer_version`. It was bumped from `"rag_v1"` because normalized output changed shape: content produced by the old version is not comparable with content produced by this one, and any index built under `rag_v1` should be rebuilt.
