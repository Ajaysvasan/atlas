# File loading and text extraction (`TextFileProcessor/`)

## What this module does

`FileLoader` finds candidate documents in a directory tree. `TextExtractor`
turns any one of them into plain text.

## Why the policy is a denylist

`FileLoader` offers the extractor **anything that is not a known binary
format**. An allowlist would have to be extended for every format the extractor
learns, and would silently drop everything else — source code, logs, config,
TeX, subtitles, formats nobody has thought of. `NON_DOCUMENT_EXTENSIONS` is the
list of things that are definitely not text.

It also skips VCS and build directories, dotfiles, empty files, and files over
64 MB; and it resolves symlinked directories against a visited set, because a
link pointing at an ancestor otherwise recurses until the stack limit — and that
`RecursionError` used to be caught as if the folder were unreadable, so the scan
returned a partial tree and reported success.

## Why extraction is structured, then falls back

Formats with structure worth exploiting get a dedicated reader: PDF, Word,
PowerPoint, Excel, HTML/XML, JSON, notebooks, and the textract-backed binaries.
**Everything else is decoded as text.** Only bytes that are not text at all are
rejected.

Word heading styles, HTML `<h1>`–`<h6>` and PowerPoint slide titles are emitted
as markdown `#` headings, so formats with no heading syntax of their own still
produce sections downstream.

JSON is flattened to one `a.b.c: value` line per leaf, so keys stay attached to
what they label once the document is split into chunks.

## Encoding, in order

1. **Byte order mark** — the only trustworthy signal for the wide encodings.
   utf-16 accepts almost any even-length input, so trying it blind turns a
   Latin-1 document into CJK without raising anything.
2. A NUL byte in the first 8 KB means binary: raise `InvalidFileType`.
3. utf-8.
4. chardet, but only at confidence ≥ 0.7 — it guesses from statistics and will
   happily label a binary blob as some obscure codepage.
5. latin-1 with replacement, which cannot fail.

## Tests

`test/data_layer_testing/test_ingestion.py`.
