# Normalizer (`normalizer/`)

## What this module does

Cleans extracted text and reports where each section landed in the result, as
`SectionSpan` offsets into the normalized content.

## Why it works line by line

The original implementation collapsed all whitespace with `\s+`, which flattened
every document to a single line **before** the chunkers ran. After that no
heading regex, paragraph split or `\n\n` separator downstream could match
anything, and every "hierarchical" document degenerated into one section of
fixed-width slices. `_remove_extra_whitespace` now collapses spaces and tabs
only (`[^\S\n]+`), leaving line structure intact.

## Why headings are detected before the lexical transforms

`__group_lines` runs on the raw lines, before lowercasing or punctuation
stripping. Otherwise an ALL CAPS heading is unrecognisable by the time the
profile has lowercased it, and a markdown `#` is gone by the time punctuation
has been stripped.

Four shapes are recognised because real documents use all of them: markdown
ATX (`# Title`), setext underlines, numbered (`1.`, `2.3`), and ALL CAPS. The
ALL CAPS rule is guarded by a letter-ratio test, or it fires on any line without
a lowercase letter — a spreadsheet row like `Q1 | 1.2M` included. Code fences
are skipped.

## What it produces

`NormalizedContent` — the cleaned text, a `has_section` flag, the metadata, and
a tuple of `SectionSpan`. Every offset is absolute into `content`, so a chunk
can be traced back to its source after the hierarchy is flattened.

## Known gaps

- Section **headings** fall between `heading_end` and `content_start`, so no
  chunk ever contains them (**bug 5.4**).
- `document_id` concatenates its fields without a separator (**bug 5.6**), and
  the URL/email patterns contain unintended character ranges (**bug 5.7**).
