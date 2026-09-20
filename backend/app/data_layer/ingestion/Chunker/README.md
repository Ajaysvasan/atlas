# Chunkers (`Chunker/`)

## What this module does

Splits normalized documents into chunks and persists the structure to SQLite.

## Routing

`Chunker` sends each document down one of two paths, on `has_section`:

| Path | For | Produces |
| :--- | :--- | :--- |
| `HierarchicalChunker` | documents with headings | Documents → Sections → Contexts → Chunks |
| `RecursiveChunker` | flat documents | Documents → RecursiveChunks |

Both end in `windowing.sliding_windows`, which breaks on word boundaries and
guarantees `len(chunk) <= chunk_size`.

## Why hierarchical chunking keeps three levels

A section is a heading's body; a context is a paragraph within it; a chunk is a
window within that. Keeping the levels means a retrieved chunk can be widened to
its paragraph, or its whole section, without re-reading the source document.

Sections are built from the normalizer's spans rather than re-derived here —
this stage sees text whose line structure has already been reshaped, so any
heading regex run at this point disagrees with the one that decided the document
was hierarchical in the first place.

## Why recursive chunking splits on separators, then falls back

`__split` tries each separator in turn (`\n\n`, `\n`, `. `, …) and recurses into
any piece still larger than `chunk_size`. Only when no separator appears does it
cut on character windows. Pieces tile the text exactly — an earlier version
rebuilt them as `part + separator`, which appended a separator the document
never had.

Overlap is applied **once**, over the finished spans. Applying it inside the
recursion duplicated the same text into a chunk several times over.

## Why ids bind position

Section, context and chunk ids all include an ordinal or an offset alongside the
content hash. Two `NOTES` headings in one document hashed to a single
`sectionId`, and the second insert died on the primary key. A repeated paragraph
did the same to chunks — and the embedder derives its vector id from the chunk
id, so a collision there made chunks unreachable in the index.

## Why `sliding_windows` clamps its step

A window pulled back to a word boundary can be shorter than the overlap, and
`end - overlap` would then step backwards forever. `max(end - overlap, start + 1)`
keeps progress strictly positive.

## Storage

`DB_Manager.Manager` owns the SQLite schema. All inserts are
`on conflict do nothing`: every id is derived from content plus position, so a
conflict means the identical row is already there — which is what re-ingesting
an unchanged folder does.
