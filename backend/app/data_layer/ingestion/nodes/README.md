# Pipeline data models (`nodes/`)

## What this module does

Holds the immutable dataclasses passed between ingestion stages. Each stage
consumes the previous stage's type and produces the next one; nothing reaches
back.

## Why they are frozen and why offsets are absolute

| Type | Produced by | Holds |
| :--- | :--- | :--- |
| `NormalizedContent` | normalizer | cleaned text, `has_section`, `SectionSpan` offsets |
| `SectionSpan` | normalizer | a heading and its body, as offsets into the content |
| `Document`, `Section`, `Context` | hierarchical chunker | the three levels above a chunk |
| `HChunk` / `RChunk` | chunkers | one chunk, its id, and its offsets |
| `EmbeddedChunk` | embedder | the vector, its id, and the chunk metadata |

All are `frozen=True`. A stage that wants to change something returns a new
object, so a bug cannot reach backwards and mutate what an earlier stage
produced.

**Offsets are absolute** into `NormalizedContent.content`, on both chunk types,
so `content[start_off_set:end_off_set] == chunk` holds regardless of which
chunker produced it. That invariant is what lets a retrieved chunk be widened to
its paragraph or located in the source, and it is checked by a property test
over random documents rather than assumed.
