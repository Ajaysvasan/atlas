# Chunk metadata (`metadata/`)

## What this module does

Holds the metadata carried alongside each pipeline type.

## Why it is separate from `nodes/`

Keeping provenance out of the content dataclasses lets a stage pass it forward
without widening the object it is producing, and lets the content types be
constructed in a test without inventing a plausible history for them.

| Type | Records |
| :--- | :--- |
| `NormalizedTextMetaData` | document id, source path, file name and type, ingestion time, normalizer version, content hash |
| `ChunkMetaData` | document name and id, which chunking algorithm ran, section name |
| `EmbeddedChunkMetaData` | chunk id, the chunk text, the model used |

`normalizer_version` exists so chunks produced by different normalizer
generations can be told apart without re-reading the source — a change in
cleaning rules changes the offsets, and nothing else would record that.

## Known gaps

- `EmbeddedChunkMetaData.modelUsedForChunking` holds the *embedding* model
  (**bug 5.9**).
- `NormalizedTextMetaData` has two source-path fields, one always `None`
  (**bug 5.10**).
