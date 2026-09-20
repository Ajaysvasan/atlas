# Data layer (`data_layer/`)

## What this module does

Turns files on disk into vectors that can be searched. Everything about
*documents* lives here; everything about *conversations* lives in `memory/`.

## Structure

```
ingestion/            files -> text -> chunks -> vectors
  TextFileProcessor/    find files, decode them to text
  normalizer/           clean text, find section boundaries
  Chunker/              split into chunks, store them in SQLite
  embedding/            text -> 128-dim float32 vectors
  nodes/, metadata/     the dataclasses passed between stages
vector_db_manager/    the DiskANN index and the pgvector store
datalayer_exceptions/ every exception this layer raises
```

## Why it is split this way

Each stage takes the previous stage's output and produces the next stage's
input, as an immutable dataclass. Nothing reaches back. That is what lets the
chunkers be tested against hand-written `NormalizedContent` without a file, and
the embedder against a hand-written chunk without a chunker.

The one piece of shared knowledge crossing stages is **offsets**: a chunk knows
where it came from, as absolute positions into the normalized document, so
`content[start_off_set:end_off_set] == chunk` holds all the way down. That is
checked by a property test rather than assumed.

## Known gaps

- Nothing maps a DiskANN vector id back to its chunk, so a search returns ids
  that cannot be resolved to text (**bug 5.3**).
- The pgvector paths cannot work against a real server — no adapter is
  registered (**bug 5.1**).
- Re-ingesting an edited document leaves the old version's rows and vectors
  behind (**bug 5.5**).

## Tests

`test/data_layer_testing/`.
