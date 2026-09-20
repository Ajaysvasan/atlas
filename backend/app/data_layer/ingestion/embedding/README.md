# Embedding (`embedding/`)

## What this module does

Turns chunks and raw strings into 128-dimensional float32 vectors, and assigns
each one a vector id.

Model: `sentence-transformers/all-MiniLM-L6-v2`, truncated to 128 dims, batch
size 64.

## Why vector ids come from the chunk id, never the chunk text

Chunk text repeats — a licence header, a boilerplate paragraph, a "yes" turn.
Hashing the text produced one vector id for several distinct chunks, so all but
one of them were unreachable in the index. The chunk ids the chunkers emit
already bind position as well as content, so hashing those is collision-free by
construction.

## Why the id is masked

`vector_id = md5(chunk_id)[:8]` is unsigned and 64 bits wide. Both storage
backends use a *signed* 64-bit column (SQLite INTEGER, Postgres bigint), so
about half of all ids overflow with `Python int too large to convert to SQLite
INTEGER`. `& Config.VECTOR_ID_MASK` — that is `(1 << 63) - 1` — clears the top
bit and keeps every id inside the range both accept.

## `embed_text` vs `embed`

`embed()` takes pipeline chunks (`HChunk` / `RChunk`). `embed_text()` takes a
raw string, because the memory layer embeds conversation summaries and turns
that never pass through a chunker. When no chunk id is given it derives one from
the content — acceptable there, because the caller supplying ids is the normal
path.
