# Conversation storage (`fullconversation_repository/`)

## What this module does

Stores conversation turns and reads them back, in SQLite.

## Schema

```sql
summary_chunks      chunk_id PK, conversation_id, chunk, created_at, chunker_type
                    UNIQUE (chunk_id, conversation_id)
full_conversation   PK (conversation_id, sequence_number), project_id, chunk_id,
                    role CHECK in (user, assistant, system), created_at
                    FK project_id -> project_table
                    FK (chunk_id, conversation_id) -> summary_chunks
```

The text lives in `summary_chunks`; the position and the speaker live in
`full_conversation`. They are split because `summary_chunks` is shared with the
snapshot metadata repository, which references the same rows. This module owns
it and is the only one that creates it.

A turn's text and its position each carry `conversation_id`, and the pair is the
foreign key, so the two copies cannot disagree. The role `CHECK` holds the
bucket's validation for writes that bypass the bucket through `add()`. A turn
for a project that was never registered is refused as `ProjectNotFound`.

`summary_chunks` must be created **before** `full_conversation`, and its rows
must be inserted before the rows that reference them — the foreign key is
enforced, and an orphan meta row is accepted only when `foreign_keys` is off,
after which every reader's JOIN silently drops it while it keeps its
`sequence_number` forever.

## Why `full_conversation` is indexed on `chunk_id`

`get_sequence_number()` finds a turn by its chunk id, which the primary key
`(conversation_id, sequence_number)` cannot serve.

It used to be justified by the summarised watermark, which joined on `chunk_id`
project-wide: the planner drove from `summary_vector_meta_data` and searched
here, so an index on the other side of the join was measured and rejected
(bug 4.68). The watermark is now per conversation, which reverses the drive —
the planner walks this table by primary key and probes the other side — so the
index that pays for it now is `idx_summary_vector_chunk` there, not this one.
See `conversation_data_management/README.md` for the numbers.

## Why sequence numbers are allocated under `BEGIN IMMEDIATE`

`append_turns` reads `MAX(sequence_number)` and then inserts. Without the write
lock taken up front, two concurrent appends both read the same maximum and hand
out the same sequence number. `MemoryDatabase.writing()` opens with `BEGIN
IMMEDIATE`, which closes that window for writers in other processes; in this
one, the database's lock already serialises them.

## Why a turn's id binds its conversation

`chunk_id` is `sha256(project_id, conversation_id, sequence_number, text)`. It
was the same without `conversation_id`, so two conversations in one project that
opened with the same words — "hello" — produced one id, and the second failed on
`summary_chunks`' primary key.

## Why `created_at` is stamped here

It is a caller-supplied TEXT column with no format enforcement, which is what
made Bug 4.31 possible. `utc_now()` stamps it in one ISO-8601 UTC format,
sortable as text. Ordering still keys off `sequence_number`; this is for display
and auditing.

## Reading

| Returns text only | Returns `Turn` (with the speaker) |
| :--- | :--- |
| `fetch_all`, `get_ranged_chunks`, `get_n_chunks`, `get_sequence_after` | `get_all_turns`, `get_turns`, `get_last_n_turns`, `get_turns_after` |

The text-only readers came first and drop the speaker, which makes them useless
for prompt assembly. They now have no callers in the source tree; see
`todo.md` 1.1. Prefer the `Turn` readers.

`get_last_n_turns` rejects a non-positive `n` explicitly — it would otherwise
reach SQLite as `LIMIT -1`, which means "no limit" and returns everything.

## Tests

`test/memory_layer_testing/test_full_conversation.py`. The ordering guards
deliberately insert rows whose `created_at` order contradicts their
`sequence_number` order, so an accidental `ORDER BY created_at` fails them.
