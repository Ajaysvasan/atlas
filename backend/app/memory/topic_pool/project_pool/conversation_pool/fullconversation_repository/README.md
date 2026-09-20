# Conversation storage (`fullconversation_repository/`)

## What this module does

Stores conversation turns and reads them back, in SQLite.

## Schema

```sql
summary_chunks      chunk_id PK, chunk, created_at, chunker_type
full_conversation   sequence_number PK, project_id, chunk_id, role, created_at
                    FK chunk_id -> summary_chunks
```

The text lives in `summary_chunks`; the position and the speaker live in
`full_conversation`. They are split because `summary_chunks` is shared with the
snapshot metadata repository, which references the same rows.

`summary_chunks` must be created **before** `full_conversation`, and its rows
must be inserted before the rows that reference them — the foreign key is
enforced, and an orphan meta row is accepted only when `foreign_keys` is off,
after which every reader's JOIN silently drops it while it keeps its
`sequence_number` forever.

## Why sequence numbers are allocated under `BEGIN IMMEDIATE`

`append_turns` reads `MAX(sequence_number)` and then inserts. Without the write
lock taken up front, two concurrent appends both read the same maximum and hand
out the same sequence number. `BEGIN IMMEDIATE` closes that window.

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

`test/memory_layer_testing/test_full_converation.py`. The ordering guards
deliberately insert rows whose `created_at` order contradicts their
`sequence_number` order, so an accidental `ORDER BY created_at` fails them.
