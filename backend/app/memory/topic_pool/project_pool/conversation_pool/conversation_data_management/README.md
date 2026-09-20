# Snapshot metadata and vectors (`conversation_data_management/`)

## What this module does

| Class | Store | Holds |
| :--- | :--- | :--- |
| `ConversationVectorMetaDataRepository` | SQLite | what a snapshot covers |
| `ConversationVectorManager` | pgvector | the embeddings |

## Schema

```sql
summary_chunks             shared with FullConversationRepository
summary_vector_meta_data   summary_vector_id PK, chunk_id, project_id
cumulative_vector_meta_data cumulative_vector_id PK, cumulative_summary,
                            created_at, project_id, len_of_the_summary
summary_snapshot_map       (cumulative_vector_id, summary_vector_id) UNIQUE
```

Two levels per snapshot: one **cumulative** vector for the summary as a whole,
which is what search ranks on, and one **summary** vector per covered chunk,
reachable afterwards through the map.

## Why this repository is thread-safe and how

One connection, opened with `check_same_thread=False`, and every statement under
an `RLock`. The lock is not optional: `commit()` and `rollback()` apply to the
whole connection rather than to one cursor, so unsynchronised threads sharing a
connection publish each other's half-written transactions and roll back each
other's finished ones. Removing the lock in a stress test produces spurious
errors, orphan rows and lost snapshots; removing it from `close()` segfaults the
interpreter rather than raising.

The fetch happens inside the lock too, not just the execute — a cursor is a view
onto the shared connection, and another thread's write can invalidate it between
the two.

The lock does not extend to other connections on the same file.
`FullConversationRepository` writes there too, on short-lived connections, and
those are serialised by SQLite's own file locking.

## Why a snapshot is written in one transaction

`insert_snapshot()` writes chunks, the cumulative row, the vector rows and the
map in a single transaction. They used to be four independently committing
calls, so a failure partway through left a snapshot that half-existed — chunks
and vector metadata committed with no cumulative row to reach them by — and
every retry added more orphans.

Order matters inside it: `summary_chunks` is the FK parent of
`summary_vector_meta_data`, which `summary_snapshot_map` references in turn.

## Why ordering uses `datetime()` plus tiebreakers

`created_at` is TEXT, so a plain sort is lexical. `datetime()` normalises it,
but truncates to whole seconds — and snapshots taken in the same second would
then tie, which matters because `SnapShot`'s cursors index into this list. The
raw column and `rowid` break the tie.
