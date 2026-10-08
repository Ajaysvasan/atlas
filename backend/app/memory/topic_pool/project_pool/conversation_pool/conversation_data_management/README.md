# Snapshot metadata and vectors (`conversation_data_management/`)

## What this module does

| Class | Store | Holds |
| :--- | :--- | :--- |
| `ConversationVectorMetaDataRepository` | SQLite | what a snapshot covers |
| `ConversationVectorManager` | pgvector | the embeddings |

## Schema

```sql
summary_chunks              owned and created by FullConversationRepository;
                            this repository writes snapshot chunks into it
summary_vector_meta_data    summary_vector_id PK, chunk_id, project_id
                            FK chunk_id -> summary_chunks, project_id -> project_table
                            INDEX (chunk_id)
cumulative_vector_meta_data cumulative_vector_id PK, conversation_id, seq,
                            cumulative_summary, created_at, project_id,
                            len_of_the_summary
                            FK project_id -> project_table; UNIQUE (project_id, seq)
summary_snapshot_map        (cumulative_vector_id, summary_vector_id) UNIQUE
                            FK each -> its table
```

It requires the turn and project schemas, so `summary_chunks` and
`project_table` exist before its own tables whichever repository opens first —
the order-dependence behind bug 4.65, when both classes created
`summary_chunks` and the first one decided its shape.

Two levels per snapshot: one **cumulative** vector for the summary as a whole,
which is what search ranks on, and one **summary** vector per covered chunk,
reachable afterwards through the map.

## Why `seq` is per project

`seq` is the project snapshot's watermark: "every conversation summary in this
project after seq N". It is allocated as `MAX(seq) + 1` for the row's project,
inside a write that opens with `BEGIN IMMEDIATE`, so the read and the insert
cannot interleave with another writer — two repositories once both claimed the
same seq because each held only its own lock. It is shared by every
conversation in the project, so their summaries fall into one order, and
`UNIQUE (project_id, seq)` both holds that and serves the watermark read.

## Why the watermark is per conversation

`get_highest_summarised_sequence()` is the highest turn in *this* conversation
that a snapshot covers. It filtered on the project alone, so a conversation
with no snapshot read a sibling's progress as its own: ten unsummarised turns
reported a watermark of fifty, and the snapshot trigger stayed silent until the
new conversation overtook the old one. Scoped to the conversation, the planner
walks the conversation's turns by primary key and probes this table through
`idx_summary_vector_chunk`: 5.3 ms per call without that index, 3-5 us with it,
for +75% on an insert that costs 0.37 us.

## Thread safety

This repository holds no connection; it uses the memory database's, under that
database's lock, and fetches inside the lock as well as executing — a cursor is
a view onto the shared connection and another thread's write can invalidate it
between the two. Its `close()` releases nothing.

## Why a snapshot is written in one transaction

`insert_snapshot()` writes chunks, the cumulative row, the vector rows and the
map in a single transaction. They used to be four independently committing
calls, so a failure partway through left a snapshot that half-existed — chunks
and vector metadata committed with no cumulative row to reach them by — and
every retry added more orphans.

Order matters inside it: `summary_chunks` is the FK parent of
`summary_vector_meta_data`, which `summary_snapshot_map` references in turn.

## Why snapshots are ordered by `seq`

`SnapShot`'s cursors index into the list of a conversation's snapshots, so its
order has to be total and stable. `created_at` is caller-supplied TEXT, and
`datetime()` truncates it to whole seconds, so two snapshots in one second tied
and their order was arbitrary. `seq` is allocated on write and cannot tie.
