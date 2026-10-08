# Project storage (`project_data_repo/`)

## What this module does

Stores what a project *is* and what represents it, across two stores that cannot
share a transaction:

| Store | Holds |
| :--- | :--- |
| SQLite (the memory database) | `project_table`, `project_description_table`, `project_mapping_table` |
| PostgreSQL / pgvector | the embeddings themselves |

The embeddings go to pgvector, never to the DiskANN index — DiskANN indexes
ingested documents and has nothing to do with project summaries.

## The two classes

| Class | Owns |
| :--- | :--- |
| `ProjectMetaData` | The SQLite registry, scoped to one project and its topic. Writes vectors too, through an injected repository. |
| `ProjectVectorHandler` | pgvector only. Derives vector ids from their source; reads batches back. |

Module-level `list_topic_projects()` and `list_topic_vector_ids()` answer
topic-wide questions, which no single project-scoped instance owns.
`is_registered()` and `project_topic()` answer the question other owners ask
when one of their foreign keys onto `project_table` is refused, so they never
query this module's tables themselves.

## Schema

```sql
project_table              project_id PK, project_name, topic_id, created_at,
                           updated_at, project_summary, user_id
                           FK topic_id -> topics_mapping_table
                           UNIQUE (project_id, topic_id)

project_description_table  PK (project_id, project_description_id)
                           + topic_id, project_description, created_at
                           FK (project_id, topic_id) -> project_table ON UPDATE CASCADE

project_mapping_table      PK (project_id, project_summary_vector_id)
                           + topic_id, created_at
                           FK (project_id, topic_id) -> project_table ON UPDATE CASCADE
```

`UNIQUE (project_id, topic_id)` adds nothing as a key — `project_id` already is
one — and exists because a foreign key may only reference columns that are
declared unique.

One summary **text** per project; many summary **vectors** — one for the summary
and one per description.

## How a write flows

```
add_project_vector(vector, vector_id, name, summary)
        |
        1. pgvector insert            <- the embedding
        |
        2. one SQLite transaction:    <- project row + mapping row
             upsert project_table      (a topic change cascades to the children)
             insert or ignore mapping
        |
   on failure at step 2 -> compensating delete of the vector
```

## Why it was designed this way

**Vectors first, metadata second, with a compensating delete.** The two stores
cannot share a transaction, so one of the two failure modes has to be chosen. An
unreachable vector is recoverable — it can be deleted or overwritten. A mapping
row pointing at an embedding that does not exist is not: every read through it
fails. This is the same ordering `SnapShot.__add_snap_shot` settled on.

**`topic_id` is denormalised into both child tables.** It is derivable by a join
from `project_table`, and it is copied anyway so that routing a query can read
every vector id in a topic with one statement and no join — the hot path of the
project layer. A denormalised copy is only safe if it cannot disagree with the
original, so the copy is half of the foreign key: `(project_id, topic_id)`
references the project's own pair. A child written under a topic its project is
not in is refused (`ProjectInAnotherTopic`), and moving a project cascades to
both children. That used to be an UPDATE loop after every upsert, which a write
that skipped the upsert never ran.

**A refused write says why.** SQLite reports only that *a* foreign key failed.
`ProjectMetaData` asks which: no such topic (`TopicNotFound`), no such project
(`ProjectNotFound`), or a project registered under another topic
(`ProjectInAnotherTopic`), with the database error kept as `__cause__`. A
project refused after its vector was written still gets the compensating
delete.

**Vector ids are derived from their source, not supplied.** No handler method
takes a vector id:

| Source | Id |
| :--- | :--- |
| the project's summary | `summary_vector_id(project_id)` |
| one description | `description_vector_id(project_id, description_id)` |

The payload hashed is `kind \x00 project_id \x00 source_id`, masked into
`Config.VECTOR_ID_MASK`. The NUL separators keep `("a","bc")` and `("ab","c")`
distinct — the rule `chunk_id` already follows. The `kind` field is what stops a
future third source (a title, a tag) colliding with a description that shares
its id. Deriving rather than storing means add, update, get and delete agree on
which row they mean without a lookup, and a vector can be regenerated from the
text it came from after an embedding model change.

**`NOT NULL` where it means something.** `project_summary` is required on every
write rather than nullable, so a project cannot exist without the text its
vector was embedded from. It is validated *before* the embedding is written, so
a bad summary leaves no orphaned vector behind.

**`created_at` survives a rewrite; `updated_at` does not.** Rewriting a
description's text does not change when that description first appeared.
Descriptions deliberately leave `updated_at` alone — the summary-vector paths
own it.

**An UPDATE matching nothing is not an error to SQLite**, so
`set_project_summary` checks `rowcount` and raises instead of reporting success
for a project that was never written.

## Known limitation

`ProjectMetaData` and `ProjectVectorHandler` still write vectors through
separate paths — the manager keeps them in step by passing
`summary_vector_id(project_id)` and sharing one repository. Full delegation is
the remaining step.

## Tests

`test/memory_layer_testing/test_project_meta_data.py` and
`test_project_vector_handler.py`. The vector store is a stateful fake rather
than a `MagicMock`: every bug that mattered here was the two stores disagreeing,
and a mock records calls without holding state, so it cannot show a disagreement.


## The project snapshot registry (`project_snapshot_repo.py`)

| Table | Holds |
| :--- | :--- |
| `project_snapshot` | the summary, its length, the `last_seq_included` watermark, and when it was taken |
| `project_snapshot_mapping` | the ordered chain of every snapshot a project has had |

The mapping is **append-only**, keyed `(project_id, project_snapshot_id)`. It is
not a pointer at the current snapshot: overwriting one row would leave the
earlier snapshots in `project_snapshot` with nothing ordering them.

A snapshot references its project (`ProjectNotFound` for one never registered),
and a mapping row references its snapshot by `(project_snapshot_id,
project_id)`, so the chain cannot list another project's snapshot. The mapping
needs no index of its own: its primary key already leads with `project_id`, and
a separate one measured slower for `latest()` (15.9 us against 12.7).

`project_snapshot_id` is derived from `(project_id, created_at, summary)` and
masked into the signed 64-bit range, because **it is also the pgvector
`vector_id`**. One id for both stores means nothing has to be looked up to go
from a row to its embedding. The timestamp is part of the derivation so that a
project summarised twice to the same words is still two snapshots.
