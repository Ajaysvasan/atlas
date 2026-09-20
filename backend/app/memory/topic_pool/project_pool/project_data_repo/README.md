# Project storage (`project_data_repo/`)

## What this module does

Stores what a project *is* and what represents it, across two stores that cannot
share a transaction:

| Store | Holds |
| :--- | :--- |
| SQLite (`project.sql`) | `project_table`, `project_description_table`, `project_mapping_table` |
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

## Schema

```sql
project_table              project_id PK, project_name, topic_id, created_at,
                           updated_at, project_summary, user_id

project_description_table  PK (project_id, project_description_id)
                           + topic_id, project_description, created_at
                           FK project_id -> project_table

project_mapping_table      PK (project_id, project_summary_vector_id)
                           + topic_id, created_at
                           FK project_id -> project_table
```

One summary **text** per project; many summary **vectors** — one for the summary
and one per description.

## How a write flows

```
add_project_vector(vector, vector_id, name, summary)
        |
        1. pgvector insert            <- the embedding
        |
        2. one SQLite transaction:    <- project row + mapping row
             upsert project_table
             sync child topic_id
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
project layer. A denormalised copy is only safe if it moves when the project
moves, so the upsert rewrites both child tables' `topic_id` in the same
transaction whenever a write carries a different topic. Left unsynchronised,
those rows keep answering for the topic the project has left.

`topic_id` has no foreign key: there is no topic table yet. The only guard is a
non-empty check in the constructor.

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

The connection is a plain `sqlite3.connect`: no WAL, no lock, and usable only
from the thread that opened it — while the file is shared by every project. Two
instances writing at once will contend. This is the problem already fixed for
the conversation database in `sqlite_setup.connect()`. Tracked as **bug 4.45**.

`ProjectMetaData` and `ProjectVectorHandler` still write vectors through
separate paths — the manager keeps them in step by passing
`summary_vector_id(project_id)` and sharing one repository. Full delegation is
the remaining step.

## Tests

`test/memory_layer_testing/test_project_meta_data.py` and
`test_project_vector_handler.py`. The vector store is a stateful fake rather
than a `MagicMock`: every bug that mattered here was the two stores disagreeing,
and a mock records calls without holding state, so it cannot show a disagreement.
