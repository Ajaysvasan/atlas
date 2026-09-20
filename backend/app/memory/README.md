# Memory layer (`memory/`)

## What this module does

Everything the system remembers about its own conversations, organised as a
hierarchy:

```
Topic
  └── Project            what a body of work is about
        └── Conversation turns, in order
              └── Snapshot   a rolling summary of the turns so far
```

`data_layer/` handles *documents*. This layer handles *what was said*.

## Structure

```
memory_manager.py        PENDING - the top-level entry point
topic_pool/
  topic_manager.py       PENDING - create/load a topic
  project_pool/
    project_manager.py   routes a query to a project  (built)
    project_data_repo/   the project registry          (built)
    conversation_pool/   turns, summaries, snapshots   (built)
memory_pool_exceptions.py
```

## What is built and what is not

| Piece | State |
| :--- | :--- |
| Conversation storage, summarisation, snapshots | Built and tested |
| Project registry and query routing | Built and tested |
| `MemoryManager`, `TopicManager` | Empty stubs (**bug 4.1**) |
| Thinking layer and its wiring | Not started — `ProjectManager.route()` stops at a `pass` |

The layers are being built bottom-up, so the storage is finished before the
things that call it. `todo.md` tracks what remains.

## Why it is organised this way

The hierarchy exists so that scope is always explicit. A query is answered
against one project's memory, not everything ever said; a snapshot summarises
one conversation, not a topic. Every id in the layer therefore carries the scope
above it, and every read is filtered by it.

## Decisions that run through the whole layer

**Ids that act as primary keys are never hashed from content alone.** Content
repeats — a "yes" turn, an unchanged summary, two projects with the same name.
`chunk_id` binds `(project_id, sequence_number, text)`;
`cumulative_vector_id` binds `(project_id, timestamp, summary)`; `project_id` is
a uuid. Fields are separated by `\x00` so they cannot run together into a
colliding payload.

**Order by `sequence_number`, never `created_at`.** `created_at` is a
caller-supplied TEXT column with no format enforcement, so it is not a reliable
sort key. Where a timestamp must be ordered, it goes through `datetime()` with
the raw column and `rowid` as tiebreakers.

**Vectors first, metadata second, with a compensating delete.** The two stores
cannot share a transaction. An unreachable vector is recoverable; a metadata row
pointing at a missing embedding is not.

## Known gaps

- PostgreSQL connections opened by `SnapShot` are never closed (**bug 4.44**).
- Nothing in this layer can delete anything — no retention policy exists yet.
