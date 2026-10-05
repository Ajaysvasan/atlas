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
sqlite_setup.py          every SQLite open in this layer goes through it
topic_pool/
  topic_manager.py       create/load/soft-delete a topic  (built)
  topic_pool_repo/       the topics table                 (built)
  project_pool/
    project_manager.py   routes a query to a project      (built)
    project_data_repo/   the project registry             (built)
    conversation_pool/   turns, summaries, snapshots      (built)
memory_pool_exceptions.py
```

## What is built and what is not

| Piece | State |
| :--- | :--- |
| Conversation storage, summarisation, snapshots | Built and tested |
| Project registry and query routing | Built and tested |
| Topics: create, read, soft delete | Built and tested |
| `MemoryManager` | Empty stub (**bug 4.1**) |
| `TopicManager` -> `ProjectManager` handoff | Not wired — the two layers exist and do not talk |
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


## Identifier validation (`identifiers.py`)

`require_identifier(value, name)` rejects anything that cannot scope a row — a
non-string, or a string that is blank once stripped — and returns the value
stripped so that `' c1'` and `'c1'` cannot become two conversations. It raises
`InvalidIdentifier`, which names the field and shows the value it got.

It lives here rather than in the conversation pool because `memory/snapshot.py`
needs it as well, and the project layer does. Validation is applied where an id
is stored or written, not at every hop: the repositories, `SnapShot` and
`ConversationSummary` check, and the pass-through layers above them inherit it.


## Conversation mapping (`memory_mapping_handler.py`)

Answers "where does this conversation belong" — its topic, its project, and the
project snapshot that was current when it was last touched. This is the lookup
`MemoryManager` needs to resume a conversation without routing it again, which
is why it sits here beside `memory_manager.py` rather than inside the
conversation pool: the pool is entered *after* this table has said which project
to open.

It takes no `conversation_id`. Every method names the conversation it acts on,
so one handler serves the whole table — the alternative, an instance per
conversation, would open a connection per conversation to a table that has one
row for each.

The write is two steps because the information arrives in two steps. A row is
opened when a conversation starts, with topic and project still unset; routing
fills them in later. That makes "never seen" (`search` returns `None`) and
"seen, not yet routed" (`MemoryMapping(None, None, None)`) different answers,
which is deliberate — the caller needs to tell them apart.

`latest_project_snapshot_id` is a **cache** of
`ProjectSnapshotRepository.latest()`, not a second source of truth. The ordered
chain of a project's snapshots lives in `project_snapshot_mapping`; this column
exists so resuming a conversation does not have to open the project registry,
the same trade as denormalising `topic_id` onto the project tables for the read
the router needs.

One open question: `user_id` is the only user-scoped column in the layer.
Nothing else — topic, project, conversation — has a notion of a user. Either
that reaches the other tables or this column is ahead of the decision.
