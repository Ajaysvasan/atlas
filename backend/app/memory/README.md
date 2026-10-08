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
memory_database.py       the one SQLite database every table here lives in
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

## One database

Every table in this layer lives in one SQLite file,
`data/memory/memory_layer/memory_layer.db` (`Config.MEMORY_DB`). It used to be
five: topics, the project registry, the mapping table, and one conversation file
per project. Across files a relationship can only be kept by hand, and several
were — a loop rewriting `topic_id` on two child tables after every project
upsert, a cached snapshot id rewritten whenever a project moved on — while
others were not kept at all.

**Ownership did not move.** `memory_database.py` holds the connection and the
transaction discipline and no table's SQL. Each owner module declares its own
tables as a `Schema` and keeps every query against them; `requires` names the
schemas whose tables it references, so a parent's tables always exist before a
child's, whichever owner is constructed first.

| Owner | Tables |
| :--- | :--- |
| `topic_pool_repo/topic_pool_meta_handler.py` | `topics_mapping_table` |
| `project_data_repo/project_meta_data.py` | `project_table`, `project_description_table`, `project_mapping_table` |
| `project_data_repo/project_snapshot_repo.py` | `project_snapshot`, `project_snapshot_mapping` |
| `fullconversation_repository.py` | `summary_chunks`, `full_conversation` |
| `conversationVectorMetaManager.py` | `summary_vector_meta_data`, `cumulative_vector_meta_data`, `summary_snapshot_map` |
| `memory_mapping_handler.py` | `memory_mapping_table` |

**The relationships are the database's to keep.**

```
topics_mapping_table
 └─ project_table.topic_id
     ├─ project_description_table (project_id, topic_id)   cascades on a topic move
     ├─ project_mapping_table     (project_id, topic_id)   cascades on a topic move
     ├─ memory_mapping_table      (project_id, topic_id)   cascades on a topic move
     ├─ project_snapshot.project_id
     │   └─ project_snapshot_mapping (project_snapshot_id, project_id)
     ├─ full_conversation.project_id
     ├─ summary_vector_meta_data.project_id
     └─ cumulative_vector_meta_data.project_id
summary_chunks
 ├─ full_conversation (chunk_id, conversation_id)
 └─ summary_vector_meta_data.chunk_id
summary_snapshot_map → cumulative_vector_meta_data, summary_vector_meta_data
```

Where a child carries a copy of its parent's column — `topic_id` on the project
tables, `conversation_id` on a turn — the foreign key is the pair, so the copy
cannot disagree with the original, and moving a project to another topic
carries its children with it. `CHECK` constraints hold what used to be checked
only in Python: a turn's role, a topic's active flag, a routing row whose topic
and project are set together or not at all.

SQLite reports a refused foreign key as `FOREIGN KEY constraint failed`, never
which one. The owners translate it into what the caller can act on —
`TopicNotFound`, `ProjectNotFound`, `ProjectInAnotherTopic` — by asking the
parent's owner, and keep the database error as `__cause__`.

**What deliberately has no parent.** `conversation_id` is a key in several
tables and a foreign key in none: there is no conversation table, because
registering conversations is not this layer's job. Vector ids point into
PostgreSQL, which a SQLite foreign key cannot reach.

**One connection.** `MemoryDatabase.shared()` keeps one per file per process;
every owner given no database uses it. Reads and writes go through one `RLock`;
a write opens with `BEGIN IMMEDIATE`, so a read-then-insert such as allocating
the next `seq` cannot interleave with another writer, and a nested write runs in
a savepoint, so a failure the caller catches undoes only its own part. Two
owners can therefore write in one transaction, which separate files never
allowed. An owner's `close()` releases nothing: the database outlives its
owners and is closed by whoever opened it. Opening a connection per call, as
the turn store used to, cost 52 us against 2 us on a shared one.

**Child columns left unindexed, on purpose.** SQLite searches a child table when
the parent's key changes or the parent row goes. `full_conversation.project_id`,
`summary_vector_meta_data.project_id` and `summary_snapshot_map.summary_vector_id`
have no index because nothing deletes a project or a vector and a topic move
changes `topic_id`, not `project_id`. An index there would be paid on every
insert for an operation that does not exist. If a delete path is added, they
need one.

## Decisions that run through the whole layer

**Ids that act as primary keys are never hashed from content alone.** Content
repeats — a "yes" turn, an unchanged summary, two projects with the same name.
`chunk_id` binds `(project_id, conversation_id, sequence_number, text)` —
without `conversation_id`, two conversations in one project could not both open
with "hello";
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
so one handler serves the whole table rather than an instance per row.

The write is two steps because the information arrives in two steps. A row is
opened when a conversation starts, with topic and project still unset; routing
fills them in later. That makes "never seen" (`search` returns `None`) and
"seen, not yet routed" (`MemoryMapping(None, None, None)`) different answers,
which is deliberate — the caller needs to tell them apart.

The latest project snapshot is not stored here. `search` asks
`ProjectSnapshotRepository.latest()` each time, so it is never stale. The column
that used to cache it existed so resuming a conversation would not open a second
database file; with one file that reason is gone, and the cache had to be
rewritten by hand every time a project moved on. It is a second lookup rather
than a join because what counts as "latest" — the highest watermark, then the
newest — belongs to the snapshot repository, and a join here would repeat it.

One open question: `user_id` is the only user-scoped column in the layer.
Nothing else — topic, project, conversation — has a notion of a user. Either
that reaches the other tables or this column is ahead of the decision.
