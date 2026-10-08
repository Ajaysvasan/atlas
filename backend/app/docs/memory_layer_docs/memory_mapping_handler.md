# Conversation mapping (`MemoryMappingHandler`)

## Overview & Purpose

Answers "where does this conversation belong" — which topic, which project, and
that project's latest snapshot. It is the lookup `MemoryManager` needs to resume
a conversation without routing it again.

Architecture and rationale live in `memory/README.md`. This page is the API.

---

## `MemoryMappingHandler`

```python
MemoryMappingHandler(database: MemoryDatabase | str | Path | None = None)
```

| Parameter | Meaning |
| :--- | :--- |
| `database` | A `MemoryDatabase`, or a path to one. `None` uses the shared memory database at `Config.MEMORY_DB` |

**It takes no `conversation_id`.** One handler serves the whole table; every
method names the conversation it acts on. It holds no connection: statements run
through the memory database's, under its lock.

| Method | Returns | Raises |
| :--- | :--- | :--- |
| `populate_conversation_id(conversation_id, user_id)` | `None` | `InvalidIdentifier`, `sqlite3.IntegrityError` if the pair already has a row |
| `insert_into_mapping_table(conversation_id, user_id, topic_id, project_id, created_at)` | `None` | `InvalidIdentifier` (any of the four ids blank), `ProjectNotFound`, `ProjectInAnotherTopic` |
| `search(conversation_id, user_id)` | `MemoryMapping` or `None` | `InvalidIdentifier` |
| `close()` | `None`. Releases nothing — the database is shared | — |

`update_latest_project_snapshot_id` and the `new_latest_project_snapshot_id`
argument are gone with the column they wrote; see below.

### `MemoryMapping`

```python
MemoryMapping(topic_id, project_id, latest_project_snapshot_id)
```

A `NamedTuple`, so it compares equal to a plain 3-tuple. `topic_id` and
`project_id` are `str | None`; `latest_project_snapshot_id` is `int | None` —
the integer the snapshot is stored under, in SQLite and pgvector alike. A row
opened by `populate_conversation_id` has all three unset until routing fills
them in, and a routed project with no snapshot yet has the last one `None`.

---

## The two-step write

A conversation gets a row before it has been routed anywhere, and the routing
arrives later:

```python
handler = MemoryMappingHandler()

handler.populate_conversation_id("conv_1", "user_1")
handler.search("conv_1", "user_1")
# MemoryMapping(topic_id=None, project_id=None, latest_project_snapshot_id=None)

handler.insert_into_mapping_table(
    conversation_id="conv_1", user_id="user_1",
    topic_id="topic_1", project_id="project_1", created_at=utc_now(),
)
handler.search("conv_1", "user_1")
# MemoryMapping('topic_1', 'project_1', <its latest snapshot id, or None>)
```

`insert_into_mapping_table` is an UPDATE despite the name, so it needs the row
to exist. A zero-row UPDATE is not an error to SQLite, so calling it first would
otherwise look like success — it **logs a warning** instead, naming the
conversation and telling you to call `populate_conversation_id`.

Routing to a project that was never registered raises `ProjectNotFound`; to a
project registered under a different topic, `ProjectInAnotherTopic`, whose
`actual_topic_id` names the right one. Both keep the database error as
`__cause__`.

`search` on a conversation with no row returns `None`, not a tuple of `None`s.
The two are different: the first means "never seen", the second "seen, not yet
routed".

### Where the snapshot comes from

`search` asks `ProjectSnapshotRepository(project_id).latest()` every time, so a
snapshot taken after routing is seen without any call on this handler. The
column that used to cache it is gone: it existed so resuming would not open a
second database file, and it was stale whenever the explicit update was missed.

---

## Schema

```sql
create table if not exists memory_mapping_table(
    conversation_id text not null,
    user_id text not null,
    topic_id text,
    project_id text,
    created_at text,
    primary key (conversation_id, user_id),
    foreign key (project_id, topic_id)
        references project_table(project_id, topic_id) on update cascade,
    check ((topic_id is null) = (project_id is null))
);

create index if not exists idx_memory_mapping_project
on memory_mapping_table(project_id, topic_id);
```

The primary key is what makes "which project is this conversation in" answerable
— without it the same conversation could hold two rows and the question would
have two answers.

The foreign key is the pair, so a routing row cannot name a project under the
wrong topic, and moving a project to another topic carries its conversations
with it. A pair with a null in it is not checked as a foreign key, which is what
lets the unrouted row exist; the `CHECK` is what stops a half-routed one. The
index serves the cascade, which looks rows up by that pair.

---

## Scope note

`user_id` is the only user-scoped column in the memory layer; the topic, project
and conversation tables have no notion of a user. Either that dimension has to
reach them too, or this column is ahead of a decision that has not been made.
See `bugs.md` 4.83 and the scope note in Section 4c.

---

## Tests

`test/memory_layer_testing/test_memory_mapping_handler.py` — the regression
guards for Bugs 4.73-4.83, the relationships the database holds, and the
snapshot being read from the chain rather than cached.
