# Conversation mapping (`MemoryMappingHandler`)

## Overview & Purpose

Answers "where does this conversation belong" — which topic, which project, and
which project snapshot was current when it was last touched. It is the lookup
`MemoryManager` needs to resume a conversation without routing it again.

Architecture and rationale live in `memory/README.md`. This page is the API.

---

## `MemoryMappingHandler`

```python
MemoryMappingHandler(db_path: str | Path | None = None)
```

| Parameter | Meaning |
| :--- | :--- |
| `db_path` | Where the table lives. `None` uses `data/memory_mapping/memory_mapping.sql` |

**It takes no `conversation_id`.** One handler serves the whole table; every
method names the conversation it acts on. The connection is opened
`check_same_thread=False`, WAL is enabled, and every statement runs under an
`RLock`.

| Method | Returns | Raises |
| :--- | :--- | :--- |
| `populate_conversation_id(conversation_id, user_id)` | `None` | `InvalidIdentifier`, `sqlite3.IntegrityError` if the pair already has a row |
| `insert_into_mapping_table(conversation_id, user_id, topic_id, project_id, new_latest_project_snapshot_id, created_at)` | `None` | `InvalidIdentifier` |
| `update_latest_project_snapshot_id(user_id, project_id, new_latest_project_snapshot_id)` | `None` | `InvalidIdentifier` |
| `search(conversation_id, user_id)` | `MemoryMapping` or `None` | `InvalidIdentifier` |
| `close()` | `None` | — |

`journal_mode` is set on construction and reports the journal actually in force
(`"wal"` unless the filesystem refused it).

### `MemoryMapping`

```python
MemoryMapping(topic_id, project_id, latest_project_snapshot_id)
```

A `NamedTuple`, so it compares equal to a plain 3-tuple. Every field is
`str | None`: a row opened by `populate_conversation_id` has all three unset
until routing fills them in.

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
    topic_id="topic_1", project_id="project_1",
    new_latest_project_snapshot_id="snap_1", created_at=utc_now(),
)
handler.search("conv_1", "user_1")
# MemoryMapping('topic_1', 'project_1', 'snap_1')
```

`insert_into_mapping_table` is an UPDATE despite the name, so it needs the row
to exist. A zero-row UPDATE is not an error to SQLite, so calling it first would
otherwise look like success — it **logs a warning** instead, naming the
conversation and telling you to call `populate_conversation_id`.

`search` on a conversation with no row returns `None`, not a tuple of `None`s.
The two are different: the first means "never seen", the second "seen, not yet
routed".

---

## Schema

```sql
memory_mapping_table
    conversation_id             text not null
    user_id                     text not null
    topic_id                    text
    project_id                  text
    latest_project_snapshot_id  text
    created_at                  text
    primary key (conversation_id, user_id)

idx_memory_mapping_project on (project_id, user_id)
```

The primary key is what makes "which project is this conversation in" answerable
— without it the same conversation could hold two rows and the question would
have two answers. The index serves `update_latest_project_snapshot_id`, which
rewrites by `(project_id, user_id)`, a pair no part of the key covers.

`latest_project_snapshot_id` is a **cache** of
`ProjectSnapshotRepository.latest()`, which owns the ordered chain in
`project_snapshot_mapping`. It is held here so resuming a conversation does not
have to open the project registry. It is not the history, and nothing should
read it as one.

---

## Scope note

`user_id` is the only user-scoped column in the memory layer; the topic, project
and conversation tables have no notion of a user. Either that dimension has to
reach them too, or this column is ahead of a decision that has not been made.
See `bugs.md` 4.83 and the scope note in Section 4c.

---

## Tests

`test/memory_layer_testing/test_memory_mapping_handler.py` — 16 tests, the
regression guards for Bugs 4.73-4.83. Reverting any one fix fails between 1 and
11 of them.
