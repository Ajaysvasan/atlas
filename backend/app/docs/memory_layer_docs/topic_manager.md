# Topics (`TopicManager` & `TopicPoolMetaHandler`)

## Overview & Purpose

A topic is the outermost grouping in the memory hierarchy: a set of projects
that belong together. `TopicManager` owns the rules about when a topic may be
created, read or removed; `TopicPoolMetaHandler` owns the SQLite table behind it.

Architecture and rationale live in `memory/topic_pool/README.md` and
`memory/topic_pool/topic_pool_repo/README.md`. This page is the API.

---

## `TopicManager`

```python
TopicManager(topic: str, query: str, topic_pool_path: str | Path | None = None)
```

| Parameter | Meaning |
| :--- | :--- |
| `topic` | The topic name. Empty or `None` raises `ValueError` |
| `query` | Optional. Held for the topic -> project handoff, which does not exist yet |
| `topic_pool_path` | Where the registry lives. `None` uses `data/topic_db/topic.sql` |

| Method | Returns | Raises |
| :--- | :--- | :--- |
| `create_new_topic()` | `str` — the new topic id | `TopicAlreadyExists` |
| `get_topic_id()` | `str` | `TopicNotFound` |
| `soft_delete()` | `str` — the id deactivated | `TopicNotFound` |
| `list_topics()` | `List[Topic]` — active topics, oldest first | — |
| `close()` | `None` | — |

Also a context manager, which is the recommended form:

```python
with TopicManager("retrieval", "how does routing work?") as manager:
    manager.create_new_topic()
    topic_id = manager.get_topic_id()
```

`TopicNotFound` and `TopicAlreadyExists` live in `memory/memory_pool_exceptions.py`
and both carry the topic name.

---

## `TopicPoolMetaHandler`

```python
TopicPoolMetaHandler(topic_pool_path: str | Path | None)
```

| Method | Returns |
| :--- | :--- |
| `is_topic_exists(topic)` | `bool` — whether an **active** topic has that name |
| `get_topic_id(topic)` | `str` or `None` |
| `create_new_topic(topic_name, topic_id, created_at)` | `None`. Raises `TopicAlreadyExists`. `created_at` may be `str`, `date`, `datetime` or `None` (now) |
| `soft_delete_by_name(topic_name)` | `str` — the id deactivated. Raises `TopicNotFound` |
| `soft_delete(topic_id)` | `None`. Raises `ValueError` if no row has that id |
| `get_all_topics()` | `List[Topic]` — `(topic_id, topic_name, created_at)`, active only, oldest first |
| `close()` | `None` |

### Schema

```sql
create table if not exists topics_mapping_table(
    topic_id text primary key not null,
    topic_name text not null,
    created_at DATE NOT NULL,
    is_active CHAR(2)
);

create unique index if not exists idx_active_topic_name
on topics_mapping_table(topic_name) where is_active = 't';
```

`is_active` is `'t'` or `'f'`. Every read filters on `'t'`.

The index does two jobs. It makes "one active topic per name" a database
constraint rather than something the caller checks first, and it turns every
name lookup from a `SCAN` into a `SEARCH`. It is **partial** so that soft delete
still works: inactive rows are unconstrained, and a name can be created and
deleted repeatedly.

### Concurrency

One connection, opened through `memory/sqlite_setup.connect()` with
`check_same_thread=False` (WAL, `synchronous=NORMAL`, `foreign_keys=ON`). Every
statement — and `close()` — runs under an `RLock`.

Both halves matter. Without `check_same_thread=False`, SQLite rejects use from
any other thread and the lock protects something unreachable. Without the lock,
`close()` racing a statement segfaults the interpreter rather than raising.

---

## Known limitations

| Bug | |
| :--- | :--- |
| 4.55 | `query` is optional now, but still unread — it closes when the topic -> project handoff is built |
| 4.57 | The project tables' `topic_id` has no foreign key onto this table, and cannot have one while the two registries are separate SQLite files |

Bugs 4.52, 4.53, 4.54, 4.56 and 4.58 were fixed here; see `bugs.md`.

## Tests

`test/memory_layer_testing/test_topic_manager.py` — 37 tests. The close-under-write
test runs in a subprocess and asserts on the exit signal, because the failure it
guards against takes the interpreter down rather than raising.
