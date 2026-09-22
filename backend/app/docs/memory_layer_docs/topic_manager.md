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
| `query` | The query this manager was opened for. Validated, stored, **not yet read** (bug 4.55) |
| `topic_pool_path` | Where the registry lives. `None` uses `data/topic_db/topic.sql` |

| Method | Returns | Raises |
| :--- | :--- | :--- |
| `create_new_topic()` | `None` | `Exception` if an active topic with that name exists |
| `get_topic_id()` | `str` | `Exception` if no active topic has that name |
| `soft_delete()` | `None` | `Exception` if no active topic has that name; `ValueError` if the stored id is not a `str` |
| `close()` | `None` | — |

Also a context manager, which is the recommended form:

```python
with TopicManager("retrieval", "how does routing work?") as manager:
    manager.create_new_topic()
    topic_id = manager.get_topic_id()
```

The exception type is `Exception` in all three cases, so callers cannot
currently tell "no such topic" from "already exists" without matching the
message — see bug 4.53.

---

## `TopicPoolMetaHandler`

```python
TopicPoolMetaHandler(topic_pool_path: str | Path | None)
```

| Method | Returns |
| :--- | :--- |
| `is_topic_exists(topic)` | `bool` — whether an **active** topic has that name |
| `get_topic_id(topic)` | `str` or `None` |
| `create_new_topic(topic_name, topic_id, created_at)` | `None`. `created_at` may be `str`, `date`, `datetime` or `None` (now) |
| `soft_delete(topic_id)` | `None`. Raises `ValueError` if no row has that id |
| `close()` | `None` |

### Schema

```sql
create table if not exists topics_mapping_table(
    topic_id text primary key not null,
    topic_name text not null,
    created_at DATE NOT NULL,
    is_active CHAR(2)
);
```

`is_active` is `'t'` or `'f'`. Every read filters on `'t'`.

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
| 4.52 | `create_new_topic()` checks then writes, so two threads can create two active rows with the same name |
| 4.53 | All failures raise bare `Exception` |
| 4.54 | `get_topic_id()` runs two SELECTs; `soft_delete()` runs two plus an UPDATE |
| 4.55 | `query` is required, validated and never read |
| 4.56 | Nothing can list the topics |
| 4.57 | The project tables' `topic_id` has no foreign key onto this table — different SQLite files |

## Tests

`test/memory_layer_testing/test_topic_manager.py` — 24 tests. The close-under-write
test runs in a subprocess and asserts on the exit signal, because the failure it
guards against takes the interpreter down rather than raising.
