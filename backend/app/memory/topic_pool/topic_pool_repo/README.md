# Topic storage (`topic_pool_repo/`)

## What this module does

Stores the topics themselves: one row per topic, in SQLite. `TopicPoolMetaHandler`
is the only thing that touches that table; `TopicManager` sits on top of it and
owns the rules about when each operation is allowed.

## Schema

```sql
create table if not exists topics_mapping_table(
    topic_id text primary key not null,
    topic_name text not null,
    created_at DATE NOT NULL,
    is_active CHAR(2)
);
```

`is_active` holds `'t'` or `'f'`. Every read filters on `'t'`, so a soft-deleted
topic is invisible to `is_topic_exists` and `get_topic_id` while its row stays on
disk.

The default file is `data/topic_db/topic.sql`; a caller can pass its own path,
which is what the tests do.

## Why deletion is soft

A topic owns projects, which own conversations and vectors in two stores. Hard
deleting the topic row would orphan all of that with no way to find it again.
Flipping `is_active` keeps the history addressable and leaves the decision about
what to do with the descendants to a retention policy that does not exist yet
(`todo.md` 1.3).

A consequence worth knowing: creating a topic, soft-deleting it and creating it
again leaves **two rows with the same `topic_name`** — one inactive, one active,
with different ids. That is intended, and it is why the uniqueness constraint has
to be *partial*:

```sql
create unique index idx_active_topic_name
on topics_mapping_table(topic_name) where is_active = 't';
```

Only active rows are constrained, so a name can cycle through create and delete
any number of times while never having two live rows. A plain
`unique(topic_name, is_active)` looks equivalent and is not — it allows one
soft-deleted row per name, so the *second* delete of a recreated topic fails.

The same index answers every name lookup. Without it each check was a `SCAN` of
the whole table: 211 microseconds at ten thousand topics against 2.3 with it.
Creating a topic is therefore a bare INSERT that converts `IntegrityError` into
`TopicAlreadyExists` — there is no check to race.

## Why the connection is shared and locked

One connection, opened through `memory/sqlite_setup.connect()` with
`check_same_thread=False`, and every statement taken under an `RLock` through
`__reading()` / `__writing()`.

Both halves are needed. Without `check_same_thread=False` SQLite rejects any use
from another thread outright, so the lock would be protecting something nothing
else could reach. Without the lock, `commit()` and `rollback()` apply to the
whole connection rather than to one cursor, so threads publish each other's
half-written transactions.

`close()` takes the same lock. Closing a connection while another thread is
mid-statement does not raise — it segfaults the interpreter, which is why the
test for it runs in a subprocess and asserts on how that process died.

## Why an UPDATE that matches nothing raises

SQLite treats it as success. `soft_delete` therefore checks `rowcount` and raises
`ValueError`, or deleting a topic id that does not exist would silently report
that it worked.

## Tests

`test/memory_layer_testing/test_topic_manager.py` — 24 tests covering both this
module and `TopicManager`.
