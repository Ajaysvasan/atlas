# The memory database (`MemoryDatabase`)

## Overview & Purpose

`memory/memory_database.py` is the one SQLite database every table in the memory
layer lives in, and the one connection every table owner uses. It holds no
table's SQL: each owner module declares its tables as a `Schema` and keeps its
own queries. Architecture and rationale are in `memory/README.md`; this page is
the API.

The file is `Config.MEMORY_DB`, by default
`data/memory/memory_layer/memory_layer.db`.

---

## `MemoryDatabase`

```python
MemoryDatabase(path: str | Path | None = None)
```

Opens `path` (default `Config.MEMORY_DB`, read when called, not at import),
creating its directory. Every open goes through `storage.sqlite_setup.connect()`
with `check_same_thread=False` and `isolation_level=None` — transactions are
opened explicitly by `writing()`. Puts the file into WAL and stamps the schema
version.

| Member | Behaviour |
| :--- | :--- |
| `MemoryDatabase.shared(path=None)` | The process-wide instance for `path`, keyed on the resolved path, so two spellings of one file share it. Reopens one that was closed |
| `MemoryDatabase.of(database)` | What owners call: a `MemoryDatabase` is returned as is; a path or `None` gives `shared(path)` |
| `MemoryDatabase.close_shared()` | Closes and forgets every shared instance |
| `ensure(schema)` | Creates `schema`'s tables, after every schema in its `requires`, once per instance |
| `reading()` | Context manager yielding a cursor under the lock |
| `writing()` | Context manager yielding a cursor inside a transaction — see below |
| `close()` | Closes the connection; safe to call twice. Later use raises `sqlite3.ProgrammingError` |
| `path`, `journal_mode`, `connection` | The file, the journal actually in force, the connection |

Owners take `database: MemoryDatabase | str | Path | None = None` and pass it to
`of()`. An owner's own `close()` releases nothing: the database outlives its
owners and is closed by whoever opened it.

### `writing()`

| Situation | What happens |
| :--- | :--- |
| Outermost call | `BEGIN IMMEDIATE`; `COMMIT` on success, `ROLLBACK` on any exception |
| Nested call | `SAVEPOINT`; released on success, rolled back to on an exception. A failure the caller catches undoes only the nested part |
| `COMMIT` itself fails (a deferred constraint, `SQLITE_BUSY`) | `ROLLBACK`, then the error. Otherwise the shared connection would stay inside the failed transaction and every later `BEGIN` would fail |

`BEGIN IMMEDIATE` takes the write lock before the first read, so a
read-then-insert — the next `sequence_number`, the next `seq` — cannot
interleave with a writer in another process. In this one, the lock already
serialises it.

### `Schema`

```python
Schema(name: str, create: Callable[[sqlite3.Cursor], None], requires: Tuple[Schema, ...] = ())
```

Each owner module defines one at module level and calls
`self.database.ensure(SCHEMA)` in its constructor. `requires` lists the schemas
whose tables its foreign keys reference, so they exist whichever owner is built
first. `create` runs inside one `writing()` transaction and should use
`create table if not exists` / `create index if not exists`, so reopening an
existing file adds a missing index without a migration.

| Schema | Owner | Requires |
| :--- | :--- | :--- |
| `topics` | `topic_pool_meta_handler.py` | — |
| `projects` | `project_meta_data.py` | `topics` |
| `project_snapshots` | `project_snapshot_repo.py` | `projects` |
| `conversation_turns` | `fullconversation_repository.py` | `projects` |
| `conversation_snapshots` | `conversationVectorMetaManager.py` | `projects`, `conversation_turns` |
| `memory_mapping` | `memory_mapping_handler.py` | `projects`, `project_snapshots` |

### Schema version

`SCHEMA_VERSION = 1`, kept in `PRAGMA user_version`. A new file is stamped; a
file with a higher version raises `NewerMemorySchema` and the connection is
closed, since older code writing into a newer layout could corrupt it. A future
change to a table's shape bumps the version and adds its step here.

---

## Connection setup

| Pragma | Value | Scope | Set by |
|---|---|---|---|
| `journal_mode` | `WAL` | Written into the **file**; survives every later open, across processes | `enable_wal()`, at construction |
| `synchronous` | `NORMAL` | Per **connection**; resets to `FULL` on every open | `connect()` |
| `foreign_keys` | `ON` | Per **connection**; resets to off on every open | `connect()` |
| `busy_timeout` | 5 s | Per connection | `sqlite3.connect`'s default `timeout` |

A raw `sqlite3.connect()` would get `synchronous=FULL` and foreign keys off, so
nothing in `memory/` opens its own; a test reads every module and fails on any
`connect(` outside this one.

`enable_wal()` cannot convert the file while another connection holds a write
transaction. Losing that race returns the current mode instead of raising: the
database is still correct in a rollback journal, just slower, and the next open
tries again. A file left in the rollback journal converts on the next open.

### Why WAL and NORMAL

Under the default rollback journal a reader holding a transaction open blocks
**every** writer with `database is locked`. `synchronous=FULL` then fsyncs the
log on every commit on top of that. Measured before the move to one database,
through the real conversation classes on an NVMe/btrfs filesystem — one writer
appending turns against six readers, plus a separate metadata commit loop:

| journal | synchronous | appends/s | reads/s | metadata commits/s |
|---|---|---|---|---|
| `delete` | `FULL` | ~330 | ~5 | ~390 |
| `wal` | `FULL` | ~850 | ~6500 | ~1000 |
| `wal` | `NORMAL` | ~2200 | ~5300 | ~20000–100000 |

WAL is what unblocks reads; `NORMAL` is what unblocks commits. **Measure these on
real storage, not `/tmp`** — it is `tmpfs` on most Linux systems, where `fsync`
is free and both settings look like they do nothing.

### The durability trade

`synchronous = NORMAL` is a deliberate weakening:

- **Process crash — safe.** The write-ahead log is on disk and intact; the next open recovers every committed transaction.
- **OS crash or power loss — the last few committed transactions can be rolled back.** They were written but not fsynced.
- **Corruption — not a risk either way.** WAL keeps the database consistent; the exposure is bounded to losing recent commits.

Set `SYNCHRONOUS = "FULL"` in `storage/sqlite_setup.py` to reverse it.

### Operational notes

- **The database is not one file.** SQLite keeps `memory_layer.db-wal` and `memory_layer.db-shm` beside it. Copying the `.db` alone can lose recent commits.
- **WAL needs shared memory**, so it does not work over most network filesystems; `enable_wal()` then keeps the existing mode rather than failing.

---

## In tests

The root `conftest.py` points `Config.MEMORY_DB` at a fresh file per test and
closes the shared instances afterwards, so a store built without a database
never reaches the developer's own. Tests that need several owners in one
database pass the same path or instance to each, and register topics and
projects first through the `seed_topics` / `seed_projects` fixtures in
`test/memory_layer_testing/conftest.py`, since every child row now needs its
parent.

## Tests

`test/memory_layer_testing/test_memory_database.py`.
