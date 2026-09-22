# Project Registry (`ProjectMetaData`)

## Overview & Purpose

`memory/topic_pool/project_pool/project_data_repo/project_meta_data.py` is the
storage half of the project layer: what a project is, what describes it, and
which summary vectors belong to it. `ProjectManager` is the domain object that
sits on top of it, routing a query to one of these projects.

One instance is scoped to one `project_id` **and its `topic_id`** —
`ProjectMetaData(project_id, topic_id, ...)`. A project belongs to exactly one
topic, and `topic_id` is `NOT NULL` in all three tables, so it is supplied once
at construction rather than per call. The SQLite file itself is a shared registry
holding every project, which is what makes "list all projects" answerable later.

---

## What lives where

| Store | Holds |
| :--- | :--- |
| `project_table` (SQLite) | One row per project: name, topic, timestamps, the summary **text**, owner |
| `project_description_table` (SQLite) | Several descriptions per project |
| `project_mapping_table` (SQLite) | `project_id` → summary vector id |
| `vectors` (PostgreSQL/pgvector) | The summary **embedding** |

The embedding goes to pgvector, not to the DiskANN index — DiskANN indexes
ingested documents and has nothing to do with project summaries.

### Schema

```sql
create table if not exists project_table(
    project_id text primary key,
    project_name text not null,
    topic_id text not null,
    created_at date not null,
    updated_at date not null,
    project_summary text not null,
    user_id text
);

create table if not exists project_description_table(
    topic_id text not null,
    project_id text not null,
    project_description_id text not null,
    project_description text not null,
    created_at date not null,
    primary key (project_id, project_description_id),
    foreign key (project_id) references project_table(project_id)
);

create table if not exists project_mapping_table(
    project_id text not null,
    topic_id text not null,
    project_summary_vector_id integer not null,
    created_at date not null,
    primary key (project_id, project_summary_vector_id),
    foreign key (project_id) references project_table(project_id)
);
```

### The topic column in the child tables

Both child tables carry their own copy of `topic_id`, which is derivable from
`project_table` by a join. It is denormalised on purpose: routing a query scores
every summary vector under one topic, and that read is then a single statement
against `project_mapping_table` with no join.

A denormalised copy is only safe if it moves when the project moves, so
`__upsert_project` rewrites both child tables' `topic_id` in the same
transaction whenever a write carries a different topic than the row holds. Left
out, those rows keep answering for the topic the project has left. Note this
holds for writes that go through `ProjectMetaData` — a raw `UPDATE` elsewhere
can still desynchronise them.

`topic_id` has no foreign key, because there is no topic table yet
(`TopicManager` owns `topics_mapping_table`, but it lives in a different SQLite
file, so no foreign key can reach it — bug 4.57). The only thing standing
between a typo and a
project filed under a topic that does not exist is the non-empty check in the
constructor.

Both child tables key on `project_id` first, so two projects may reuse the same
description id or the same vector id without colliding — and content-derived
vector ids do collide across projects.

---

## Writing

```python
meta.add_project_vector(vector, vector_id, project_name, project_summary)
meta.add_batch_project_vector(vectors, vector_ids, project_name, project_summary)
```

`project_summary` is the text the vector was embedded from. It is required
because `project_table` stores it `NOT NULL`, and it is validated **before** the
embedding is written — a bad summary must not leave an orphaned vector for the
compensating delete to clean up.

Writes go **vectors first, metadata second, with a compensating delete on
failure**. The two stores cannot share a transaction, so one of the two failure
modes has to be chosen: an unreachable vector is recoverable, a mapping row
pointing at a missing embedding is not. This is the same ordering settled for
snapshots in `SnapShot.__add_snap_shot`.

### What an existing project keeps

`project_name`, `project_summary` and `updated_at` land on every write. The
summary is regenerated as the project moves on, and the row holds the current
one, the way it holds the current name. `created_at` and `user_id` are left
alone: `created_at` is by definition the first write, and adding a summary
vector is not a transfer of ownership.

---

## Descriptions

```python
meta.add_description("goal", "Answer questions over a private corpus.")
meta.add_descriptions([("goal", "..."), ("scope", "...")])
meta.get_description("goal")     # -> str | None
meta.get_descriptions()          # -> [(description_id, description, created_at)]
```

- A project holds several, told apart by a caller-supplied id. The table is
  keyed `(project_id, project_description_id)`, so the same id means the same
  description and rewriting it is an update, not a second row.
- `created_at` survives a rewrite, for the same reason `project_table`'s does:
  it records when the description first appeared.
- The project row must already exist. The foreign key rejects a description for
  a project nothing has written yet.
- Ids and text must be non-empty strings, and a batch is validated in full
  before any row is written, so a bad pair partway through cannot leave the
  first half committed. A repeated id inside one batch raises `MisMatchCount`.
- `updated_at` on `project_table` is left alone; the summary-vector paths own it.

`get_descriptions()` returns oldest first, ordered by `datetime(created_at)`,
then the raw `created_at`, then `rowid`. `datetime()` lets a caller-supplied
stamp in another format still sort chronologically; it truncates to whole
seconds, so the raw column is what orders everything written inside one second;
`rowid` keeps identical stamps in insertion order.

---

## Reading

| Method | Returns |
| :--- | :--- |
| `get_project()` | A `ProjectRow` — `project_id, project_name, topic_id, created_at, updated_at, project_summary, user_id` — or `None`. A `NamedTuple`, so `row.topic_id` works and indexing still does too |
| `get_topic_summary_vector_ids()` | `[(project_id, vector_id)]` for **every project under this topic**, which is what routing needs: it scores all the topic's vectors and must know which project each one belongs to |
| `get_summary_vector(vector_id)` | One embedding. Raises `VectorNotFoundEror` if absent |
| `get_all_summary_vector_id()` | Every vector id, oldest first — from SQLite alone, no PostgreSQL round trip |
| `get_all_summary_vector()` | Every embedding, oldest first; row *i* matches id *i* |
| `get_description(id)` / `get_descriptions()` | See above |

---

## `ProjectVectorHandler` (`project_vector_handler.py`)

A sibling module, and the project-level counterpart to
`ConversationVectorManager`. A project holds **one summary vector and one vector
per description** — the count `project_mapping_table` has always allowed.

```python
handler = ProjectVectorHandler()
handler.add_project_summary_vector(project_id, vector)
handler.add_project_description_vector(project_id, description_id, vector)
handler.add_project_description_vectors(project_id, [(description_id, vector), ...])
handler.get_project_vectors(project_id, vector_ids)   # batch, in the order given
handler.close()                                       # or use it as a context manager
```

Each kind has the same four operations (`add`, `update`, `delete`, `get`).

**Ids are derived from the source, not passed.** No method takes a vector id:

| Source | Id |
| :--- | :--- |
| the project's summary | `summary_vector_id(project_id)` |
| one description | `description_vector_id(project_id, description_id)` |

Both fold `sha256` into `Config.VECTOR_ID_MASK`. The payload is
`kind \x00 project_id \x00 source_id`: NUL separates the fields so `("a","bc")`
and `("ab","c")` stay different inputs, and the kind is what keeps a future
third source — a title, a tag — from colliding with a description that shares
its id. Deriving rather than storing means add, update, get and delete agree on
which row they mean without a lookup, and a vector can be regenerated from the
text it came from after an embedding model change.

**Enumeration lives in SQLite, not here.** `VectorRepository` can only fetch by
id — it has no "list this project's vectors" — so the router gets its ids from
`ProjectMetaData.get_topic_summary_vector_ids()` and passes them to
`get_project_vectors()`, which returns row *i* for `vector_ids[i]`. That
ordering is what lets a similarity score be traced back to the vector, and
through it the project, that produced it.

| Call | Behaviour |
| :--- | :--- |
| `add` | Raises `DuplicateVectorException` if the project already has one — the slot is taken, and replacing it is `update`'s job rather than something an add does silently |
| `update` | One UPDATE, not delete-then-insert: the pair is two commits and a failure between them loses the vector. Raises `VectorNotFoundEror` if there is nothing to update |
| `delete` | Idempotent — a project with no vector is not an error |
| `get` | Raises `VectorNotFoundEror` if absent |

Project ids and vector widths are validated **before** the store is touched, so
a malformed call costs no connection. A bad width raises the memory layer's
`InvalidVectorDimension`, not the data layer's exception of the same name.

`VectorRepository` opens its PostgreSQL connection in the constructor, so the
handler keeps one repository per project it has touched rather than building one
per call. `close()` releases them all.

### How it lines up with `ProjectMetaData`

They now agree on the count: many vectors per project, one per source. What is
still split is who writes what. `ProjectMetaData` owns the SQLite side and takes
a vector id from the caller; the handler owns the pgvector side and derives its
own. Nothing routes between them yet, so a project's vectors can still be
written through two paths — pass `summary_vector_id(project_id)` into
`ProjectMetaData.add_project_vector` to keep the two in step by hand.

The remaining step is delegation: `ProjectMetaData` calling the handler for its
vector half instead of holding its own `VectorRepository`. Worth doing when
`ProjectManager` is written, since that is the first caller that touches both.

---

## Known limitation

The connection is a plain `sqlite3.connect`: no lock, no WAL, and usable only
from the thread that opened it — while the file is shared by every project. Two
`ProjectMetaData` instances writing at once will contend. This is the same
problem already fixed for the conversation database in `sqlite_setup.connect()`,
and it should be fixed here before `ProjectManager` opens projects concurrently.
See `todo.md` section 2.

---

## Tests

`test/memory_layer_testing/test_project_meta_data.py` — 139 tests. The vector
store is a stateful fake rather than a `MagicMock`: every bug that mattered in
this layer was the two stores disagreeing, and a mock records calls without
holding state, so it cannot show a disagreement.

Checked against deliberate regressions, each of which fails at least one test:
the description foreign key dropped, its primary key narrowed to `project_id`, a
rewrite resetting `created_at`, the ordering tiebreaker removed, summary
validation made a no-op or moved after the vector write, the summary not
refreshed on conflict, a batch validated row by row instead of up front, and
either description reader ignoring the project scope.
