# Project snapshots (`ProjectSnapshot` & `ProjectSnapshotRepository`)

## Overview & Purpose

A project snapshot is a rolling description of a project: what it is about and
what has been done so far. It is not a conversation summary — it reads across
every conversation in the project and describes the project, not the chats.

`ProjectSnapshot` builds them; `ProjectSnapshotRepository` stores them.
Architecture and rationale live in
`memory/topic_pool/project_pool/README.md` and
`memory/topic_pool/project_pool/project_data_repo/README.md`. This page is the API.

---

## `ProjectSnapshot`

```python
ProjectSnapshot(
    project_id: str,
    project_name: str,
    meta_repo: ConversationVectorMetaDataRepository,
    embed: Callable[[str], ndarray],
    snapshot_repo: ProjectSnapshotRepository | None = None,
    database: MemoryDatabase | str | Path | None = None,
)
```

| Parameter | Meaning |
| :--- | :--- |
| `meta_repo` | Supplies the conversation summaries. Injected rather than built, because a project-wide read needs no `conversation_id` |
| `embed` | `str -> ndarray`. Cheap and stable, so it is held |
| `snapshot_repo` | The project's snapshot chain. Built when not given |
| `database` | Defaults to `meta_repo.database`, so the conversation summaries and the project's snapshots are read from one database |

| Method | Returns |
| :--- | :--- |
| `pending()` | `List[Tuple[seq, cumulative_vector_id, summary]]` — summaries not yet folded in |
| `take(summarise)` | `int` (the new snapshot id) or `None` if nothing is new |
| `close()` | `None` |

### Why `summarise` is a parameter of `take`, not the constructor

```python
snapshot.take(lambda system, user: run_inference(model, system, user))
```

The draft model costs seconds to load and its lifetime belongs to the caller.
Passing the summariser per call lets `ConversationSummary` roll the project
description forward from inside a window where a model is **already resident**,
instead of loading a second copy. `take` supplies `PROJECT_SNAPSHOT_SYSTEM` as
the system half and `render_prompt(...)` as the user half.

### Incremental, and what that guarantees

Snapshot *N* summarises snapshot *N-1* plus the conversation summaries written
since it. Cost tracks what changed rather than the size of the project: fifty
conversations never go into one prompt.

The watermark is `seq` on `cumulative_vector_meta_data`, allocated monotonically
on write under `BEGIN IMMEDIATE` — not `created_at`, which is caller-supplied
`TEXT` with no format enforcement.

**An empty summary does not advance the watermark.** Advancing it would step
"new since" past those conversation summaries and drop them from every future
snapshot, so `take` returns `None` and logs instead.

---

## `ProjectSnapshotRepository`

```python
ProjectSnapshotRepository(project_id: str,
                          database: MemoryDatabase | str | Path | None = None)
```

`database` is a `MemoryDatabase` or a path to one; `None` uses the shared memory
database. `project_id` must be non-blank (`InvalidIdentifier`).

| Method | Returns |
| :--- | :--- |
| `add_snapshot(summary, last_seq_included, created_at=None)` | `int` — the snapshot id. Raises `ProjectNotFound` for a project never registered, with nothing written |
| `latest()` | `ProjectSnapshotRow` or `None` |
| `last_seq_included()` | `int` — the watermark, `0` for a project with no snapshots |
| `history()` | `List[ProjectSnapshotRow]`, oldest first |
| `get_snapshot(snapshot_id)` | `ProjectSnapshotRow` or `None` |
| `close()` | `None`. Releases nothing — the database is shared |

```python
ProjectSnapshotRow(project_snapshot_id, project_id, summary,
                   len_of_the_summary, last_seq_included, created_at)
```

### `project_snapshot_id`

```python
project_snapshot_id(project_id: str, created_at: str, summary: str) -> int
```

Derived, never passed, and masked into the signed 64-bit range because **it is
also the pgvector `vector_id`** — one id for both stores, so a row needs no
lookup to reach its embedding. `created_at` is part of the derivation, so a
project summarised twice to the same words is still two snapshots.

---

## Schema

```sql
project_snapshot
    project_snapshot_id   integer primary key
    project_id            text not null references project_table(project_id)
    summary               text not null
    len_of_the_summary    integer not null
    last_seq_included     integer not null
    created_at            text not null
    unique (project_snapshot_id, project_id)

project_snapshot_mapping
    project_id            text not null
    project_snapshot_id   integer not null
    created_at            text not null
    primary key (project_id, project_snapshot_id)
    foreign key (project_snapshot_id, project_id)
        references project_snapshot(project_snapshot_id, project_id)

idx_project_snapshot_project          on project_snapshot(project_id)
```

The mapping's foreign key is the pair, so the chain cannot list another
project's snapshot. It has no index of its own: its primary key already leads
with `project_id`.

The mapping is **append-only**: it holds the ordered chain of every snapshot a
project has had, not a pointer at the newest. Overwriting one row would leave
the earlier snapshots in `project_snapshot` with nothing ordering them.

Writes go vectors-first, metadata-second, with a compensating delete — so a
failure leaves an unreachable vector rather than a row pointing at nothing.

---

## Tests

`test/memory_layer_testing/test_project_snapshot.py` (15) and
`test_project_snapshot_repo.py` (19).
