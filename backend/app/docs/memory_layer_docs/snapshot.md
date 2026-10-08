# Snapshot history (`memory/snapshot.py`)

## Overview & Purpose

`SnapShot` walks a history of snapshots with two cursors and finds the one most
similar to a query. It serves two scopes with the same cursors and search:

| Scope | History | Needs |
| :--- | :--- | :--- |
| `"conversation"` (default) | this conversation's cumulative summaries, from `ConversationVectorMetaDataRepository` | a `conversation_id` |
| `"project"` | the project's snapshot chain, from `ProjectSnapshotRepository` | no `conversation_id` |

Each scope stores its vector under the id its listing returns, which is why one
search serves both. Architecture is in `memory/topic_pool/project_pool/conversation_pool/README.md`;
this page is the API.

The older `SnapShotNode` and the parameterless constructor this page used to
describe no longer exist.

---

## `SnapShot`

```python
SnapShot(
    project_id: str,
    project_name: str,
    conversation_id: str = "",
    meta_repo: ConversationVectorMetaDataRepository | None = None,
    scope: str = "conversation",
    snapshot_repo: ProjectSnapshotRepository | None = None,
    database: MemoryDatabase | str | Path | None = None,
)
```

| Parameter | Meaning |
| :--- | :--- |
| `conversation_id` | Required for the conversation scope (`InvalidIdentifier` if blank); ignored for the project scope |
| `meta_repo` | The conversation scope's metadata. Built from `database` when not given; a given one is borrowed and never closed here |
| `scope` | `"conversation"` or `"project"`; anything else raises `InvalidSnapshotScope` |
| `snapshot_repo` | The project scope's chain. Built from `database` when not given |
| `database` | A `MemoryDatabase`, or a path to one; `None` uses the shared memory database. Only used to build a repository that was not given |

Each scope builds only the repository it reads, so a project snapshot opens no
conversation metadata and a conversation snapshot opens no snapshot chain. The
pgvector connection is built on first use (`vector_manager`).

| Method | Returns | Notes |
| :--- | :--- | :--- |
| `add(time_of_snapshot, len_of_the_summary, summary_vector_ids, summary_vectors, chunk_ids, chunks, summary, cumulative_summary_vector_id, cumulative_summary_vector, reset_right_pointer=False, reset_left_pointer=False)` | `None` | Conversation scope only, else `WrongSnapshotScope`. Vectors first, then one metadata transaction, with a compensating delete. The first add sets both cursors to 0; later ones move the right cursor |
| `add_project_snapshot(summary, last_seq_included, summary_vector, created_at=None)` | `int` — the snapshot id | Project scope only. Same ordering and compensation |
| `advance()` | `None` | Moves the left cursor right; `InvalidCursorException("left", …)` past the right cursor or the end |
| `prev()` | `None` | Moves the right cursor left; `InvalidCursorException("right", …)` past the left cursor or the start |
| `sync_cursors()` | `None` | Points the cursors at the whole stored history; `NullPointerException` if there is none |
| `search(query)` | the best-matching snapshot id in the cursor range, or `None` | Cosine similarity (torch). Does not move the cursors (bug 4.3). Cursors never set are opened onto the stored history first |
| `close()` | `None` | Closes the pgvector connection and any repository this instance built; borrowed ones stay open. The memory database is shared and is not closed |

---

## Tests

`test/memory_layer_testing/test_snapshot.py`.
