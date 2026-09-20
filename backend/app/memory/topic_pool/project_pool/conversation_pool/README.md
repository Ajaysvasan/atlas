# Conversation pool (`conversation_pool/`)

## What this module does

One conversation's memory: the turns as they were said, a rolling summary that
keeps up with them, and snapshots that make the summary searchable.

## The pieces

```
ConversationPoolManager        the only thing a caller should need
  ├── FullConversation         append and read turns
  │     └── FullConversationRepository      SQLite
  ├── ConversationSummary      draft-model summarisation
  │     └── SnapShot           snapshot history + similarity search
  │           ├── ConversationVectorMetaDataRepository   SQLite
  │           └── ConversationVectorManager              pgvector
  └── sqlite_setup.connect()   every SQLite open goes through here
```

## How a turn flows

```
record_turn(role, text)
   -> append_turn            one row in summary_chunks + one in full_conversation
   -> maybe_snapshot         has SNAPSHOT_EVERY_N_TURNS accumulated?
        -> take_snapshot     window -> draft model -> summary
             -> embed        cumulative vector + one per covered chunk
             -> SnapShot.add vectors first, then one metadata transaction
```

## Why it is wired through one manager

`FullConversation`, `ConversationSummary` and `SnapShot` each work alone but
have to agree on three things: the same directory, the same project identity,
and **one shared `SnapShot`** so its cursors survive between calls. Building
them ad hoc is how the summariser ended up reading a different database than the
snapshot writer, and how a second `SnapShot` ended up with its own cursors that
drifted from the first. `ConversationPoolManager` is the only place that wiring
lives.

## Why turns and snapshot metadata share one database file

`{project_id}_conversation.db` holds both. That is what lets
`get_highest_summarised_sequence()` join snapshot metadata against
`full_conversation` in one statement, with no second connection and no
cross-database consistency problem.

Because two classes open the same file, **every open goes through
`sqlite_setup.connect()`**. `journal_mode` is written into the file and survives
every later open; `synchronous` and `foreign_keys` are per-connection and reset
to their defaults each time. A new method with a raw `sqlite3.connect()` would
silently get `FULL` and no foreign keys — which has happened, in `__add_chunks`,
where it let orphan rows into `full_conversation`. A test asserts neither module
contains a raw `sqlite3.connect()`.

WAL is used so a reader does not block a writer; `synchronous=NORMAL` trades
durability against an OS crash — never corruption — for roughly two orders of
magnitude in commit throughput.

## Why the summariser reads role-tagged turns

Turns are read back as `Turn(sequence_number, role, text, created_at, chunk_id)`
and rendered as a speaker-labelled transcript. Before that, the draft model
received `" ".join(...)` over bare text and could not tell who asked from who
answered. Batching breaks **between** turns for the same reason: a character
split leaves the rest of a turn opening the next batch with no speaker attached.

## Why snapshot search uses cursors

`SnapShot` keeps a left and a right cursor into the snapshot list so a caller can
narrow the range it searches — the conversation's recent past rather than all of
it. Searching must not move them: `__find_best_snapshot` scans with local copies,
because a search that consumed its own cursors left the next search with an
empty range (**bug 4.3**).

## Sub-modules

| Directory | Does |
| :--- | :--- |
| `fullconversation_repository/` | The turns table and its reads |
| `conversation_data_management/` | Snapshot metadata (SQLite) and vectors (pgvector) |
| `conversation_summary_pipeline/` | The draft-model summariser |

## Known gaps

- The pgvector connection `SnapShot` opens is never closed (**bug 4.44**).
- The chunk-level drill-down is written on every snapshot and read by nothing
  (`todo.md` 1.2).
