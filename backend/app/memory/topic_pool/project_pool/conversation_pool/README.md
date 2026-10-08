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
  └── MemoryDatabase           the one connection every owner shares
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
have to agree on three things: the same database, the same project identity,
and **one shared `SnapShot`** so its cursors survive between calls. Building
them ad hoc is how the summariser ended up reading a different database than the
snapshot writer, and how a second `SnapShot` ended up with its own cursors that
drifted from the first. `ConversationPoolManager` is the only place that wiring
lives.

## Where the conversation tables live

In the memory database, with every other table in this layer
(`memory/README.md`). Turns and snapshot metadata used to share a file per
project; they now share one with the projects they belong to, so every
conversation row references its project and a turn's text and position must
agree on their conversation.

Nothing in `memory/` opens its own connection. `synchronous` and `foreign_keys`
are per-connection pragmas that reset on every open, and a raw
`sqlite3.connect()` silently gets `FULL` and foreign keys off — which once let
orphan rows into `full_conversation` through `__add_chunks`. A test reads every
module under `memory/` and fails on any `connect(` outside `memory_database.py`.

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

- The chunk-level drill-down is written on every snapshot and read by nothing
  (`todo.md` 1.2).


## Schema versions

`schema_migrations.py` is retired. It rebuilt per-project conversation files
that predated `conversation_id`, and with one memory database there are no such
files. The database stamps `PRAGMA user_version` when it is created and refuses
a file written by a newer schema (`NewerMemorySchema`); the next change to a
table's shape adds its step there, in `memory_database.py`.
