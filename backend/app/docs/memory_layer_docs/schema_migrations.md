# Conversation database migrations (`schema_migrations`)

## Overview & Purpose

Brings an existing conversation database up to the current schema. Needed
because the schema is created with `CREATE TABLE IF NOT EXISTS`, which does
nothing to a table that already exists — so without this, every database written
before a change fails its next insert.

Architecture and rationale live in
`memory/topic_pool/project_pool/conversation_pool/README.md`. This page is the API.

---

## API

```python
migrate(db_path: str | Path) -> int
```

Returns the schema version now in force. Idempotent, and safe to call from more
than one place: `FullConversationRepository` and
`ConversationVectorMetaDataRepository` share one database file and either may
open it first, so **both** call it before any statement.

| Name | Meaning |
| :--- | :--- |
| `SCHEMA_VERSION` | `2` — the version `migrate` brings a database to |
| `LEGACY_CONVERSATION_ID` | `"legacy"` — the conversation pre-`conversation_id` rows are assigned to |

Versioning is `PRAGMA user_version`. It is what distinguishes one schema
generation from the next; the "does this table already have the column" check
cannot, which matters as soon as there is more than one version.

---

## The versions

| Version | Change |
| :--- | :--- |
| 1 | `conversation_id` on `summary_chunks` and `full_conversation`; `full_conversation`'s key becomes `(conversation_id, sequence_number)` |
| 2 | `conversation_id` and the monotonic `seq` on `cumulative_vector_meta_data` |

`migrate` steps through every pending version in order, so a version-0 database
and a version-1 database both end at 2.

## Why tables are rebuilt rather than altered

`full_conversation` needed a new primary key and SQLite cannot alter one in
place. Each affected table is therefore recreated — create, copy, drop, rename —
inside one `BEGIN IMMEDIATE`, with foreign keys off for the duration. The pragma
has to be issued outside the transaction, since a pragma inside one is silently
ignored, which is why `migrate` opens its own connection rather than borrowing
the caller's.

A migrated database comes out schema-identical to a fresh one, column order
included. `PRAGMA foreign_key_check` runs afterwards and any violation is
**logged, not raised** — the rows are still there and readable, and failing the
open would lock the user out of their own history.

## Why `'legacy'` is not a placeholder

Before `conversation_id` existed, a project database held exactly one
conversation. Collapsing its rows into a single id is what they always meant, so
the old history stays readable:

```python
repo = FullConversationRepository(path, project_id, name, LEGACY_CONVERSATION_ID)
repo.get_all_turns()        # every turn written before the change
```

The value is pinned by a test against its literal, because it is written into
real databases — changing it would orphan every row a previous migration
assigned rather than migrating anything.

---

## Tests

`test/memory_layer_testing/test_schema_migrations.py` — 18 tests, including that
a migrated database matches a fresh one and that the stamped version is
authoritative.
