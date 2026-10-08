# Identifier validation (`identifiers`)

## Overview & Purpose

Validates the ids that scope rows across the memory layer. `conversation_id`,
`user_id` and `project_id` are written into `not null` columns and filtered on
by every reader, so a blank one partitions nothing and a `None` fails at the
database with a message that names a column rather than the caller.

Architecture and rationale live in `memory/README.md`. This page is the API.

---

## API

```python
require_identifier(value, name: str) -> str
```

Returns `value` stripped, or raises `InvalidIdentifier`.

| Input | Result |
| :--- | :--- |
| `"conv_1"` | `"conv_1"` |
| `"  conv_1  "` | `"conv_1"` — stripped, so `' c1'` and `'c1'` cannot become two conversations |
| `""`, `"   "`, `"\t\n"` | `InvalidIdentifier` |
| `None`, `0`, `123`, `[]`, any non-`str` | `InvalidIdentifier` |

`InvalidIdentifier` lives in `memory/memory_pool_exceptions.py` and carries both
the field name and the value it got:

```
conversation_id must be a non-empty string, got ''. It scopes rows in the
database, so an empty or missing value would silently partition nothing.
```

---

## Where it is applied

At the points that **store or write** an id, not at every hop:

| Class | Validates |
| :--- | :--- |
| `FullConversationRepository` | `conversation_id` |
| `ConversationVectorMetaDataRepository` | `conversation_id` |
| `ConversationSummary` | `conversation_id` |
| `SnapShot` | `conversation_id` (conversation scope only) |
| `ProjectSnapshotRepository` | `project_id` |
| `MemoryMappingHandler` | `conversation_id`, `user_id`; `topic_id` and `project_id` when routing |

Pass-through layers — `FullConversation`, `ConversationPoolManager` — inherit
it: they hand the id straight down, so construction still fails immediately with
a message naming the field. Validating at every layer would duplicate the check
without catching anything new.

It lives in `memory/` beside `memory_manager.py` rather than in the conversation
pool because `memory/snapshot.py` and the project layer both need it.

---

## Tests

`test/memory_layer_testing/test_identifiers.py` — 26 tests (10 functions, most
parameterised), covering the validator and each class that applies it.
