# SQLite connection setup (`storage/`)

## What this module does

`connect()` opens a SQLite connection with the pragmas every store in this
project needs — `synchronous=NORMAL` and `foreign_keys=ON` — and `enable_wal()`
puts it into WAL mode, reporting the journal actually in force rather than
assuming the switch took.

## Why it is not in `memory/`

It was, until `knowledge_sufficiency` needed it too. The memory layer already
imports that subsystem (`cosine_scores`, and `MemoryManager` imports
`KSVManager`), so importing the other way would have made the two packages
mutually dependent for the sake of a ten-line helper.

The same reasoning moved `cosine_scores` out of `ProjectManager`: when two
packages need the same small thing, it belongs below both rather than inside
one of them.

## `timestamps.py`

`utc_now()` is the one stamp format every row in this project carries: ISO-8601
UTC with microseconds, sortable as text so two writes in the same second do not
tie. `as_timestamp()` normalises a caller-supplied value into it.

It moved here for the same reason as `connect()`. It had already been written to
stop `utc_now` being defined four separate times (bug 4.58), so letting
`knowledge_sufficiency` keep its own copy would have been the fifth — and two
formats in one database do not sort against each other.

## Who uses it

Every SQLite store: the conversation database, the topic registry, the project
registry, the conversation mapping table, and the KSV acquisition store.
