# Data layer exceptions (`datalayer_exceptions/`)

## What this module does

Holds every exception the data layer raises, in one module so a caller can
import the set without reaching into the store that raises it.

## Why some of these are separate types

**`DuplicateVectorException` vs `VectorInsertionError`.** A caller needs to tell
"already written" from "the write failed": the first needs no compensating
delete, the second does. That distinction is what the snapshot and project write
paths depend on. `batch_insert` never raises it — that path is
`on conflict do nothing` by design.

**`VectorInsertionError.cause`** carries the driver exception separately from
`vector_id`, which used to hold whichever of the two the raising site happened
to have — the DiskANN driver passed an id, the pgvector repository passed the
psycopg error, so anything reading the attribute got one or the other.

**`MissingDatabaseConfiguration`** exists so a missing `.env` setting is named
where it is missing, rather than surfacing as a confusing "role does not exist"
from the server.

## Known naming defects (bug 5.11)

- `VectorNotFoundEror` is misspelled and is part of the public surface.
- `InsertionError.message` holds a *table name*, not a message.
- `InvalidVectorDimension` is defined here **and** in
  `memory/memory_pool_exceptions.py`, with no relationship between them, so
  `except` on one silently misses the other.
- `InvalidVectorID` says invalid; the condition is missing.
