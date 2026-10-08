# Vector repositories (`repository/`)

## What this module does

`VectorRepository` is the pgvector store: one connection, scoped to one project.
`VectorMetaDataRepository` allocates each DiskANN label, maps it back to its
chunk, and stores the vector itself — the rows the in-memory index is rebuilt
from.

## Why the vector table is keyed `(project_id, vector_id)`

Vector ids are content-derived, so two projects legitimately produce the same
id for the same text. A composite key lets both exist; a bare `vector_id`
primary key would make one project's write silently collide with another's.

## Why the settings are validated before connecting

psycopg substitutes libpq's defaults for anything passed as `None` — including
the OS username — so a missing setting does not fail, it connects *somewhere
else*. `MissingDatabaseConfiguration` names the missing keys instead.

For the same reason the setting is `DB_USER`, not `USER`: every login shell
exports `USER`, and `load_dotenv()` does not override a variable already in the
environment, so the `.env` line was ignored and the connection was made as
whoever ran the process.

## Why an update checks `rowcount`

An UPDATE matching nothing is not an error to psycopg, so updating an id that
was never inserted would report success and leave the caller believing the new
embedding is stored. `VectorNotFoundEror` is raised instead.

## Why `vector_meta_data` hands out the labels

A DiskANN label is `uint32`; the project's derived vector ids are 63-bit, for
pgvector. Hashing chunk ids down to 32 bits would collide about 116 times at
`MAX_VECTORS`, and a collision is a hit resolving to the wrong text. A
sequential id from this table cannot collide, and the table is what turns it
back into a chunk (bug 5.16).

## Why it keeps the vectors

diskannpy cannot load an index it saved (bug 5.20; the index side is in
`../README.md`), so something else has to survive a restart. The vector goes
into the row that allocates its label, in one transaction: an index rebuilt
from these rows cannot hold a label the table has not heard of, nor miss one it
has.

Only one model's vectors at one width are ever read back. Another model's
vectors are in a different space, and mixed into one index they would answer
queries with neighbours that mean nothing. Rows left out — another model, or
written before vectors were kept — are counted by `missing_vectors()` and
reported when the index is built.

## Why it lives in the chunk store, without a foreign key

In the chunk store's own database file, a search result becomes text through a
single join rather than a second connection. There is no foreign key on
`chunkId` because a chunk lives in `Chunks` or in `RecursiveChunks`, and SQLite
cannot reference whichever of two tables holds it; the previous declaration
named a table in another file and made every insert fail (bug 5.2).

## Why a chunk has one label

A chunk id binds its content and position, so the same id with the same model
is the same vector. Allocation used to insert unconditionally, and re-ingesting
an unchanged folder — which writes no new chunks — still gave every chunk a new
label; once vectors were stored, every copy was rebuilt into the index and could
fill retrieval's candidate pool with one chunk (bug 5.21). `unique (chunkId,
embeddingModelUsed)` makes the database refuse a second label, and allocation
is an upsert that hands the existing one back.

## Known gaps

- There is no delete path, here or anywhere in the data layer (**bug 5.5**), so
  a re-ingested *edited* document — new chunk ids — leaves its old vectors stored
  and indexed.
- Rows written before vectors were kept stay unsearchable by meaning until their
  chunks are re-embedded.
