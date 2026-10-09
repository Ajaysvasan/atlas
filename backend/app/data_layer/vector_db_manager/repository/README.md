# Vector repositories (`repository/`)

## What this module does

`VectorRepository` is the pgvector store: one connection, scoped to one project,
where every vector the project has is kept. `VectorMetaDataRepository` maps each
DiskANN label to the vector id it stands for and to its chunk, and hands the
labels out. It holds no vectors.

## Why the vector table is keyed `(project_id, vector_id)`

Vector ids are content-derived, so two projects legitimately produce the same
id for the same text. A composite key lets both exist; a bare `vector_id`
primary key would make one project's write silently collide with another's.
Document chunks belong to no project, so they are stored under
`Config.GLOBAL_VECTOR_SCOPE` (`"global"`); real project ids are uuid4 hex and
cannot be that.

## Why the bulk read casts to `bigint[]`

`vectors_for` reads a page of vectors in one query, `vector_id = any(%s)`. The
ids use all 63 bits, and psycopg sends a list of ints that large as `numeric[]`;
`bigint = numeric` cannot use the primary key's index. The cast keeps it a
lookup. The live tests check that ids past 32 bits come back exactly.

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

## Why `vector_meta_data` hands out labels, and requires the vector id

A DiskANN label is `uint32`; the project's vector ids are 63-bit. Hashing chunk
ids down to 32 bits would collide about 116 times at `MAX_VECTORS`, and a
collision is a hit resolving to the wrong text (bug 5.16). So the table
allocates a sequential label — the only number it generates, never reused, and
refused past `uint32` rather than wrapped.

The vector id is the other way round: it is the key every store shares, derived
from the chunk id by the embedder, so the table never makes one up. A missing,
`NULL`, non-integer or negative id is refused by `checked_vector_id` before any
write, and by the column's `NOT NULL` and `CHECK` if something writes around it.
Both matter: SQLite fills in an omitted `INTEGER PRIMARY KEY` silently, and
accepts `NULL` in any other primary key, which is what the two earlier versions
of this column did (bug 5.22).

## Why a chunk has one row

A vector id belongs to a chunk, so `chunkId` and `vectorId` are each unique.
Inserting a chunk already stored hands back its label — re-ingesting an
unchanged folder used to mint a new label every time (bug 5.21). A chunk under
a different vector id, or a vector id under a different chunk, is a
`VectorIdConflict`; a chunk stored by another embedding model is an
`EmbeddingModelMismatch`, because one id cannot point at two models' vectors.
A batch is checked whole and written in one transaction, so a refused row
leaves nothing behind.

## Why it lives in the chunk store, without a foreign key

In the chunk store's own database file, a search result becomes text through a
single join rather than a second connection. There is no foreign key on
`chunkId` because a chunk lives in `Chunks` or in `RecursiveChunks`, and SQLite
cannot reference whichever of two tables holds it; the previous declaration
named a table in another file and made every insert fail (bug 5.2).

## Why an old table is rebuilt, not altered

The previous versions kept the label in `vectorId` and, from bug 5.20 on, the
vector itself. SQLite cannot turn a column into a primary key in place, so the
table is rebuilt on open: labels kept, vector ids derived from the chunk ids,
one row per chunk, blobs dropped, and the sequence kept above every label the
old table handed out. Rows kept have no vector in pgvector until their
documents are ingested again.

## Known gaps

- There is no delete path, here or anywhere in the data layer (**bug 5.5**), so
  a re-ingested *edited* document — new chunk ids — leaves its old vectors stored
  and indexed.
- `batch_search` is one query per id; only `vectors_for` is a bulk read.
