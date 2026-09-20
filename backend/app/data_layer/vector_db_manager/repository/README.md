# Vector repositories (`repository/`)

## What this module does

`VectorRepository` is the pgvector store: one connection, scoped to one project.
`VectorMetaDataRepository` is a SQLite sidecar meant to map vector ids back to
chunk ids.

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

## Known gaps

- **Bug 5.1** — no pgvector adapter is registered, so numpy arrays cannot be
  adapted at all, lists are sent as Postgres arrays rather than vectors, and
  reads come back as text. Every test mocks `psycopg`, so nothing catches it.
- **Bug 5.2** — `VectorMetaDataRepository`'s foreign key references `Chunks`,
  which lives in a different SQLite file, so every insert raises
  `no such table: main.Chunks`. It has no callers.
