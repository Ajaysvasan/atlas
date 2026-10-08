# Topic pool (`topic_pool/`)

## What this module does

A topic is the outermost grouping: a set of projects that belong together. It is
what scopes the project router — `ProjectManager` decides among the projects of
**one** topic, never all of them.

## Why topics scope the router

`ProjectManager` decides among the projects of one topic. Without that scope it
would score a query against every project a user has ever had, where unrelated
work competes for the same match and the similarity floor has to be set against
noise instead of against alternatives. A topic is the boundary that keeps the
comparison meaningful.

## State

`TopicManager` is built: it creates a topic, reads its id back, lists the active
ones, and soft-deletes it. Each of those is a single SQL statement, and
uniqueness is enforced by a partial index rather than by a check before the
write. `topic_pool_repo/` holds the storage behind it — see its README.

`project_table.topic_id` is a foreign key onto `topics_mapping_table`, so a
project cannot name a topic that does not exist; that became possible once both
tables moved into the one memory database (see `memory/README.md`). A
soft-deleted topic keeps its row, so its projects stay valid.

What is still missing: **nothing hands `(topic_id, query)` down to
`ProjectManager` yet.** The two layers exist and do not talk.

## Sub-modules

| Directory | Does |
| :--- | :--- |
| `topic_pool_repo/` | The topics table and the handler that owns it |
| `project_pool/` | Projects: routing, registry, and the conversations under them |
