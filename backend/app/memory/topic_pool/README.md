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

What is still missing:

- **Nothing hands `(topic_id, query)` down to `ProjectManager` yet.** The two
  layers exist and do not talk.
- `topic_id` on the project tables is still plain text with no foreign key onto
  `topics_mapping_table`, so a project can name a topic that does not exist. The
  two databases are separate files, which is why the constraint cannot simply be
  added.
- The on-disk scheme for topic -> project -> conversation directories is still
  undecided (`todo.md` section 2), and `Config.TOPIC` was added for it but is not
  read by anything — the handler builds its own default path.

That on-disk scheme is the open decision blocking this layer; see `todo.md`
section 2. Related: **4.1 Conversation identity** in `todo.md` — there is no
`conversation_id` anywhere, so two conversations in one project are separated
only by the directory the caller passes.

## Sub-modules

| Directory | Does |
| :--- | :--- |
| `topic_pool_repo/` | The topics table and the handler that owns it |
| `project_pool/` | Projects: routing, registry, and the conversations under them |
