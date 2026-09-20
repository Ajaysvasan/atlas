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

`topic_manager.py` is an empty stub (**bug 4.1**). There is no topic table
either: `topic_id` is stored as plain text on every project row and on both
project child tables, with no foreign key behind it. The only guard against a
typo is a non-empty check in `ProjectMetaData`.

## What `TopicManager` has to do when it is written

- Create and load a topic, and decide where a topic's data lives on disk.
- Hand `(topic_id, query)` down to `ProjectManager`.
- Settle whether topics get their own table, which is what would let `topic_id`
  become a foreign key.

That on-disk scheme is the open decision blocking this layer; see `todo.md`
section 2. Related: **4.1 Conversation identity** in `todo.md` — there is no
`conversation_id` anywhere, so two conversations in one project are separated
only by the directory the caller passes.

## Sub-modules

| Directory | Does |
| :--- | :--- |
| `project_pool/` | Projects: routing, registry, and the conversations under them |
