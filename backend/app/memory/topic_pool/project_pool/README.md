# Project layer (`project_pool/`)

## What this module does

Given a topic and a query, it answers one question: **does this query belong to
a project that already exists?** If it does, the caller gets a `project_id`. If
it does not, the query needs a new project.

It also owns the project list it answers from — creating projects and keeping
each one's summary current.

## Where it sits

```
              TopicManager
                   |
          (topic_id, query)
                   |
                   v
            ProjectManager  <----->  list of projects
                   |
               Exists?
          yes /         \ no
      project_id         PENDING: thinking layer names and
           |             summarises a new project
           v
   PENDING: thinking layer <--> conversation layer
                   |
            project summary flows back to
            ProjectManager.update_project_summary()
```

**Pending.** Neither the thinking layer nor its wiring to the conversation layer
exists yet. `route()` returns a `project_id` when the project exists and stops
at a `pass` when it does not — that `pass` is where the thinking layer will be
called. `create_project()` and `update_project_summary()` are already built and
tested, so the thinking layer has doors to come back through when it arrives.

## How a query flows through

1. `ProjectManager(topic_id, query)` — nothing loads yet; the embedder and the
   PostgreSQL connection are built on first use.
2. `resolve()` embeds the query **once** and caches it.
3. `list_topic_vector_ids(topic_id)` (SQLite) returns `(project_id, vector_id)`
   for every vector in the topic.
4. `ProjectVectorHandler.get_project_vectors()` (pgvector) turns those ids into
   embeddings, per project.
5. Each project scores as the **highest** cosine any of its vectors reaches.
6. The result is a `ProjectMatch`: `exists`, `project_id`, `score`, `margin`,
   `ambiguous`, and the ranked `candidates`.

The split in steps 3 and 4 is forced: `VectorRepository` can only fetch by id —
it has no "list this project's vectors" — so SQLite says *which* vectors exist
and whose they are, and pgvector says what they hold.

## Why it was designed this way

**A project scores on its best vector, not its average.** A project holds one
summary vector and one per description, and it covers several things. A query
matching one corner of a project should score as that corner; averaging would
dilute a sharp match into a weak one and lose to a blander project.

**Two numbers, not one.** `similarity_floor` decides whether anything in the
topic is close enough at all. `ambiguity_margin` decides whether the winner is
far enough above the runner-up to be trusted. Raw cosine scores are not
comparable between different queries — one query's 0.6 is another's 0.3 — but
the gap between the top two *within one query* is meaningful. A match inside the
margin returns `ambiguous=True` with its candidates, which is where a caller can
escalate to the draft model with the top few summaries instead of all of them.

**Both defaults are guesses.** `SIMILARITY_FLOOR = 0.35` and
`AMBIGUITY_MARGIN = 0.05` are starting points for MiniLM cosine scores, not
measured values. They need a labelled set of queries — 30 to 50, each tagged
with the project it belongs to — before anyone should trust them.

**Embedding is cheap, inference is not.** Routing by vector costs one embedding
on a model already in memory. Asking the draft model to pick among every project
summary costs a full prompt that grows with the project count. The cascade —
vectors first, model only on a near-tie — keeps the model for the cases where
cosine is actually bad.

**Ids are generated, not derived.** `create_project()` mints a uuid.
`project_id` is a primary key, while names and summaries both change and repeat,
so deriving the id from either would collide or drift. This is the same rule
`chunk_id` and `cumulative_vector_id` follow in the conversation layer.

**One connection per project.** The manager hands its `ProjectVectorHandler`'s
repository down to `ProjectMetaData`, which would otherwise open a second
connection to the same store for the same project in the same call.

## Files

| File | Holds |
| :--- | :--- |
| `project_manager.py` | The routing decision, project creation, summary refresh |
| `project_data_repo/` | Storage — see its own README |

## Tests

`test/memory_layer_testing/test_project_manager.py`. The embedder is a fake that
maps text to a chosen axis, so "this query is about project A" is something the
test states rather than something a model decides; SQLite is real.
