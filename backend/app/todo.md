# Todo

## Status

The conversation layer stores turns, summarises them on a trigger, persists
snapshots atomically across SQLite and pgvector, and searches them — verified
end to end. 499 memory-layer tests, 681 across the project.

Two gaps remain **inside** the conversation layer; everything after that is
new construction on top of it.

---

## 1. Finish the conversation layer

### 1.1 Make `role` readable — done

`append_turn` validates and stores a role on every turn, but no read path
returns it. `fetch_all`, `get_ranged_chunks`, `get_ranged_rows`, `recent` and
`since` all return chunk text only (`get_ranged_rows` reads `summary_chunks`,
which has no role column).

The result is an ordered blob of text with user and assistant turns
indistinguishable — which is unusable for prompt assembly, the thing this layer
exists to feed. Worth doing before the query pipeline is built, because prompt
assembly will bake in assumptions about the shape either way.

- [x] `FullConversationRepository`: `get_turns`, `get_last_n_turns`,
      `get_turns_after`, `get_all_turns` return
      `Turn(sequence_number, role, text, created_at, chunk_id)`.
      `get_ranged_rows` was left alone — `insert_snapshot` writes its rows into
      `summary_chunks` verbatim, so widening it would have broken that contract.
- [x] Surfaced on `FullConversation`. On `ConversationPoolManager`,
      `history`/`recent`/`context`/`since` now return `Turn` — a deliberate
      breaking change, made while they had no callers.
- [x] `ConversationSummary` feeds the model a speaker-labelled transcript, and
      batches between turns so no batch opens mid-turn without a speaker.
- [x] Tests: role round-trips; ordering ignores `created_at`; interleaved
      user/assistant reconstructed exactly; batching properties over random
      shapes. Every one checked against a deliberate regression.
- [ ] The text-only readers (`fetch_all`, `get_ranged_chunks`, `get_n_chunks`,
      `get_sequence_after`, and their bucket counterparts) no longer have a
      caller in the source tree — only tests. Remove them, or keep them as the
      cheap path, once the query layer shows whether anything wants text alone.

### 1.2 Wire the chunk-level drill-down — or stop paying for it

Nothing calls `get_summary_vector_ids_from_map` or
`batch_get_summary_vector_meta_data`. Every snapshot embeds N chunks and writes
N rows to `summary_vector_meta_data` plus N to `summary_snapshot_map`, and no
code path reads any of it.

The two-level search designed in Step 2 is currently one level: you can find the
best snapshot but not descend into its chunks. Decide one way or the other —
the current state is pure write amplification.

- [ ] Decide: wire the drill-down, or drop per-chunk embedding from
      `__persist_snapshot` and simplify the schema
- [ ] If wiring: `SnapShot.chunks_for(cumulative_vector_id)` →
      map → `summary_vector_meta_data` → `summary_chunks` text
- [ ] If wiring: rank chunks within a snapshot by similarity to the query, so
      `search()` can return passages rather than just a snapshot id
- [ ] Expose through `ConversationPoolManager.search()`

### 1.3 Deletion and retention — design decision first

No `delete`, `prune`, or `DELETE FROM` anywhere in `memory/`. Conversations grow
without bound, and there is no way to remove one, prune old snapshots, or honour
a delete request. Not urgent unless the project has a data-retention
requirement — but note it is a *design* decision (what retention policy?) before
it is a coding task.

- [ ] Decide a retention policy
- [ ] Settle 4.1 (conversation identity) first — until then, deleting one
      conversation can delete vectors another conversation in the same project
      still uses
- [ ] `delete_conversation(project_id)` spanning SQLite **and** the vector store
      (same compensation problem as Bug 4.38 — vectors and metadata must not
      diverge)
- [ ] Optional: prune snapshots older than N, or keep only the last K

### 1.4 Loose ends

- [ ] 7 open P3 bugs in this layer — see `bugs.md` (4.2, 4.19, 4.20, 4.21, 4.22,
      4.41, 4.43). All cosmetic or type-hint level.
- [ ] `Config.CONVERSATION` is now dead — the only mention left is a docstring
      explaining why it is *not* used. Remove it, or repurpose it in 2.1.
- [x] `test/stress_testing/test_stress.py` referenced
      `Config.CONVERSATION_SNAPSHOT_DB`, which does not exist — 3 failing tests.
      Rewritten against the current APIs and against temporary directories
      instead of Config paths, so the suite no longer depends on the 2.1 path
      decision at all. 10 tests, whole suite now green.
- [x] No logger in `snapshot.py`, so the compensating-delete failure path in
      `__add_snap_shot` swallowed errors silently. It now logs the orphaned
      vector ids with `logger.exception`.

---

## 2. The layers above (Bug 4.1)

`MemoryManager` is still `pass`. `TopicManager` and `ProjectManager` are built.
`ConversationPoolManager` is done and is what they should hand back.

`ProjectMetaData` (`project_data_repo/project_meta_data.py`) is done: it is the
storage half of the project layer — the vector, the `project_table` row and the
`project_mapping_table` row written together, vectors-first with a compensating
delete, same rule as snapshots. `project_table` also holds the summary text
(`project_summary`, required on every write), and `project_description_table`
holds several descriptions per project, keyed `(project_id,
project_description_id)`. 107 tests. `ProjectManager` is the domain object that
should sit on top of it.

Mostly path and identity resolution now that the layer below is settled.

- [ ] Decide the on-disk scheme: topic → project → conversation directories,
      and how ids map to paths
- [ ] `ProjectMetaData.__project_db` is a single shared registry file, which is
      right for "list all projects" but is the only global path left. Confirm it
      when the scheme is decided.
- [ ] `ProjectMetaData` opens that shared file with a plain `sqlite3.connect` —
      no lock, no WAL, and bound to the thread that opened it. Route it through
      the same `connect()` treatment the conversation database got, before
      `ProjectManager` opens projects concurrently.
- [x] `ProjectManager`: routes a query to an existing project, or reports that
      none matches. `route()` returns a `project_id` or `None`; `resolve()`
      carries the score, margin, ambiguity flag and ranked candidates.
      `create_project()` and `update_project_summary()` are the doors the
      thinking layer comes back through. Architecture in
      `memory/topic_pool/project_pool/README.md`.
- [ ] **Pending: thinking layer integration.** `ProjectManager.route()` stops at
      a `pass` on the no branch — that is where the thinking layer will name and
      summarise a new project before `create_project()` stores it. Neither the
      thinking layer nor its wiring to the conversation layer exists yet.
      - When replacing that `pass`, **write the `return`**. `route()` currently
        returns `None` by falling off the end, not because of the `pass`, so
        `self.thinking_layer.create_project_for(...)` without a `return` still
        returns `None` and no test or type checker complains.
      - The same seam is where the conversation layer gets handed the project,
        and where the project summary flows back through
        `update_project_summary()`.
- [ ] Decide `ProjectManager`'s public surface. `route()`, `resolve()`,
      `projects()`, `score_projects()`, `create_project()`,
      `update_project_summary()` and `query_vector()` are all public. The flow
      only needs `route()`; the rest exist for the thinking layer and for
      escalating an ambiguous match. Narrow it once the thinking layer shows
      which it actually calls.
- [ ] The routing thresholds (`SIMILARITY_FLOOR`, `AMBIGUITY_MARGIN`) are
      guesses. They need 30-50 queries labelled with the project they belong to
      before they mean anything.
- [x] `TopicManager`: create, load and soft-delete a topic, over
      `topic_pool_repo/TopicPoolMetaHandler`. 24 tests. Architecture in
      `memory/topic_pool/README.md` and `topic_pool_repo/README.md`.
- [ ] **Pending: topic -> project handoff.** `TopicManager` holds `query` and
      never reads it; `ProjectManager` takes `(topic_id, query)`. Nothing passes
      one to the other yet (bug 4.55).
- [x] `get_all_topics()` on `TopicManager` / `TopicPoolMetaHandler` (bug 4.56).
      The reason is **enumeration, not caching**: `MemoryManager` has to resolve
      a topic before it can name one, and the CLI has to show the user what
      exists. Neither is answerable today — the handler can only test one name
      at a time.

      It is *not* worth adding to save database hits, which was the original
      motivation. Measured on a 10,000-topic table: loading every topic costs
      16x a single existence check as the code stands, and 1,492x once
      `topic_name` is indexed, while a `TopicManager` performs two or three
      lookups in its entire life. A cached list would also go stale, and since
      `topic_name` cannot carry a UNIQUE constraint (soft delete keeps old rows
      with the same name), a stale "does not exist" leads straight to the
      duplicate insert in bug 4.52. The crossover where loading once wins is
      around 1,500 lookups against an unchanging set — a bulk import, not this.
- [x] Index `topic_name` — done as a **partial unique** index, which also made
      the create race (bug 4.52) impossible rather than merely unlikely.
      This *is* the fix for lookup cost: the column has no
      index, so every existence check is a `SCAN` of the whole table, growing
      linearly with the topic count. Measured at 10,000 topics: 211 us per check
      as a scan, 2.3 us with `create index idx_topic_name on
      topics_mapping_table(topic_name)` — 93x, for one statement in `__db_init`
      and nothing to keep in step.
- [ ] `Config.TOPIC` was added for the on-disk scheme and is read by nothing —
      `TopicPoolMetaHandler` builds `data/topic_db/topic.sql` itself.
- [ ] `MemoryManager`: top-level entry point returning a
      `ConversationPoolManager` for a given topic/project/conversation
- [ ] Fix the per-project path collision `Config.CONVERSATION` used to cause —
      no global shared directory

---

## 2b. Ingestion — done

The submodule now takes any file type and the chunkers agree with the
normalizer about where sections are. 85 tests in
`test/data_layer_testing/test_ingestion.py`.

What changed, and why each mattered:

- The `\s+` whitespace collapse flattened every document to one line **before**
  the chunkers ran, so `has_section` was decided on the raw text while
  `HierarchicalChunker` saw text with no line structure left. Every "hierarchical"
  document degenerated to a single MAIN section, one context, and fixed-width
  slices. Normalization is now line-by-line and the normalizer hands down
  `SectionSpan` offsets, so there is only one heading detector in the pipeline.
- Section, context and chunk ids hashed content alone. A document with two
  `NOTES` headings, or a repeated paragraph, aborted on the primary key — the
  same rule already recorded under "Decisions on record". Ids now bind position.
- Re-running the pipeline over an unchanged folder died on `Documents.documentId`.
  All writes are `on conflict do nothing`.
- `RecursiveChunker` appended a separator the document never had, and applied
  overlap again at every level of the recursion, duplicating text into chunks.
- `FileLoader` recursed forever on a symlink to an ancestor; the `RecursionError`
  was caught as if the folder were unreadable, so the scan silently returned a
  partial tree.
- HTML and XML were embedded with their tags and inline scripts. PDF pages with
  no text layer raised `TypeError` on `"\n".join`. `.docx` tables were dropped.

- [ ] `openpyxl` is commented out in `requirements.txt`; `.xlsx` currently goes
      through textract. Install it if spreadsheets matter.
- [ ] `chunkAlgorithmTypes.ChunkingAlgorithmType` is still unused — the routing
      decision is `has_section`. Use the enum or drop it.

---

## 3. Retrieval and delivery

- [ ] **RAG query pipeline** — query → embed → DiskANN search → retrieve chunks
      → assemble context (with roles: build chat messages from `Turn`, not
      from the summariser's transcript text) → respond
- [ ] Bridge the two vector stores: the data layer uses DiskANN, the memory
      layer uses pgvector. Decide whether retrieval spans both.
- [ ] `Config.MODEL_PATH` is `None` — there is a draft model for summarisation
      but no main model to answer queries
- [ ] **CLI** (Bugs 2.1, 2.2) — `cli_interface()` still contains `# some stuff`;
      `main.py --query` prints a placeholder
- [ ] **FastAPI routes** — no HTTP surface exists yet

---

## 4. Later optimizations

Deferred deliberately. Neither blocks the retrieval design, the query layer or
MVP_V1.

### 4.1 Conversation identity

No `conversation_id` exists anywhere in `memory/`. Both conversation databases
are named `{project_id}_conversation.db`, so two conversations in one project
are kept apart only by the directory the caller passes in.

The vector store does not have even that. pgvector's `vectors` table is keyed
`(project_id, vector_id)`, and a chunk-level summary vector id is derived from
`chunk_id`, which binds `(project_id, sequence_number, text)`. Two conversations
in the same project whose turn 1 is the same text produce the same vector id,
and share one row (`on conflict do nothing`). Harmless while nothing deletes —
the embedding is identical — but a delete for one conversation removes the
other's vector.

- [ ] Decide: a `conversation_id` column in the schema and in every derived id,
      or one directory per conversation with the directory's id bound into
      `chunk_id` — together with the on-disk scheme in section 2
- [ ] Must land before any deletion in 1.3

### 4.2 Snapshot search round trips

`SnapShot.__find_best_snapshot` calls `vector_manager.get_vector` once per
candidate snapshot inside its scan, and `VectorRepository.batch_search` loops
`__get_vector` in the same way — one PostgreSQL round trip per snapshot. Fine at
tens of snapshots; linear in round trips once memory is a retrieval source.

- [ ] Either fetch every candidate in one query
      (`WHERE project_id = %s AND vector_id = ANY(%s)`), or push the ranking
      into pgvector (`ORDER BY embedding <=> %s LIMIT k`) and stop pulling
      vectors into Python at all
- [ ] The second option moves similarity below the store interface — decide it
      with the vector-store protocol from the retrieval design, not on its own

---

## Decisions on record

- Draft model: **Qwen2.5-3B-Instruct-GGUF** (`q4_k_m`), 128K context, ~2GB RAM.
  Downloaded via `download_models/download_draft_model.py`.
- One conversation turn = one chunk (`chunker_type="turn"`); the summariser does
  its own batching rather than relying on pre-split turns.
- Ids that act as primary keys are never hashed from content alone — content
  repeats. `chunk_id` binds `(project_id, sequence_number, text)`;
  `cumulative_vector_id` binds `(project_id, timestamp, summary)`.
- Snapshot writes go vectors-first, metadata-second, with a compensating delete:
  a failure then leaves unreachable vectors rather than metadata pointing at
  missing ones.
- Indexes are added on measurement, not on shape. Two plausible ones were
  measured and rejected: `summary_vector_meta_data(project_id, chunk_id)` — the
  watermark join is already served by `idx_full_conversation_chunk`, so it won
  ~2% (inside noise) while costing +93% on every insert — and
  `cumulative_vector_meta_data(datetime(created_at))`, which saved 4.7us at
  fifty snapshots and has to be maintained on every write. Do not re-add either
  without a benchmark that contradicts this.
