# Todo

## Status

The conversation layer stores turns, summarises them on a trigger, persists
snapshots atomically across SQLite and pgvector, and searches them — verified
end to end. 374 memory-layer tests, 397 across the project.

Three gaps remain **inside** the conversation layer; everything after that is
new construction on top of it.

---

## 1. Finish the conversation layer

### 1.1 Make `role` readable — **blocking**

`append_turn` validates and stores a role on every turn, but no read path
returns it. `fetch_all`, `get_ranged_chunks`, `get_ranged_rows`, `recent` and
`since` all return chunk text only (`get_ranged_rows` reads `summary_chunks`,
which has no role column).

The result is an ordered blob of text with user and assistant turns
indistinguishable — which is unusable for prompt assembly, the thing this layer
exists to feed. Worth doing before the query pipeline is built, because prompt
assembly will bake in assumptions about the shape either way.

- [ ] `FullConversationRepository`: return `role` alongside text — either widen
      `get_ranged_rows` (join `full_conversation` rather than reading
      `summary_chunks` alone) or add a `get_turns(start, end)` returning
      `(sequence_number, role, chunk, created_at)`
- [ ] Surface it on `FullConversation` and `ConversationPoolManager`
- [ ] Have `ConversationSummary` build its prompt from role-tagged turns instead
      of `" ".join(...)` over bare text
- [ ] Tests: role round-trips; ordering preserved; interleaved user/assistant
      reconstructed correctly

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
- [ ] No logger in `snapshot.py`, so the compensating-delete failure path in
      `__add_snap_shot` swallows errors silently. Route it somewhere once
      logging exists.

---

## 2. The layers above (Bug 4.1)

`MemoryManager`, `TopicManager`, `ProjectManager` are still `pass`.
`ConversationPoolManager` is done and is what they should hand back.

`ProjectMetaData` (`project_data_repo/project_meta_data.py`) is done: it is the
storage half of the project layer — the vector, the `project_table` row and the
`project_mapping_table` row written together, vectors-first with a compensating
delete, same rule as snapshots. 67 tests. `ProjectManager` is the domain object
that should sit on top of it.

Mostly path and identity resolution now that the layer below is settled.

- [ ] Decide the on-disk scheme: topic → project → conversation directories,
      and how ids map to paths
- [ ] `ProjectMetaData.__project_db` is a single shared registry file, which is
      right for "list all projects" but is the only global path left. Confirm it
      when the scheme is decided.
- [ ] `TopicManager`: create/load a topic
- [ ] `ProjectManager`: create/load a project under a topic
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
      → assemble context (with roles, from 1.1) → respond
- [ ] Bridge the two vector stores: the data layer uses DiskANN, the memory
      layer uses pgvector. Decide whether retrieval spans both.
- [ ] `Config.MODEL_PATH` is `None` — there is a draft model for summarisation
      but no main model to answer queries
- [ ] **CLI** (Bugs 2.1, 2.2) — `cli_interface()` still contains `# some stuff`;
      `main.py --query` prints a placeholder
- [ ] **FastAPI routes** — no HTTP surface exists yet

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
