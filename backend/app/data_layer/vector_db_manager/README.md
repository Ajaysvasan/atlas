# Vector stores (`vector_db_manager/`)

## What this module does

Holds the two vector stores this project uses, behind thin wrappers.

| Wrapper | Store | Holds |
| :--- | :--- | :--- |
| `VectorDbManager` → `VectorDb_diskann` | DiskANN, in memory, rebuilt from `vector_meta_data` | document chunk vectors |
| `repository/VectorRepository` | PostgreSQL + pgvector | conversation and project vectors |

## Why there are two

DiskANN is an approximate index built for many vectors and fast nearest-neighbour
search over an unchanging corpus. The memory layer's vectors are few, change
constantly, and need transactional behaviour next to their metadata — which is
what a relational store gives. They were never meant to serve the same access
pattern.

This is real operational surface, though: two stores, two failure modes, two
things to run. Whether retrieval spans both is an open design question.

## Why `VectorDbManager` takes a lock

`diskannpy.DynamicMemoryIndex` is not safe for concurrent mutation. Every insert,
delete, restore, save and load runs under one `threading.Lock`. Searches do not
take it — they do not mutate. Removing the lock and running the stress test
produces dozens of concurrent entries into the index.

## Why the index is rebuilt rather than loaded

diskannpy 0.7.0 cannot reload a dynamic index it saved. `save()` writes every
label as `0`, and `from_file()` in a new process returns garbage — still garbage
after the right labels were written into `ann.tags` by hand, so the loader is
broken as well as the writer. A reload in the same process happens to look
right, which is how it went unseen. 0.7.0 is the last release, and the cp311
ceiling rules out anything newer (bug 5.20).

So the index is derived data. Each vector is stored in `vector_meta_data` in the
transaction that allocates its label, and `restore()` builds the index from
those rows: the label map and the index cannot disagree, because one is built
from the other. Labels only grow, so after an ingest `restore(store, after=last)`
indexes only what is new rather than rebuilding the graph.

The price is a graph build at startup, which grows faster than linearly — 0.9 s
for 10k vectors, 6.3 s for 50k, 16.6 s for 100k at `Config`'s settings
(complexity 100, degree 120, 4 threads). Reading the vectors out of SQLite is
under 0.1 s per 100k. At the corpus sizes this project targets that is seconds,
paid on the first search.

## Why the graph is saturated

With diskannpy's default `saturate_graph=False`, the dynamic index builds a graph
too sparse to walk: recall@10 of 0.09 on 5,000 vectors against 0.96 with it on,
and an inserted vector cannot find itself (bug 5.19). A small index hides this —
its start node links to every point — so the test that guards it uses 2,000.

## Why a search never asks for more than the index holds

diskannpy returns `k` slots whatever the index holds, and fills the surplus from
uninitialised memory: labels that can be real ones, at distance `0.0`, sorted
ahead of every true result (bug 5.18). `search_vector` caps `k` at `count()`.
A new install's corpus is smaller than retrieval's 32-candidate pool, so this is
the usual case rather than the edge.

## Known gaps

- The startup rebuild grows with the corpus (above). A snapshot through the
  static index (`build_memory_index`, whose ids are row positions this project
  could map to labels) would avoid it, at the cost of a full rebuild per ingest.
- `save()` and `load()` remain, unused and unable to persist an index. They are
  kept for their tests; `load()` also still discards its own return value
  (**bug 5.8**).
