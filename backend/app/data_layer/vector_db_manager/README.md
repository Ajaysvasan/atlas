# Vector stores and the document index (`vector_db_manager/`)

## What this module does

Holds every vector the project has, and the index retrieval searches over the
document chunks.

| Piece | Where | Holds |
| :--- | :--- | :--- |
| `repository/VectorRepository` | PostgreSQL + pgvector, `vectors` | every vector: document chunks under project id `"global"`, the memory layer's under theirs |
| `repository/VectorMetaDataRepository` | SQLite, `vector_meta_data` in the chunk store | DiskANN label → vector id → chunk id. No vectors |
| `index_generations.py` | `data/disk_ann_index/gen-N/` | the built index, opened from disk at startup |
| `VectorDbManager` → `VectorDb_diskann` | memory | the built generation, plus the vectors ingested since |
| `memory_guard.py` | — | what the index may allocate before it degrades instead |

## How a vector travels

```
ingest:   EmbeddedChunk(vector, vector_id, chunk_id)
            -> pgvector  vectors(project "global", vector_id)       first
            -> SQLite    vector_meta_data(label, vector_id, chunk)   then
            -> every INDEX_REBUILD_AT vectors: build gen-N from both, swap CURRENT

startup:  open CURRENT's generation from disk               (no rebuild)
          load labels > its `through` from both stores into the recent index

search:   built generation + recent index, merged by distance -> labels
          labels -> chunks through vector_meta_data, in one join
```

## Why there are two numbers per vector

The vector id is the project's id for a vector: derived from the chunk id
(`ingestion/embedding/vector_ids.py`), 63 bits, the key in pgvector. DiskANN
cannot hold it — its labels are `uint32`, and it refuses anything wider. Cutting
the id to 32 bits would collide about 116 times at a million vectors (bug 5.16).
So `vector_meta_data` hands out a sequential label beside the vector id, and is
what turns a hit back into a chunk. The vector id is passed in and required —
by the code and by the database; the label is the only number the table makes
up (bug 5.22).

## Why the vectors live in PostgreSQL

pgvector is the project's vector store. The document vectors once lived in
`vector_meta_data` instead, because they had to survive a restart and nothing
else stored them (bug 5.22). The lookup table went back to looking up.

The price is that ingestion needs PostgreSQL: a vector is stored before its
label is written, so a label never exists for a vector that was not stored, and
a store that cannot be reached fails the ingest with `VectorStoreUnavailable`.
Search does not need it for what is already built — that is files on disk.

## Why the index is built in generations

diskannpy 0.7.0 cannot save the dynamic index: every label is written as `0`,
and the loader returns garbage in a new process (bug 5.20). Its **static**
indexes save and load correctly, but cannot be added to. So ingestion builds a
static index — a *generation* — every `INDEX_REBUILD_AT` (10,000) new vectors,
and what arrives between builds goes into a small dynamic index that startup
fills from the stores. Half of `RECENT_VECTOR_CAPACITY`, so one failed build
still leaves room.

| At 100k vectors, measured | Startup | RAM | Search p50 |
| :--- | :--- | :--- | :--- |
| Before: dynamic index rebuilt from every vector | 5.89 s | 1,236 MB | 0.84 ms |
| Memory generation + 5k recent | 0.25 s | +130 MB | 1.18 ms |
| Disk generation + 5k recent, budget squeezed to 100 MB | 2.48 s | +81 MB | 3.06 ms |

Recall@9 was 0.95–0.96 throughout. The 1.2 GB was the dynamic index reserving
every slot for `MAX_VECTORS` up front, however few it held (bug 5.23); the
recent index is capped at 20,000 slots, 34 MB.

## Why memory or disk is chosen by a budget

`VECTOR_INDEX_RAM_MB` (512) bounds the index whatever the corpus does. A build
chooses:

- **A memory index** while it fits — about 0.94 KB a vector, so roughly 500k
  vectors. Fastest: 0.50 ms per search at 100k.
- **A disk index** past that. The graph and full vectors stay on disk; RAM holds
  compressed vectors (a quarter of the budget) and a cache of the nodes every
  search passes through, sized to what is left.

Measured at 100k on the NVMe drive, searching the disk index costs 4.33 ms with
nothing cached in 17 MB, 2.63 ms with a tenth cached, 1.33 ms with half — but
with everything cached it holds 124 MB against the memory index's 90 MB and is
still slower. So caching is the dial once memory is short, not a replacement for
the memory index. For scale: embedding a query takes 3.7 ms and reranking 32
candidates 136 ms on the same CPU.

## Why every allocation is admitted first

Linux overcommits. A native allocation the machine cannot back does not fail —
it succeeds, and when the pages are touched the OOM killer picks a process, quite
possibly this one. A `MemoryError` never arrives to be caught. So nothing here
allocates and hopes: `memory_guard` reads `MemAvailable` and the tightest cgroup
v2 limit above the process, keeps `MEMORY_HEADROOM_MB` back, and every index the
module makes is admitted against that first. What does not fit degrades:

| Short of | Falls back to |
| :--- | :--- |
| memory for the recent index | dense search off, keyword search carries on; `refresh()` retries |
| memory for the built generation | a disk generation opens with a smaller cache, down to none; a memory one is skipped — searched again on refresh |
| a usable current generation (corrupt, wrong model, files missing) | the previous generation, kept for this |
| memory to build a memory index | a disk index, whose build is bounded |
| memory for any build, or disk space | no build; the current generation stays |
| a build that dies, fails or hangs | the current generation stays; the reason is logged |
| PostgreSQL at startup | the built generation alone; recent vectors on refresh |

The builder runs as its own process (`python -m …index_build`) and sets its
`oom_score_adj` to 1000: if the estimates are ever wrong, the kernel kills the
builder, not the application. An address-space limit cannot do this job —
importing diskannpy reserves 1.4 GB of address space to use 62 MB.

It is a separate interpreter rather than `multiprocessing` because spawn
re-imports the caller's main script, and an entry point without a `__main__`
guard ran itself again inside the builder.

## Why a build is crash-safe

The generation is written into `gen-N.building/`, fsynced, renamed, and only
then named in `CURRENT` — replaced atomically, never rewritten in place. A crash
at any step leaves `CURRENT` naming a complete generation; the debris is swept by
the next build. Readers check every file's size against the manifest before
opening, and fall back to `previous` when one is wrong. One build runs at a
time, by `flock`; a second one skips rather than waits.

## Why the graph is saturated, and searches are capped

The recent index is diskannpy's dynamic index, which defaults to
`saturate_graph=False` and then cannot find its own vectors: recall@10 of 0.09
(bug 5.19). Static builds are not affected.

Every search asks each part for no more than it holds. Over that, the dynamic
index fills the surplus from uninitialised memory (bug 5.18), and a disk index
pads with position 0 — a real vector, so a wrong but valid label. A disk index
is also never searched with one thread: it does not return (bug 5.24).

## Known gaps

- Nothing deletes a vector (**bug 5.5**). A built generation cannot drop one at
  all until the next build; `delete_vector` reaches the recent index only.
- A label whose vector was missing when a generation was built is not looked
  for again until the next build, even if its document is re-ingested.
- One store holds one embedding model's vectors. The vector id belongs to the
  chunk, so changing models means ingesting into a fresh store.
- `VectorRepository.batch_search` is still one query per id; `vectors_for` is
  the bulk read.
