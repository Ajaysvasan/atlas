# Vector stores (`vector_db_manager/`)

## What this module does

Holds the two vector stores this project uses, behind thin wrappers.

| Wrapper | Store | Holds |
| :--- | :--- | :--- |
| `VectorDbManager` → `VectorDb_diskann` | DiskANN, on disk | document chunk vectors |
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
delete, save and load runs under one `threading.Lock`. Searches do not take it —
they do not mutate. Removing the lock and running the stress test produces
dozens of concurrent entries into the index.

## Known gaps

- The DiskANN index is never persisted or reloaded by the pipeline (**P3**), and
  `VectorDbManager.load()` discards what it loaded (**bug 5.8**).
- The pgvector paths cannot work against a real server (**bug 5.1**).
