"""Check that the ingested corpus is consistent and that retrieval finds the right material in it.

    uv run python scripts/validate_corpus.py

Read-only on the corpus: it counts and cross-checks rows, then runs representative
queries through the real retriever and KSV, in a fresh process (so the vector index
is opened from disk, not rebuilt).
"""

import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config, configure_logging  # noqa: E402

QUERIES = [
    ("How does PostgreSQL MVCC work?", ("mvcc", "transaction-iso", "routine-vacuuming", "isolation")),
    ("What is the difference between B-tree and B+ tree indexing?", ("btree", "fileformat2", "indexes")),
    ("How does write-ahead logging help crash recovery?", ("wal", "atomiccommit")),
    ("What is the difference between sharding and partitioning?", ("partitioning", "dynamo")),
    ("How do isolation levels affect concurrent transactions?", ("transaction-iso", "isolation", "mvcc")),
]


def main() -> int:
    configure_logging(log_file=Config.LOG_FILE)
    from data_layer.ingestion.corpus_ingestion import read_provenance, source_urls
    from data_layer.ingestion.quota import IngestionQuota
    from data_layer.vector_db_manager import stored_vectors
    from data_layer.vector_db_manager.index_generations import IndexGenerations
    from knowledge_sufficiency.ksv_manager import KSVManager
    from memory.memory_manager import shared_models
    from retrieval_layer.retriever import Retriever

    failures = []
    chunks = sqlite3.connect(f"file:{Config.DB_PATH}?mode=ro", uri=True)
    count = lambda sql: chunks.execute(sql).fetchone()[0]  # noqa: E731
    documents = count("select count(*) from Documents")
    chunk_rows = count("select count(*) from Chunks") + count("select count(*) from RecursiveChunks")
    labels = count("select count(*) from vector_meta_data")
    unlabelled = count(
        "select count(*) from (select chunkId from Chunks union all select chunkId from RecursiveChunks) c "
        "where c.chunkId not in (select chunkId from vector_meta_data)")
    dangling = count(
        "select count(*) from vector_meta_data where chunkId not in "
        "(select chunkId from Chunks union all select chunkId from RecursiveChunks)")
    vector_ids = [row[0] for row in chunks.execute("select vectorId from vector_meta_data")]
    store = stored_vectors.chunk_vector_store()
    try:
        stored = store.vectors_for(vector_ids)
    finally:
        store.close()
    missing_vectors = len(set(vector_ids) - set(stored))
    print(f"Documents {documents}; chunks {chunk_rows}; labels {labels}; vectors in PostgreSQL {len(stored)}")
    print(f"Chunks without a label {unlabelled}; labels without a chunk {dangling}; "
          f"labels without a vector {missing_vectors}")
    if dangling or missing_vectors or labels != len(stored):
        failures.append("label/vector/chunk mismatch")

    provenance = [e for e in read_provenance() if e.get("document_id")]
    print(f"Documents with provenance (url, topic, fetched_at, sha256): {len(provenance)}")

    generations = IndexGenerations(Config.INDEX_PATH)
    print(f"Index generation {generations.current()} covers labels through {generations.through()}")

    usage = IngestionQuota().usage()
    limit = Config.MAX_INGESTION_STORAGE_BYTES
    print(f"Corpus storage {usage.total:,} bytes of {'unlimited' if limit is None else f'{limit:,}'}")
    if limit is not None and usage.total > limit:
        failures.append("over quota")

    models = shared_models()
    started = time.monotonic()
    retriever = Retriever(models.settings, preparation=models.preparation, reranker=models.reranker)
    print(f"Retriever opened in {time.monotonic() - started:.1f} s; searching generation "
          f"{retriever.vector_search.generation}")
    ksv = KSVManager()
    urls = source_urls()
    for query, expected in QUERIES:
        started = time.monotonic()
        passages = retriever.retrieve(query).passages
        took = (time.monotonic() - started) * 1000
        vectors = models.preparation.encode([query] + [p.text for p in passages])
        verdict = ksv.sufficiency_verification(vectors[0], [(p.vector_id, vectors[i + 1]) for i, p in enumerate(passages)])
        sources = [urls.get(p.document_id or "", p.document_id or "?").rsplit("/", 1)[-1] for p in passages[:5]]
        hit = any(any(word in source for word in expected) for source in sources)
        print(f"\n{query}\n  {len(passages)} passage(s) in {took:.0f} ms; KSV {verdict.sufficiency.value} "
              f"(best {verdict.best:.2f}); top sources: {', '.join(sources)}"
              f"\n  relevant source in top 5: {'yes' if hit else 'NO'}")
        if passages:
            print(f"  top passage: {passages[0].text[:160]!r}")
        if not hit:
            failures.append(f"no relevant source for {query!r}")
    retriever.close()
    print("\nFAILURES: " + "; ".join(failures) if failures else "\nAll checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
