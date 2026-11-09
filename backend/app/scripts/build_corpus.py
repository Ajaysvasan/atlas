"""Fetch a curated database-engineering corpus from trusted sources and ingest it through the pipeline.

Named pages only, never a crawl. Every page is fetched through KSV's trusted-source
fetcher (https, allowlisted host, streamed byte cap), stored under Config.CORPUS_DIR
with its provenance in the manifest, and ingested inside the storage quota.

    uv run python scripts/build_corpus.py            # fetch what is missing, ingest it
    uv run python scripts/build_corpus.py --list     # show the source list and exit

Licences: PostgreSQL documentation, PostgreSQL Licence; SQLite documentation, public
domain; Apache Cassandra documentation, Apache License 2.0. Each host's robots.txt
allows these paths (postgresql.org disallows /docs/devel/, so /docs/current/ is used).
"""

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config, configure_logging  # noqa: E402
from storage.timestamps import utc_now  # noqa: E402

PG = "https://www.postgresql.org/docs/current/"
SQLITE = "https://sqlite.org/"
CASSANDRA = "https://cassandra.apache.org/doc/latest/cassandra/"

# (url, topic, subtopic)
SOURCES = [
    (PG + "tutorial-arch.html", "Relational fundamentals", "Client/server architecture"),
    (PG + "tutorial-concepts.html", "Relational fundamentals", "Relations and tables"),
    (PG + "tutorial-join.html", "Relational fundamentals", "Joins"),
    (PG + "tutorial-fk.html", "Relational fundamentals", "Foreign keys"),
    (PG + "tutorial-transactions.html", "Transactions and ACID", "Transactions"),
    (PG + "mvcc-intro.html", "MVCC and isolation", "MVCC introduction"),
    (PG + "transaction-iso.html", "MVCC and isolation", "Isolation levels"),
    (PG + "explicit-locking.html", "Concurrency control", "Explicit locking"),
    (PG + "applevel-consistency.html", "Concurrency control", "Application-level consistency"),
    (PG + "mvcc-serialization-failure-handling.html", "MVCC and isolation", "Serialization failures"),
    (PG + "routine-vacuuming.html", "MVCC and isolation", "VACUUM and dead tuples"),
    (PG + "storage-hot.html", "Storage engines", "Heap-only tuples"),
    (PG + "indexes-intro.html", "Indexing", "Index basics"),
    (PG + "indexes-types.html", "Indexing", "Index types"),
    (PG + "indexes-multicolumn.html", "Indexing", "Multicolumn indexes"),
    (PG + "indexes-index-only-scans.html", "Indexing", "Index-only scans"),
    (PG + "btree.html", "Indexing", "B-tree indexes"),
    (PG + "btree-implementation.html", "Indexing", "B-tree implementation"),
    (PG + "using-explain.html", "Query planning", "EXPLAIN"),
    (PG + "planner-stats.html", "Query planning", "Planner statistics"),
    (PG + "explicit-joins.html", "Query planning", "Join order"),
    (PG + "planner-optimizer.html", "Query planning", "Planner and optimizer"),
    (PG + "query-path.html", "Query planning", "Path of a query"),
    (PG + "wal-intro.html", "WAL and recovery", "Write-ahead logging"),
    (PG + "wal-reliability.html", "WAL and recovery", "Reliability"),
    (PG + "wal-configuration.html", "WAL and recovery", "Checkpoints"),
    (PG + "wal-internals.html", "WAL and recovery", "WAL internals"),
    (PG + "continuous-archiving.html", "WAL and recovery", "Point-in-time recovery"),
    (PG + "high-availability.html", "Replication", "High availability"),
    (PG + "different-replication-solutions.html", "Replication", "Replication solutions"),
    (PG + "warm-standby.html", "Replication", "Streaming replication"),
    (PG + "hot-standby.html", "Replication", "Hot standby"),
    (PG + "logical-replication.html", "Replication", "Logical replication"),
    (PG + "ddl-partitioning.html", "Partitioning and sharding", "Table partitioning"),
    (PG + "storage-file-layout.html", "Storage engines", "File layout"),
    (PG + "storage-page-layout.html", "Storage engines", "Page layout"),
    (PG + "storage-toast.html", "Storage engines", "TOAST"),
    (PG + "populate.html", "Performance engineering", "Bulk loading"),
    (PG + "performance-tips.html", "Performance engineering", "Performance tips"),
    (PG + "routine-reindex.html", "Performance engineering", "Reindexing"),
    (PG + "monitoring-locks.html", "Performance engineering", "Lock monitoring"),
    (SQLITE + "atomiccommit.html", "Transactions and ACID", "Atomic commit"),
    (SQLITE + "wal.html", "WAL and recovery", "SQLite WAL"),
    (SQLITE + "isolation.html", "MVCC and isolation", "SQLite isolation"),
    (SQLITE + "lockingv3.html", "Concurrency control", "SQLite locking"),
    (SQLITE + "fileformat2.html", "Storage engines", "SQLite file format and B-trees"),
    (SQLITE + "queryplanner.html", "Query planning", "How the SQLite planner works"),
    (SQLITE + "optoverview.html", "Query planning", "SQLite optimizer"),
    (SQLITE + "arch.html", "Storage engines", "SQLite architecture"),
    (SQLITE + "howitworks.html", "Storage engines", "How SQLite works"),
    (CASSANDRA + "architecture/overview.html", "Distributed databases", "Cassandra overview"),
    (CASSANDRA + "architecture/dynamo.html", "Partitioning and sharding", "Consistent hashing and replication"),
    (CASSANDRA + "architecture/guarantees.html", "Distributed databases", "Consistency guarantees"),
    (CASSANDRA + "architecture/storage-engine.html", "Storage engines", "LSM storage engine"),
    (CASSANDRA + "managing/operating/compaction/overview.html", "Storage engines", "Compaction"),
    (CASSANDRA + "managing/operating/hints.html", "Distributed databases", "Hinted handoff"),
    (CASSANDRA + "managing/operating/read_repair.html", "Distributed databases", "Read repair"),
    (CASSANDRA + "developing/data-modeling/data-modeling_rdbms.html", "Distributed databases", "Data modelling vs RDBMS"),
]


def target_file(directory: Path, url: str) -> Path:
    host, _, path = url.removeprefix("https://").partition("/")
    slug = path.strip("/").replace("/", "__") or "index.html"
    return directory / "sources" / host / slug


def fetch_missing(directory: Path) -> list:
    from knowledge_sufficiency.fetching import fetch_document
    from knowledge_sufficiency.ksv_exceptions import FetchFailed, UntrustedSource

    fetched = []
    for url, topic, subtopic in SOURCES:
        path = target_file(directory, url)
        if path.exists():
            continue
        try:
            document = fetch_document(url)
        except (FetchFailed, UntrustedSource) as error:
            print(f"  skipped {url}: {error}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        data = document.text.encode("utf-8")
        path.write_bytes(data)
        fetched.append({
            "file": str(path.relative_to(directory)), "source_url": document.url, "topic": topic,
            "subtopic": subtopic, "fetched_at": utc_now(), "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data), "origin": "curated_corpus",
        })
        print(f"  fetched {url} ({len(data):,} bytes)")
    return fetched


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="print the sources and exit")
    parser.add_argument("--no-fetch", action="store_true", help="ingest what is already downloaded")
    args = parser.parse_args(argv)
    if args.list:
        for url, topic, subtopic in SOURCES:
            print(f"{topic:<28} {subtopic:<36} {url}")
        return 0

    configure_logging(log_file=Config.LOG_FILE)
    from data_layer.ingestion.corpus_ingestion import ingest_files, read_provenance, record_provenance

    directory = Path(Config.CORPUS_DIR)
    print(f"Corpus directory: {directory}")
    fetched = [] if args.no_fetch else fetch_missing(directory)
    if fetched:
        # Recorded before ingesting, so a file whose ingestion fails is retried
        # by the next run rather than sitting on disk unknown.
        record_provenance(fetched, directory)
    known = {entry["file"]: entry for entry in read_provenance(directory) if "file" in entry}
    pending = {}
    files = [directory / name for name in sorted(known) if not known[name].get("document_id")]
    print(f"Ingesting {len(files)} document(s) (quota: "
          f"{'unlimited' if Config.MAX_INGESTION_STORAGE_BYTES is None else f'{Config.MAX_INGESTION_STORAGE_BYTES:,} bytes'})")
    report = ingest_files(files)
    by_path = {Path(d.path).resolve(): d for d in report.documents}
    entries = []
    for name in sorted({**known, **pending}):
        entry = {**known.get(name, {}), **pending.get(name, {})}
        document = by_path.get((directory / name).resolve())
        if document is not None:
            entry.update(document_id=document.document_id, chunks=document.chunks,
                         vectors=document.vectors, ingested_at=utc_now())
            entries.append(entry)
    if entries:
        record_provenance(entries, directory)
    print(f"Documents ingested: {len(report.documents)}; chunks {report.chunks}; vectors {report.vectors}")
    print(f"Skipped: {len(report.skipped)}" + "".join(f"\n  {p}: {why}" for p, why in report.skipped))
    print(f"Corpus storage: {report.usage_before.total:,} -> {report.usage_after.total:,} bytes")
    for name, size in report.usage_after.files.items():
        print(f"  {name:<16} {size:>14,}")
    print(f"  {'postgres':<16} {report.usage_after.postgres_bytes or 0:>14,}")
    print(f"Index generation built: {report.index_built}; stopped: {report.stopped}; "
          f"{report.seconds:.1f} s")
    return 0 if report.stopped is None else 2


if __name__ == "__main__":
    sys.exit(main())
