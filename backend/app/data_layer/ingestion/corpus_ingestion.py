"""Ingest files into the document corpus, one document at a time, inside the storage quota."""

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Sequence, Tuple

from config import Config, get_logger
from data_layer.datalayer_exceptions.datalayer_exceptions import (
    IngestionQuotaExceeded,
    InsufficientDiskSpace,
    InvalidFileType,
    VectorInsertionError,
    VectorStoreUnavailable,
)
from data_layer.ingestion.quota import IngestionQuota, StorageUsage
from storage.timestamps import utc_now

logger = get_logger(__name__)

# Bytes the corpus grows by per byte of extracted text: sections, contexts and
# overlapping chunks in SQLite, and a vector row with its index share in
# PostgreSQL. 17.1 was measured over the initial 57-document corpus (1.07 MB of
# text); 24 leaves 40% for denser documents.
GROWTH_PER_TEXT_BYTE = 24
# New pages in each chunk-store table and index, whatever the document's size;
# about 100 KB was measured for a 2 KB document.
GROWTH_PER_DOCUMENT = 128 * 1024
# The chunk store's write-ahead log grows to ~4.7 MB before SQLite checkpoints
# it, landing on whichever document is being written; every reservation keeps
# room for it.
WAL_HEADROOM = 8 * 1024 * 1024
# A built index generation, per vector, while the previous one is kept beside
# it: 1,237 bytes a vector were measured.
INDEX_BYTES_PER_VECTOR = 4096

MANIFEST = "manifest.jsonl"


class DocumentRecord(NamedTuple):
    path: str
    document_id: str | None
    chunks: int
    vectors: int
    estimated_bytes: int
    added_bytes: int


class IngestionReport(NamedTuple):
    documents: Tuple[DocumentRecord, ...]
    skipped: Tuple[Tuple[str, str], ...]
    stopped: str | None
    usage_before: StorageUsage
    usage_after: StorageUsage
    limit_bytes: int | None
    index_built: bool
    seconds: float

    @property
    def chunks(self) -> int:
        return sum(d.chunks for d in self.documents)

    @property
    def vectors(self) -> int:
        return sum(d.vectors for d in self.documents)


def estimate_bytes(text: str) -> int:
    return GROWTH_PER_DOCUMENT + len(text.encode("utf-8")) * GROWTH_PER_TEXT_BYTE


def ingest_files(
    files: Sequence[Path],
    *,
    pipeline=None,
    quota: IngestionQuota | None = None,
    build_index: bool = True,
    estimate: Callable[[str], int] = estimate_bytes,
    admit: Callable[[Path, int], None] | None = None,
    on_document: Callable[["DocumentRecord"], None] | None = None,
) -> IngestionReport:
    """Every file through the ingestion pipeline, stopping before the quota would be passed.

    The quota is checked before a document's chunks are written, against an
    estimate that errs high, and the measured growth is reported after it.
    A document that cannot be read is skipped; a full quota, a full disk or an
    unreachable vector store stops the run and leaves what was ingested intact.

    `admit` and `on_document` run inside the ingestion lock, before a document
    is written and after it is, so a caller's own limit (acquisition's) is
    checked and recorded with no other ingestion in between.
    """
    from data_layer.ingestion.ingestion_pipeline import IngestionPipeline

    started = time.monotonic()
    owns_pipeline = pipeline is None
    pipeline = IngestionPipeline() if pipeline is None else pipeline
    quota = pipeline.quota if quota is None else quota
    before = quota.usage()
    documents: List[DocumentRecord] = []
    skipped: List[Tuple[str, str]] = []
    stopped = None
    index_built = False
    try:
        for position, path in enumerate(files):
            path = Path(path)
            try:
                _, text = pipeline.extract_text_from_file(str(path))
            except (InvalidFileType, FileNotFoundError, OSError, ValueError) as error:
                skipped.append((str(path), f"unreadable: {error}"))
                continue
            if not text.strip():
                skipped.append((str(path), "no text"))
                continue
            needed = estimate(text)
            try:
                with quota.reserve(needed + WAL_HEADROOM) as reservation:
                    if admit is not None:
                        admit(path, needed)
                    normalized = pipeline.normalize_doc(str(path), text)
                    chunks = pipeline.chunk_text(normalized)
                    embedded = pipeline.embed(list(chunks)) if chunks else []
                    labels = pipeline.batch_insert_vectors(embedded) if embedded else []
                    added = quota.usage().total - reservation.used_at_start
                    record = DocumentRecord(
                        str(path), getattr(normalized.meta_data, "document_id", None),
                        len(chunks), len(labels), needed, added,
                    )
                    if on_document is not None:
                        on_document(record)
            except (IngestionQuotaExceeded, InsufficientDiskSpace) as error:
                stopped = str(error)
                skipped.extend((str(p), "not ingested: storage limit") for p in files[position:])
                logger.warning("Ingestion stopped before %s: %s", path, error)
                break
            except (VectorStoreUnavailable, VectorInsertionError, ConnectionError) as error:
                # This document's chunk rows may already exist without vectors or
                # labels; nothing points at them, and re-ingesting completes them.
                stopped = f"vector store unavailable: {error}"
                skipped.extend((str(p), "not ingested: vector store down") for p in files[position:])
                logger.error("Ingestion stopped at %s: %s", path, error)
                break
            documents.append(record)
            if added > needed + WAL_HEADROOM:
                logger.warning("%s grew the corpus by %d bytes, over its %d-byte estimate",
                               path, added, needed + WAL_HEADROOM)
            if quota.limit_bytes is not None and reservation.used_at_start + added > quota.limit_bytes:
                stopped = (f"the limit was passed by {reservation.used_at_start + added - quota.limit_bytes} "
                           f"bytes at {path.name}; the estimate was too low")
                skipped.extend((str(p), "not ingested: storage limit") for p in files[position + 1:])
                logger.error("Ingestion stopped: %s", stopped)
                break
        if build_index and documents and stopped is None:
            vectors = sum(d.vectors for d in documents)
            try:
                with quota.reserve(vectors * INDEX_BYTES_PER_VECTOR):
                    index_built = pipeline.maintain_index(force=True) is not None
            except (IngestionQuotaExceeded, InsufficientDiskSpace) as error:
                stopped = f"index not rebuilt: {error}"
                logger.warning("The vector index was not rebuilt: %s", error)
    finally:
        if owns_pipeline:
            pipeline.close()
    after = quota.usage()
    report = IngestionReport(tuple(documents), tuple(skipped), stopped, before, after,
                             quota.limit_bytes, index_built, time.monotonic() - started)
    logger.info("Ingested %d document(s), %d chunk(s), %d vector(s); corpus %d -> %d bytes%s",
                len(documents), report.chunks, report.vectors, before.total, after.total,
                f"; stopped: {stopped}" if stopped else "")
    return report


def record_provenance(entries: Sequence[Dict], directory: Path | None = None) -> Path:
    """Append provenance lines (url, topic, fetched_at, sha256, file) to the corpus manifest."""
    directory = Path(Config.CORPUS_DIR if directory is None else directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / MANIFEST
    with open(manifest, "a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps({"recorded_at": utc_now(), **entry}, sort_keys=True) + "\n")
    return manifest


def read_provenance(directory: Path | None = None) -> List[Dict]:
    manifest = Path(Config.CORPUS_DIR if directory is None else directory) / MANIFEST
    if not manifest.exists():
        return []
    with open(manifest, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def source_urls(directory: Path | None = None, chunk_store: str | Path | None = None) -> Dict[str, str]:
    """Source URL by the id retrieval reports as a passage's parent: a document id, or a context id."""
    import sqlite3

    by_document = {
        entry["document_id"]: entry["source_url"]
        for entry in read_provenance(directory)
        if entry.get("document_id") and entry.get("source_url")
    }
    urls = dict(by_document)
    path = Path(Config.DB_PATH if chunk_store is None else chunk_store)
    if not by_document or not path.exists():
        return urls
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "select c.contextId, s.documentId from Contexts c join Sections s on c.sectionId = s.sectionId"
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        connection.close()
    for context_id, document_id in rows:
        if document_id in by_document:
            urls[context_id] = by_document[document_id]
    return urls


ACQUIRED = "ksv_acquisition"


def acquired_bytes(directory: Path | None = None) -> int:
    """What acquired material occupies: its files plus the growth measured when each was ingested."""
    return sum(
        int(entry.get("stored_bytes") or 0)
        for entry in read_provenance(directory)
        if entry.get("origin") == ACQUIRED and entry.get("document_id")
    )
