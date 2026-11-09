"""A user's uploaded files into the corpus, privately, through the one quota-checked ingestion path.

Each file is kept at `<corpus>/uploads/<sha256[:2]>/<sha256>.<ext>`, so a
re-upload of the same bytes is the same document; ownership lives in
`document_owners`, not in the path. The owner row is written before the
document's first chunk, so an interrupted ingestion never leaves a private
document readable by everyone.
"""

import os
import shutil
import time
from pathlib import Path
from typing import Callable, List, NamedTuple, Sequence

from config import Config, get_logger
from data_layer.datalayer_exceptions.datalayer_exceptions import VectorStoreUnavailable
from data_layer.ingestion.corpus_ingestion import ingest_files, record_provenance
from data_layer.ingestion.document_owners import DocumentOwners

logger = get_logger(__name__)

UPLOADS = "uploads"
ORIGIN = "upload"

INGESTED, FAILED, NOT_INGESTED = "INGESTED", "FAILED", "NOT_INGESTED"


class UploadedFile(NamedTuple):
    """A validated file, staged on the data volume, awaiting ingestion."""

    index: int
    filename: str
    extension: str
    path: Path
    sha256: str
    size: int


class FileOutcome(NamedTuple):
    index: int
    filename: str
    status: str
    document_id: str = ""
    chunks: int = 0
    vectors: int = 0
    stored_bytes: int = 0
    error_code: str = ""
    error_message: str = ""


class UploadReport(NamedTuple):
    files: List[FileOutcome]
    stopped: str | None
    usage_bytes: int
    limit_bytes: int | None
    seconds: float


def _skip_reason(reason: str) -> tuple:
    if "storage limit" in reason:
        return NOT_INGESTED, "QUOTA_EXCEEDED", "the ingestion storage quota is exhausted"
    if "vector store" in reason:
        return NOT_INGESTED, "VECTOR_STORE_UNAVAILABLE", "the vector store is unavailable; try again later"
    if reason == "no text":
        return FAILED, "NO_TEXT", "no text could be extracted from the file"
    return FAILED, "MALFORMED", "the file could not be read"


class UploadIngestion:
    """Owns one ingestion pipeline for the process; callers serialise access (the operation gate)."""

    def __init__(
        self,
        refresh_retrieval: Callable[[], None] = lambda: None,
        owners: DocumentOwners | None = None,
        corpus_dir: str | Path | None = None,
        pipeline_factory: Callable[[], object] | None = None,
        ingest: Callable = ingest_files,
    ) -> None:
        self.refresh_retrieval = refresh_retrieval
        self.owners = owners
        self.corpus_dir = corpus_dir
        self.pipeline_factory = pipeline_factory
        self.ingest_files = ingest
        self.__pipeline = None

    def __directory(self) -> Path:
        return Path(Config.CORPUS_DIR if self.corpus_dir is None else self.corpus_dir)

    def __owners(self) -> DocumentOwners:
        return self.owners if self.owners is not None else DocumentOwners()

    def pipeline(self):
        if self.__pipeline is None:
            if self.pipeline_factory is not None:
                self.__pipeline = self.pipeline_factory()
            else:
                from data_layer.ingestion.ingestion_pipeline import IngestionPipeline

                self.__pipeline = IngestionPipeline()
        return self.__pipeline

    def __drop_pipeline(self) -> None:
        if self.__pipeline is not None:
            try:
                self.__pipeline.close()
            except Exception:
                logger.debug("Closing a failed ingestion pipeline raised", exc_info=True)
            self.__pipeline = None

    def ingest(self, user_id: str, files: Sequence[UploadedFile]) -> UploadReport:
        started = time.monotonic()
        owners = self.__owners()
        pipeline = self.pipeline()
        outcomes: dict = {}
        placed: dict = {}        # index -> (final path, newly placed)
        claimed: dict = {}       # index -> (document id, owner row added)
        by_path: dict = {}
        for upload in files:
            final = self.__directory() / UPLOADS / upload.sha256[:2] / f"{upload.sha256}.{upload.extension}"
            final.parent.mkdir(parents=True, exist_ok=True)
            newly = not final.exists()
            if newly:
                shutil.move(str(upload.path), str(final))
                os.chmod(final, 0o600)
            else:
                upload.path.unlink(missing_ok=True)
            placed[upload.index] = (final, newly)
            try:
                _, text = pipeline.extract_text_from_file(str(final))
            except Exception as error:
                logger.info("Upload %d (%s) could not be read: %s", upload.index, upload.filename, error)
                outcomes[upload.index] = FileOutcome(upload.index, upload.filename, FAILED,
                                                     error_code="MALFORMED",
                                                     error_message="the file could not be read")
                continue
            if not text.strip():
                outcomes[upload.index] = FileOutcome(upload.index, upload.filename, FAILED,
                                                     error_code="NO_TEXT",
                                                     error_message="no text could be extracted from the file")
                continue
            document_id = pipeline.normalize_doc(str(final), text).meta_data.document_id
            claimed[upload.index] = (document_id, owners.add(document_id, user_id))
            by_path[str(final)] = upload

        stopped = None
        report = None
        if by_path:
            try:
                report = self.ingest_files([Path(p) for p in by_path], pipeline=pipeline, build_index=False)
                stopped = report.stopped
            except VectorStoreUnavailable as error:
                self.__drop_pipeline()
                stopped = f"vector store unavailable: {error}"
                for upload in by_path.values():
                    outcomes[upload.index] = FileOutcome(
                        upload.index, upload.filename, NOT_INGESTED, error_code="VECTOR_STORE_UNAVAILABLE",
                        error_message="the vector store is unavailable; try again later")
            if report is not None:
                if report.stopped and "vector store" in report.stopped:
                    self.__drop_pipeline()
                for document in report.documents:
                    upload = by_path[document.path]
                    outcomes[upload.index] = FileOutcome(
                        upload.index, upload.filename, INGESTED, document.document_id or claimed[upload.index][0],
                        document.chunks, document.vectors, max(0, document.added_bytes))
                for path, reason in report.skipped:
                    upload = by_path.get(path)
                    if upload is not None and upload.index not in outcomes:
                        status, code, message = _skip_reason(reason)
                        outcomes[upload.index] = FileOutcome(upload.index, upload.filename, status,
                                                             error_code=code, error_message=message)

        ingested = [o for o in outcomes.values() if o.status == INGESTED]
        for upload in files:
            outcome = outcomes.get(upload.index)
            if outcome is None:
                outcomes[upload.index] = FileOutcome(upload.index, upload.filename, FAILED,
                                                     error_code="INTERNAL", error_message="the file was not ingested")
            if outcomes[upload.index].status != INGESTED:
                final, newly = placed[upload.index]
                document_id, added = claimed.get(upload.index, (None, False))
                if added:
                    owners.remove(document_id, user_id)
                if newly:
                    final.unlink(missing_ok=True)
        if ingested:
            record_provenance([
                {"origin": ORIGIN, "document_id": o.document_id, "sha256": u.sha256,
                 "file": str(placed[u.index][0].relative_to(self.__directory())), "stored_bytes": o.stored_bytes}
                for u in files for o in [outcomes[u.index]] if o.status == INGESTED
            ], self.__directory())
            try:
                self.refresh_retrieval()
            except Exception:
                logger.exception("Retrieval could not catch up with the upload; it will on the next refresh")
        usage = report.usage_after.total if report is not None else 0
        limit = report.limit_bytes if report is not None else Config.MAX_INGESTION_STORAGE_BYTES
        logger.info("Upload by %s: %d of %d file(s) ingested%s", user_id, len(ingested), len(files),
                    f"; stopped: {stopped}" if stopped else "")
        return UploadReport([outcomes[u.index] for u in files], stopped, usage, limit,
                            time.monotonic() - started)

    def close(self) -> None:
        self.__drop_pipeline()
