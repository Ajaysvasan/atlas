"""UploadIngestion: ownership before chunks, content-addressed storage, honest per-file outcomes, cleanup."""

import hashlib
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from data_layer.datalayer_exceptions.datalayer_exceptions import VectorStoreUnavailable
from data_layer.ingestion.corpus_ingestion import DocumentRecord, IngestionReport
from data_layer.ingestion.document_owners import DocumentOwners
from data_layer.ingestion.quota import StorageUsage
from pipeline.uploads import UploadedFile, UploadIngestion

ALICE, BOB = "alice", "bob"


class FakeIngestionPipeline:
    closed = 0

    def extract_text_from_file(self, path):
        text = Path(path).read_text()
        if text.startswith("CORRUPT"):
            raise ValueError("cannot parse")
        return ".txt", text

    def normalize_doc(self, path, text):
        return SimpleNamespace(meta_data=SimpleNamespace(document_id="doc-" + hashlib.md5(path.encode()).hexdigest()[:8]))

    def close(self):
        FakeIngestionPipeline.closed += 1


class FakeIngest:
    """Stands in for ingest_files; checks the owner rows already exist when it runs."""

    def __init__(self, owners, skip_reason=None, raises=None):
        self.owners = owners
        self.skip_reason = skip_reason
        self.raises = raises
        self.calls = []
        self.owned_at_call = []

    def __call__(self, paths, pipeline, build_index):
        self.calls.append(([str(p) for p in paths], build_index))
        if self.raises:
            raise self.raises
        docs, skipped = [], []
        for path in paths:
            document_id = pipeline.normalize_doc(str(path), "").meta_data.document_id
            self.owned_at_call.append(self.owners.owners(document_id))
            if self.skip_reason:
                skipped.append((str(path), self.skip_reason))
            else:
                docs.append(DocumentRecord(str(path), document_id, 4, 4, 1000, 2048))
        usage = StorageUsage({"x": 5000}, None)
        return IngestionReport(tuple(docs), tuple(skipped), "stopped" if skipped else None, usage, usage,
                               1 << 30, False, 0.01)


@pytest.fixture
def owners(tmp_path):
    return DocumentOwners(tmp_path / "chunks.db")


def staged(tmp_path, index, name, data: bytes):
    path = tmp_path / "staging" / f"{index}-{name}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return UploadedFile(index, name, name.rsplit(".", 1)[-1], path, hashlib.sha256(data).hexdigest(), len(data))


def service(tmp_path, owners, ingest, refreshed=None):
    return UploadIngestion(refresh_retrieval=(lambda: refreshed.append(1)) if refreshed is not None else (lambda: None),
                           owners=owners, corpus_dir=tmp_path / "corpus",
                           pipeline_factory=FakeIngestionPipeline, ingest=ingest)


def test_the_owner_exists_before_any_chunk_is_written(tmp_path, owners):
    ingest = FakeIngest(owners)
    report = service(tmp_path, owners, ingest).ingest(ALICE, [staged(tmp_path, 0, "a.txt", b"alpha")])
    assert ingest.owned_at_call == [{ALICE}]
    assert ingest.calls[0][1] is False  # joins the recent index; no full rebuild per upload
    assert report.files[0].status == "INGESTED"


def test_an_ingested_file_is_kept_by_content_private_and_recorded(tmp_path, owners):
    refreshed = []
    upload = staged(tmp_path, 0, "notes.md", b"# heading\ntext")
    report = service(tmp_path, owners, FakeIngest(owners), refreshed).ingest(ALICE, [upload])
    kept = tmp_path / "corpus" / "uploads" / upload.sha256[:2] / f"{upload.sha256}.md"
    assert kept.read_bytes() == b"# heading\ntext"
    assert stat.S_IMODE(kept.stat().st_mode) == 0o600
    assert not upload.path.exists()
    outcome = report.files[0]
    assert (outcome.status, outcome.chunks, outcome.vectors, outcome.stored_bytes) == ("INGESTED", 4, 4, 2048)
    assert owners.owners(outcome.document_id) == {ALICE}
    (entry,) = [json.loads(line) for line in (tmp_path / "corpus" / "manifest.jsonl").read_text().splitlines()]
    assert (entry["origin"], entry["sha256"], entry["document_id"]) == ("upload", upload.sha256, outcome.document_id)
    assert ALICE not in json.dumps(entry)
    assert refreshed == [1]


def test_a_quota_stop_reports_not_ingested_and_leaves_nothing_behind(tmp_path, owners):
    upload = staged(tmp_path, 0, "a.txt", b"alpha")
    report = service(tmp_path, owners, FakeIngest(owners, skip_reason="not ingested: storage limit")).ingest(
        ALICE, [upload])
    outcome = report.files[0]
    assert (outcome.status, outcome.error_code) == ("NOT_INGESTED", "QUOTA_EXCEEDED")
    assert report.stopped
    assert not any((tmp_path / "corpus" / "uploads").rglob("*.txt"))
    assert not owners.any_private()


def test_an_unreadable_file_fails_alone(tmp_path, owners):
    files = [staged(tmp_path, 0, "bad.txt", b"CORRUPT data"), staged(tmp_path, 1, "good.txt", b"fine")]
    report = service(tmp_path, owners, FakeIngest(owners)).ingest(ALICE, files)
    assert [(f.status, f.error_code) for f in report.files] == [("FAILED", "MALFORMED"), ("INGESTED", "")]
    assert len(list((tmp_path / "corpus" / "uploads").rglob("*.txt"))) == 1


def test_a_file_with_no_text_fails_as_no_text(tmp_path, owners):
    report = service(tmp_path, owners, FakeIngest(owners)).ingest(ALICE, [staged(tmp_path, 0, "blank.txt", b"   \n")])
    assert (report.files[0].status, report.files[0].error_code) == ("FAILED", "NO_TEXT")


def test_the_same_bytes_from_two_users_are_one_document_with_two_owners(tmp_path, owners):
    first = service(tmp_path, owners, FakeIngest(owners)).ingest(ALICE, [staged(tmp_path, 0, "a.txt", b"same")])
    second = service(tmp_path, owners, FakeIngest(owners)).ingest(BOB, [staged(tmp_path, 0, "b.txt", b"same")])
    assert first.files[0].document_id == second.files[0].document_id
    assert owners.owners(first.files[0].document_id) == {ALICE, BOB}


def test_a_failed_reupload_removes_only_the_new_owner_and_keeps_the_file(tmp_path, owners):
    service(tmp_path, owners, FakeIngest(owners)).ingest(ALICE, [staged(tmp_path, 0, "a.txt", b"same")])
    failed = service(tmp_path, owners, FakeIngest(owners, skip_reason="not ingested: storage limit")).ingest(
        BOB, [staged(tmp_path, 0, "b.txt", b"same")])
    assert failed.files[0].status == "NOT_INGESTED"
    assert len(list((tmp_path / "corpus" / "uploads").rglob("*.txt"))) == 1
    (document_id,) = {d for d in [f"doc-{hashlib.md5(str(p).encode()).hexdigest()[:8]}"
                                  for p in (tmp_path / "corpus" / "uploads").rglob("*.txt")]}
    assert owners.owners(document_id) == {ALICE}


def test_an_unavailable_vector_store_is_reported_and_the_pipeline_rebuilt(tmp_path, owners):
    uploads = service(tmp_path, owners, FakeIngest(owners, raises=VectorStoreUnavailable("down")))
    before = FakeIngestionPipeline.closed
    report = uploads.ingest(ALICE, [staged(tmp_path, 0, "a.txt", b"alpha")])
    assert (report.files[0].status, report.files[0].error_code) == ("NOT_INGESTED", "VECTOR_STORE_UNAVAILABLE")
    assert FakeIngestionPipeline.closed == before + 1
    assert not owners.any_private()
