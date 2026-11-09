"""The ingestion storage quota: configuration, accounting, enforcement on every path, and races."""

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from config import Config, parse_storage_limit
from data_layer.datalayer_exceptions.datalayer_exceptions import (
    IngestionQuotaExceeded,
    InsufficientDiskSpace,
)
from data_layer.ingestion.corpus_ingestion import ingest_files
from data_layer.ingestion.quota import IngestionQuota, corpus_paths

GIB = 1 << 30


class TestConfiguration:
    def test_the_default_is_one_binary_gibibyte(self):
        assert parse_storage_limit(GIB) == 1_073_741_824

    @pytest.mark.parametrize("value", [None, "null", "None", " unlimited ", ""])
    def test_null_means_unlimited(self, value):
        assert parse_storage_limit(value) is None

    @pytest.mark.parametrize("value", [0, -1, "-5", "1.5GB", True, 2.5, "lots"])
    def test_invalid_limits_are_refused(self, value):
        with pytest.raises(ValueError):
            parse_storage_limit(value)

    def test_a_string_of_digits_is_bytes(self):
        assert parse_storage_limit("2048") == 2048


def quota(tmp_path, limit, used=0, free=10 * GIB, reserve=0):
    q = IngestionQuota(limit_bytes=limit, disk_reserve_bytes=reserve, postgres_usage=lambda: used,
                       paths=lambda: {}, lock_path=tmp_path / "lock")
    q.free_disk_bytes = lambda: free
    return q


class TestEnforcement:
    def test_a_reservation_inside_the_limit_is_granted(self, tmp_path):
        with quota(tmp_path, 1000, used=400).reserve(600) as reservation:
            assert reservation.used_at_start == 400

    def test_a_reservation_past_the_limit_is_refused(self, tmp_path):
        with pytest.raises(IngestionQuotaExceeded):
            with quota(tmp_path, 1000, used=400).reserve(601):
                pass

    def test_unlimited_still_keeps_the_disk_reserve(self, tmp_path):
        with quota(tmp_path, None, used=10 * GIB).reserve(10 * GIB - 1):
            pass
        with pytest.raises(InsufficientDiskSpace):
            with quota(tmp_path, None, free=100, reserve=50).reserve(60):
                pass

    def test_a_full_quota_is_checked_again_mid_ingestion(self, tmp_path):
        used = {"bytes": 0}
        q = IngestionQuota(limit_bytes=1000, disk_reserve_bytes=0,
                           postgres_usage=lambda: used["bytes"], paths=lambda: {},
                           lock_path=tmp_path / "lock")
        with q.reserve(10) as reservation:
            used["bytes"] = 995
            with pytest.raises(IngestionQuotaExceeded):
                reservation.check(10)

    def test_concurrent_ingestions_are_serialised(self, tmp_path):
        q = quota(tmp_path, None)
        spans = []

        def ingest():
            with q.reserve(1):
                start = time.monotonic()
                time.sleep(0.05)
                spans.append((start, time.monotonic()))

        threads = [threading.Thread(target=ingest) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        spans.sort()
        assert all(a_end <= b_start for (_, a_end), (b_start, _) in zip(spans, spans[1:]))

    def test_usage_counts_each_file_once_and_ignores_symlinks(self, tmp_path):
        directory = tmp_path / "corpus"
        directory.mkdir()
        (directory / "a.txt").write_bytes(b"x" * 5000)
        (directory / "link.txt").symlink_to(directory / "a.txt")
        q = IngestionQuota(postgres_usage=None, paths=lambda: {"one": directory, "again": directory / "a.txt"})
        usage = q.usage()
        assert usage.files["one"] >= 5000 and usage.files["again"] == 0

    def test_the_counted_paths_are_the_corpus_only(self):
        paths = corpus_paths()
        assert Path(Config.MEMORY_DB) not in paths.values()
        assert Path(Config.DRAFT_MODEL_PATH) not in paths.values()


DOCS = {
    "mvcc.md": "# MVCC\n\n" + "PostgreSQL keeps several versions of each row so readers never block writers. " * 30,
    "wal.md": "# WAL\n\n" + "Write-ahead logging records changes before data pages are written to disk. " * 30,
    "btree.md": "# B-tree\n\n" + "A B-tree keeps keys sorted in pages so lookups take logarithmic time. " * 30,
}


@pytest.fixture
def files(tmp_path):
    paths = []
    for name, text in DOCS.items():
        path = tmp_path / "docs" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
        paths.append(path)
    return paths


def chunk_rows():
    with sqlite3.connect(Config.DB_PATH) as connection:
        tables = {r[0] for r in connection.execute("select name from sqlite_master where type='table'")}
        total = 0
        for table in ("Chunks", "RecursiveChunks"):
            if table in tables:
                total += connection.execute(f"select count(*) from {table}").fetchone()[0]
        labels = connection.execute("select count(*) from vector_meta_data").fetchone()[0]
    return total, labels


class TestIngestion:
    def test_everything_is_ingested_inside_a_generous_quota(self, files, chunk_vectors, tmp_path):
        q = IngestionQuota(limit_bytes=GIB, postgres_usage=chunk_vectors.storage_bytes,
                           lock_path=tmp_path / "lock")
        report = ingest_files(files, quota=q, build_index=False)
        assert report.stopped is None
        assert len(report.documents) == 3 and report.vectors == report.chunks > 0
        chunks, labels = chunk_rows()
        assert chunks == labels == report.vectors == len(chunk_vectors.rows)
        assert report.usage_after.total > report.usage_before.total
        from data_layer.ingestion.corpus_ingestion import WAL_HEADROOM

        assert all(d.added_bytes <= d.estimated_bytes + WAL_HEADROOM for d in report.documents)

    def test_ingestion_stops_before_the_quota_and_keeps_what_it_has(self, files, chunk_vectors, tmp_path):
        from data_layer.ingestion import corpus_ingestion

        q = IngestionQuota(limit_bytes=1000, postgres_usage=chunk_vectors.storage_bytes,
                           paths=lambda: {}, lock_path=tmp_path / "lock")
        corpus_ingestion.WAL_HEADROOM, saved = 0, corpus_ingestion.WAL_HEADROOM
        try:
            report = ingest_files(files, quota=q, build_index=False, estimate=lambda text: 1)
        finally:
            corpus_ingestion.WAL_HEADROOM = saved
        # The first document fits its 1-byte estimate, overshoots in fact, and
        # the measured check stops the run there.
        assert len(report.documents) == 1
        assert "estimate was too low" in report.stopped
        assert "storage limit" in report.skipped[-1][1] and len(report.skipped) == 2
        assert report.stopped is not None
        chunks, labels = chunk_rows()
        # Nothing of the refused documents was written: no chunk without its vector.
        assert chunks == labels == report.documents[0].vectors == len(chunk_vectors.rows)

    def test_the_pipeline_itself_refuses_vectors_past_the_quota(self, files, chunk_vectors, tmp_path):
        """A caller driving the stages directly is still stopped at the vector write."""
        from data_layer.ingestion.ingestion_pipeline import IngestionPipeline

        q = IngestionQuota(limit_bytes=100, postgres_usage=chunk_vectors.storage_bytes,
                           paths=lambda: {}, lock_path=tmp_path / "lock")
        pipeline = IngestionPipeline(quota=q)
        try:
            _, text = pipeline.extract_text_from_file(str(files[0]))
            chunks = pipeline.chunk_text(pipeline.normalize_doc(str(files[0]), text))
            with pytest.raises(IngestionQuotaExceeded):
                pipeline.batch_insert_vectors(pipeline.embed(list(chunks)))
        finally:
            pipeline.close()
        assert chunk_vectors.rows == {}

    def test_unlimited_ingests_everything(self, files, chunk_vectors, tmp_path):
        q = IngestionQuota(limit_bytes=None, postgres_usage=chunk_vectors.storage_bytes,
                           lock_path=tmp_path / "lock")
        report = ingest_files(files, quota=q, build_index=False)
        assert len(report.documents) == 3 and report.limit_bytes is None

    def test_reingesting_the_same_files_writes_nothing_new(self, files, chunk_vectors, tmp_path):
        q = IngestionQuota(limit_bytes=GIB, postgres_usage=chunk_vectors.storage_bytes,
                           lock_path=tmp_path / "lock")
        ingest_files(files, quota=q, build_index=False)
        before = (chunk_rows(), len(chunk_vectors.rows))
        ingest_files(files, quota=q, build_index=False)
        assert (chunk_rows(), len(chunk_vectors.rows)) == before

    def test_an_unreadable_file_is_skipped_not_fatal(self, files, chunk_vectors, tmp_path):
        broken = tmp_path / "docs" / "binary.txt"
        broken.write_bytes(b"\x00\x01\x02" * 100)
        q = IngestionQuota(limit_bytes=GIB, postgres_usage=chunk_vectors.storage_bytes,
                           lock_path=tmp_path / "lock")
        report = ingest_files([broken, *files], quota=q, build_index=False)
        assert len(report.documents) == 3
        assert report.skipped[0][0] == str(broken)

    def test_a_down_vector_store_stops_without_orphan_labels(self, files, chunk_vectors, tmp_path):
        chunk_vectors.down = True
        q = IngestionQuota(limit_bytes=GIB, postgres_usage=None, lock_path=tmp_path / "lock")
        report = ingest_files(files, quota=q, build_index=False)
        assert report.documents == () and "unavailable" in report.stopped
        _, labels = chunk_rows()
        assert labels == 0


def test_a_document_that_would_not_fit_is_refused_before_anything_is_written(files, chunk_vectors, tmp_path):
    q = IngestionQuota(limit_bytes=None, postgres_usage=chunk_vectors.storage_bytes,
                       lock_path=tmp_path / "lock")
    ingest_files(files[:1], quota=q, build_index=False)
    before = (chunk_rows(), len(chunk_vectors.rows))
    q.limit_bytes = q.usage().total + 1000  # far less than the next document's estimate
    report = ingest_files(files[1:], quota=q, build_index=False)
    assert report.documents == () and "needs" in report.stopped
    assert (chunk_rows(), len(chunk_vectors.rows)) == before
    assert report.usage_after.total <= q.limit_bytes
