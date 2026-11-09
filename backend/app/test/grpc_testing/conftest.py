"""A real gRPC server on a loopback port, with the pipeline and uploads replaced by fakes."""

import hashlib
import threading
import uuid
from typing import List

import grpc
import pytest

from grpc_server.gate import OperationGate
from grpc_server.generated import atlas_service_pb2 as pb
from grpc_server.generated import atlas_service_pb2_grpc as pb_grpc
from grpc_server.server import build_server
from grpc_server.service import AtlasServicer
from grpc_server.settings import GrpcSettings
from memory.conversation_registry import ConversationRecord
from observability.telemetry import RequestRecord
from pipeline.atlas import AtlasResponse
from pipeline.uploads import FileOutcome, UploadReport

TOKEN = "test-service-token-0123456789"
ALICE = "6f1c2c0e-9a43-4b8e-8d2c-111111111111"
BOB = "6f1c2c0e-9a43-4b8e-8d2c-222222222222"


class FakeRegistry:
    def __init__(self) -> None:
        self.rows = {}

    def create(self, user_id, name):
        record = ConversationRecord(uuid.uuid4().hex, user_id, name, "2026-11-09T00:00:00+00:00")
        self.rows[record.conversation_id] = record
        return record

    def get(self, conversation_id, user_id):
        record = self.rows.get(conversation_id)
        return record if record is not None and record.user_id == user_id else None


class FakePipeline:
    """new_conversation + ask, the two calls the servicer makes, with a recordable outcome."""

    def __init__(self) -> None:
        self.registry = FakeRegistry()
        self.asked: List[tuple] = []
        self.fail_category: str | None = None
        self.raise_on_new: Exception | None = None
        self.warnings = ()
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def new_conversation(self, user_id, first_query):
        if self.raise_on_new is not None:
            raise self.raise_on_new
        return self.registry.create(user_id, "PostgreSQL MVCC")

    def ask(self, conversation_id, user_id, query, request_id=None):
        self.started.set()
        self.release.wait(10)
        self.asked.append((conversation_id, user_id, query, request_id))
        record = RequestRecord(request_id, conversation_id, "2026-11-09T00:00:00+00:00")
        record.total_ms = 1250.4
        record.job_retries, record.plan_revisions, record.runtime_retries = 1, 1, 0
        record.mode, record.path = "doubt", "direct"
        record.stages.update({"decomposition": 100.2, "generation": 900.0})
        if self.fail_category:
            record.status, record.error_category = "failed", self.fail_category
            return AtlasResponse(request_id, conversation_id, None, "failed", False, (),
                                 "/srv/secret/path.py line 3: boom", record)
        record.status = "completed"
        return AtlasResponse(request_id, conversation_id, "PostgreSQL uses MVCC.", "completed", True,
                             tuple(self.warnings), None, record)


class FakeUploads:
    def __init__(self) -> None:
        self.calls = []
        self.outcome = "INGESTED"
        self.error_code = ""

    def ingest(self, user_id, files):
        self.calls.append((user_id, [(f.filename, f.extension, f.sha256, f.path.read_bytes()) for f in files]))
        outcomes = [
            FileOutcome(f.index, f.filename, self.outcome,
                        document_id=f"doc-{f.sha256[:8]}" if self.outcome == "INGESTED" else "",
                        chunks=3, vectors=3, stored_bytes=4096, error_code=self.error_code,
                        error_message="the ingestion storage quota is exhausted" if self.error_code else "")
            for f in files
        ]
        stopped = "quota" if self.error_code == "QUOTA_EXCEEDED" else None
        return UploadReport(outcomes, stopped, 10_000, 1 << 30, 0.1)


@pytest.fixture
def settings():
    return GrpcSettings(service_token=TOKEN, max_waiting=1, max_file_bytes=64 * 1024,
                        max_request_bytes=256 * 1024, max_query_chars=200)


@pytest.fixture
def fakes():
    return FakePipeline(), FakeUploads()


@pytest.fixture
def server(settings, fakes, tmp_path):
    pipeline, uploads = fakes
    servicer = AtlasServicer(pipeline, uploads, settings, OperationGate(settings.max_waiting),
                             staging_dir=tmp_path / "staging")
    grpc_server, port = build_server(servicer, settings, port=0)
    grpc_server.start()
    yield port
    grpc_server.stop(None)


@pytest.fixture
def channel(server):
    with grpc.insecure_channel(f"127.0.0.1:{server}") as channel:
        yield channel


@pytest.fixture
def stub(channel):
    return pb_grpc.AtlasServiceStub(channel)


AUTH = (("authorization", f"Bearer {TOKEN}"),)


def upload_stream(user_id, files, request_id="req-ingest-1", tamper=False, count=None):
    yield pb.IngestRequest(header=pb.IngestHeader(request_id=request_id, user_id=user_id,
                                                  file_count=len(files) if count is None else count))
    for index, (name, data) in enumerate(files):
        yield pb.IngestRequest(file_start=pb.FileStart(file_index=index, filename=name,
                                                       content_type="application/octet-stream",
                                                       size_bytes=len(data)))
        for offset in range(0, len(data), 16 * 1024):
            yield pb.IngestRequest(chunk=pb.FileChunk(file_index=index, data=data[offset:offset + 16 * 1024]))
        digest = hashlib.sha256(data + (b"x" if tamper else b"")).hexdigest()
        yield pb.IngestRequest(file_end=pb.FileEnd(file_index=index, sha256=digest))


def trailer(error: grpc.RpcError, key="atlas-error-code"):
    return dict(error.trailing_metadata() or ()).get(key)
