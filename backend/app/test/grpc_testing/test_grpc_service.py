"""AtlasService over a real loopback gRPC server: auth, Query, Ingest, failures and the operation gate."""

import io
import threading
import time
import zipfile

import grpc
import pytest

from grpc_server.generated import atlas_service_pb2 as pb
from inference.inference_exceptions import InferenceError

from .conftest import ALICE, AUTH, BOB, trailer, upload_stream


def query(stub, user=ALICE, text="Explain how PostgreSQL MVCC works.", conversation=None, metadata=AUTH,
          timeout=10, request_id="req-1"):
    request = pb.QueryRequest(request_id=request_id, user_id=user, query=text)
    if conversation is not None:
        request.conversation_id = conversation
    return stub.Query(request, metadata=metadata, timeout=timeout)


def office(folder="word/", body=b"<w:document>hello world</w:document>"):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr(f"{folder}document.xml", body)
    return buffer.getvalue()


class TestServiceAuthentication:
    def test_no_token_is_unauthenticated(self, stub, fakes):
        with pytest.raises(grpc.RpcError) as caught:
            query(stub, metadata=())
        assert caught.value.code() == grpc.StatusCode.UNAUTHENTICATED
        assert fakes[0].asked == []

    def test_a_wrong_token_is_unauthenticated_for_streams_too(self, stub, fakes):
        bad = (("authorization", "Bearer not-the-token-at-all"),)
        with pytest.raises(grpc.RpcError) as caught:
            query(stub, metadata=bad)
        assert caught.value.code() == grpc.StatusCode.UNAUTHENTICATED
        with pytest.raises(grpc.RpcError) as caught:
            stub.Ingest(upload_stream(ALICE, [("a.txt", b"hello")]), metadata=bad, timeout=10)
        assert caught.value.code() == grpc.StatusCode.UNAUTHENTICATED
        assert fakes[1].calls == []

    def test_health_needs_the_token_and_reports_state(self, stub):
        response = stub.Health(pb.HealthRequest(), metadata=AUTH, timeout=5)
        assert response.serving and response.pipeline_ready and not response.busy


class TestQuery:
    def test_a_new_conversation_is_created_and_answered(self, stub, fakes):
        response = query(stub)
        assert response.request_id == "req-1"
        assert response.new_conversation
        assert response.conversation_name == "PostgreSQL MVCC"
        assert response.answer == "PostgreSQL uses MVCC."
        assert response.status == pb.QUERY_STATUS_COMPLETED
        assert response.execution_time_ms == 1250
        assert response.retry_count == 2
        assert response.stage_ms["generation"] == 900
        assert (response.mode, response.thinking_path, response.persisted) == ("doubt", "direct", True)
        assert fakes[0].asked == [(response.conversation_id, ALICE, "Explain how PostgreSQL MVCC works.", "req-1")]

    def test_the_owner_continues_their_conversation(self, stub):
        first = query(stub)
        second = query(stub, conversation=first.conversation_id, text="And vacuum?")
        assert second.conversation_id == first.conversation_id
        assert not second.new_conversation

    def test_another_users_conversation_does_not_exist_for_them(self, stub, fakes):
        mine = query(stub, user=ALICE)
        with pytest.raises(grpc.RpcError) as caught:
            query(stub, user=BOB, conversation=mine.conversation_id)
        assert caught.value.code() == grpc.StatusCode.NOT_FOUND
        assert trailer(caught.value) == "CONVERSATION_NOT_FOUND"
        assert len(fakes[0].asked) == 1

    def test_warnings_are_reported(self, stub, fakes):
        fakes[0].warnings = ("The answer was NOT saved to conversation memory",)
        response = query(stub)
        assert response.status == pb.QUERY_STATUS_COMPLETED_WITH_WARNINGS
        assert list(response.warnings) == ["The answer was NOT saved to conversation memory"]

    @pytest.mark.parametrize("user, text, code", [
        ("", "hello", "INVALID_REQUEST"),
        ("../etc/passwd", "hello", "INVALID_REQUEST"),
        (ALICE, "   ", "INVALID_REQUEST"),
        (ALICE, "x" * 201, "QUERY_TOO_LARGE"),
    ])
    def test_invalid_requests_never_reach_the_pipeline(self, stub, fakes, user, text, code):
        with pytest.raises(grpc.RpcError) as caught:
            query(stub, user=user, text=text)
        assert caught.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert trailer(caught.value) == code
        assert fakes[0].asked == []

    def test_a_malformed_conversation_id_is_simply_not_found(self, stub):
        with pytest.raises(grpc.RpcError) as caught:
            query(stub, conversation="'; drop table conversations; --")
        assert caught.value.code() == grpc.StatusCode.NOT_FOUND

    @pytest.mark.parametrize("category, status, code", [
        ("model", grpc.StatusCode.UNAVAILABLE, "MODEL_UNAVAILABLE"),
        ("database", grpc.StatusCode.UNAVAILABLE, "DATABASE_UNAVAILABLE"),
        ("input", grpc.StatusCode.INVALID_ARGUMENT, "INVALID_REQUEST"),
        ("internal", grpc.StatusCode.INTERNAL, "INTERNAL"),
    ])
    def test_pipeline_failures_map_to_status_codes_without_internals(self, stub, fakes, category, status, code):
        fakes[0].fail_category = category
        with pytest.raises(grpc.RpcError) as caught:
            query(stub)
        assert caught.value.code() == status
        assert trailer(caught.value) == code
        assert trailer(caught.value, "atlas-conversation-id")
        assert "/srv/secret" not in caught.value.details() and "line 3" not in caught.value.details()

    def test_a_missing_dependency_while_starting_a_conversation_is_unavailable(self, stub, fakes):
        fakes[0].raise_on_new = InferenceError("model file /models/x.gguf missing")
        with pytest.raises(grpc.RpcError) as caught:
            query(stub)
        assert caught.value.code() == grpc.StatusCode.UNAVAILABLE
        assert trailer(caught.value) == "MODEL_UNAVAILABLE"
        assert "/models" not in caught.value.details()

    def test_an_unexpected_exception_is_internal(self, stub, fakes):
        fakes[0].raise_on_new = KeyError("secret-key")
        with pytest.raises(grpc.RpcError) as caught:
            query(stub)
        assert caught.value.code() == grpc.StatusCode.INTERNAL
        assert "secret-key" not in caught.value.details()


class TestGate:
    def test_one_runs_one_waits_the_next_is_told_busy(self, stub, fakes):
        pipeline = fakes[0]
        pipeline.release.clear()
        results = {}

        def run(name):
            try:
                results[name] = query(stub, request_id=name, timeout=20).answer
            except grpc.RpcError as error:
                results[name] = error.code()

        first = threading.Thread(target=run, args=("first",))
        first.start()
        assert pipeline.started.wait(5)
        second = threading.Thread(target=run, args=("second",))
        second.start()
        time.sleep(0.5)
        with pytest.raises(grpc.RpcError) as caught:
            query(stub, request_id="third")
        assert caught.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
        assert trailer(caught.value) == "BUSY"
        pipeline.release.set()
        first.join(10)
        second.join(10)
        assert results == {"first": "PostgreSQL uses MVCC.", "second": "PostgreSQL uses MVCC."}

    def test_a_request_whose_deadline_passes_while_queued_never_runs(self, stub, fakes):
        pipeline = fakes[0]
        pipeline.release.clear()
        blocker = threading.Thread(target=lambda: query(stub, request_id="blocker", timeout=20))
        blocker.start()
        assert pipeline.started.wait(5)
        with pytest.raises(grpc.RpcError) as caught:
            query(stub, request_id="late", timeout=1)
        assert caught.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
        # The server's copy of the deadline lands a transit time after the client's;
        # release only once it has passed there too.
        time.sleep(0.3)
        pipeline.release.set()
        blocker.join(10)
        time.sleep(0.5)
        assert [asked[3] for asked in pipeline.asked] == ["blocker"]


class TestIngest:
    def test_several_files_are_staged_checked_and_handed_on(self, stub, fakes):
        files = [("notes.md", b"# MVCC\nPostgreSQL keeps row versions."), ("report.docx", office()),
                 ("paper.pdf", b"%PDF-1.7\n...")]
        response = stub.Ingest(upload_stream(ALICE, files), metadata=AUTH, timeout=10)
        assert response.status == pb.INGEST_STATUS_COMPLETED
        assert (response.succeeded, response.failed) == (3, 0)
        assert [f.filename for f in response.files] == ["notes.md", "report.docx", "paper.pdf"]
        assert all(f.document_id.startswith("doc-") for f in response.files)
        assert response.storage_limit_bytes == 1 << 30
        ((user, handed),) = fakes[1].calls
        assert user == ALICE
        assert [(name, ext, data) for name, ext, _, data in handed] == [
            ("notes.md", "md", files[0][1]), ("report.docx", "docx", files[1][1]), ("paper.pdf", "pdf", files[2][1])]

    def test_bad_files_are_rejected_one_by_one_and_the_rest_still_ingest(self, stub, fakes):
        bomb = office(body=b"0" * (3 * 1024 * 1024))
        files = [
            ("good.txt", b"plain text"),
            ("tool.exe", b"MZ\x90\x00"),
            ("fake.pdf", b"not a pdf at all"),
            ("binary.txt", b"abc\x00\x01\x02"),
            ("../../etc/passwd.txt", b"root:x:0:0"),
            ("empty.md", b""),
            ("bad.json", b"{not json"),
            ("bomb.docx", bomb),
        ]
        response = stub.Ingest(upload_stream(ALICE, files), metadata=AUTH, timeout=20)
        by_name = {f.filename: f for f in response.files}
        assert response.status == pb.INGEST_STATUS_PARTIAL
        assert (response.succeeded, response.failed) == (2, 6)
        assert by_name["good.txt"].status == pb.FILE_STATUS_INGESTED
        assert by_name["passwd.txt"].status == pb.FILE_STATUS_INGESTED
        expected = {"tool.exe": "UNSUPPORTED_TYPE", "fake.pdf": "MALFORMED", "binary.txt": "MALFORMED",
                    "empty.md": "EMPTY", "bad.json": "MALFORMED", "bomb.docx": "MALFORMED"}
        for name, code in expected.items():
            assert (by_name[name].status, by_name[name].error_code) == (pb.FILE_STATUS_REJECTED, code), name
        assert [h[0] for h in fakes[1].calls[0][1]] == ["good.txt", "passwd.txt"]

    def test_a_damaged_transfer_is_rejected(self, stub, fakes):
        response = stub.Ingest(upload_stream(ALICE, [("a.txt", b"hello")], tamper=True), metadata=AUTH, timeout=10)
        assert response.status == pb.INGEST_STATUS_FAILED
        assert response.files[0].error_code == "CHECKSUM_MISMATCH"
        assert fakes[1].calls == []

    def test_an_oversized_file_is_rejected_and_an_oversized_request_refused(self, stub, settings):
        big = b"a" * (settings.max_file_bytes + 1)
        response = stub.Ingest(upload_stream(ALICE, [("big.txt", big), ("ok.txt", b"ok")]),
                               metadata=AUTH, timeout=10)
        assert [f.error_code for f in response.files] == ["TOO_LARGE", ""]
        huge = [(f"part{i}.txt", b"b" * settings.max_file_bytes) for i in range(5)]
        with pytest.raises(grpc.RpcError) as caught:
            stub.Ingest(upload_stream(ALICE, huge), metadata=AUTH, timeout=10)
        assert caught.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
        assert trailer(caught.value) == "TOO_LARGE"

    def test_the_quota_stops_ingestion_and_says_so(self, stub, fakes):
        fakes[1].outcome, fakes[1].error_code = "NOT_INGESTED", "QUOTA_EXCEEDED"
        response = stub.Ingest(upload_stream(ALICE, [("a.txt", b"hello")]), metadata=AUTH, timeout=10)
        assert response.status == pb.INGEST_STATUS_FAILED
        assert response.files[0].status == pb.FILE_STATUS_NOT_INGESTED
        assert response.files[0].error_code == "QUOTA_EXCEEDED"
        assert response.stopped_reason == "quota"

    @pytest.mark.parametrize("stream", [
        lambda: iter([pb.IngestRequest(chunk=pb.FileChunk(file_index=0, data=b"x"))]),
        lambda: upload_stream(ALICE, [("a.txt", b"x")], count=2),
        lambda: upload_stream(ALICE, [("a.txt", b"x")] * 11),
        lambda: upload_stream("", [("a.txt", b"x")]),
    ])
    def test_malformed_streams_are_invalid(self, stub, fakes, stream):
        with pytest.raises(grpc.RpcError) as caught:
            stub.Ingest(stream(), metadata=AUTH, timeout=10)
        assert caught.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert fakes[1].calls == []

    def test_staging_is_cleaned_up_whatever_happens(self, stub, tmp_path):
        stub.Ingest(upload_stream(ALICE, [("a.txt", b"hello")]), metadata=AUTH, timeout=10)
        with pytest.raises(grpc.RpcError):
            stub.Ingest(upload_stream(ALICE, [("a.txt", b"x")], count=2), metadata=AUTH, timeout=10)
        assert list((tmp_path / "staging").iterdir()) == []
