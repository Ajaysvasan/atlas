"""AtlasService: Query, Ingest and Health, as thin adapters over the pipeline the CLI runs.

No intelligence lives here. Query is `AtlasPipeline.new_conversation` + `ask`,
exactly as the CLI calls them; Ingest stages and validates the stream, then
hands the files to `UploadIngestion`, which uses the quota-checked
`ingest_files`. Errors leave as status codes with a stable `atlas-error-code`
trailer and a message written for the end user, never a stack trace or path.
"""

import hashlib
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List

import grpc

from config import Config, get_logger, log_context
from grpc_server import uploads as upload_checks
from grpc_server.auth import ERROR_CODE_KEY
from grpc_server.gate import Abandoned, Busy, OperationGate
from grpc_server.generated import atlas_service_pb2 as pb
from grpc_server.generated import atlas_service_pb2_grpc as pb_grpc
from grpc_server.settings import GrpcSettings
from pipeline.uploads import INGESTED, NOT_INGESTED, UploadedFile, UploadIngestion

logger = get_logger(__name__)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@-]{0,127}$")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

# Pipeline error category -> (status, atlas-error-code, message for the user).
_CATEGORIES = {
    "conversation": (grpc.StatusCode.NOT_FOUND, "CONVERSATION_NOT_FOUND", "no such conversation"),
    "input": (grpc.StatusCode.INVALID_ARGUMENT, "INVALID_REQUEST", "the query cannot be processed as given"),
    "model": (grpc.StatusCode.UNAVAILABLE, "MODEL_UNAVAILABLE", "the language model is unavailable"),
    "database": (grpc.StatusCode.UNAVAILABLE, "DATABASE_UNAVAILABLE", "the database is unavailable"),
    "internal": (grpc.StatusCode.INTERNAL, "INTERNAL", "the pipeline failed"),
}

_FILE_STATUS = {
    INGESTED: pb.FILE_STATUS_INGESTED,
    NOT_INGESTED: pb.FILE_STATUS_NOT_INGESTED,
    "FAILED": pb.FILE_STATUS_FAILED,
    "REJECTED": pb.FILE_STATUS_REJECTED,
}


def _abort(context, status: grpc.StatusCode, code: str, message: str, **trailers):
    metadata = [(ERROR_CODE_KEY, code)] + [(f"atlas-{k.replace('_', '-')}", str(v)) for k, v in trailers.items() if v]
    context.set_trailing_metadata(tuple(metadata))
    context.abort(status, message)


def _category(error: BaseException) -> str:
    from pipeline.atlas import _category as pipeline_category

    return pipeline_category(error)


class AtlasServicer(pb_grpc.AtlasServiceServicer):
    def __init__(
        self,
        pipeline,
        uploads: UploadIngestion,
        settings: GrpcSettings,
        gate: OperationGate | None = None,
        ready: Callable[[], bool] = lambda: True,
        staging_dir: str | Path | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.uploads = uploads
        self.settings = settings
        self.gate = OperationGate(settings.max_waiting) if gate is None else gate
        self.ready = ready
        self.staging_dir = staging_dir

    # ------------------------------------------------------------------ helpers

    def __request_id(self, value: str) -> str:
        return value if _REQUEST_ID.match(value or "") else uuid.uuid4().hex

    def __user(self, value: str, context) -> str:
        if not _IDENTIFIER.match(value or ""):
            _abort(context, grpc.StatusCode.INVALID_ARGUMENT, "INVALID_REQUEST", "a valid user id is required")
        return value

    def __admitted(self, context, work: Callable[[], object]):
        """Run `work` once the gate lets this request in; refusals become status codes."""
        try:
            with self.gate.admit(context.is_active, context.time_remaining()):
                if not context.is_active():
                    _abort(context, grpc.StatusCode.CANCELLED, "CANCELLED", "the caller went away")
                return work()
        except Busy:
            _abort(context, grpc.StatusCode.RESOURCE_EXHAUSTED, "BUSY", "Atlas is busy; try again shortly")
        except Abandoned:
            _abort(context, grpc.StatusCode.DEADLINE_EXCEEDED, "CANCELLED",
                   "the request was abandoned while it waited")

    # -------------------------------------------------------------------- Query

    def Query(self, request: pb.QueryRequest, context) -> pb.QueryResponse:
        request_id = self.__request_id(request.request_id)
        user_id = self.__user(request.user_id, context)
        query = (request.query or "").strip()
        if not query:
            _abort(context, grpc.StatusCode.INVALID_ARGUMENT, "INVALID_REQUEST", "the query is empty")
        if len(query) > self.settings.max_query_chars:
            _abort(context, grpc.StatusCode.INVALID_ARGUMENT, "QUERY_TOO_LARGE",
                   f"the query is longer than {self.settings.max_query_chars} characters")
        conversation_id = request.conversation_id if request.HasField("conversation_id") else None
        if conversation_id is not None and not _IDENTIFIER.match(conversation_id):
            # Malformed ids get the same answer as another user's: nothing to find.
            _abort(context, grpc.StatusCode.NOT_FOUND, "CONVERSATION_NOT_FOUND", "no such conversation")

        with log_context(request_id=request_id, user_id=user_id):
            logger.info("Query received (%d chars, %s conversation)", len(query),
                        "new" if conversation_id is None else "existing")
            return self.__admitted(context, lambda: self.__query(context, request_id, user_id,
                                                                 conversation_id, query))

    def __query(self, context, request_id, user_id, conversation_id, query) -> pb.QueryResponse:
        try:
            if conversation_id is None:
                conversation = self.pipeline.new_conversation(user_id, query)
                created = True
            else:
                conversation = self.pipeline.registry.get(conversation_id, user_id)
                created = False
        except Exception as error:
            category = _category(error)
            logger.error("Conversation lookup failed (%s): %s", category, error,
                         exc_info=category == "internal")
            status, code, message = _CATEGORIES.get(category, _CATEGORIES["internal"])
            _abort(context, status, code, message, request_id=request_id)
        if conversation is None:
            _abort(context, grpc.StatusCode.NOT_FOUND, "CONVERSATION_NOT_FOUND", "no such conversation",
                   request_id=request_id)

        response = self.pipeline.ask(conversation.conversation_id, user_id, query, request_id=request_id)
        record = response.record
        if response.text is None:
            status, code, message = _CATEGORIES.get(record.error_category, _CATEGORIES["internal"])
            _abort(context, status, code, message, request_id=request_id,
                   conversation_id=conversation.conversation_id)
        return pb.QueryResponse(
            request_id=request_id,
            conversation_id=conversation.conversation_id,
            conversation_name=conversation.conversation_name,
            new_conversation=created,
            answer=response.text,
            status=(pb.QUERY_STATUS_COMPLETED_WITH_WARNINGS if response.warnings
                    else pb.QUERY_STATUS_COMPLETED),
            execution_time_ms=int(round(record.total_ms or 0)),
            retry_count=record.job_retries + record.plan_revisions + record.runtime_retries,
            warnings=list(response.warnings),
            persisted=response.persisted,
            stage_ms={name: int(round(ms)) for name, ms in record.stages.items()},
            mode=record.mode or "",
            thinking_path=record.path or "",
        )

    # ------------------------------------------------------------------- Ingest

    def Ingest(self, request_iterator, context) -> pb.IngestResponse:
        started = time.monotonic()
        root = Path(self.staging_dir or Path(Config.DATA_DIR) / ".upload_staging")
        root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="ingest-", dir=root))
        received: List[dict] = []
        try:
            return self.__ingest(request_iterator, context, staging, started, received)
        finally:
            for item in received:
                item["handle"].close()
            shutil.rmtree(staging, ignore_errors=True)

    def __ingest(self, request_iterator, context, staging: Path, started: float,
                 received: List[dict]) -> pb.IngestResponse:
        def invalid(message: str):
            _abort(context, grpc.StatusCode.INVALID_ARGUMENT, "INVALID_REQUEST", message)

        messages = iter(request_iterator)
        first = next(messages, None)
        if first is None or first.WhichOneof("part") != "header":
            invalid("the stream must start with a header")
        header = first.header
        request_id = self.__request_id(header.request_id)
        user_id = self.__user(header.user_id, context)
        if not 1 <= header.file_count <= self.settings.max_files:
            invalid(f"between 1 and {self.settings.max_files} files are accepted per request")

        current = None
        total = 0
        with log_context(request_id=request_id, user_id=user_id):
            for message in messages:
                part = message.WhichOneof("part")
                if part == "file_start":
                    start = message.file_start
                    if current is not None or start.file_index != len(received) \
                            or len(received) >= header.file_count:
                        invalid("files must be sent one at a time, in order")
                    name = upload_checks.safe_filename(start.filename)
                    current = {"index": start.file_index, "name": name, "path": staging / f"{start.file_index}.part",
                               "hash": hashlib.sha256(), "size": 0, "too_large": False}
                    current["handle"] = open(current["path"], "wb")
                    received.append(current)
                elif part == "chunk":
                    if current is None or message.chunk.file_index != current["index"]:
                        invalid("a chunk arrived outside its file")
                    data = message.chunk.data
                    total += len(data)
                    if total > self.settings.max_request_bytes:
                        _abort(context, grpc.StatusCode.RESOURCE_EXHAUSTED, "TOO_LARGE",
                               f"the upload is larger than {self.settings.max_request_bytes} bytes")
                    current["size"] += len(data)
                    current["hash"].update(data)
                    if current["size"] > self.settings.max_file_bytes:
                        current["too_large"] = True
                    elif not current["too_large"]:
                        current["handle"].write(data)
                elif part == "file_end":
                    if current is None or message.file_end.file_index != current["index"]:
                        invalid("a file ended that was not open")
                    current["handle"].close()
                    current["claimed_sha"] = message.file_end.sha256.lower()
                    current = None
                else:
                    invalid("unexpected message in the stream")
            if current is not None or len(received) != header.file_count:
                invalid("the stream ended before every file was complete")
            if not context.is_active():
                _abort(context, grpc.StatusCode.CANCELLED, "CANCELLED", "the caller went away")

            results: Dict[int, pb.FileResult] = {}
            accepted: List[UploadedFile] = []
            for item in received:
                sha = item["hash"].hexdigest()
                rejection = None
                if item["too_large"]:
                    rejection = upload_checks.Rejection(
                        "TOO_LARGE", f"the file is larger than {self.settings.max_file_bytes} bytes")
                elif not _SHA256.match(item.get("claimed_sha", "")) or item["claimed_sha"] != sha:
                    rejection = upload_checks.Rejection("CHECKSUM_MISMATCH", "the file arrived damaged")
                else:
                    rejection = upload_checks.validate(item["path"], item["name"], self.settings)
                if rejection is not None:
                    results[item["index"]] = pb.FileResult(
                        file_index=item["index"], filename=item["name"], status=pb.FILE_STATUS_REJECTED,
                        error_code=rejection.code, error_message=rejection.message)
                    continue
                accepted.append(UploadedFile(item["index"], item["name"], upload_checks.extension(item["name"]),
                                             item["path"], sha, item["size"]))
            logger.info("Upload received: %d file(s), %d accepted, %d bytes", len(received), len(accepted), total)

            stopped = ""
            usage = 0
            limit = Config.MAX_INGESTION_STORAGE_BYTES
            if accepted:
                report = self.__admitted(context, lambda: self.uploads.ingest(user_id, accepted))
                stopped = report.stopped or ""
                usage, limit = report.usage_bytes, report.limit_bytes
                for outcome in report.files:
                    results[outcome.index] = pb.FileResult(
                        file_index=outcome.index, filename=outcome.filename,
                        status=_FILE_STATUS[outcome.status], document_id=outcome.document_id,
                        chunks=outcome.chunks, vectors=outcome.vectors, stored_bytes=outcome.stored_bytes,
                        error_code=outcome.error_code, error_message=outcome.error_message)

        files = [results[i] for i in sorted(results)]
        succeeded = sum(1 for f in files if f.status == pb.FILE_STATUS_INGESTED)
        failed = len(files) - succeeded
        status = (pb.INGEST_STATUS_COMPLETED if failed == 0 else
                  pb.INGEST_STATUS_FAILED if succeeded == 0 else pb.INGEST_STATUS_PARTIAL)
        response = pb.IngestResponse(
            request_id=request_id, status=status, files=files, succeeded=succeeded, failed=failed,
            storage_used_bytes=usage, stopped_reason=stopped,
            execution_time_ms=int((time.monotonic() - started) * 1000))
        if limit is not None:
            response.storage_limit_bytes = limit
        return response

    # ------------------------------------------------------------------- Health

    def Health(self, request, context) -> pb.HealthResponse:
        return pb.HealthResponse(serving=True, pipeline_ready=bool(self.ready()),
                                 queued=self.gate.waiting, busy=self.gate.busy)
