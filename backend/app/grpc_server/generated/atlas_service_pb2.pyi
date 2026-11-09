from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class QueryStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    QUERY_STATUS_UNSPECIFIED: _ClassVar[QueryStatus]
    QUERY_STATUS_COMPLETED: _ClassVar[QueryStatus]
    QUERY_STATUS_COMPLETED_WITH_WARNINGS: _ClassVar[QueryStatus]

class IngestStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    INGEST_STATUS_UNSPECIFIED: _ClassVar[IngestStatus]
    INGEST_STATUS_COMPLETED: _ClassVar[IngestStatus]
    INGEST_STATUS_PARTIAL: _ClassVar[IngestStatus]
    INGEST_STATUS_FAILED: _ClassVar[IngestStatus]

class FileStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    FILE_STATUS_UNSPECIFIED: _ClassVar[FileStatus]
    FILE_STATUS_INGESTED: _ClassVar[FileStatus]
    FILE_STATUS_REJECTED: _ClassVar[FileStatus]
    FILE_STATUS_FAILED: _ClassVar[FileStatus]
    FILE_STATUS_NOT_INGESTED: _ClassVar[FileStatus]
QUERY_STATUS_UNSPECIFIED: QueryStatus
QUERY_STATUS_COMPLETED: QueryStatus
QUERY_STATUS_COMPLETED_WITH_WARNINGS: QueryStatus
INGEST_STATUS_UNSPECIFIED: IngestStatus
INGEST_STATUS_COMPLETED: IngestStatus
INGEST_STATUS_PARTIAL: IngestStatus
INGEST_STATUS_FAILED: IngestStatus
FILE_STATUS_UNSPECIFIED: FileStatus
FILE_STATUS_INGESTED: FileStatus
FILE_STATUS_REJECTED: FileStatus
FILE_STATUS_FAILED: FileStatus
FILE_STATUS_NOT_INGESTED: FileStatus

class QueryRequest(_message.Message):
    __slots__ = ("request_id", "user_id", "conversation_id", "query", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    CONVERSATION_ID_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    user_id: str
    conversation_id: str
    query: str
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, request_id: _Optional[str] = ..., user_id: _Optional[str] = ..., conversation_id: _Optional[str] = ..., query: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class QueryResponse(_message.Message):
    __slots__ = ("request_id", "conversation_id", "conversation_name", "new_conversation", "answer", "status", "execution_time_ms", "retry_count", "warnings", "persisted", "stage_ms", "mode", "thinking_path")
    class StageMsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    CONVERSATION_ID_FIELD_NUMBER: _ClassVar[int]
    CONVERSATION_NAME_FIELD_NUMBER: _ClassVar[int]
    NEW_CONVERSATION_FIELD_NUMBER: _ClassVar[int]
    ANSWER_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    RETRY_COUNT_FIELD_NUMBER: _ClassVar[int]
    WARNINGS_FIELD_NUMBER: _ClassVar[int]
    PERSISTED_FIELD_NUMBER: _ClassVar[int]
    STAGE_MS_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    THINKING_PATH_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    conversation_id: str
    conversation_name: str
    new_conversation: bool
    answer: str
    status: QueryStatus
    execution_time_ms: int
    retry_count: int
    warnings: _containers.RepeatedScalarFieldContainer[str]
    persisted: bool
    stage_ms: _containers.ScalarMap[str, int]
    mode: str
    thinking_path: str
    def __init__(self, request_id: _Optional[str] = ..., conversation_id: _Optional[str] = ..., conversation_name: _Optional[str] = ..., new_conversation: _Optional[bool] = ..., answer: _Optional[str] = ..., status: _Optional[_Union[QueryStatus, str]] = ..., execution_time_ms: _Optional[int] = ..., retry_count: _Optional[int] = ..., warnings: _Optional[_Iterable[str]] = ..., persisted: _Optional[bool] = ..., stage_ms: _Optional[_Mapping[str, int]] = ..., mode: _Optional[str] = ..., thinking_path: _Optional[str] = ...) -> None: ...

class IngestRequest(_message.Message):
    __slots__ = ("header", "file_start", "chunk", "file_end")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    FILE_START_FIELD_NUMBER: _ClassVar[int]
    CHUNK_FIELD_NUMBER: _ClassVar[int]
    FILE_END_FIELD_NUMBER: _ClassVar[int]
    header: IngestHeader
    file_start: FileStart
    chunk: FileChunk
    file_end: FileEnd
    def __init__(self, header: _Optional[_Union[IngestHeader, _Mapping]] = ..., file_start: _Optional[_Union[FileStart, _Mapping]] = ..., chunk: _Optional[_Union[FileChunk, _Mapping]] = ..., file_end: _Optional[_Union[FileEnd, _Mapping]] = ...) -> None: ...

class IngestHeader(_message.Message):
    __slots__ = ("request_id", "user_id", "file_count")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    FILE_COUNT_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    user_id: str
    file_count: int
    def __init__(self, request_id: _Optional[str] = ..., user_id: _Optional[str] = ..., file_count: _Optional[int] = ...) -> None: ...

class FileStart(_message.Message):
    __slots__ = ("file_index", "filename", "content_type", "size_bytes")
    FILE_INDEX_FIELD_NUMBER: _ClassVar[int]
    FILENAME_FIELD_NUMBER: _ClassVar[int]
    CONTENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    file_index: int
    filename: str
    content_type: str
    size_bytes: int
    def __init__(self, file_index: _Optional[int] = ..., filename: _Optional[str] = ..., content_type: _Optional[str] = ..., size_bytes: _Optional[int] = ...) -> None: ...

class FileChunk(_message.Message):
    __slots__ = ("file_index", "data")
    FILE_INDEX_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    file_index: int
    data: bytes
    def __init__(self, file_index: _Optional[int] = ..., data: _Optional[bytes] = ...) -> None: ...

class FileEnd(_message.Message):
    __slots__ = ("file_index", "sha256")
    FILE_INDEX_FIELD_NUMBER: _ClassVar[int]
    SHA256_FIELD_NUMBER: _ClassVar[int]
    file_index: int
    sha256: str
    def __init__(self, file_index: _Optional[int] = ..., sha256: _Optional[str] = ...) -> None: ...

class FileResult(_message.Message):
    __slots__ = ("file_index", "filename", "status", "document_id", "chunks", "vectors", "stored_bytes", "error_code", "error_message")
    FILE_INDEX_FIELD_NUMBER: _ClassVar[int]
    FILENAME_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    DOCUMENT_ID_FIELD_NUMBER: _ClassVar[int]
    CHUNKS_FIELD_NUMBER: _ClassVar[int]
    VECTORS_FIELD_NUMBER: _ClassVar[int]
    STORED_BYTES_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    file_index: int
    filename: str
    status: FileStatus
    document_id: str
    chunks: int
    vectors: int
    stored_bytes: int
    error_code: str
    error_message: str
    def __init__(self, file_index: _Optional[int] = ..., filename: _Optional[str] = ..., status: _Optional[_Union[FileStatus, str]] = ..., document_id: _Optional[str] = ..., chunks: _Optional[int] = ..., vectors: _Optional[int] = ..., stored_bytes: _Optional[int] = ..., error_code: _Optional[str] = ..., error_message: _Optional[str] = ...) -> None: ...

class IngestResponse(_message.Message):
    __slots__ = ("request_id", "status", "files", "succeeded", "failed", "storage_used_bytes", "storage_limit_bytes", "stopped_reason", "execution_time_ms")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    FILES_FIELD_NUMBER: _ClassVar[int]
    SUCCEEDED_FIELD_NUMBER: _ClassVar[int]
    FAILED_FIELD_NUMBER: _ClassVar[int]
    STORAGE_USED_BYTES_FIELD_NUMBER: _ClassVar[int]
    STORAGE_LIMIT_BYTES_FIELD_NUMBER: _ClassVar[int]
    STOPPED_REASON_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    status: IngestStatus
    files: _containers.RepeatedCompositeFieldContainer[FileResult]
    succeeded: int
    failed: int
    storage_used_bytes: int
    storage_limit_bytes: int
    stopped_reason: str
    execution_time_ms: int
    def __init__(self, request_id: _Optional[str] = ..., status: _Optional[_Union[IngestStatus, str]] = ..., files: _Optional[_Iterable[_Union[FileResult, _Mapping]]] = ..., succeeded: _Optional[int] = ..., failed: _Optional[int] = ..., storage_used_bytes: _Optional[int] = ..., storage_limit_bytes: _Optional[int] = ..., stopped_reason: _Optional[str] = ..., execution_time_ms: _Optional[int] = ...) -> None: ...

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("serving", "pipeline_ready", "queued", "busy")
    SERVING_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_READY_FIELD_NUMBER: _ClassVar[int]
    QUEUED_FIELD_NUMBER: _ClassVar[int]
    BUSY_FIELD_NUMBER: _ClassVar[int]
    serving: bool
    pipeline_ready: bool
    queued: int
    busy: bool
    def __init__(self, serving: _Optional[bool] = ..., pipeline_ready: _Optional[bool] = ..., queued: _Optional[int] = ..., busy: _Optional[bool] = ...) -> None: ...

class ListConversationsRequest(_message.Message):
    __slots__ = ("request_id", "user_id", "limit", "cursor")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    CURSOR_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    user_id: str
    limit: int
    cursor: str
    def __init__(self, request_id: _Optional[str] = ..., user_id: _Optional[str] = ..., limit: _Optional[int] = ..., cursor: _Optional[str] = ...) -> None: ...

class ConversationSummary(_message.Message):
    __slots__ = ("conversation_id", "conversation_name", "created_at", "last_activity_at", "exchange_count")
    CONVERSATION_ID_FIELD_NUMBER: _ClassVar[int]
    CONVERSATION_NAME_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    LAST_ACTIVITY_AT_FIELD_NUMBER: _ClassVar[int]
    EXCHANGE_COUNT_FIELD_NUMBER: _ClassVar[int]
    conversation_id: str
    conversation_name: str
    created_at: str
    last_activity_at: str
    exchange_count: int
    def __init__(self, conversation_id: _Optional[str] = ..., conversation_name: _Optional[str] = ..., created_at: _Optional[str] = ..., last_activity_at: _Optional[str] = ..., exchange_count: _Optional[int] = ...) -> None: ...

class ListConversationsResponse(_message.Message):
    __slots__ = ("request_id", "conversations", "next_cursor")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    CONVERSATIONS_FIELD_NUMBER: _ClassVar[int]
    NEXT_CURSOR_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    conversations: _containers.RepeatedCompositeFieldContainer[ConversationSummary]
    next_cursor: str
    def __init__(self, request_id: _Optional[str] = ..., conversations: _Optional[_Iterable[_Union[ConversationSummary, _Mapping]]] = ..., next_cursor: _Optional[str] = ...) -> None: ...

class ListExchangesRequest(_message.Message):
    __slots__ = ("request_id", "user_id", "conversation_id", "limit", "cursor")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    CONVERSATION_ID_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    CURSOR_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    user_id: str
    conversation_id: str
    limit: int
    cursor: str
    def __init__(self, request_id: _Optional[str] = ..., user_id: _Optional[str] = ..., conversation_id: _Optional[str] = ..., limit: _Optional[int] = ..., cursor: _Optional[str] = ...) -> None: ...

class ExchangeRecord(_message.Message):
    __slots__ = ("exchange_id", "query", "asked_at", "answer", "answered_at")
    EXCHANGE_ID_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    ASKED_AT_FIELD_NUMBER: _ClassVar[int]
    ANSWER_FIELD_NUMBER: _ClassVar[int]
    ANSWERED_AT_FIELD_NUMBER: _ClassVar[int]
    exchange_id: str
    query: str
    asked_at: str
    answer: str
    answered_at: str
    def __init__(self, exchange_id: _Optional[str] = ..., query: _Optional[str] = ..., asked_at: _Optional[str] = ..., answer: _Optional[str] = ..., answered_at: _Optional[str] = ...) -> None: ...

class ListExchangesResponse(_message.Message):
    __slots__ = ("request_id", "conversation_id", "conversation_name", "exchanges", "next_cursor")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    CONVERSATION_ID_FIELD_NUMBER: _ClassVar[int]
    CONVERSATION_NAME_FIELD_NUMBER: _ClassVar[int]
    EXCHANGES_FIELD_NUMBER: _ClassVar[int]
    NEXT_CURSOR_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    conversation_id: str
    conversation_name: str
    exchanges: _containers.RepeatedCompositeFieldContainer[ExchangeRecord]
    next_cursor: str
    def __init__(self, request_id: _Optional[str] = ..., conversation_id: _Optional[str] = ..., conversation_name: _Optional[str] = ..., exchanges: _Optional[_Iterable[_Union[ExchangeRecord, _Mapping]]] = ..., next_cursor: _Optional[str] = ...) -> None: ...
