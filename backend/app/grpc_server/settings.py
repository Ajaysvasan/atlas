"""The gRPC server's settings, from the environment (and .env); nothing secret has a default."""

import os
from dataclasses import dataclass, field
from typing import FrozenSet

MIB = 1024 * 1024
ALLOWED_EXTENSIONS = frozenset({"pdf", "docx", "pptx", "xlsx", "txt", "md", "html", "htm", "json", "csv"})
MIN_TOKEN_LENGTH = 16


class SettingsError(ValueError):
    pass


def _int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise SettingsError(f"{name} must be an integer") from error
    if value < minimum:
        raise SettingsError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True)
class GrpcSettings:
    service_token: str
    bind: str = "127.0.0.1"
    port: int = 50051
    tls_cert: str | None = None
    tls_key: str | None = None
    tls_client_ca: str | None = None
    workers: int = 8
    max_waiting: int = 4
    max_message_bytes: int = 4 * MIB
    max_files: int = 10
    max_file_bytes: int = 25 * MIB
    max_request_bytes: int = 100 * MIB
    max_query_chars: int = 8000
    max_uncompressed_bytes: int = 200 * MIB
    max_archive_entries: int = 10_000
    max_compression_ratio: int = 100
    shutdown_grace_seconds: float = 30.0
    allowed_extensions: FrozenSet[str] = field(default=ALLOWED_EXTENSIONS)

    def __post_init__(self) -> None:
        if not self.service_token or len(self.service_token) < MIN_TOKEN_LENGTH:
            raise SettingsError(
                f"ATLAS_GRPC_SERVICE_TOKEN must be set to at least {MIN_TOKEN_LENGTH} characters "
                "(e.g. `openssl rand -hex 32`), the same value the Java service uses")
        if bool(self.tls_cert) != bool(self.tls_key):
            raise SettingsError("ATLAS_GRPC_TLS_CERT and ATLAS_GRPC_TLS_KEY go together")
        if self.tls_client_ca and not self.tls_cert:
            raise SettingsError("ATLAS_GRPC_TLS_CLIENT_CA (mTLS) needs ATLAS_GRPC_TLS_CERT and _KEY")
        if self.bind not in ("127.0.0.1", "::1", "localhost") and not self.tls_cert:
            raise SettingsError(
                f"Binding to {self.bind} exposes the pipeline beyond this machine; set "
                "ATLAS_GRPC_TLS_CERT/_KEY (and ATLAS_GRPC_TLS_CLIENT_CA for mTLS) first")

    @property
    def tls(self) -> bool:
        return bool(self.tls_cert)

    @classmethod
    def from_environment(cls) -> "GrpcSettings":
        from dotenv import load_dotenv

        load_dotenv()
        return cls(
            service_token=os.environ.get("ATLAS_GRPC_SERVICE_TOKEN", ""),
            bind=os.environ.get("ATLAS_GRPC_BIND", "127.0.0.1"),
            port=_int("ATLAS_GRPC_PORT", 50051),
            tls_cert=os.environ.get("ATLAS_GRPC_TLS_CERT") or None,
            tls_key=os.environ.get("ATLAS_GRPC_TLS_KEY") or None,
            tls_client_ca=os.environ.get("ATLAS_GRPC_TLS_CLIENT_CA") or None,
            workers=_int("ATLAS_GRPC_WORKERS", 8),
            max_waiting=_int("ATLAS_GRPC_MAX_WAITING", 4, minimum=0),
            max_files=_int("ATLAS_UPLOAD_MAX_FILES", 10),
            max_file_bytes=_int("ATLAS_UPLOAD_MAX_FILE_BYTES", 25 * MIB),
            max_request_bytes=_int("ATLAS_UPLOAD_MAX_REQUEST_BYTES", 100 * MIB),
            max_query_chars=_int("ATLAS_QUERY_MAX_CHARS", 8000),
        )
