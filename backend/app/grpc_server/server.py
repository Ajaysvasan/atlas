"""Starting and stopping the gRPC server around the pipeline the CLI uses."""

import signal
import threading
from concurrent import futures
from pathlib import Path
from typing import Tuple

import grpc

from config import get_logger
from grpc_server.auth import ServiceTokenInterceptor
from grpc_server.gate import OperationGate
from grpc_server.generated import atlas_service_pb2_grpc as pb_grpc
from grpc_server.service import AtlasServicer
from grpc_server.settings import GrpcSettings

logger = get_logger(__name__)


def build_server(servicer: AtlasServicer, settings: GrpcSettings, port: int | None = None) -> Tuple[grpc.Server, int]:
    """A server with the service-token check, message limits and TLS when configured; not started."""
    options = [
        ("grpc.max_receive_message_length", settings.max_message_bytes),
        ("grpc.max_send_message_length", settings.max_message_bytes),
        ("grpc.keepalive_permit_without_calls", 1),
        ("grpc.http2.min_ping_interval_without_data_ms", 60_000),
    ]
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=settings.workers, thread_name_prefix="atlas-grpc"),
        interceptors=[ServiceTokenInterceptor(settings.service_token)],
        options=options,
        maximum_concurrent_rpcs=settings.workers,
    )
    pb_grpc.add_AtlasServiceServicer_to_server(servicer, server)
    address = f"{settings.bind}:{settings.port if port is None else port}"
    if settings.tls:
        credentials = grpc.ssl_server_credentials(
            [(Path(settings.tls_key).read_bytes(), Path(settings.tls_cert).read_bytes())],
            root_certificates=Path(settings.tls_client_ca).read_bytes() if settings.tls_client_ca else None,
            require_client_auth=bool(settings.tls_client_ca),
        )
        bound = server.add_secure_port(address, credentials)
    else:
        bound = server.add_insecure_port(address)
    if bound == 0:
        raise OSError(f"could not bind the gRPC server to {address}")
    return server, bound


def serve(settings: GrpcSettings | None = None, acquisition: bool = True) -> int:
    """Load the pipeline once, serve until SIGINT/SIGTERM, then drain and release everything."""
    from pipeline.bootstrap import Components
    from pipeline.uploads import UploadIngestion

    settings = GrpcSettings.from_environment() if settings is None else settings
    components = Components(acquisition=acquisition)
    uploads = UploadIngestion(refresh_retrieval=components.retriever.refresh)
    server = None
    try:
        pipeline = components.pipeline()
        for problem in components.warm_up():
            logger.warning("Starting without: %s", problem)
        servicer = AtlasServicer(pipeline, uploads, settings, OperationGate(settings.max_waiting))
        server, port = build_server(servicer, settings)
        stop = threading.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda number, frame: stop.set())
        server.start()
        logger.info("Atlas gRPC listening on %s:%d (%s)", settings.bind, port, "TLS" if settings.tls else "plaintext")
        print(f"Atlas gRPC server listening on {settings.bind}:{port}", flush=True)
        stop.wait()
        logger.info("Shutting down; in-flight requests get %.0f s", settings.shutdown_grace_seconds)
        server.stop(settings.shutdown_grace_seconds).wait()
        server = None
        return 0
    finally:
        if server is not None:
            server.stop(None)
        uploads.close()
        components.close()
