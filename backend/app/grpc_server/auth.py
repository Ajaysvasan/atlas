"""Every call must prove it comes from the Java service: `authorization: Bearer <service token>`."""

import hmac

import grpc

from config import get_logger

logger = get_logger(__name__)

ERROR_CODE_KEY = "atlas-error-code"


def _deny(code: grpc.StatusCode, details: str):
    def abort(request_or_iterator, context):
        context.set_trailing_metadata(((ERROR_CODE_KEY, "UNAUTHENTICATED"),))
        context.abort(code, details)

    return abort


class ServiceTokenInterceptor(grpc.ServerInterceptor):
    """Refuses any call without the shared token, compared in constant time.

    The token is what lets the server trust the user_id inside a request: only
    the Java service holds it, and Java takes user_id from a verified JWT.
    """

    def __init__(self, token: str) -> None:
        self.__expected = f"Bearer {token}".encode()

    def intercept_service(self, continuation, handler_call_details):
        presented = ""
        for key, value in handler_call_details.invocation_metadata or ():
            if key == "authorization":
                presented = value
                break
        if hmac.compare_digest(presented.encode(), self.__expected):
            return continuation(handler_call_details)
        logger.warning("Refused %s: missing or wrong service token", handler_call_details.method)
        handler = continuation(handler_call_details)
        deny = _deny(grpc.StatusCode.UNAUTHENTICATED, "service authentication required")
        if handler is None:
            return None
        if handler.request_streaming and handler.response_streaming:
            return grpc.stream_stream_rpc_method_handler(deny)
        if handler.request_streaming:
            return grpc.stream_unary_rpc_method_handler(deny)
        if handler.response_streaming:
            return grpc.unary_stream_rpc_method_handler(deny)
        return grpc.unary_unary_rpc_method_handler(deny)
