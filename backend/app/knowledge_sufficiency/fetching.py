"""Fetching a document from a trusted source, and refusing anything else."""

from typing import NamedTuple, Sequence

from config import get_logger
from knowledge_sufficiency.ksv_exceptions import FetchFailed, UntrustedSource
from knowledge_sufficiency.sources import DEFAULT_SOURCES, TrustedSource, check_trusted

logger = get_logger(__name__)

MAX_BYTES = 2 * 1024 * 1024
TIMEOUT_SECONDS = 15.0
MAX_REDIRECTS = 5

ALLOWED_CONTENT_TYPES = (
    "text/plain",
    "text/html",
    "text/markdown",
    "application/json",
    "application/xhtml+xml",
)


class Document(NamedTuple):
    url: str
    text: str
    content_type: str


def _content_type_allowed(header: str) -> bool:
    kind = header.split(";")[0].strip().lower()
    return kind in ALLOWED_CONTENT_TYPES


def fetch_document(
    url: str,
    sources: Sequence[TrustedSource] = DEFAULT_SOURCES,
    max_bytes: int = MAX_BYTES,
    timeout: float = TIMEOUT_SECONDS,
) -> Document:
    """Fetch `url`, or raise. Only https, only a trusted host, only so many bytes."""
    import httpx

    check_trusted(url, sources)

    try:
        # follow_redirects stays off: httpx would follow them itself and the
        # destination would never pass through check_trusted. A trusted host
        # redirecting somewhere untrusted is the ordinary way an allowlist
        # leaks, so each hop is validated here instead.
        with httpx.Client(follow_redirects=False, timeout=timeout) as client:
            current = url
            for _ in range(MAX_REDIRECTS + 1):
                response = client.get(current)
                if not response.is_redirect:
                    break
                location = response.headers.get("location")
                if not location:
                    raise FetchFailed(current, "redirect without a location")
                current = str(response.next_request.url)
                check_trusted(current, sources)
                logger.debug("Following redirect to %s", current)
            else:
                raise FetchFailed(url, f"more than {MAX_REDIRECTS} redirects")

            if response.status_code >= 400:
                raise FetchFailed(current, f"HTTP {response.status_code}")

            content_type = response.headers.get("content-type", "")
            if not _content_type_allowed(content_type):
                raise FetchFailed(current, f"content type {content_type!r} is not text")

            body = response.content[:max_bytes]
            if len(response.content) > max_bytes:
                logger.warning(
                    "Truncated %s at %d bytes", current, max_bytes
                )
    except (UntrustedSource, FetchFailed):
        raise
    except Exception as error:
        raise FetchFailed(url, f"{type(error).__name__}: {error}") from error

    text = body.decode(response.encoding or "utf-8", errors="replace")
    logger.info("Fetched %d character(s) from %s", len(text), current)
    return Document(url=current, text=text, content_type=content_type)
