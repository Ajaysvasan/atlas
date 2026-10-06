"""The sources knowledge may be acquired from, and nothing else."""

import ipaddress
import socket
from pathlib import Path
from typing import NamedTuple, Sequence, Tuple
from urllib.parse import urlparse

from config import get_logger
from knowledge_sufficiency.ksv_exceptions import UntrustedSource

logger = get_logger(__name__)


DOCS = "docs"
DATASET = "dataset"


class TrustedSource(NamedTuple):
    name: str
    host: str
    description: str
    search: str = ""
    kind: str = DOCS

    @property
    def searchable(self) -> bool:
        return bool(self.search)


# A starting set, meant to be curated rather than grown casually: every entry is
# a primary or standards source for the material this project answers over.
# Anything not listed here is refused, so adding one is a deliberate act.
SOURCES_FILE = Path(__file__).with_name("sources.toml")


def load_sources(path: Path | str = SOURCES_FILE) -> Tuple[TrustedSource, ...]:
    """Read the trusted sources from the versioned config beside this module."""
    import tomllib

    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    sources = tuple(
        TrustedSource(
            name=entry["name"],
            host=entry["host"],
            description=entry["description"],
            search=entry.get("search", ""),
            kind=entry.get("kind", DOCS),
        )
        for entry in data.get("source", ())
    )
    logger.debug("Loaded %d trusted source(s) from %s", len(sources), path)
    return sources


DEFAULT_SOURCES: Sequence[TrustedSource] = load_sources()


def sources_of_kind(
    kind: str, sources: Sequence[TrustedSource] = DEFAULT_SOURCES
) -> Tuple[TrustedSource, ...]:
    return tuple(source for source in sources if source.kind == kind)


ALLOWED_SCHEMES = frozenset({"https"})


def _is_private(host: str) -> bool:
    """Whether the host resolves to an address inside this network.

    Checked after resolution, not on the name: an allowlisted name whose DNS
    answer points at 127.0.0.1 or 169.254.169.254 would otherwise turn a fetch
    into a request against this machine or a cloud metadata service. It is not
    airtight — a name can resolve differently between this check and the
    request — but it closes the straightforward case.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
        ):
            return True
    return False


def host_is_listed(host: str, sources: Sequence[TrustedSource]) -> bool:
    """Exact host, or a subdomain of one.

    Compared label by label rather than by `endswith`, which would accept
    `docs.python.org.attacker.example` for `docs.python.org`.
    """
    host = host.lower().rstrip(".")
    for source in sources:
        listed = source.host.lower().rstrip(".")
        if host == listed or host.endswith("." + listed):
            return True
    return False


def check_trusted(
    url: str,
    sources: Sequence[TrustedSource] = DEFAULT_SOURCES,
    allow_private: bool = False,
) -> str:
    """Return the host, or raise `UntrustedSource` saying why not."""
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise UntrustedSource(url, f"scheme {parsed.scheme!r} is not https")
    if not parsed.hostname:
        raise UntrustedSource(url, "no host")
    if not host_is_listed(parsed.hostname, sources):
        raise UntrustedSource(url, f"{parsed.hostname} is not a trusted source")
    if not allow_private and _is_private(parsed.hostname):
        raise UntrustedSource(url, f"{parsed.hostname} resolves inside this network")
    return parsed.hostname


def is_trusted(
    url: str, sources: Sequence[TrustedSource] = DEFAULT_SOURCES
) -> bool:
    try:
        check_trusted(url, sources)
        return True
    except UntrustedSource:
        return False
