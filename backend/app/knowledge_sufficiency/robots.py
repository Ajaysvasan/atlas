"""robots.txt, read once per host and obeyed before discovery fetches anything."""

import re
import threading
import time
from typing import Callable, Dict, List, Tuple
from urllib.parse import unquote, urlparse

from config import get_logger
from knowledge_sufficiency.fetching import Document, fetch_document
from knowledge_sufficiency.ksv_exceptions import FetchFailed, UntrustedSource

logger = get_logger(__name__)

ROBOTS_TTL_SECONDS = 3600.0
AGENT = "*"
_ALLOW_ALL = "allow"
_DENY_ALL = "deny"


class RobotsPolicy:
    """What each host's robots.txt allows, following RFC 9309 when it cannot be read.

    A robots.txt that is missing or refused (4xx) allows everything; one that
    cannot be reached (5xx, network) is treated as disallowing everything until
    it can be read, because "unknown" must not mean "permitted".
    """

    def __init__(
        self,
        fetch: Callable[[str], Document] = fetch_document,
        ttl_seconds: float = ROBOTS_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.fetch = fetch
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self.__rules: Dict[str, Tuple[float, object]] = {}
        self.__lock = threading.Lock()

    def __rules_for(self, host: str):
        with self.__lock:
            cached = self.__rules.get(host)
            if cached is not None and self.clock() - cached[0] < self.ttl_seconds:
                return cached[1]
        url = f"https://{host}/robots.txt"
        try:
            rules = RobotsRules.parse(self.fetch(url).text)
        except FetchFailed as error:
            rules = _ALLOW_ALL if str(error.reason).startswith("HTTP 4") else _DENY_ALL
            logger.info("robots.txt of %s unreadable (%s): %s", host, error.reason,
                        "allowing" if rules is _ALLOW_ALL else "refusing")
        except UntrustedSource:
            rules = _DENY_ALL
        with self.__lock:
            self.__rules[host] = (self.clock(), rules)
        return rules

    def allowed(self, url: str) -> bool:
        host = (urlparse(url).hostname or "").lower()
        if not host:
            return False
        rules = self.__rules_for(host)
        if rules is _ALLOW_ALL:
            return True
        if rules is _DENY_ALL:
            return False
        return rules.allowed(url)


class RobotsRules:
    """The `User-agent: *` group of a robots.txt, evaluated as RFC 9309 says.

    urllib.robotparser applies the first rule that matches, so
    cassandra.apache.org's "Disallow: /doc" followed by "Allow: /doc/latest"
    refuses the very pages the second line allows. RFC 9309 (and every major
    crawler) takes the longest matching rule, and allow wins a tie.
    """

    def __init__(self, rules: List[Tuple[bool, str]]) -> None:
        self.rules = rules

    @staticmethod
    def parse(text: str) -> "RobotsRules":
        rules: List[Tuple[bool, str]] = []
        agents: List[str] = []
        in_rules = False
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            field, value = (part.strip() for part in line.split(":", 1))
            field = field.lower()
            if field == "user-agent":
                if in_rules:
                    agents, in_rules = [], False
                agents.append(value.lower())
            elif field in ("allow", "disallow"):
                in_rules = True
                if AGENT in agents and value:
                    rules.append((field == "allow", value))
        return RobotsRules(rules)

    def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        path = (parsed.path or "/") + (f"?{parsed.query}" if parsed.query else "")
        best: Tuple[int, bool] | None = None
        for allow, pattern in self.rules:
            if _matches(pattern, path):
                key = (len(pattern), allow)
                if best is None or key > best:
                    best = key
        return True if best is None else best[1]


def _matches(pattern: str, path: str) -> bool:
    regex = "".join(".*" if c == "*" else re.escape(c) for c in pattern.rstrip("$"))
    return re.match(regex + ("$" if pattern.endswith("$") else ""), unquote(path)) is not None
