"""Candidate pages from a source's published sitemap, ranked by the target's terms."""

import math
import re
import threading
import time
from typing import Callable, Dict, List, Sequence, Tuple
from urllib.parse import unquote, urlparse

from config import get_logger
from knowledge_sufficiency.fetching import Document, fetch_document
from knowledge_sufficiency.ksv_exceptions import FetchFailed, UntrustedSource
from knowledge_sufficiency.robots import RobotsPolicy
from knowledge_sufficiency.sources import TrustedSource, check_trusted

logger = get_logger(__name__)

SITEMAP_TTL_SECONDS = 24 * 3600.0
# postgresql.org's is 4.6 MB; larger than any page, so it has its own cap.
SITEMAP_MAX_BYTES = 8 * 1024 * 1024
MAX_CHILD_SITEMAPS = 10
MAX_URLS = 100_000

# A pattern, not an XML parser: a sitemap needs only its <loc> values, and a
# pattern cannot be made to expand entities or fetch an external DTD.
_LOC = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>", re.IGNORECASE)
_INDEX = re.compile(r"<sitemapindex\b", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9][a-z0-9+#]*")


def _unescape(url: str) -> str:
    return url.replace("&amp;", "&").replace("&apos;", "'").replace("&quot;", '"')


def path_words(url: str) -> List[str]:
    """The words of a URL's path: `/docs/current/brin-intro.html` -> docs, current, brin, intro, html."""
    path = unquote(urlparse(url).path).lower().replace("_", " ").replace("-", " ")
    return _WORD.findall(path)


class SitemapIndex:
    """Each source's sitemap URLs, read once a day, obeying robots.txt and the source's scope."""

    def __init__(
        self,
        fetch: Callable[..., Document] = fetch_document,
        robots: RobotsPolicy | None = None,
        ttl_seconds: float = SITEMAP_TTL_SECONDS,
        max_bytes: int = SITEMAP_MAX_BYTES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.fetch = fetch
        self.robots = RobotsPolicy(fetch) if robots is None else robots
        self.ttl_seconds = ttl_seconds
        self.max_bytes = max_bytes
        self.clock = clock
        self.__urls: Dict[str, Tuple[float, Tuple[str, ...]]] = {}
        self.__lock = threading.Lock()

    def __read(self, url: str, sources: Sequence[TrustedSource]) -> str | None:
        try:
            check_trusted(url, sources)
            if not self.robots.allowed(url):
                logger.info("robots.txt disallows the sitemap %s", url)
                return None
            return self.fetch(url, max_bytes=self.max_bytes).text
        except (UntrustedSource, FetchFailed) as error:
            logger.warning("Could not read sitemap %s: %s", url, error)
            return None

    def urls(self, source: TrustedSource, sources: Sequence[TrustedSource] = ()) -> Tuple[str, ...]:
        """In-scope page URLs on the source's own host; empty when the sitemap cannot be read."""
        if not source.has_sitemap:
            return ()
        sources = tuple(sources) or (source,)
        with self.__lock:
            cached = self.__urls.get(source.name)
            if cached is not None and self.clock() - cached[0] < self.ttl_seconds:
                return cached[1]
        text = self.__read(source.sitemap, sources)
        found: List[str] = []
        if text is not None:
            locations = [_unescape(loc) for loc in _LOC.findall(text)]
            if _INDEX.search(text):
                children = [loc for loc in locations if self.__same_host(loc, source)
                            and not loc.lower().endswith(".gz")][:MAX_CHILD_SITEMAPS]
                locations = []
                for child in children:
                    child_text = self.__read(child, sources)
                    if child_text is not None:
                        locations.extend(_unescape(loc) for loc in _LOC.findall(child_text))
            scope = source.sitemap_scope
            for loc in locations:
                loc = loc.partition("#")[0]
                if self.__same_host(loc, source) and (not scope or loc.startswith(scope)):
                    found.append(loc)
                if len(found) >= MAX_URLS:
                    break
        urls = tuple(dict.fromkeys(found))
        with self.__lock:
            # An unreadable sitemap is cached too, so a down host is not
            # re-requested on every acquisition; it is retried after the TTL.
            self.__urls[source.name] = (self.clock(), urls)
        logger.info("Sitemap of %s: %d in-scope URL(s)", source.name, len(urls))
        return urls

    @staticmethod
    def __same_host(url: str, source: TrustedSource) -> bool:
        parsed = urlparse(url)
        return parsed.scheme == "https" and (parsed.hostname or "").lower() == source.host.lower()

    def candidates(
        self,
        source: TrustedSource,
        tokens: Sequence[str],
        limit: int,
        sources: Sequence[TrustedSource] = (),
    ) -> List[str]:
        """The sitemap URLs whose paths best match `tokens`, robots-allowed, best first.

        Each matched word counts by its rarity across the sitemap: "index" is
        in a hundred paths of a database manual and says little, "brin" is in
        three and says which page is wanted.
        """
        urls = self.urls(source, sources)
        if not urls or not tokens:
            return []
        words = [set(path_words(url)) for url in urls]
        frequency = {token: sum(1 for w in words if _matches(token, w)) for token in tokens}
        longest_first = sorted(tokens, key=len, reverse=True)
        scored = []
        for url, w in zip(urls, words):
            # A token inside a longer one already matched on this URL adds
            # nothing: "stat" and "statements" must not outvote
            # "pgstatstatements" on the page that is exactly that.
            matched: List[str] = []
            score = 0.0
            for token in longest_first:
                if any(token in longer for longer in matched) or not _matches(token, w):
                    continue
                matched.append(token)
                score += math.log((1 + len(urls)) / (1 + frequency[token]))
            if score > 0:
                scored.append((-score, len(url), url))
        scored.sort()
        chosen = []
        for _, _, url in scored:
            if self.robots.allowed(url):
                chosen.append(url)
            if len(chosen) >= limit:
                break
        return chosen


def _matches(token: str, words: set) -> bool:
    """A token matches a path word it begins: "index" matches "indexes", "replicat" would match both forms."""
    return any(word.startswith(token) or token.startswith(word) and len(word) >= 4 for word in words)
