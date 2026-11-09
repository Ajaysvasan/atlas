"""Finding candidate documents by searching the trusted sources themselves."""

import re
from typing import Callable, List, Sequence
from urllib.parse import quote_plus, urljoin, urlparse

from config import get_logger
from knowledge_sufficiency.fetching import Document, fetch_document
from knowledge_sufficiency.ksv_exceptions import FetchFailed, UntrustedSource
from knowledge_sufficiency.robots import RobotsPolicy
from knowledge_sufficiency.sitemaps import SitemapIndex
from knowledge_sufficiency.sources import DEFAULT_SOURCES, TrustedSource, check_trusted
from knowledge_sufficiency.target import AcquisitionTarget

logger = get_logger(__name__)

MAX_LINKS_PER_SOURCE = 5

# Site chrome: the header, the menus and the footer link the same pages from
# every page, search results included, and come before the results in the HTML.
_CHROME = ("nav", "header", "footer")
# The block around a result link, whose text says what the result is about. A
# results page puts the title and snippet beside the link, not inside it.
_RESULT_BLOCKS = ("li", "tr", "dd", "dt", "p", "article")
_CONTEXT_CHARS = 400
_NOT_DOCUMENTS = ("/search", "/login", "/logout", "/signup", "/register", "/submit",
                  "/account", "/advanced")
_STOPWORDS = {
    "a", "an", "and", "are", "can", "does", "for", "from", "how", "into", "its", "the",
    "this", "that", "what", "when", "where", "which", "who", "why", "with", "use",
    "used", "using", "work", "works", "between", "difference", "explain",
}
_WORD = re.compile(r"[a-z0-9][a-z0-9+#]*")
_COMPOUND = re.compile(r"[a-z0-9]+(?:[_-][a-z0-9]+)+")


def search_url(source: TrustedSource, terms: str) -> str:
    return source.search.format(terms=quote_plus(terms))


def term_tokens(terms: str, source: TrustedSource) -> List[str]:
    """The words a relevant result must mention, without the source's own name.

    "sqlite" matches every page on sqlite.org, so it says nothing about which
    of them is relevant.
    """
    host = source.host.lower()
    tokens = []
    # pg_stat_statements is also searched as pgstatstatements: slugs and
    # titles write compounds both ways.
    compounds = ["".join(_WORD.findall(c)) for c in _COMPOUND.findall(terms.lower())]
    for word in _WORD.findall(terms.lower()) + compounds:
        if len(word) < 3 or word in _STOPWORDS or word in host or word in tokens:
            continue
        tokens.append(word)
    return tokens


def _normalised(text: str) -> str:
    return " ".join(_WORD.findall(text.lower().replace("_", " ").replace("-", " ")))


def _context(anchor) -> str:
    for parent in anchor.parents:
        if parent.name in _RESULT_BLOCKS:
            return parent.get_text(" ", strip=True)[:_CONTEXT_CHARS]
    return ""


def extract_links(
    html: str,
    base_url: str,
    source: TrustedSource,
    limit: int = MAX_LINKS_PER_SOURCE,
    terms: str | None = None,
) -> List[str]:
    """Result links from a search page, on the source's own host, best match first.

    Every link is taken and those that leave the host are discarded, which
    needs no knowledge of each site's markup; the host rule is what makes it
    safe. Navigation, header and footer links are dropped, as are links back
    into search, login or submission pages. With `terms`, a link is kept only
    if its text, its URL or the result block around it mentions at least one
    of them, and links are ranked by how many they mention: the first links on
    a results page are the site's menus, not its results.
    """
    from bs4 import BeautifulSoup

    tokens = term_tokens(terms, source) if terms is not None else None
    search_path = urlparse(base_url).path
    seen, scored = set(), []
    for position, anchor in enumerate(BeautifulSoup(html, "html.parser").find_all("a", href=True)):
        # Before de-duplication: a page the menu links to may also be a result,
        # and only its occurrence among the results says so.
        if any(parent.name in _CHROME for parent in anchor.parents):
            continue
        candidate = urljoin(base_url, anchor["href"].strip())
        candidate, _, _ = candidate.partition("#")
        if candidate in seen or candidate == base_url:
            continue
        seen.add(candidate)
        parsed = urlparse(candidate)
        if parsed.scheme != "https":
            continue
        if (parsed.hostname or "").lower().rstrip(".") != source.host.lower():
            continue
        path = parsed.path.lower()
        if path in ("", "/") or path == search_path.lower() or any(
            path.startswith(prefix) for prefix in _NOT_DOCUMENTS
        ):
            continue
        if tokens is None:
            scored.append((0, position, candidate))
            continue
        text = _normalised(" ".join((anchor.get_text(" ", strip=True), parsed.path, _context(anchor))))
        score = sum(1 for token in tokens if token in text)
        if score:
            scored.append((-score, position, candidate))
    scored.sort()
    return [candidate for _, _, candidate in scored[:limit]]


def discover(
    target: AcquisitionTarget,
    sources: Sequence[TrustedSource] = DEFAULT_SOURCES,
    fetch: Callable[[str], Document] = fetch_document,
    per_source: int = MAX_LINKS_PER_SOURCE,
    robots: RobotsPolicy | None = None,
    sitemaps: SitemapIndex | None = None,
) -> List[str]:
    """Candidate document URLs for this target, from the sources' own searches or sitemaps.

    A source is asked through its search when it has one, and through its
    published sitemap when it has no search or the search found nothing; a
    source with neither cannot be asked what it holds, and guessing paths on
    it would be a crawl. Nothing robots.txt disallows is fetched or offered.
    """
    robots = RobotsPolicy(fetch) if robots is None else robots
    sitemaps = SitemapIndex(fetch, robots) if sitemaps is None else sitemaps
    keywords = " ".join(part for part in (target.topic, *target.subtopics) if part and part.strip())
    candidates: List[str] = []
    for source in sources:
        links: List[str] = []
        if source.searchable:
            # The full terms first; a keyword search box that matches titles
            # finds nothing for a whole sentence, so the subject alone is tried
            # once more.
            for terms in dict.fromkeys((target.terms, keywords)):
                found = _search(source, terms, sources, fetch, robots, per_source)
                if found is None:
                    break
                if found:
                    links = found
                    break
        if not links and source.has_sitemap:
            links = sitemaps.candidates(source, term_tokens(target.terms, source), per_source, sources)
            logger.debug("%s's sitemap offered %d candidate(s)", source.name, len(links))
        candidates.extend(links)
    return candidates


def _search(source, terms, sources, fetch, robots, per_source) -> List[str] | None:
    """Relevant, robots-allowed candidates from one search; None when the source cannot be searched."""
    if not terms.strip():
        return []
    url = search_url(source, terms)
    try:
        check_trusted(url, sources)
        if not robots.allowed(url):
            logger.info("robots.txt of %s disallows its search page; skipping it", source.host)
            return None
        results = fetch(url)
    except (UntrustedSource, FetchFailed) as error:
        logger.warning("Could not search %s: %s", source.name, error)
        return None
    links = extract_links(results.text, results.url, source, per_source, terms)
    allowed = [link for link in links if robots.allowed(link)]
    logger.debug("%s offered %d relevant candidate(s) for %r, %d allowed by robots.txt",
                 source.name, len(links), terms, len(allowed))
    return allowed
