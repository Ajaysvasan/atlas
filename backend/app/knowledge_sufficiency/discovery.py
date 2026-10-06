"""Finding candidate documents by searching the trusted sources themselves."""

from typing import Callable, List, Sequence
from urllib.parse import quote_plus, urljoin, urlparse

from config import get_logger
from knowledge_sufficiency.fetching import Document, fetch_document
from knowledge_sufficiency.ksv_exceptions import FetchFailed, UntrustedSource
from knowledge_sufficiency.sources import DEFAULT_SOURCES, TrustedSource, check_trusted
from knowledge_sufficiency.target import AcquisitionTarget

logger = get_logger(__name__)

MAX_LINKS_PER_SOURCE = 5


def search_url(source: TrustedSource, terms: str) -> str:
    return source.search.format(terms=quote_plus(terms))


def extract_links(
    html: str,
    base_url: str,
    source: TrustedSource,
    limit: int = MAX_LINKS_PER_SOURCE,
) -> List[str]:
    """Links from a search results page that stay on the source's own host.

    Deliberately generic rather than a parser per source: taking every link and
    then discarding those that leave the host needs no knowledge of each site's
    markup, and the host rule is what makes it safe. A results page that links
    outward simply yields fewer candidates.
    """
    from bs4 import BeautifulSoup

    seen, found = set(), []
    for anchor in BeautifulSoup(html, "html.parser").find_all("a", href=True):
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
        found.append(candidate)
        if len(found) >= limit:
            break
    return found


def discover(
    target: AcquisitionTarget,
    sources: Sequence[TrustedSource] = DEFAULT_SOURCES,
    fetch: Callable[[str], Document] = fetch_document,
    per_source: int = MAX_LINKS_PER_SOURCE,
) -> List[str]:
    """Candidate document URLs for this target, from the sources' own searches.

    Only sources that expose a search are used; the rest have no way to be asked
    what they hold, and guessing paths on them would be a crawl.
    """
    terms = target.terms
    candidates: List[str] = []
    for source in sources:
        if not source.searchable:
            continue
        url = search_url(source, terms)
        try:
            check_trusted(url, sources)
            results = fetch(url)
        except (UntrustedSource, FetchFailed) as error:
            logger.warning("Could not search %s: %s", source.name, error)
            continue
        links = extract_links(results.text, results.url, source, per_source)
        logger.debug("%s offered %d candidate(s)", source.name, len(links))
        candidates.extend(links)
    return candidates
