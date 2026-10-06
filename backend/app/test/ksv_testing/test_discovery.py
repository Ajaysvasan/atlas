"""Finding candidates by searching the trusted sources themselves.

No general web search is involved: every candidate comes from inside a source
that is already on the allowlist, and the same-host rule on extracted links is
what keeps it that way.
"""

import numpy as np
import pytest

from knowledge_sufficiency.discovery import discover, extract_links, search_url
from knowledge_sufficiency.fetching import Document
from knowledge_sufficiency.ksv_exceptions import FetchFailed
from knowledge_sufficiency.sources import TrustedSource
from knowledge_sufficiency.target import AcquisitionTarget

DIMENSIONS = 128
PYTHON = TrustedSource(
    "python", "docs.python.org", "docs",
    "https://docs.python.org/3/search.html?q={terms}",
)
NO_SEARCH = TrustedSource("w3c", "www.w3.org", "standards")


def vector():
    return np.ones(DIMENSIONS, dtype=np.float32)


def target(topic="databases", subtopics=("write-ahead logging",), query="how does WAL work"):
    return AcquisitionTarget(
        topic=topic, query=query, topic_vector=vector(), query_vector=vector(),
        subtopics=tuple(subtopics), subtopic_vectors=tuple(vector() for _ in subtopics),
    )


def serving(html, url="https://docs.python.org/3/search.html?q=x"):
    return lambda _: Document(url=url, text=html, content_type="text/html")


class TestTheSearchUrl:
    def test_the_terms_are_url_encoded(self):
        built = search_url(PYTHON, "how does WAL work")
        assert " " not in built
        assert "docs.python.org" in built

    def test_special_characters_cannot_break_out_of_the_query(self):
        built = search_url(PYTHON, "a&b=c #frag")
        assert built.count("?") == 1
        assert "#" not in built

    def test_the_terms_carry_topic_subtopics_and_query(self):
        built = search_url(PYTHON, target().terms)
        for part in ("databases", "logging", "WAL"):
            assert part.replace(" ", "+") in built or part in built


class TestExtractingLinks:
    def test_links_on_the_source_are_kept(self):
        html = '<a href="/3/library/sqlite3.html">sqlite3</a>'
        links = extract_links(html, "https://docs.python.org/3/search.html", PYTHON)
        assert links == ["https://docs.python.org/3/library/sqlite3.html"]

    def test_links_that_leave_the_source_are_dropped(self):
        """A results page linking outward must not pull us off the allowlist."""
        html = '<a href="https://evil.example/x">elsewhere</a>'
        assert extract_links(html, "https://docs.python.org/3/search.html", PYTHON) == []

    def test_a_lookalike_host_is_dropped(self):
        html = '<a href="https://docs.python.org.evil.example/x">lookalike</a>'
        assert extract_links(html, "https://docs.python.org/3/search.html", PYTHON) == []

    def test_plain_http_links_are_dropped(self):
        html = '<a href="http://docs.python.org/3/x.html">insecure</a>'
        assert extract_links(html, "https://docs.python.org/3/search.html", PYTHON) == []

    def test_duplicates_are_collapsed(self):
        html = '<a href="/a">1</a><a href="/a">2</a><a href="/a#frag">3</a>'
        links = extract_links(html, "https://docs.python.org/s", PYTHON)
        assert len(links) == 1

    def test_fragments_are_stripped(self):
        html = '<a href="/a#section">x</a>'
        links = extract_links(html, "https://docs.python.org/s", PYTHON)
        assert links == ["https://docs.python.org/a"]

    def test_the_search_page_does_not_link_to_itself(self):
        base = "https://docs.python.org/3/search.html"
        html = f'<a href="{base}">this page</a><a href="/other">other</a>'
        assert extract_links(html, base, PYTHON) == ["https://docs.python.org/other"]

    def test_the_number_of_links_is_capped(self):
        html = "".join(f'<a href="/p{i}">{i}</a>' for i in range(50))
        assert len(extract_links(html, "https://docs.python.org/s", PYTHON, limit=3)) == 3

    def test_a_page_with_no_links_yields_nothing(self):
        assert extract_links("<p>nothing here</p>", "https://docs.python.org/s", PYTHON) == []


class TestDiscovery:
    def test_it_searches_each_searchable_source(self):
        asked = []

        def fetch(url):
            asked.append(url)
            return Document(url=url, text='<a href="/doc">d</a>', content_type="text/html")

        discover(target(), [PYTHON], fetch)
        assert len(asked) == 1
        assert "docs.python.org" in asked[0]

    def test_a_source_without_a_search_is_skipped(self):
        """There is no way to ask it what it holds, and guessing paths is a crawl."""
        asked = []

        def fetch(url):
            asked.append(url)
            return Document(url=url, text="", content_type="text/html")

        discover(target(), [NO_SEARCH], fetch)
        assert asked == []

    def test_a_failing_search_does_not_stop_the_others(self):
        def fetch(url):
            if "python" in url:
                raise FetchFailed(url, "HTTP 503")
            return Document(
                url=url, text='<a href="/doc">d</a>', content_type="text/html"
            )

        other = TrustedSource(
            "sqlite", "sqlite.org", "docs", "https://sqlite.org/search?q={terms}"
        )
        found = discover(target(), [PYTHON, other], fetch)
        assert found == ["https://sqlite.org/doc"]

    def test_candidates_from_several_sources_are_combined(self):
        other = TrustedSource(
            "sqlite", "sqlite.org", "docs", "https://sqlite.org/search?q={terms}"
        )

        def fetch(url):
            host = "docs.python.org" if "python" in url else "sqlite.org"
            return Document(
                url=url, text=f'<a href="https://{host}/doc">d</a>',
                content_type="text/html",
            )

        assert len(discover(target(), [PYTHON, other], fetch)) == 2

    def test_no_sources_yields_no_candidates(self):
        assert discover(target(), [], lambda u: None) == []
