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
            return Document(url=url, text='<a href="/wal.html">Write-ahead logging</a>',
                            content_type="text/html")

        discover(target(), [PYTHON], fetch)
        searches = [url for url in asked if not url.endswith("/robots.txt")]
        assert len(searches) == 1
        assert "docs.python.org" in searches[0]

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
                url=url, text='<a href="/wal.html">WAL</a>', content_type="text/html"
            )

        other = TrustedSource(
            "sqlite", "sqlite.org", "docs", "https://sqlite.org/search?q={terms}"
        )
        found = discover(target(), [PYTHON, other], fetch)
        assert found == ["https://sqlite.org/wal.html"]

    def test_candidates_from_several_sources_are_combined(self):
        other = TrustedSource(
            "sqlite", "sqlite.org", "docs", "https://sqlite.org/search?q={terms}"
        )

        def fetch(url):
            host = "docs.python.org" if "python" in url else "sqlite.org"
            return Document(
                url=url, text=f'<a href="https://{host}/wal">write-ahead logging</a>',
                content_type="text/html",
            )

        assert len(discover(target(), [PYTHON, other], fetch)) == 2

    def test_no_sources_yields_no_candidates(self):
        assert discover(target(), [], lambda u: None) == []


SQLITE = TrustedSource("sqlite", "sqlite.org", "docs", "https://sqlite.org/search?q={terms}")
RESULTS = """
<header><a href="/wal.html">WAL in the menu</a></header>
<nav><a href="/about.html">About</a> <a href="/lang_vacuum.html">VACUUM</a></nav>
<div><a href="/index.html">Home</a> <a href="/download.html">Download</a></div>
<a href="/search?q=next">Next page</a> <a href="/login">Log in</a> <a href="/">Root</a>
<ul>
  <li><a href="/r1.html">Result one</a> How SQLite uses write-ahead logging for crash recovery</li>
  <li><a href="/r2.html">Result two</a> Unrelated notes on the release history</li>
  <li><a href="/wal.html">The WAL file</a> write-ahead logging and checkpoints</li>
</ul>
<footer><a href="/copyright.html">logging policy</a></footer>
"""


class TestRankingResults:
    def links(self, html=RESULTS, terms="SQLite write-ahead logging crash recovery"):
        return extract_links(html, "https://sqlite.org/search?q=x", SQLITE, 5, terms)

    def test_menus_headers_and_footers_never_count_as_results(self):
        links = self.links()
        assert "https://sqlite.org/about.html" not in links
        assert "https://sqlite.org/copyright.html" not in links

    def test_search_login_and_root_links_are_not_documents(self):
        links = self.links()
        assert not any(part in link for link in links for part in ("/search", "/login"))
        assert "https://sqlite.org/" not in links

    def test_only_links_about_the_terms_are_kept_best_first(self):
        assert self.links() == ["https://sqlite.org/r1.html", "https://sqlite.org/wal.html"]

    def test_the_text_beside_a_link_counts_not_just_the_anchor(self):
        assert "https://sqlite.org/r1.html" in self.links(terms="crash recovery")

    def test_the_sources_own_name_says_nothing_about_relevance(self):
        html = '<ul><li><a href="/x.html">SQLite home</a> all about SQLite</li></ul>'
        assert self.links(html, "SQLite") == []

    def test_without_terms_every_non_menu_link_is_kept(self):
        links = extract_links(RESULTS, "https://sqlite.org/search?q=x", SQLITE, 10)
        assert "https://sqlite.org/r2.html" in links
        assert "https://sqlite.org/about.html" not in links


def robots_fetch(robots, results=RESULTS, asked=None):
    def fetch(url):
        if asked is not None:
            asked.append(url)
        if url.endswith("/robots.txt"):
            if isinstance(robots, Exception):
                raise robots
            return Document(url=url, text=robots, content_type="text/plain")
        return Document(url=url, text=results, content_type="text/html")
    return fetch


class TestRobots:
    def test_a_disallowed_search_page_is_never_fetched(self):
        asked = []
        found = discover(target(), [SQLITE], robots_fetch("User-agent: *\nDisallow: /search", asked=asked))
        assert found == []
        assert asked == ["https://sqlite.org/robots.txt"]

    def test_disallowed_candidates_are_dropped(self):
        found = discover(target(), [SQLITE], robots_fetch("User-agent: *\nDisallow: /wal.html"))
        assert "https://sqlite.org/wal.html" not in found
        assert "https://sqlite.org/r1.html" in found

    def test_a_missing_robots_file_allows_everything(self):
        found = discover(target(), [SQLITE], robots_fetch(FetchFailed("u", "HTTP 404")))
        assert found

    def test_an_unreachable_robots_file_allows_nothing(self):
        asked = []
        found = discover(target(), [SQLITE], robots_fetch(FetchFailed("u", "HTTP 503"), asked=asked))
        assert found == [] and len(asked) == 1

    def test_robots_rules_are_cached_per_host(self):
        from knowledge_sufficiency.robots import RobotsPolicy

        asked = []
        policy = RobotsPolicy(robots_fetch("User-agent: *\nAllow: /", asked=asked))
        for path in ("/a", "/b", "/c"):
            assert policy.allowed(f"https://sqlite.org{path}")
        assert asked == ["https://sqlite.org/robots.txt"]


class TestKeywordRetry:
    def test_a_sentence_that_finds_nothing_is_retried_with_the_subject(self):
        asked = []

        def fetch(url):
            asked.append(url)
            if url.endswith("/robots.txt"):
                return Document(url=url, text="", content_type="text/plain")
            hit = "how" not in url.lower()
            return Document(url=url, content_type="text/html",
                            text='<li><a href="/wal.html">write-ahead logging</a></li>' if hit else "")

        found = discover(target(), [SQLITE], fetch)
        assert found == ["https://sqlite.org/wal.html"]
        searches = [u for u in asked if "/search" in u]
        assert len(searches) == 2 and "how" not in searches[1].lower()

    def test_no_retry_once_the_first_search_found_something(self):
        asked = []
        discover(target(), [SQLITE], robots_fetch("", asked=asked))
        assert len([u for u in asked if "/search" in u]) == 1


class TestRobotsFollowsRfc9309:
    def rules(self, text):
        from knowledge_sufficiency.robots import RobotsRules

        return RobotsRules.parse(text)

    def test_the_longest_match_wins_not_the_first(self):
        rules = self.rules("User-agent: *\nDisallow: /doc\nAllow: /doc/latest\n")
        assert rules.allowed("https://x.org/doc/latest/repair.html")
        assert not rules.allowed("https://x.org/doc/4.0/repair.html")

    def test_allow_wins_a_tie(self):
        assert self.rules("User-agent: *\nDisallow: /a\nAllow: /a\n").allowed("https://x.org/a")

    def test_wildcards_and_end_anchors(self):
        rules = self.rules("User-agent: *\nDisallow: /*.pdf$\nDisallow: /private*/x\n")
        assert not rules.allowed("https://x.org/files/a.pdf")
        assert rules.allowed("https://x.org/files/a.pdf.html")
        assert not rules.allowed("https://x.org/private-area/x")

    def test_only_the_star_group_applies(self):
        text = "User-agent: Googlebot\nDisallow: /\n\nUser-agent: *\nDisallow: /search\n"
        rules = self.rules(text)
        assert rules.allowed("https://x.org/docs") and not rules.allowed("https://x.org/search?q=1")

    def test_an_empty_disallow_allows_everything(self):
        assert self.rules("User-agent: *\nDisallow:\n").allowed("https://x.org/anything")

    def test_grouped_user_agents_share_their_rules(self):
        rules = self.rules("User-agent: bot\nUser-agent: *\nDisallow: /x\n")
        assert not rules.allowed("https://x.org/x")
