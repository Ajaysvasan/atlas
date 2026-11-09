"""Discovery through a source's published sitemap: what is read, what is offered, in what order."""

import numpy as np
import pytest

from knowledge_sufficiency.discovery import discover, term_tokens
from knowledge_sufficiency.fetching import Document
from knowledge_sufficiency.ksv_exceptions import FetchFailed
from knowledge_sufficiency.robots import RobotsPolicy
from knowledge_sufficiency.sitemaps import SITEMAP_MAX_BYTES, SitemapIndex, path_words
from knowledge_sufficiency.sources import TrustedSource
from knowledge_sufficiency.target import AcquisitionTarget

BASE = "https://www.postgresql.org"
PG = TrustedSource("postgresql", "www.postgresql.org", "PostgreSQL manual",
                   sitemap=f"{BASE}/sitemap.xml", sitemap_scope=f"{BASE}/docs/current/")
PAGES = ["indexes-intro", "indexes-types", "indexes-multicolumn", "indexes-unique",
         "indexes-partial", "brin", "brin-intro", "gin", "wal-intro", "tutorial-arch"]
URLSET = ("<?xml version='1.0'?><urlset>"
          + "".join(f"<url><loc>{BASE}/docs/current/{p}.html</loc></url>" for p in PAGES)
          + f"<url><loc>{BASE}/docs/9.3/brin.html</loc></url>"
          + "<url><loc>https://elsewhere.example/docs/current/brin.html</loc></url>"
          + f"<url><loc>{BASE}/docs/current/q.html?a=1&amp;b=2</loc></url>"
          + "</urlset>")


class Web:
    def __init__(self, pages, robots="") -> None:
        self.pages = pages
        self.robots = robots
        self.asked = []

    def __call__(self, url, max_bytes=None):
        self.asked.append((url, max_bytes))
        if url.endswith("/robots.txt"):
            return Document(url, self.robots, "text/plain")
        if url not in self.pages:
            raise FetchFailed(url, "HTTP 404")
        return Document(url, self.pages[url], "text/xml")


def index(web, **options):
    return SitemapIndex(web, RobotsPolicy(web), **options)


def vector():
    return np.ones(4, dtype=np.float32)


def target(query, topic="PostgreSQL", subtopics=()):
    return AcquisitionTarget(topic, query, vector(), vector(), tuple(subtopics),
                             tuple(vector() for _ in subtopics))


class TestReading:
    def test_only_in_scope_urls_on_the_sources_host(self):
        urls = index(Web({PG.sitemap: URLSET})).urls(PG)
        assert f"{BASE}/docs/current/brin.html" in urls
        assert f"{BASE}/docs/9.3/brin.html" not in urls
        assert not any("elsewhere.example" in url for url in urls)
        assert f"{BASE}/docs/current/q.html?a=1&b=2" in urls

    def test_the_sitemap_has_its_own_size_cap(self):
        web = Web({PG.sitemap: URLSET})
        index(web).urls(PG)
        assert (PG.sitemap, SITEMAP_MAX_BYTES) in web.asked

    def test_an_index_is_followed_one_level_on_the_same_host_and_never_into_gzip(self):
        child = f"{BASE}/sitemap-docs.xml"
        sitemap_index = (f"<sitemapindex><sitemap><loc>{child}</loc></sitemap>"
                         f"<sitemap><loc>{BASE}/sitemap-big.xml.gz</loc></sitemap>"
                         "<sitemap><loc>https://elsewhere.example/s.xml</loc></sitemap></sitemapindex>")
        web = Web({PG.sitemap: sitemap_index, child: URLSET})
        urls = index(web).urls(PG)
        assert f"{BASE}/docs/current/gin.html" in urls
        fetched = [url for url, _ in web.asked]
        assert not any(url.endswith(".gz") or "elsewhere" in url for url in fetched)

    def test_entities_are_never_expanded(self):
        bomb = ('<?xml version="1.0"?><!DOCTYPE l [<!ENTITY a "aaaaaaaaaa">'
                '<!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">]>'
                f"<urlset><url><loc>{BASE}/docs/current/&b;.html</loc></url></urlset>")
        urls = index(Web({PG.sitemap: bomb})).urls(PG)
        assert urls == (f"{BASE}/docs/current/&b;.html",)

    def test_a_disallowed_sitemap_is_not_fetched(self):
        web = Web({PG.sitemap: URLSET}, robots="User-agent: *\nDisallow: /sitemap.xml")
        assert index(web).urls(PG) == ()
        assert [url for url, _ in web.asked] == [f"{BASE}/robots.txt"]

    def test_read_once_per_ttl_even_when_unreadable(self):
        now = [0.0]
        web = Web({})
        sitemaps = index(web, clock=lambda: now[0], ttl_seconds=100)
        sitemaps.urls(PG)
        sitemaps.urls(PG)
        assert [u for u, _ in web.asked].count(PG.sitemap) == 1
        now[0] = 101
        web.pages[PG.sitemap] = URLSET
        assert sitemaps.urls(PG)
        assert [u for u, _ in web.asked].count(PG.sitemap) == 2

    def test_path_words(self):
        assert path_words(f"{BASE}/docs/current/brin-intro.html") == ["docs", "current", "brin", "intro", "html"]


class TestRanking:
    def candidates(self, query, robots=""):
        web = Web({PG.sitemap: URLSET}, robots)
        return index(web).candidates(PG, term_tokens(query, PG), 3)

    def test_a_rare_word_outranks_a_common_one(self):
        found = self.candidates("How do BRIN indexes work?")
        assert found[0].endswith("/brin.html")
        assert found.index(f"{BASE}/docs/current/brin-intro.html") < 2

    def test_the_sources_own_name_does_not_match_everything(self):
        assert self.candidates("PostgreSQL") == []

    def test_nothing_matching_offers_nothing(self):
        assert self.candidates("quantum chromodynamics") == []

    def test_disallowed_pages_are_not_offered(self):
        found = self.candidates("BRIN", robots="User-agent: *\nDisallow: /docs/current/brin.html")
        assert f"{BASE}/docs/current/brin.html" not in found
        assert found


class TestDiscoveryUsesSitemaps:
    def test_a_source_without_search_is_discovered_through_its_sitemap(self):
        web = Web({PG.sitemap: URLSET})
        found = discover(target("How do BRIN indexes work?", subtopics=("BRIN",)), [PG], web)
        assert found and found[0].endswith("/brin.html")

    def test_a_search_that_finds_results_leaves_the_sitemap_unread(self):
        searchable = PG._replace(search=f"{BASE}/find?q={{terms}}")

        def fetch(url, max_bytes=None):
            if url.endswith("/robots.txt"):
                return Document(url, "", "text/plain")
            if "/find" in url:
                return Document(url, '<li><a href="/docs/current/gin.html">GIN indexes</a></li>', "text/html")
            raise AssertionError(f"unexpected fetch {url}")

        assert discover(target("GIN indexes"), [searchable], fetch) == [f"{BASE}/docs/current/gin.html"]

    def test_a_search_that_finds_nothing_falls_back_to_the_sitemap(self):
        searchable = PG._replace(search=f"{BASE}/find?q={{terms}}")
        pages = {PG.sitemap: URLSET}

        def fetch(url, max_bytes=None):
            if url.endswith("/robots.txt"):
                return Document(url, "", "text/plain")
            if "/find" in url:
                return Document(url, "<p>No results</p>", "text/html")
            return Document(url, pages[url], "text/xml")

        found = discover(target("BRIN", subtopics=("BRIN",)), [searchable], fetch)
        assert found[0].endswith("/brin.html")

    def test_a_source_with_neither_is_not_asked(self):
        bare = TrustedSource("w3c", "www.w3.org", "standards")
        assert discover(target("anything"), [bare], lambda url, **kw: pytest.fail("fetched")) == []


def test_a_compound_name_matches_a_slug_that_joins_it():
    sitemap = (f"<urlset><url><loc>{BASE}/docs/current/pgstatstatements.html</loc></url>"
               f"<url><loc>{BASE}/docs/current/plpgsql-statements.html</loc></url></urlset>")
    web = Web({PG.sitemap: sitemap})
    found = index(web).candidates(PG, term_tokens("What does pg_stat_statements track?", PG), 2)
    assert found[0].endswith("/pgstatstatements.html")
