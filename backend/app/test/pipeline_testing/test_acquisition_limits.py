"""KSV acquisition through the real ingestion driver: what it stores, and the two limits that bound it."""

import numpy as np
import pytest

from config import Config, parse_storage_limit
from data_layer.ingestion.corpus_ingestion import acquired_bytes, ingest_files, read_provenance
from data_layer.ingestion.quota import IngestionQuota
from knowledge_sufficiency.fetching import Document
from knowledge_sufficiency.ksv_manager import KSVManager
from knowledge_sufficiency.sources import DEFAULT_SOURCES
from pipeline.acquisition import Acquirer

PAGE = ("SQLite VACUUM INTO writes a vacuumed copy of the database to a new file, "
        "which makes it a consistent online backup. ") * 40
SQLITE = next(s for s in DEFAULT_SOURCES if s.name == "sqlite")
SEARCH = ('<ul><li><a href="/lang_vacuum.html">VACUUM</a> backup with VACUUM INTO</li>'
          '<li><a href="/backup.html">Online backup</a> the backup API and VACUUM INTO</li></ul>')


def same_vector(texts):
    v = np.ones(Config.EMBEDDING_DIMENSIONS, dtype=np.float32)
    return np.array([v / np.linalg.norm(v)] * len(texts))


class Web:
    """sqlite.org as far as acquisition sees it; records every request and its byte cap."""

    def __init__(self) -> None:
        self.requests = []

    def __call__(self, url, max_bytes=None):
        self.requests.append((url, max_bytes))
        if url.endswith("/robots.txt"):
            return Document(url, "", "text/plain")
        text = SEARCH if "/search" in url else PAGE
        if max_bytes is not None:
            text = text.encode()[:max_bytes].decode(errors="ignore")
        return Document(url, text, "text/html")


@pytest.fixture
def acquirer(tmp_path, chunk_vectors):
    quota = IngestionQuota(limit_bytes=1 << 30, postgres_usage=chunk_vectors.storage_bytes,
                           lock_path=tmp_path / "lock")
    web = Web()

    def build(**limits):
        made = Acquirer(
            KSVManager(), same_vector,
            ingest=lambda paths, **hooks: ingest_files(paths, quota=quota, build_index=False, **hooks),
            fetch=web, directory=tmp_path / "corpus", sources=[SQLITE], **limits,
        )
        made.web = web
        return made

    return build


def acquire(acquirer):
    return acquirer.acquire("SQLite", "How does VACUUM INTO create a backup?", ("VACUUM", "backup"))


def test_what_answered_the_query_is_stored_with_its_provenance(acquirer, chunk_vectors, tmp_path):
    made = acquirer()
    outcome = acquire(made)
    assert outcome.sufficiency == "full" and outcome.stored_documents == 1
    (entry,) = [e for e in read_provenance(tmp_path / "corpus") if e.get("document_id")]
    assert entry["origin"] == "ksv_acquisition" and entry["topic"] == "SQLite"
    assert entry["source_url"].startswith("https://sqlite.org/")
    assert entry["stored_bytes"] > 0
    assert made.stored_bytes() == entry["stored_bytes"]
    assert len(chunk_vectors.rows) == entry["vectors"] > 0


def test_the_storage_limit_refuses_before_anything_is_written(acquirer, chunk_vectors, tmp_path):
    outcome = acquire(acquirer(storage_limit=1000))
    assert outcome.stored_documents == 0
    assert "acquisition limit" in outcome.stopped
    assert chunk_vectors.rows == {}
    assert list((tmp_path / "corpus" / "acquired").iterdir()) == []
    assert acquired_bytes(tmp_path / "corpus") == 0


def test_the_storage_limit_counts_every_earlier_acquisition(acquirer, tmp_path):
    first = acquirer()
    acquire(first)
    used = first.stored_bytes()
    second = acquirer(storage_limit=used + 1000)
    second.web.requests.clear()
    outcome = second.acquire("SQLite", "What does the online backup API copy?", ("backup",))
    assert outcome.stored_documents == 0 and "acquisition limit" in outcome.stopped
    assert second.stored_bytes() == used


def test_no_storage_limit_still_stores(acquirer):
    assert acquire(acquirer(storage_limit=None)).stored_documents == 1


def unrelated_pages(texts):
    """The query and subject on one axis, every fetched page on another: nothing ever suffices."""
    out = []
    for text in texts:
        v = np.zeros(Config.EMBEDDING_DIMENSIONS, dtype=np.float32)
        v[1 if "VACUUM INTO writes" in text else 0] = 1.0
        out.append(v)
    return np.array(out)


def test_the_download_budget_caps_and_then_refuses_fetches(acquirer):
    budget = len(SEARCH) + 300
    made = acquirer(download_limit=budget)
    made.encode = unrelated_pages
    outcome = acquire(made)
    assert outcome.downloaded_bytes <= budget
    caps = [cap for url, cap in made.web.requests if not url.endswith("robots.txt")]
    assert all(cap is not None and cap <= budget for cap in caps)
    # Once the search pages and a slice of the first page spend it, the rounds
    # left are refused without a request ever being made.
    documents = [url for url, _ in made.web.requests if url.endswith(".html")]
    assert outcome.failures >= 1
    assert len(documents) == outcome.rounds - outcome.failures
    assert outcome.stored_documents == 0


def test_an_unlimited_download_budget_passes_no_cap(acquirer):
    made = acquirer(download_limit=None)
    acquire(made)
    assert all(cap is None for _, cap in made.web.requests)


@pytest.mark.parametrize("name", ["MAX_ACQUISITION_STORAGE_BYTES", "MAX_ACQUISITION_DOWNLOAD_BYTES"])
def test_the_limits_are_configured_like_the_ingestion_quota(name):
    assert isinstance(getattr(Config, name), int) and getattr(Config, name) > 0
    assert parse_storage_limit("null") is None


def test_markup_never_reaches_what_is_stored(acquirer, tmp_path):
    made = acquirer()
    page = ("<html><nav><a href='/'>Home</a> Menu</nav><body><p>" + PAGE + "</p>"
            "<script>track()</script></body></html>")
    original = made.web.__call__

    def fetch(url, max_bytes=None):
        document = original(url, max_bytes)
        return document._replace(text=page) if url.endswith(".html") else document

    made.fetch = fetch
    acquire(made)
    (stored,) = list((tmp_path / "corpus" / "acquired").iterdir())
    text = stored.read_text()
    assert "<" not in text and "track()" not in text and "Menu" not in text
    assert "VACUUM INTO" in text


def test_readable_leaves_plain_text_alone():
    from pipeline.acquisition import readable

    assert readable("a < b and c > d") == "a < b and c > d"
