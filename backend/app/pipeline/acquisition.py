"""KSV's controlled acquisition, with what it keeps stored through the quota-checked ingestion path."""

import hashlib
import re
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Sequence

from config import Config, get_logger
from knowledge_sufficiency.acquisition import KnowledgeAcquisition, split_into_chunks
from knowledge_sufficiency.acquisition_store import AcquisitionStore
from knowledge_sufficiency.fetching import MAX_BYTES, fetch_document
from knowledge_sufficiency.ksv_exceptions import AcquisitionLimitReached, FetchFailed
from knowledge_sufficiency.ksv_manager import KSVManager
from knowledge_sufficiency.robots import RobotsPolicy
from knowledge_sufficiency.selection import SourceIndex
from knowledge_sufficiency.sitemaps import SitemapIndex
from knowledge_sufficiency.sources import DEFAULT_SOURCES, TrustedSource
from knowledge_sufficiency.target import build_target
from storage.timestamps import utc_now

logger = get_logger(__name__)


_MARKUP = re.compile(r"<\s*(html|body|div|p|li|a|span|h[1-6])\b", re.IGNORECASE)


def readable(text: str) -> str:
    """A fetched page as the text a reader sees: no tags, scripts, menus or footers."""
    if not _MARKUP.search(text):
        return text
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer"]):
        tag.decompose()
    lines = (line.strip() for line in soup.get_text("\n").splitlines())
    return "\n".join(line for line in lines if line)


def readable_chunks(text: str) -> List[str]:
    """What acquisition embeds, verifies and stores: windows of the readable text, never of markup."""
    return split_into_chunks(readable(text))


class AcquiredChunk(NamedTuple):
    url: str
    text: str


class AcquisitionOutcome(NamedTuple):
    sufficiency: str
    rounds: int
    stored_documents: int
    urls: tuple
    failures: int
    stopped: str | None
    downloaded_bytes: int = 0


_UNSET = object()


class Acquirer:
    """Fetches only from the trusted sources KSV allows, and stores only what answered the query.

    Two limits bound it: what one acquisition may download, and what acquired
    material may occupy in the corpus in all. Both default to Config, and None
    lifts either; the corpus-wide ingestion quota applies regardless.
    """

    def __init__(
        self,
        ksv: KSVManager,
        encode: Callable[[Sequence[str]], object],
        ingest: Callable[..., object],
        fetch=fetch_document,
        max_rounds: int = 3,
        directory: Path | None = None,
        acquisition_store: AcquisitionStore | None = None,
        download_limit=_UNSET,
        storage_limit=_UNSET,
        robots: RobotsPolicy | None = None,
        sources: Sequence[TrustedSource] = DEFAULT_SOURCES,
    ) -> None:
        self.ksv = ksv
        self.encode = encode
        self.ingest = ingest
        self.fetch = fetch
        self.max_rounds = max_rounds
        self.directory = directory
        self.acquisition_store = acquisition_store
        self.__download_limit = download_limit
        self.__storage_limit = storage_limit
        # One per acquirer, so each host's robots.txt is read once an hour, not per query.
        self.sources = tuple(sources)
        self.robots = RobotsPolicy(fetch) if robots is None else robots
        # Long-lived too: a 4.6 MB sitemap is read once a day, not per question.
        self.sitemaps = SitemapIndex(self.__budgeted_fetch, self.robots)
        self.__spent: Dict[str, int] | None = None
        self.__source_index: SourceIndex | None = None

    @property
    def download_limit(self) -> int | None:
        return Config.MAX_ACQUISITION_DOWNLOAD_BYTES if self.__download_limit is _UNSET else self.__download_limit

    @property
    def storage_limit(self) -> int | None:
        return Config.MAX_ACQUISITION_STORAGE_BYTES if self.__storage_limit is _UNSET else self.__storage_limit

    @property
    def corpus_directory(self) -> Path:
        return Path(Config.CORPUS_DIR if self.directory is None else self.directory)

    def stored_bytes(self) -> int:
        """What acquired material occupies in the corpus now."""
        from data_layer.ingestion.corpus_ingestion import acquired_bytes

        return acquired_bytes(self.corpus_directory)

    def __embed_one(self, text: str):
        return self.encode([text])[0]

    def __budgeted_fetch(self, url, max_bytes: int | None = None):
        """Every download of an acquisition, sitemaps included, counted against its budget."""
        spent = self.__spent if self.__spent is not None else {"bytes": 0}
        limit = self.download_limit
        remaining = None if limit is None else limit - spent["bytes"]
        if remaining is not None and remaining <= 0:
            raise FetchFailed(url, f"the {limit}-byte download budget of this acquisition is spent")
        caps = [cap for cap in (max_bytes, remaining) if cap is not None]
        document = self.fetch(url, max_bytes=min(caps)) if caps else self.fetch(url)
        spent["bytes"] += len(document.text.encode("utf-8"))
        return document

    def acquire(self, topic: str, query: str, subtopics: Sequence[str] = ()) -> AcquisitionOutcome:
        current: Dict[str, str] = {}
        stopped: List[str] = []
        stored: List[Path] = []
        spent = {"bytes": 0}
        self.__spent = spent
        download_limit = self.download_limit

        def fetch(url):
            document = self.__budgeted_fetch(url, None if download_limit is None else MAX_BYTES)
            current["url"] = document.url
            return document

        def embed(chunks):
            vectors = self.encode(list(chunks))
            return [(AcquiredChunk(current.get("url", ""), text), vector)
                    for text, vector in zip(chunks, vectors)]

        def store(candidates):
            stored.extend(self.__store(candidates, topic, stopped))

        if self.__source_index is None:
            # Ranked among the sources discovery can name pages on; a source
            # with neither a search nor a sitemap would spend a slot and find nothing.
            self.__source_index = SourceIndex(
                self.__embed_one, [s for s in self.sources if s.discoverable]
            )
        target = build_target(topic, query, self.__embed_one, subtopics)
        acquisition = KnowledgeAcquisition(
            verify=self.ksv.sufficiency_verification,
            embed=embed,
            store=store,
            fetch=fetch,
            chunk=readable_chunks,
            sources=self.sources,
            max_rounds=self.max_rounds,
            source_index=self.__source_index,
            acquisition_store=self.acquisition_store,
            robots=self.robots,
            sitemaps=self.sitemaps,
        ).acquire_until_sufficient(target)
        self.__spent = None
        urls = tuple(dict.fromkeys(c.url for c, _ in acquisition.acquired if isinstance(c, AcquiredChunk)))
        logger.info("Acquisition downloaded %d bytes (budget %s); stored %d document(s)",
                    spent["bytes"], download_limit, len(stored))
        return AcquisitionOutcome(
            acquisition.verdict.sufficiency.value,
            acquisition.rounds,
            len(stored),
            urls,
            len(acquisition.failures),
            "; ".join(stopped) or None,
            spent["bytes"],
        )

    def __store(self, candidates, topic: str, stopped: List[str]) -> List[Path]:
        """Ingest the kept material; the acquisition limit is checked and recorded inside the ingestion lock."""
        from data_layer.ingestion.corpus_ingestion import ACQUIRED, acquired_bytes, record_provenance

        directory = self.corpus_directory
        files = self.__write(candidates, topic)
        limit = self.storage_limit
        ingested: List[Path] = []

        def admit(path, needed):
            used = acquired_bytes(directory)
            if limit is not None and used + needed + path.stat().st_size > limit:
                raise AcquisitionLimitReached(used, needed, limit)

        def on_document(record):
            path = Path(record.path)
            entry = dict(files[path], document_id=record.document_id, chunks=record.chunks,
                         vectors=record.vectors, ingested_at=utc_now(), origin=ACQUIRED,
                         stored_bytes=record.added_bytes + path.stat().st_size)
            record_provenance([entry], directory)
            ingested.append(path)

        report = self.ingest(list(files), admit=admit, on_document=on_document)
        if getattr(report, "stopped", None):
            stopped.append(report.stopped)
        for path in files:
            if path not in ingested:
                # Never ingested, so never counted: leaving it would be an
                # orphan the corpus quota pays for and nothing can find.
                path.unlink(missing_ok=True)
        return ingested

    def __write(self, candidates, topic: str) -> Dict[Path, Dict]:
        """One text file per source URL, and the provenance it will be recorded with."""
        target = self.corpus_directory / "acquired"
        target.mkdir(parents=True, exist_ok=True)
        by_url: Dict[str, List[str]] = {}
        for chunk, _ in candidates:
            if isinstance(chunk, AcquiredChunk) and chunk.text.strip():
                by_url.setdefault(chunk.url, []).append(chunk.text.strip())
        files: Dict[Path, Dict] = {}
        for url, texts in by_url.items():
            body = "\n\n".join(dict.fromkeys(texts))
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
            path = target / f"{digest[:16]}.txt"
            path.write_text(body, encoding="utf-8")
            files[path] = {"file": str(path.relative_to(self.corpus_directory)), "source_url": url,
                           "topic": topic, "fetched_at": utc_now(), "sha256": digest}
        return files
