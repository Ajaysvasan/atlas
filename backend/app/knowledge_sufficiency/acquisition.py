"""Acquire knowledge until it answers the query, or until the rounds run out."""

from typing import Callable, List, NamedTuple, Sequence, Tuple

import numpy as np
from numpy import float32, ndarray

from config import get_logger
from knowledge_sufficiency.acquisition_store import AcquisitionStore
from knowledge_sufficiency.discovery import discover
from knowledge_sufficiency.fetching import Document, fetch_document
from knowledge_sufficiency.ksv_exceptions import FetchFailed, UntrustedSource
from knowledge_sufficiency.relevance import (
    MAX_KEPT_CHUNKS,
    TOPIC_FLOOR,
    filter_relevant,
)
from knowledge_sufficiency.selection import TOP_SOURCES, SourceIndex
from knowledge_sufficiency.sources import DEFAULT_SOURCES, DOCS, TrustedSource
from knowledge_sufficiency.target import AcquisitionTarget
from knowledge_sufficiency.verdict import Sufficiency, Verdict

logger = get_logger(__name__)

MAX_ROUNDS = 3

Candidate = Tuple[object, ndarray]


class Acquisition(NamedTuple):
    verdict: Verdict
    acquired: List[Candidate]
    stored: bool
    rounds: int
    failures: List[Tuple[str, str]]
    discarded: int
    recorded: List[int]


def split_into_chunks(text: str, size: int = 256, overlap: int = 20) -> List[str]:
    """Window the text on word boundaries, never exceeding `size`."""
    from data_layer.ingestion.Chunker.windowing import sliding_windows

    if not text.strip():
        return []
    pieces = (text[start:end] for start, end in sliding_windows(text, size, overlap))
    return [piece for piece in pieces if piece.strip()]


class KnowledgeAcquisition:
    """Search, fetch, chunk, embed, gate, verify — while the answer is missing.

    Every collaborator is injected. The default fetcher refuses anything that is
    not https from a trusted host, but a caller can supply its own, which is how
    the tests run the whole loop without a network.
    """

    def __init__(
        self,
        verify: Callable[[ndarray, Sequence[Candidate]], Verdict],
        embed: Callable[[Sequence[str]], Sequence[Candidate]],
        store: Callable[[Sequence[Candidate]], None] | None = None,
        fetch: Callable[[str], Document] = fetch_document,
        chunk: Callable[[str], List[str]] = split_into_chunks,
        sources: Sequence[TrustedSource] = DEFAULT_SOURCES,
        max_rounds: int = MAX_ROUNDS,
        store_on_partial: bool = False,
        topic_floor: float = TOPIC_FLOOR,
        max_kept: int = MAX_KEPT_CHUNKS,
        source_index: SourceIndex | None = None,
        top_sources: int = TOP_SOURCES,
        acquisition_store: AcquisitionStore | None = None,
    ) -> None:
        if max_rounds < 1:
            raise ValueError(f"max_rounds must be at least 1, got {max_rounds}")
        if max_kept < 1:
            raise ValueError(f"max_kept must be at least 1, got {max_kept}")
        self.verify = verify
        self.embed = embed
        self.store = store
        self.fetch = fetch
        self.chunk = chunk
        self.sources = sources
        self.max_rounds = max_rounds
        self.store_on_partial = store_on_partial
        self.topic_floor = topic_floor
        self.max_kept = max_kept
        self.source_index = source_index
        self.top_sources = top_sources
        self.acquisition_store = acquisition_store

    def __round(self, url: str) -> List[Candidate]:
        document = self.fetch(url)
        chunks = self.chunk(document.text)
        if not chunks:
            logger.debug("Nothing to embed from %s", url)
            return []
        return list(self.embed(chunks))

    def acquire_until_sufficient(
        self, target: AcquisitionTarget, urls: Sequence[str] | None = None
    ) -> Acquisition:
        """Gather material about the target's subject until it answers the query.

        `urls` overrides discovery, for a caller that already knows where to
        look. Knowledge accumulates across rounds and is verified against
        everything kept so far, not just the newest document: two sources that
        each fall short alone may answer the query together.
        """
        if urls is None:
            # Narrowed to the sources that plausibly cover the subject. Without
            # an index every searchable source is asked, which for a SQLite
            # question means searching the web platform reference and a preprint
            # server too.
            sources = self.sources
            if self.source_index is not None:
                sources = self.source_index.select(
                    target, kind=DOCS, top_k=self.top_sources
                )
            urls = discover(target, sources, self.fetch)
            logger.info(
                "Discovered %d candidate(s) for topic %r from %d source(s)",
                len(urls), target.topic, len(sources),
            )

        acquired: List[Candidate] = []
        contributed: List[str] = []
        failures: List[Tuple[str, str]] = []
        verdict = Verdict(Sufficiency.NONE, -1.0, [], np.empty(0, dtype=float32))
        rounds = 0
        discarded = 0

        for url in list(urls)[: self.max_rounds]:
            rounds += 1
            try:
                embedded = self.__round(url)
            except (UntrustedSource, FetchFailed) as error:
                # One bad source does not end the attempt; the next may answer.
                logger.warning("Skipping %s: %s", url, error)
                failures.append((url, str(error)))
                continue

            # The gate runs per round, before anything accumulates: a document
            # about something else must not reach the store, and must not join
            # the running total where the cap would later have to evict it.
            filtered = filter_relevant(
                target, embedded, self.topic_floor, self.max_kept
            )
            discarded += filtered.dropped
            # Only a document that survived the gate is worth recording: one
            # that was fetched and found entirely off-subject contributed
            # nothing, and listing it would say knowledge came from somewhere
            # it did not.
            if filtered.kept:
                contributed.append(url)
            acquired.extend(filtered.kept)

            if not acquired:
                continue

            capped = filter_relevant(
                target, acquired, self.topic_floor, self.max_kept
            )
            discarded += capped.capped
            acquired = capped.kept

            verdict = self.verify(target.query_vector, acquired)
            logger.debug(
                "Round %d over %d candidate(s): %s",
                rounds, len(acquired), verdict.sufficiency.value,
            )
            if verdict.is_sufficient:
                break

        stored = self.__maybe_store(verdict, acquired)
        recorded = self.__record(target, contributed, stored)
        logger.info(
            "Acquisition finished %s after %d round(s): kept %d, discarded %d, "
            "stored=%s, recorded=%d",
            verdict.sufficiency.value, rounds, len(acquired), discarded, stored,
            len(recorded),
        )
        return Acquisition(
            verdict, acquired, stored, rounds, failures, discarded, recorded
        )

    def __record(
        self, target: AcquisitionTarget, urls: Sequence[str], stored: bool
    ) -> List[int]:
        """Note where the knowledge came from, once it has been kept.

        Tied to storage rather than to fetching: the table says what the system
        took knowledge from, and a document whose chunks were discarded as
        insufficient was never taken from.
        """
        if self.acquisition_store is None or not stored or not urls:
            return []
        return self.acquisition_store.record_many(target.topic, target.query, urls)

    def __maybe_store(
        self, verdict: Verdict, acquired: Sequence[Candidate]
    ) -> bool:
        """Only what answered the query goes into the index.

        Storing every fetch would fill the store with material that did not
        answer anything, which costs on every later search.
        """
        if self.store is None or not acquired:
            return False
        keep = verdict.is_sufficient or (
            self.store_on_partial and verdict.sufficiency is Sufficiency.PARTIAL
        )
        if not keep:
            return False
        self.store(acquired)
        return True
