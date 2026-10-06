"""Datasets as a source of knowledge, which prose caps do not bound."""

import json
from typing import Callable, List, NamedTuple, Sequence

from config import get_logger
from knowledge_sufficiency.fetching import Document, fetch_document
from knowledge_sufficiency.ksv_exceptions import FetchFailed, UntrustedSource
from knowledge_sufficiency.sources import DATASET, TrustedSource, sources_of_kind
from knowledge_sufficiency.target import AcquisitionTarget

logger = get_logger(__name__)

MAX_DATASET_BYTES = 512 * 1024 * 1024

# Licences that permit use and redistribution of derived material. A dataset
# whose licence is absent or unrecognised is refused rather than assumed open:
# the cost of being wrong is redistributing someone's work without the right to.
PERMISSIVE_LICENCES = frozenset({
    "apache-2.0", "mit", "bsd", "bsd-2-clause", "bsd-3-clause",
    "cc0-1.0", "cc-by-4.0", "cc-by-sa-4.0", "odc-by",
    "cdla-permissive-1.0", "cdla-permissive-2.0",
})


class DatasetCandidate(NamedTuple):
    name: str
    url: str
    licence: str
    size_bytes: int | None
    description: str

    @property
    def licence_is_permissive(self) -> bool:
        return self.licence.lower().strip() in PERMISSIVE_LICENCES


class DatasetVerdict(NamedTuple):
    accepted: List[DatasetCandidate]
    rejected: List[tuple]


def parse_candidates(payload: str, source: TrustedSource) -> List[DatasetCandidate]:
    """Read a dataset listing into candidates, skipping anything malformed."""
    try:
        entries = json.loads(payload)
    except json.JSONDecodeError as error:
        logger.warning("Could not read a dataset listing from %s: %s",
                       source.name, error)
        return []
    if isinstance(entries, dict):
        entries = entries.get("datasets", [])

    found: List[DatasetCandidate] = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("id") or entry.get("name")
        if not name:
            continue
        card = entry.get("cardData") or {}
        found.append(DatasetCandidate(
            name=str(name),
            url=f"https://{source.host}/datasets/{name}",
            licence=str(card.get("license") or entry.get("license") or ""),
            size_bytes=_size_of(entry),
            description=str(
                card.get("description") or entry.get("description") or name
            ),
        ))
    return found


def _size_of(entry: dict) -> int | None:
    for key in ("size_bytes", "sizeBytes", "downloads_size", "dataset_size"):
        value = entry.get(key)
        if isinstance(value, (int, float)) and value >= 0:
            return int(value)
    return None


def screen(
    candidates: Sequence[DatasetCandidate],
    max_bytes: int = MAX_DATASET_BYTES,
    require_licence: bool = True,
) -> DatasetVerdict:
    """Keep the datasets that may actually be used, and say why the rest cannot.

    Size and licence, not relevance: relevance is the caller's gate, but a
    dataset too large to hold or licensed against reuse is out regardless of
    how well it matches.
    """
    accepted: List[DatasetCandidate] = []
    rejected: List[tuple] = []
    for candidate in candidates:
        if require_licence and not candidate.licence:
            rejected.append((candidate.name, "no licence stated"))
        elif require_licence and not candidate.licence_is_permissive:
            rejected.append((candidate.name, f"licence {candidate.licence!r}"))
        elif candidate.size_bytes is None:
            # An unknown size cannot be bounded in advance, and the prose byte
            # cap does not apply to a dataset download.
            rejected.append((candidate.name, "size unknown"))
        elif candidate.size_bytes > max_bytes:
            rejected.append(
                (candidate.name, f"{candidate.size_bytes} bytes over the cap")
            )
        else:
            accepted.append(candidate)
    if rejected:
        logger.info(
            "Screened out %d dataset(s): %s", len(rejected),
            ", ".join(f"{name} ({why})" for name, why in rejected),
        )
    return DatasetVerdict(accepted, rejected)


def find_datasets(
    target: AcquisitionTarget,
    sources: Sequence[TrustedSource],
    fetch: Callable[[str], Document] = fetch_document,
    max_bytes: int = MAX_DATASET_BYTES,
    require_licence: bool = True,
) -> DatasetVerdict:
    """Datasets about this target that may be used.

    Returns references, never payloads. Downloading and ingesting a dataset is
    the data layer's pipeline; this subsystem decides whether one is worth
    handing over.
    """
    found: List[DatasetCandidate] = []
    for source in sources_of_kind(DATASET, sources):
        if not source.searchable:
            continue
        url = source.search.format(terms=target.terms.replace(" ", "+"))
        try:
            payload = fetch(url)
        except (UntrustedSource, FetchFailed) as error:
            logger.warning("Could not search %s: %s", source.name, error)
            continue
        found.extend(parse_candidates(payload.text, source))
    logger.debug("Found %d dataset candidate(s) for %r", len(found), target.topic)
    return screen(found, max_bytes, require_licence)
