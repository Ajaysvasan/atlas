"""Accepted job results, the evidence they cite, and what is still missing."""

import re
import threading
from typing import Dict, Iterable, List, Sequence, Tuple

from thinking_layer.models import Evidence

_CITATION = re.compile(r"\[([A-Z]\d{1,3})\]")


def cited_refs(text: str, known: Iterable[str]) -> Tuple[str, ...]:
    """Evidence tags `text` cites that name real evidence, in first-cited order."""
    known = set(known)
    seen: List[str] = []
    for ref in _CITATION.findall(text or ""):
        if ref in known and ref not in seen:
            seen.append(ref)
    return tuple(seen)


def unknown_refs(text: str, known: Iterable[str]) -> Tuple[str, ...]:
    """Tags `text` cites that no evidence carries: citations the model made up."""
    known = set(known)
    return tuple(sorted({ref for ref in _CITATION.findall(text or "") if ref not in known}))


class Accumulator:
    """Holds only accepted results; a model's output is never filed as evidence."""

    def __init__(self, evidence: Sequence[Evidence]) -> None:
        self.evidence: Dict[str, Evidence] = {item.ref: item for item in evidence}
        self.__results: Dict[str, str] = {}
        self.__citations: Dict[str, Tuple[str, ...]] = {}
        self.__unresolved: List[str] = []
        self.__lock = threading.Lock()

    def accept(self, job_id: str, output: str) -> Tuple[str, ...]:
        refs = cited_refs(output, self.evidence)
        with self.__lock:
            self.__results[job_id] = output
            self.__citations[job_id] = refs
        return refs

    def retract(self, job_id: str) -> None:
        """Withdraw a result a plan revision reopened, so no dependent reads it."""
        with self.__lock:
            self.__results.pop(job_id, None)
            self.__citations.pop(job_id, None)

    def result(self, job_id: str) -> str | None:
        with self.__lock:
            return self.__results.get(job_id)

    def accepted_ids(self) -> List[str]:
        with self.__lock:
            return list(self.__results)

    def dependency_context(self, job_ids: Sequence[str], budget_chars: int) -> List[Tuple[str, str]]:
        """Accepted results of `job_ids`, each trimmed to an equal share of the budget."""
        with self.__lock:
            present = [(job_id, self.__results[job_id]) for job_id in job_ids if job_id in self.__results]
        if not present:
            return []
        share = max(200, budget_chars // len(present))
        return [(job_id, trim(text, share)) for job_id, text in present]

    def cited(self) -> Tuple[Evidence, ...]:
        """Evidence any accepted result cites, in tag order."""
        with self.__lock:
            refs = {ref for refs in self.__citations.values() for ref in refs}
        return tuple(self.evidence[ref] for ref in sorted(refs, key=_tag_order))

    def mark_unresolved(self, items: Iterable[str]) -> None:
        with self.__lock:
            for item in items:
                if item and item not in self.__unresolved:
                    self.__unresolved.append(item)

    @property
    def unresolved(self) -> Tuple[str, ...]:
        with self.__lock:
            return tuple(self.__unresolved)

    def aggregate(self, order: Sequence[str]) -> List[Tuple[str, str]]:
        """Accepted results in plan order, for the final critique."""
        with self.__lock:
            return [(job_id, self.__results[job_id]) for job_id in order if job_id in self.__results]


def _tag_order(ref: str) -> Tuple[str, int]:
    return ref[0], int(ref[1:])


def trim(text: str, limit: int) -> str:
    """`text` cut to `limit` characters, marked when anything was cut."""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 15)].rstrip() + " …[truncated]"
