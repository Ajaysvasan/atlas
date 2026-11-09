"""Per-request records: stage timings, retries, model calls, cache and retrieval outcomes. Never prompt text."""

import logging
import threading
import time
from collections import OrderedDict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Iterator, List

from config import get_logger

logger = get_logger(__name__)


@dataclass
class RequestRecord:
    request_id: str
    conversation_id: str
    started_at: str
    status: str = "running"
    mode: str | None = None
    path: str | None = None
    error_category: str | None = None
    error: str | None = None
    stages: "OrderedDict[str, float]" = field(default_factory=OrderedDict)
    total_ms: float | None = None
    jobs: int = 0
    job_retries: int = 0
    plan_revisions: int = 0
    runtime_retries: int = 0
    retried_jobs: List[str] = field(default_factory=list)
    retry_reasons: List[str] = field(default_factory=list)
    retries_exhausted: bool = False
    thinking_status: str | None = None
    model_calls: Dict[str, int] = field(default_factory=dict)
    retrieval: Dict[str, object] = field(default_factory=dict)
    cache: Dict[str, object] = field(default_factory=dict)
    persisted: Dict[str, object] = field(default_factory=dict)

    @property
    def reasoning_retries(self) -> int:
        return self.job_retries + self.plan_revisions

    def summary(self) -> Dict[str, object]:
        """What goes into the log line: counts and timings, no content."""
        return {
            "request_id": self.request_id,
            "conversation_id": self.conversation_id,
            "status": self.status,
            "mode": self.mode,
            "path": self.path,
            "error_category": self.error_category,
            "stages_ms": {k: round(v, 1) for k, v in self.stages.items()},
            "total_ms": None if self.total_ms is None else round(self.total_ms, 1),
            "jobs": self.jobs,
            "job_retries": self.job_retries,
            "plan_revisions": self.plan_revisions,
            "runtime_retries": self.runtime_retries,
            "model_calls": self.model_calls,
            "retrieval": self.retrieval,
            "cache": self.cache,
            "persisted": self.persisted,
        }


class StageClock:
    """Wall-clock stage durations on a monotonic clock; stages run one after another, so they add up."""

    def __init__(self, record: RequestRecord, clock=time.monotonic) -> None:
        self.record = record
        self.clock = clock
        self.started = clock()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = self.clock()
        try:
            yield
        finally:
            self.record.stages[name] = self.record.stages.get(name, 0.0) + (self.clock() - start) * 1000.0

    def finish(self) -> None:
        self.record.total_ms = (self.clock() - self.started) * 1000.0


class Telemetry:
    """The last few requests of this process, newest last."""

    def __init__(self, keep: int = 50) -> None:
        self.__records: Deque[RequestRecord] = deque(maxlen=keep)
        self.__lock = threading.Lock()
        self.__listeners: List[Callable[[RequestRecord], None]] = []

    def subscribe(self, listener: Callable[[RequestRecord], None]) -> None:
        """Called with each record after it is added; a failing listener is logged, never raised."""
        with self.__lock:
            self.__listeners.append(listener)

    def add(self, record: RequestRecord) -> None:
        with self.__lock:
            self.__records.append(record)
            listeners = list(self.__listeners)
        if logger.isEnabledFor(logging.INFO):
            logger.info("Request finished: %s", record.summary())
        for listener in listeners:
            try:
                listener(record)
            except Exception:
                logger.exception("A telemetry listener failed")

    def last(self) -> RequestRecord | None:
        with self.__lock:
            return self.__records[-1] if self.__records else None

    def records(self) -> List[RequestRecord]:
        with self.__lock:
            return list(self.__records)
