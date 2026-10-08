"""Not searching twice for the same question."""

import threading
import time
from collections import OrderedDict
from typing import Any, Callable, NamedTuple, Optional, Tuple

from config import get_logger
from retrieval_layer.models import RetrievalRequest
from retrieval_layer.settings import CACHE_SIZE, CACHE_TTL_SECONDS

logger = get_logger(__name__)


class CacheStats(NamedTuple):
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    size: int = 0

    @property
    def hit_rate(self) -> float:
        looked_up = self.hits + self.misses
        return self.hits / looked_up if looked_up else 0.0


def cache_key(request: RetrievalRequest) -> str:
    """The parts of a request that change its answer.

    Case and spacing are folded because neither reaches the searchers: the
    embedding model is uncased and FTS5's unicode61 tokenizer folds case, so
    "Python GIL" and "python  gil" retrieve the same passages and sharing one
    entry is correct rather than merely close. Resolve a request's defaults
    before keying it, or `top_k=None` gets its own entry alongside `top_k=8`.
    """
    query = " ".join(request.query.split()).casefold()
    return f"{query}|{request.top_k}|{request.rerank}"


class ResultCache:
    """A bounded, expiring store of recent answers.

    Bounded because a long-lived process would otherwise hold every result it
    has ever returned; expiring because an answer is only as current as the
    corpus behind it and nothing here is told when that corpus grows. Ingestion
    calls `clear()`; the TTL is what covers the ingestion that forgets to.
    """

    def __init__(
        self,
        max_size: int = CACHE_SIZE,
        ttl_seconds: float = CACHE_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_size = max(0, int(max_size))
        self.ttl_seconds = float(ttl_seconds)
        self.clock = clock
        self._entries: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    def __expired(self, stored_at: float, now: float) -> bool:
        return now - stored_at >= self.ttl_seconds

    def __drop_expired(self) -> None:
        now = self.clock()
        stale = [
            key for key, (stored_at, _) in self._entries.items()
            if self.__expired(stored_at, now)
        ]
        for key in stale:
            del self._entries[key]
            self._expirations += 1

    def get(self, key: str) -> Optional[Any]:
        """The stored answer, or None if there is none worth having."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            stored_at, value = entry
            if self.__expired(stored_at, self.clock()):
                del self._entries[key]
                self._expirations += 1
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return value

    def put(self, key: str, value: Any) -> None:
        """Store an answer, evicting the least recently used if full."""
        if self.max_size == 0:
            return
        with self._lock:
            if key in self._entries:
                del self._entries[key]
            elif len(self._entries) >= self.max_size:
                # Reclaim before evicting: a cache full of entries nobody can
                # use any more should not cost a live one its place.
                self.__drop_expired()
                while len(self._entries) >= self.max_size:
                    self._entries.popitem(last=False)
                    self._evictions += 1
            self._entries[key] = (self.clock(), value)

    def clear(self) -> None:
        """Forget every answer. The lifetime counters are not forgotten."""
        with self._lock:
            dropped = len(self._entries)
            self._entries.clear()
        if dropped:
            logger.debug("Dropped %d cached result(s)", dropped)

    @property
    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                expirations=self._expirations,
                size=len(self._entries),
            )
