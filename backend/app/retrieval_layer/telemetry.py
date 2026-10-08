"""Where a retrieval spent its time."""

import time
from contextlib import contextmanager
from typing import Callable, Iterator, List, Optional

from config import get_logger
from retrieval_layer.models import StageTiming

logger = get_logger(__name__)


class Stage:
    """The handle a timed block uses to report what it produced."""

    __slots__ = ("name", "produced")

    def __init__(self, name: str) -> None:
        self.name = name
        self.produced = 0


class StageRecorder:
    """Times each stage of one retrieval, and only ever one.

    Unlocked on purpose: a recorder belongs to a single request, and sharing
    one between concurrent retrievals would interleave their stages into a
    timeline that describes neither. The clock must be monotonic, or a
    duration can come back negative and poison every total built from it.
    """

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self.clock = clock
        self._timings: List[Optional[StageTiming]] = []

    @contextmanager
    def stage(self, name: str) -> Iterator[Stage]:
        """Time a block and record how long it took and what it produced."""
        handle = Stage(name)
        # The slot is claimed on entry and filled on exit, so the timeline reads
        # in the order stages started even where one stage nests inside another.
        position = len(self._timings)
        self._timings.append(None)
        started = self.clock()
        try:
            yield handle
        except BaseException:
            elapsed = (self.clock() - started) * 1000
            self._timings[position] = StageTiming(name, elapsed, handle.produced)
            logger.warning("Retrieval stage %s failed after %.0f ms", name, elapsed)
            raise
        else:
            elapsed = (self.clock() - started) * 1000
            self._timings[position] = StageTiming(name, elapsed, handle.produced)

    @property
    def timings(self) -> List[StageTiming]:
        """The stages that have finished, in the order they started."""
        return [timing for timing in self._timings if timing is not None]

    def log_breakdown(self) -> None:
        """One line for the whole retrieval, rather than one per stage."""
        finished = self.timings
        if not finished:
            return
        breakdown = ", ".join(
            f"{timing.stage} {timing.milliseconds:.0f}ms/{timing.produced}"
            for timing in finished
        )
        total = sum(timing.milliseconds for timing in finished)
        logger.debug("Retrieval took %.0f ms: %s", total, breakdown)
