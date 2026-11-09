"""One pipeline operation at a time, with a short queue behind it.

The model, the vector index and the ingestion lock are all single-occupancy;
letting two requests interleave inside them buys nothing but contention. A
request that finds the queue full is refused at once (BUSY) instead of
holding a thread for many minutes. A waiting request whose caller has gone
away (cancelled, deadline passed) leaves the queue without running.
"""

import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator


class Busy(Exception):
    pass


class Abandoned(Exception):
    """The caller stopped waiting before the request started."""


class OperationGate:
    POLL_SECONDS = 0.25

    def __init__(self, max_waiting: int) -> None:
        self.max_waiting = max_waiting
        self.__condition = threading.Condition()
        self.__running = False
        self.__waiting = 0

    @property
    def busy(self) -> bool:
        return self.__running

    @property
    def waiting(self) -> int:
        return self.__waiting

    @contextmanager
    def admit(self, still_wanted: Callable[[], bool] = lambda: True,
              max_wait_seconds: float | None = None) -> Iterator[None]:
        deadline = None if max_wait_seconds is None else time.monotonic() + max_wait_seconds
        def abandoned() -> bool:
            return not still_wanted() or (deadline is not None and time.monotonic() >= deadline)

        with self.__condition:
            if self.__running:
                if self.__waiting >= self.max_waiting:
                    raise Busy(f"{self.__waiting} request(s) already waiting")
                self.__waiting += 1
                try:
                    while self.__running:
                        self.__condition.wait(self.POLL_SECONDS)
                        if abandoned():
                            raise Abandoned()
                finally:
                    self.__waiting -= 1
            # Checked again on the way in: a wake-up from the previous request
            # finishing must not admit a caller whose time ran out meanwhile.
            if abandoned():
                raise Abandoned()
            self.__running = True
        try:
            yield
        finally:
            with self.__condition:
                self.__running = False
                self.__condition.notify_all()
