"""A bounded pool of threads that run jobs; model calls inside them still queue on the model's lock."""

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Callable, Dict, Iterable, Set, Tuple


class WorkerPool:
    """Logical concurrency only. Physical inference is serialised by the shared model, not here."""

    def __init__(self, max_workers: int) -> None:
        if max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {max_workers}")
        self.max_workers = max_workers
        self.__executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="thinking-worker"
        )
        self.running: Dict[Future, str] = {}

    @property
    def free(self) -> int:
        return self.max_workers - len(self.running)

    def submit(self, job_id: str, task: Callable[[], object]) -> Future:
        if self.free < 1:
            raise RuntimeError("no free worker; check `free` before submitting")
        future = self.__executor.submit(task)
        self.running[future] = job_id
        return future

    def wait_any(self, timeout: float | None) -> Iterable[Tuple[str, Future]]:
        """The jobs that finished within `timeout`, removed from `running`."""
        if not self.running:
            return []
        done, _ = wait(list(self.running), timeout=timeout, return_when=FIRST_COMPLETED)
        finished = [(self.running.pop(future), future) for future in done]
        return finished

    def abandon(self) -> Set[str]:
        """Stop tracking jobs still running; their results will be ignored."""
        abandoned = set(self.running.values())
        for future in self.running:
            future.cancel()
        self.running.clear()
        return abandoned

    def shutdown(self) -> None:
        # wait=False: a job abandoned at the deadline may still be inside a
        # model call, which cannot be interrupted; it finishes on its own.
        self.__executor.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> "WorkerPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()
