"""Which jobs may run now: dependencies met, none failed, in a fixed order."""

from typing import List

from thinking_layer.job_pool import JobPool
from thinking_layer.models import FailureKind, JobState

_BLOCKING = (JobState.FAILED, JobState.CANCELLED)
_WAITING = (JobState.PENDING, JobState.RETRY_PENDING)


class Scheduler:
    def __init__(self, pool: JobPool) -> None:
        self.pool = pool

    def promote(self) -> List[str]:
        """Cancel jobs whose prerequisites failed, mark the runnable READY, and return every READY id.

        Order is priority first, then the plan's dependency order, so the same
        plan always runs the same way.
        """
        # One pass in dependency order is enough: a cancellation can only
        # cascade to jobs later in that order.
        for job in self.pool.jobs():
            if job.state not in _WAITING:
                continue
            states = [self.pool.get(dep).state for dep in job.spec.depends_on]
            blocked = [
                dep for dep, state in zip(job.spec.depends_on, states) if state in _BLOCKING
            ]
            if blocked:
                self.pool.transition(
                    job.job_id,
                    JobState.CANCELLED,
                    failure_kind=FailureKind.DEPENDENCY,
                    failure=f"prerequisite {', '.join(blocked)} did not complete",
                )
            elif all(state is JobState.COMPLETED for state in states):
                self.pool.transition(job.job_id, JobState.READY)
        ready = [job for job in self.pool.jobs() if job.state is JobState.READY]
        position = {job_id: index for index, job_id in enumerate(self.pool.order)}
        ready.sort(key=lambda job: (-job.spec.priority, position[job.job_id]))
        return [job.job_id for job in ready]

    def cancel_open(self, kind: FailureKind, reason: str) -> List[str]:
        """Cancel every job not yet finished or running; returns their ids."""
        cancelled = []
        for job in self.pool.jobs():
            if job.state in (JobState.PENDING, JobState.READY, JobState.RETRY_PENDING):
                self.pool.transition(
                    job.job_id, JobState.CANCELLED, failure_kind=kind, failure=reason
                )
                cancelled.append(job.job_id)
        return cancelled
