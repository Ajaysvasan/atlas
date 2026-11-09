"""Every job of one plan, and the only place a job's state changes."""

import threading
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from thinking_layer.models import (
    TERMINAL_STATES,
    Critique,
    FailureKind,
    JobKind,
    JobOutcome,
    JobSpec,
    JobState,
    Plan,
)
from thinking_layer.thinking_exceptions import InvalidPlan, InvalidTransition, UnknownJob

# COMPLETED -> RETRY_PENDING is a plan-level revision reopening accepted work.
_ALLOWED = {
    JobState.PENDING: {JobState.READY, JobState.CANCELLED},
    JobState.READY: {JobState.RUNNING, JobState.CANCELLED},
    JobState.RUNNING: {
        JobState.COMPLETED,
        JobState.FAILED,
        JobState.RETRY_PENDING,
        JobState.CANCELLED,
    },
    JobState.RETRY_PENDING: {JobState.READY, JobState.CANCELLED, JobState.FAILED},
    JobState.COMPLETED: {JobState.RETRY_PENDING},
    JobState.FAILED: set(),
    JobState.CANCELLED: set(),
}


@dataclass
class Job:
    spec: JobSpec
    state: JobState = JobState.PENDING
    attempts: int = 0
    reasoning_retries: int = 0
    runtime_retries: int = 0
    output: str | None = None
    rejected_output: str | None = None
    feedback: Tuple[str, ...] = ()
    failure_kind: FailureKind | None = None
    failure: str | None = None
    critique: Critique | None = None
    evidence_refs: Tuple[str, ...] = ()
    execute_ms: float = 0.0
    critique_ms: float = 0.0
    history: List[JobState] = field(default_factory=list)

    @property
    def job_id(self) -> str:
        return self.spec.job_id

    def outcome(self) -> JobOutcome:
        return JobOutcome(
            job_id=self.spec.job_id,
            kind=self.spec.kind,
            description=self.spec.description,
            state=self.state,
            output=self.output if self.state is JobState.COMPLETED else None,
            attempts=self.attempts,
            reasoning_retries=self.reasoning_retries,
            runtime_retries=self.runtime_retries,
            failure_kind=self.failure_kind,
            failure=self.failure,
            evidence_refs=self.evidence_refs,
            critique=self.critique,
            execute_ms=self.execute_ms,
            critique_ms=self.critique_ms,
        )


def topological_order(jobs: Iterable[JobSpec]) -> List[str]:
    """Job ids, dependencies first, ties kept in plan order; raises on any defect."""
    jobs = list(jobs)
    if not jobs:
        raise InvalidPlan("a plan needs at least one job")
    ids = [job.job_id for job in jobs]
    seen = set()
    for job_id in ids:
        if not isinstance(job_id, str) or not job_id.strip():
            raise InvalidPlan(f"job id {job_id!r} is empty")
        if job_id in seen:
            raise InvalidPlan(f"duplicate job id {job_id!r}")
        seen.add(job_id)
    position = {job_id: index for index, job_id in enumerate(ids)}
    waiting = {}
    for job in jobs:
        for dependency in job.depends_on:
            if dependency == job.job_id:
                raise InvalidPlan(f"job {job.job_id!r} depends on itself")
            if dependency not in position:
                raise InvalidPlan(f"job {job.job_id!r} depends on unknown job {dependency!r}")
        waiting[job.job_id] = set(job.depends_on)
    order: List[str] = []
    while waiting:
        ready = sorted((j for j, deps in waiting.items() if not deps), key=position.get)
        if not ready:
            raise InvalidPlan(f"dependency cycle among {sorted(waiting)}")
        for job_id in ready:
            order.append(job_id)
            del waiting[job_id]
        for deps in waiting.values():
            deps.difference_update(ready)
    return order


class JobPool:
    """Validates the plan's graph once, then enforces every state change under one lock."""

    def __init__(self, plan: Plan) -> None:
        self.plan = plan
        self.order = topological_order(plan.jobs)
        self.__jobs: Dict[str, Job] = {spec.job_id: Job(spec) for spec in plan.jobs}
        self.__lock = threading.RLock()

    def __job(self, job_id: str) -> Job:
        try:
            return self.__jobs[job_id]
        except KeyError:
            raise UnknownJob(job_id) from None

    def get(self, job_id: str) -> Job:
        return self.__job(job_id)

    def jobs(self) -> List[Job]:
        """Every job, in dependency order."""
        return [self.__jobs[job_id] for job_id in self.order]

    def transition(self, job_id: str, new_state: JobState, **changes) -> Job:
        with self.__lock:
            job = self.__job(job_id)
            if new_state not in _ALLOWED[job.state]:
                raise InvalidTransition(job_id, job.state, new_state)
            for name, value in changes.items():
                if not hasattr(job, name):
                    raise AttributeError(f"Job has no field {name!r}")
                setattr(job, name, value)
            job.history.append(job.state)
            job.state = new_state
            return job

    def claim(self, job_id: str) -> bool:
        """READY -> RUNNING, once: a second claim of the same job returns False."""
        with self.__lock:
            job = self.__job(job_id)
            if job.state is not JobState.READY:
                return False
            job.history.append(job.state)
            job.state = JobState.RUNNING
            job.attempts += 1
            return True

    def dependents(self, job_id: str) -> List[str]:
        """Every job that depends on `job_id`, directly or through others, in order."""
        self.__job(job_id)
        found = {job_id}
        for candidate in self.order:
            if found.intersection(self.__jobs[candidate].spec.depends_on):
                found.add(candidate)
        found.discard(job_id)
        return [j for j in self.order if j in found]

    def states(self) -> Dict[str, JobState]:
        with self.__lock:
            return {job_id: self.__jobs[job_id].state for job_id in self.order}

    def all_terminal(self) -> bool:
        with self.__lock:
            return all(job.state in TERMINAL_STATES for job in self.__jobs.values())

    def outcomes(self) -> Tuple[JobOutcome, ...]:
        with self.__lock:
            return tuple(self.__jobs[job_id].outcome() for job_id in self.order)

    def kind_of(self, job_id: str) -> JobKind:
        return self.__job(job_id).spec.kind
