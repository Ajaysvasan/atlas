"""What moves into, through and out of the thinking layer."""

from enum import Enum
from typing import Dict, NamedTuple, Tuple

from query_decomposition.models import Decomposition, Mode


class ExecutionPath(Enum):
    """How much reasoning a request gets, derived from what decomposition decided."""

    DIRECT = "direct"
    PLANNED = "planned"
    EXPANDED = "expanded"


def execution_path(mode: Mode, expanded: bool) -> ExecutionPath:
    if mode is Mode.DOUBT:
        return ExecutionPath.DIRECT
    return ExecutionPath.EXPANDED if expanded else ExecutionPath.PLANNED


class JobKind(Enum):
    ANSWER = "answer"
    STEP = "step"
    WORKER = "worker"
    STITCH = "stitch"


class JobState(Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY_PENDING = "retry_pending"
    CANCELLED = "cancelled"


TERMINAL_STATES = frozenset({JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED})


class FailureKind(Enum):
    REASONING = "reasoning"
    MODEL = "model"
    RETRIEVAL = "retrieval"
    DEPENDENCY = "dependency"
    DEADLINE = "deadline"


class Verdict(Enum):
    ACCEPT = "accept"
    REVISE = "revise"
    FAIL = "fail"


class PlanVerdict(Enum):
    ACCEPT = "accept"
    REVISE = "revise"
    LIMIT = "limit"


class Evidence(NamedTuple):
    """One retrieved passage, tagged so reasoning can cite it."""

    ref: str
    source: str
    text: str
    score: float
    vector_id: int | None = None
    chunk_id: str | None = None
    document_id: str | None = None
    origin: str | None = None


class MemoryContext(NamedTuple):
    """What the conversation already holds: its rolling summary and latest turns."""

    summary: str | None = None
    turns: Tuple[Tuple[str, str], ...] = ()


class ThinkingRequest(NamedTuple):
    request_id: str
    conversation_id: str
    decomposition: Decomposition
    evidence: Tuple[Evidence, ...] = ()
    memory: MemoryContext = MemoryContext()
    limitations: Tuple[str, ...] = ()


class JobSpec(NamedTuple):
    job_id: str
    kind: JobKind
    description: str
    expected_output: str
    depends_on: Tuple[str, ...] = ()
    priority: int = 0


class Plan(NamedTuple):
    plan_id: str
    request_id: str
    path: ExecutionPath
    jobs: Tuple[JobSpec, ...]
    source: str


class Critique(NamedTuple):
    verdict: Verdict
    reasons: Tuple[str, ...] = ()
    missing: Tuple[str, ...] = ()
    reviewed: bool = True


class PlanCritique(NamedTuple):
    verdict: PlanVerdict
    affected_jobs: Tuple[str, ...] = ()
    missing: Tuple[str, ...] = ()
    reasons: Tuple[str, ...] = ()
    reviewed: bool = True


class RetryEvent(NamedTuple):
    level: str
    job_id: str | None
    kind: FailureKind
    reason: str
    attempt: int


class JobOutcome(NamedTuple):
    job_id: str
    kind: JobKind
    description: str
    state: JobState
    output: str | None
    attempts: int
    reasoning_retries: int
    runtime_retries: int
    failure_kind: FailureKind | None
    failure: str | None
    evidence_refs: Tuple[str, ...]
    critique: Critique | None
    execute_ms: float
    critique_ms: float


class RetryStats(NamedTuple):
    job_retries: int
    plan_revisions: int
    runtime_retries: int
    events: Tuple[RetryEvent, ...]
    exhausted: bool

    @property
    def reasoning_retries(self) -> int:
        """Planner-critique retries: job revisions plus plan revisions, never runtime ones."""
        return self.job_retries + self.plan_revisions


class ThinkingStatus(Enum):
    COMPLETED = "completed"
    LIMITED = "completed_with_limitations"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class ThinkingResult(NamedTuple):
    request_id: str
    conversation_id: str
    original_query: str
    effective_query: str
    mode: Mode
    path: ExecutionPath
    status: ThinkingStatus
    plan: Plan
    jobs: Tuple[JobOutcome, ...]
    accepted: Tuple[JobOutcome, ...]
    evidence: Tuple[Evidence, ...]
    critiques: Tuple[PlanCritique, ...]
    limitations: Tuple[str, ...]
    retries: RetryStats
    timings: Dict[str, float]
    model_calls: int

    @property
    def final_output(self) -> str | None:
        """The stitched result when there is one, else None: generation combines the rest."""
        for outcome in reversed(self.accepted):
            if outcome.kind is JobKind.STITCH:
                return outcome.output
        return None
