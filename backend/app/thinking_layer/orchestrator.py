"""One reasoning request from plan to accepted results. It coordinates; each part does its own work."""

import threading
import time
from typing import Callable, Dict, List, NamedTuple, Tuple

from config import get_logger, log_context
from inference.inference_exceptions import PromptTooLarge
from inference.model_port import WORKER, TextModel
from thinking_layer.accumulator import Accumulator
from thinking_layer.critic import JobCritic, PlanCritic
from thinking_layer.job_pool import JobPool
from thinking_layer.models import (
    Critique,
    FailureKind,
    JobKind,
    JobSpec,
    JobState,
    PlanCritique,
    PlanVerdict,
    RetryEvent,
    RetryStats,
    ThinkingRequest,
    ThinkingResult,
    ThinkingStatus,
    Verdict,
    execution_path,
)
from thinking_layer.planner import Planner
from thinking_layer.prompt_text import (
    ThinkingPrompts,
    dependency_block,
    evidence_block,
    join_blocks,
    load_thinking_prompts,
    memory_block,
    request_block,
)
from thinking_layer.scheduler import Scheduler
from thinking_layer.settings import ThinkingSettings
from thinking_layer.worker_pool import WorkerPool

logger = get_logger(__name__)


class CountingModel:
    """Counts the calls one request makes, by purpose, without changing them."""

    def __init__(self, model: TextModel) -> None:
        self.model = model
        self.calls: Dict[str, int] = {}
        self.__lock = threading.Lock()

    def infer(self, system, user, **options) -> str:
        purpose = options.get("purpose", "draft")
        with self.__lock:
            self.calls[purpose] = self.calls.get(purpose, 0) + 1
        return self.model.infer(system, user, **options)

    @property
    def total(self) -> int:
        with self.__lock:
            return sum(self.calls.values())


class _Attempt(NamedTuple):
    output: str
    critique: Critique
    execute_ms: float
    critique_ms: float


class _Run:
    """Request-scoped state, discarded when the request ends."""

    def __init__(self, request, plan, settings, deadline_at, cancel) -> None:
        self.request = request
        self.plan = plan
        self.pool = JobPool(plan)
        self.scheduler = Scheduler(self.pool)
        self.accumulator = Accumulator(request.evidence)
        self.deadline_at = deadline_at
        self.cancel = cancel
        self.events: List[RetryEvent] = []
        self.critiques: List[PlanCritique] = []
        self.limitations: List[str] = list(request.limitations)
        self.plan_revisions = 0
        self.exhausted = False
        self.timed_out = False
        self.cancelled = False


class ThinkingOrchestrator:
    def __init__(
        self,
        model: TextModel,
        prompts: ThinkingPrompts | None = None,
        settings: ThinkingSettings | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.model = model
        self.prompts = load_thinking_prompts() if prompts is None else prompts
        self.settings = (ThinkingSettings() if settings is None else settings).validated()
        self.clock = clock

    def think(
        self,
        request: ThinkingRequest,
        deadline_at: float | None = None,
        cancel: threading.Event | None = None,
    ) -> ThinkingResult:
        started = self.clock()
        if deadline_at is None:
            deadline_at = started + self.settings.deadline_seconds
        model = CountingModel(self.model)
        decomposition = request.decomposition
        path = execution_path(decomposition.mode, decomposition.expanded)
        timings: Dict[str, float] = {}
        with log_context(request_id=request.request_id):
            planner = Planner(model, self.prompts, self.settings)
            plan = planner.plan(request, path)
            timings["planning"] = self.__since(started)
            run = _Run(request, plan, self.settings, deadline_at, cancel or threading.Event())
            job_critic = JobCritic(model, self.prompts, self.settings)
            plan_critic = PlanCritic(model, self.prompts, self.settings)

            execution_started = self.clock()
            plan_critique_ms = 0.0
            with WorkerPool(self.settings.max_workers) as workers:
                while True:
                    self.__run_jobs(run, model, job_critic, workers)
                    if run.timed_out or run.cancelled or len(plan.jobs) == 1:
                        break
                    critique_started = self.clock()
                    critique = plan_critic.review(
                        request,
                        run.accumulator.aggregate(run.pool.order),
                        request.evidence,
                    )
                    plan_critique_ms += self.__since(critique_started)
                    run.critiques.append(critique)
                    if not self.__revise(run, critique, path):
                        break
            timings["execution"] = self.__since(execution_started) - plan_critique_ms
            timings["plan_critique"] = plan_critique_ms
            outcomes = run.pool.outcomes()
            timings["job_execute_sum"] = sum(o.execute_ms for o in outcomes)
            timings["job_critique_sum"] = sum(o.critique_ms for o in outcomes)
            timings["total"] = self.__since(started)
            result = self.__result(run, path, outcomes, timings, model.total)
            logger.info(
                "Thinking %s: %s path, %d job(s), %d accepted, %d reasoning retr%s, %d model call(s)",
                result.status.value,
                path.value,
                len(outcomes),
                len(result.accepted),
                result.retries.reasoning_retries,
                "y" if result.retries.reasoning_retries == 1 else "ies",
                result.model_calls,
            )
            return result

    def __since(self, start: float) -> float:
        return (self.clock() - start) * 1000.0

    def __run_jobs(self, run: _Run, model, critic: JobCritic, workers: WorkerPool) -> None:
        while True:
            if run.cancel.is_set():
                run.cancelled = True
                self.__stop(run, workers, FailureKind.DEADLINE, "request cancelled")
                return
            remaining = run.deadline_at - self.clock()
            if remaining <= 0:
                run.timed_out = True
                self.__stop(run, workers, FailureKind.DEADLINE, "request deadline passed")
                return
            for job_id in run.scheduler.promote():
                if workers.free < 1:
                    break
                if run.pool.claim(job_id):
                    task = self.__task(run, model, critic, run.pool.get(job_id))
                    workers.submit(job_id, task)
            if not workers.running:
                return
            for job_id, future in workers.wait_any(timeout=min(remaining, 1.0)):
                error = future.exception()
                self.__settle(run, job_id, None if error else future.result(), error)

    def __stop(self, run: _Run, workers: WorkerPool, kind: FailureKind, reason: str) -> None:
        for job_id in workers.abandon():
            run.pool.transition(job_id, JobState.CANCELLED, failure_kind=kind, failure=reason)
        run.scheduler.cancel_open(kind, reason)
        run.limitations.append(f"Reasoning stopped early: {reason}")

    def __task(self, run: _Run, model, critic: JobCritic, job) -> Callable[[], _Attempt]:
        """The job's prompt is fixed here, on the scheduling thread, from accepted results only."""
        spec: JobSpec = job.spec
        settings = self.settings
        dependencies = run.accumulator.dependency_context(spec.depends_on, settings.dependency_chars)
        system = self.prompts.stitch if spec.kind is JobKind.STITCH else self.prompts.worker
        feedback = ""
        if job.feedback:
            previous = (
                f"\nYour previous result was:\n{job.rejected_output}"
                if job.rejected_output else ""
            )
            feedback = (
                "<reviewer_feedback>\n" + "\n".join(f"- {item}" for item in job.feedback)
                + previous + "\n</reviewer_feedback>"
            )
        limits = (
            "<known_limitations>\n" + "\n".join(run.request.limitations) + "\n</known_limitations>"
            if run.request.limitations else ""
        )
        user = join_blocks(
            request_block(run.request.decomposition, settings.query_chars),
            memory_block(run.request.memory, settings.memory_chars),
            f"<job id=\"{spec.job_id}\">\nTask: {spec.description}\n"
            f"Your result must contain: {spec.expected_output}\n</job>",
            dependency_block(dependencies),
            evidence_block(run.request.evidence, settings.evidence_chars),
            limits,
            feedback,
        )
        max_tokens = settings.stitch_max_tokens if spec.kind is JobKind.STITCH else settings.job_max_tokens
        request = run.request
        evidence = request.evidence
        clock = self.clock

        def execute() -> _Attempt:
            started = clock()
            output = model.infer(system, user, max_tokens=max_tokens, purpose=WORKER).strip()
            executed = clock()
            critique = critic.review(request, spec, output, evidence)
            return _Attempt(
                output, critique, (executed - started) * 1000.0, (clock() - executed) * 1000.0
            )

        return execute

    def __settle(self, run: _Run, job_id: str, attempt: _Attempt | None, error) -> None:
        pool = run.pool
        job = pool.get(job_id)
        if job.state is not JobState.RUNNING:
            return
        if attempt is not None:
            job.execute_ms += attempt.execute_ms
            job.critique_ms += attempt.critique_ms
        if error is not None:
            retryable = not isinstance(error, PromptTooLarge)
            if retryable and job.runtime_retries < self.settings.max_runtime_retries:
                run.events.append(
                    RetryEvent("job", job_id, FailureKind.MODEL, str(error), job.runtime_retries + 1)
                )
                pool.transition(job_id, JobState.RETRY_PENDING,
                                runtime_retries=job.runtime_retries + 1)
                logger.warning("Job %s failed in the model (%s); retrying", job_id, error)
            else:
                pool.transition(job_id, JobState.FAILED, failure_kind=FailureKind.MODEL,
                                failure=str(error))
                logger.error("Job %s failed in the model: %s", job_id, error)
            return

        critique = attempt.critique
        if critique.verdict is Verdict.ACCEPT:
            refs = run.accumulator.accept(job_id, attempt.output)
            pool.transition(job_id, JobState.COMPLETED, output=attempt.output,
                            critique=critique, evidence_refs=refs, feedback=(),
                            rejected_output=None)
            if not critique.reviewed:
                run.limitations.append(f"Job {job_id} could not be reviewed")
            return

        reason = "; ".join(critique.reasons) or critique.verdict.value
        if job.reasoning_retries < self.settings.max_job_retries:
            run.events.append(
                RetryEvent("job", job_id, FailureKind.REASONING, reason, job.reasoning_retries + 1)
            )
            pool.transition(
                job_id,
                JobState.RETRY_PENDING,
                reasoning_retries=job.reasoning_retries + 1,
                critique=critique,
                feedback=critique.reasons + critique.missing,
                rejected_output=attempt.output if critique.verdict is Verdict.REVISE else None,
            )
            logger.info("Job %s rejected by the critic (%s); retrying", job_id, reason)
        else:
            run.exhausted = True
            pool.transition(
                job_id,
                JobState.FAILED,
                critique=critique,
                failure_kind=FailureKind.REASONING,
                failure=f"rejected after {job.attempts} attempt(s): {reason}",
            )
            logger.warning("Job %s rejected with no retries left: %s", job_id, reason)

    def __revise(self, run: _Run, critique: PlanCritique, path) -> bool:
        """Reopen only the jobs the plan critique named, and what depends on them."""
        if critique.verdict is PlanVerdict.ACCEPT:
            if not critique.reviewed:
                run.limitations.append("The combined result could not be reviewed")
            return False
        budget = self.settings.max_plan_revisions.get(path, 0)
        reopenable = [
            job_id for job_id in critique.affected_jobs
            if run.pool.get(job_id).state is JobState.COMPLETED
        ]
        if critique.verdict is PlanVerdict.LIMIT or not reopenable or run.plan_revisions >= budget:
            if critique.verdict is PlanVerdict.REVISE:
                run.exhausted = run.exhausted or run.plan_revisions >= budget
            run.accumulator.mark_unresolved(critique.missing or critique.reasons)
            run.limitations.extend(
                f"Not fully resolved: {item}" for item in (critique.missing or critique.reasons)
            )
            return False
        run.plan_revisions += 1
        feedback = critique.missing + critique.reasons
        reopened: List[str] = []
        for job_id in reopenable:
            for target in [job_id] + run.pool.dependents(job_id):
                job = run.pool.get(target)
                if target in reopened or job.state is not JobState.COMPLETED:
                    continue
                run.accumulator.retract(target)
                run.pool.transition(target, JobState.RETRY_PENDING, feedback=feedback,
                                    rejected_output=job.output)
                reopened.append(target)
        run.events.append(
            RetryEvent("plan", None, FailureKind.REASONING,
                       "; ".join(feedback) or "plan critique", run.plan_revisions)
        )
        logger.info("Plan revision %d reopens %s", run.plan_revisions, ", ".join(reopened))
        return True

    def __result(self, run: _Run, path, outcomes, timings, model_calls) -> ThinkingResult:
        accepted = tuple(o for o in outcomes if o.state is JobState.COMPLETED)
        failed = [o for o in outcomes if o.state in (JobState.FAILED, JobState.CANCELLED)]
        limitations = list(run.limitations)
        for outcome in failed:
            limitations.append(f"Job {outcome.job_id} ({outcome.description}) did not complete: "
                               f"{outcome.failure}")
        if run.cancelled:
            status = ThinkingStatus.CANCELLED
        elif run.timed_out:
            status = ThinkingStatus.TIMED_OUT
        elif not accepted:
            status = ThinkingStatus.FAILED
        elif failed or limitations:
            status = ThinkingStatus.LIMITED
        else:
            status = ThinkingStatus.COMPLETED
        job_events = [e for e in run.events if e.level == "job"]
        retries = RetryStats(
            job_retries=sum(1 for e in job_events if e.kind is FailureKind.REASONING),
            plan_revisions=run.plan_revisions,
            runtime_retries=sum(1 for e in job_events if e.kind is FailureKind.MODEL),
            events=tuple(run.events),
            exhausted=run.exhausted,
        )
        decomposition = run.request.decomposition
        return ThinkingResult(
            request_id=run.request.request_id,
            conversation_id=run.request.conversation_id,
            original_query=decomposition.query,
            effective_query=decomposition.working_query,
            mode=decomposition.mode,
            path=path,
            status=status,
            plan=run.plan,
            jobs=outcomes,
            accepted=accepted,
            evidence=run.accumulator.cited(),
            critiques=tuple(run.critiques),
            limitations=tuple(dict.fromkeys(limitations)),
            retries=retries,
            timings=timings,
            model_calls=model_calls,
        )
