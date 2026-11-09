"""Plans as graphs: validation, the job state machine, scheduling and accumulation."""

import pytest

from thinking_layer.accumulator import Accumulator, cited_refs, unknown_refs
from thinking_layer.job_pool import JobPool, topological_order
from thinking_layer.models import (
    ExecutionPath,
    FailureKind,
    JobKind,
    JobSpec,
    JobState,
    Plan,
    execution_path,
)
from thinking_layer.replies import UnusableReply, read_job_critique, read_plan, read_plan_critique
from thinking_layer.scheduler import Scheduler
from thinking_layer.thinking_exceptions import InvalidPlan, InvalidTransition, UnknownJob
from query_decomposition.models import Mode
from test.thinking_testing.fakes import EVIDENCE


def spec(job_id, *deps, priority=0):
    return JobSpec(job_id, JobKind.STEP, f"do {job_id}", "a result", tuple(deps), priority)


def plan(*jobs):
    return Plan("p1", "r1", ExecutionPath.PLANNED, tuple(jobs), "test")


class TestValidation:
    def test_dependencies_come_first_and_ties_keep_plan_order(self):
        assert topological_order([spec("c", "a"), spec("a"), spec("b")]) == ["a", "b", "c"]

    @pytest.mark.parametrize(
        "jobs, reason",
        [
            ([spec("a"), spec("a")], "duplicate"),
            ([spec("a", "zz")], "unknown"),
            ([spec("a", "a")], "itself"),
            ([spec("a", "b"), spec("b", "a")], "cycle"),
            ([spec("a", "c"), spec("b", "a"), spec("c", "b")], "cycle"),
            ([], "at least one"),
        ],
    )
    def test_unschedulable_graphs_are_refused(self, jobs, reason):
        with pytest.raises(InvalidPlan, match=reason):
            topological_order(jobs)

    def test_the_pool_refuses_an_invalid_plan(self):
        with pytest.raises(InvalidPlan):
            JobPool(plan(spec("a", "b"), spec("b", "a")))


class TestStateMachine:
    def test_a_job_is_claimed_once(self):
        pool = JobPool(plan(spec("a")))
        Scheduler(pool).promote()
        assert pool.claim("a") is True
        assert pool.claim("a") is False
        assert pool.get("a").attempts == 1

    def test_a_job_cannot_run_before_it_is_ready(self):
        pool = JobPool(plan(spec("a")))
        assert pool.claim("a") is False
        with pytest.raises(InvalidTransition):
            pool.transition("a", JobState.RUNNING)

    def test_failed_is_final(self):
        pool = JobPool(plan(spec("a")))
        Scheduler(pool).promote()
        pool.claim("a")
        pool.transition("a", JobState.FAILED)
        for state in (JobState.COMPLETED, JobState.READY, JobState.RETRY_PENDING):
            with pytest.raises(InvalidTransition):
                pool.transition("a", state)

    def test_a_failed_job_is_not_reported_with_its_output(self):
        pool = JobPool(plan(spec("a")))
        Scheduler(pool).promote()
        pool.claim("a")
        pool.transition("a", JobState.FAILED, output="half an answer")
        assert pool.outcomes()[0].output is None

    def test_unknown_jobs_are_named(self):
        with pytest.raises(UnknownJob):
            JobPool(plan(spec("a"))).get("b")

    def test_dependents_are_transitive(self):
        pool = JobPool(plan(spec("a"), spec("b", "a"), spec("c", "b"), spec("d")))
        assert pool.dependents("a") == ["b", "c"]


class TestScheduler:
    def test_only_jobs_whose_prerequisites_completed_are_ready(self):
        pool = JobPool(plan(spec("a"), spec("b", "a"), spec("c")))
        scheduler = Scheduler(pool)
        assert scheduler.promote() == ["a", "c"]
        pool.claim("a")
        pool.transition("a", JobState.COMPLETED)
        # Dependency order puts c (no prerequisites) before b.
        assert scheduler.promote() == ["c", "b"]

    def test_higher_priority_runs_first(self):
        pool = JobPool(plan(spec("a"), spec("b", priority=5)))
        assert Scheduler(pool).promote() == ["b", "a"]

    def test_a_failed_prerequisite_cancels_everything_after_it(self):
        pool = JobPool(plan(spec("a"), spec("b", "a"), spec("c", "b"), spec("d")))
        scheduler = Scheduler(pool)
        scheduler.promote()
        pool.claim("a")
        pool.transition("a", JobState.FAILED)
        scheduler.promote()
        states = pool.states()
        assert states["b"] is JobState.CANCELLED and states["c"] is JobState.CANCELLED
        assert pool.get("b").failure_kind is FailureKind.DEPENDENCY
        assert states["d"] is JobState.READY

    def test_cancel_open_leaves_finished_jobs_alone(self):
        pool = JobPool(plan(spec("a"), spec("b")))
        scheduler = Scheduler(pool)
        scheduler.promote()
        pool.claim("a")
        pool.transition("a", JobState.COMPLETED)
        assert scheduler.cancel_open(FailureKind.DEADLINE, "late") == ["b"]
        assert pool.get("a").state is JobState.COMPLETED


class TestAccumulator:
    def test_only_real_evidence_is_counted_as_cited(self):
        assert cited_refs("see [G2] and [G9] and [G2]", ["G1", "G2"]) == ("G2",)
        assert unknown_refs("see [G2] and [G9]", ["G1", "G2"]) == ("G9",)

    def test_a_retracted_result_is_not_handed_to_dependents(self):
        acc = Accumulator(EVIDENCE)
        acc.accept("a", "result [G1]")
        acc.retract("a")
        assert acc.dependency_context(["a"], 1000) == []
        assert acc.cited() == ()

    def test_dependency_context_is_bounded(self):
        acc = Accumulator(EVIDENCE)
        acc.accept("a", "x" * 5000)
        acc.accept("b", "y" * 5000)
        context = acc.dependency_context(["a", "b"], 1000)
        assert sum(len(text) for _, text in context) <= 1000

    def test_cited_evidence_keeps_its_provenance(self):
        acc = Accumulator(EVIDENCE)
        acc.accept("a", "per [G2] and [G1]")
        assert [e.ref for e in acc.cited()] == ["G1", "G2"]
        assert acc.cited()[0].chunk_id == "c1"


class TestReplies:
    def test_plan_ids_are_normalised_and_dependencies_follow(self):
        jobs = read_plan(
            '{"jobs": [{"id": "x", "task": "t1", "expected_output": "o", "depends_on": []},'
            ' {"id": "y", "task": "t2", "expected_output": "", "depends_on": ["x"]}]}', 3)
        assert [j["job_id"] for j in jobs] == ["j1", "j2"]
        assert jobs[1]["depends_on"] == ("j1",)
        assert jobs[1]["expected_output"] == "t2"

    @pytest.mark.parametrize("reply", [
        "not json",
        '{"jobs": []}',
        '{"jobs": [{"id": "a", "task": "", "depends_on": []}]}',
        '{"jobs": [{"id": "a", "task": "t", "depends_on": ["b"]}]}',
        '{"jobs": [{"id": "a", "task": "t"}, {"id": "a", "task": "u"}]}',
    ])
    def test_unusable_plans_are_refused(self, reply):
        with pytest.raises(UnusableReply):
            read_plan(reply, 3)

    def test_more_jobs_than_allowed_are_refused(self):
        jobs = ",".join(f'{{"id": "j{i}", "task": "t", "depends_on": []}}' for i in range(4))
        with pytest.raises(UnusableReply):
            read_plan(f'{{"jobs": [{jobs}]}}', 3)

    def test_critiques_parse_and_unknown_affected_jobs_are_dropped(self):
        assert read_job_critique('{"verdict": "REVISE", "reasons": ["x"]}').verdict.value == "revise"
        critique = read_plan_critique(
            '{"verdict": "revise", "affected_jobs": ["j1", "nope"], "missing": ["m"]}', ["j1"])
        assert critique.affected_jobs == ("j1",)
        with pytest.raises(UnusableReply):
            read_job_critique('{"verdict": "maybe"}')


@pytest.mark.parametrize("mode, expanded, path", [
    (Mode.DOUBT, False, ExecutionPath.DIRECT),
    (Mode.DOUBT, True, ExecutionPath.DIRECT),
    (Mode.DEVELOPMENT, False, ExecutionPath.PLANNED),
    (Mode.DEVELOPMENT, True, ExecutionPath.EXPANDED),
])
def test_execution_path_follows_the_decomposer(mode, expanded, path):
    assert execution_path(mode, expanded) is path
