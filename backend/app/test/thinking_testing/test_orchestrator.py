"""The orchestrator end to end over a scripted model: paths, critique at both levels, retries, limits."""

import json
import threading

import pytest

from inference.inference_exceptions import InferenceError, PromptTooLarge
from query_decomposition.models import Mode
from thinking_layer.models import ExecutionPath, FailureKind, JobKind, JobState, ThinkingStatus
from thinking_layer.orchestrator import ThinkingOrchestrator
from thinking_layer.settings import ThinkingSettings
from test.thinking_testing.fakes import (
    PROMPTS,
    ScriptedModel,
    accept,
    decomposition,
    plan_accept,
    plan_revise,
    request,
    revise,
)


def orchestrator(model, **settings):
    return ThinkingOrchestrator(model, PROMPTS, ThinkingSettings(**settings))


def three_step_plan(user, number):
    return json.dumps({"jobs": [
        {"id": "a", "task": "explain MVCC", "expected_output": "mechanism", "depends_on": []},
        {"id": "b", "task": "explain VACUUM", "expected_output": "cleanup", "depends_on": []},
        {"id": "c", "task": "relate them", "expected_output": "link", "depends_on": ["a", "b"]},
    ]})


class TestScenarioA_SimpleQuery:
    def test_a_doubt_is_one_job_with_no_planner_and_no_plan_critique(self):
        model = ScriptedModel()
        result = orchestrator(model).think(request())
        assert result.path is ExecutionPath.DIRECT
        assert result.status is ThinkingStatus.COMPLETED
        assert [j.kind for j in result.jobs] == [JobKind.ANSWER]
        assert model.count("planner") == 0 and model.count("plan_critic") == 0
        assert result.model_calls == 2  # one worker, one job critique
        assert result.retries.reasoning_retries == 0
        assert [e.ref for e in result.evidence] == ["G1"]

    def test_the_original_request_reaches_the_worker_verbatim(self):
        model = ScriptedModel()
        orchestrator(model).think(request(decomposition("Explain MVCC. Use exactly 3 bullets.")))
        worker = next(c for c in model.calls if c["role"] == "worker")
        assert "Explain MVCC. Use exactly 3 bullets." in worker["user"]


class TestScenarioB_ComplexQuery:
    def test_dependent_jobs_run_after_and_see_only_accepted_results(self):
        model = ScriptedModel({"planner": three_step_plan})
        dec = decomposition("Design MVCC cleanup", Mode.DEVELOPMENT)
        result = orchestrator(model).think(request(dec))
        assert result.path is ExecutionPath.PLANNED
        assert result.status is ThinkingStatus.COMPLETED
        assert [j.job_id for j in result.accepted] == ["j1", "j2", "j3"]
        last_worker = [c for c in model.calls if c["role"] == "worker"][-1]
        assert "Result of j1" in last_worker["user"] and "Result of j2" in last_worker["user"]
        assert model.count("plan_critic") == 1

    def test_an_expanded_query_gets_one_worker_per_sub_query_and_a_stitch(self):
        model = ScriptedModel()
        dec = decomposition("Build a physics engine in C", Mode.DEVELOPMENT,
                            sub_queries=("vector maths", "collision detection"))
        result = orchestrator(model).think(request(dec))
        assert result.path is ExecutionPath.EXPANDED
        assert [j.kind for j in result.jobs] == [JobKind.WORKER, JobKind.WORKER, JobKind.STITCH]
        assert model.count("planner") == 0
        assert model.count("stitch") == 1
        assert result.final_output is not None

    def test_an_unusable_plan_falls_back_to_one_answer(self):
        model = ScriptedModel({"planner": lambda u, n: "garbage"})
        result = orchestrator(model).think(request(decomposition("Design X", Mode.DEVELOPMENT)))
        assert result.plan.source == "fallback"
        assert result.status is ThinkingStatus.COMPLETED


class TestScenarioC_JobCritique:
    def test_a_rejected_job_is_retried_once_with_the_feedback(self):
        model = ScriptedModel({"job_critic": lambda u, n: revise("misses vacuum") if n == 0 else accept()})
        result = orchestrator(model).think(request())
        assert result.status is ThinkingStatus.COMPLETED
        assert result.retries.job_retries == 1
        assert result.jobs[0].attempts == 2
        retry_prompt = [c for c in model.calls if c["role"] == "worker"][1]["user"]
        assert "misses vacuum" in retry_prompt and "Your previous result was" in retry_prompt

    def test_retries_are_bounded_and_exhaustion_fails_the_job(self):
        model = ScriptedModel({"job_critic": lambda u, n: revise()})
        result = orchestrator(model, max_job_retries=2).think(request())
        assert result.status is ThinkingStatus.FAILED
        assert result.retries.job_retries == 2
        assert result.retries.exhausted
        assert model.count("worker") == 3
        assert result.jobs[0].state is JobState.FAILED
        assert result.jobs[0].failure_kind is FailureKind.REASONING
        assert result.accepted == ()

    def test_a_rejected_prerequisite_never_feeds_its_dependents(self):
        def critic(user, n):
            return revise() if "relate" not in user and "explain MVCC" in user else accept()

        model = ScriptedModel({"planner": three_step_plan, "job_critic": critic})
        result = orchestrator(model, max_job_retries=1).think(
            request(decomposition("Design it", Mode.DEVELOPMENT)))
        states = {j.job_id: j.state for j in result.jobs}
        assert states == {"j1": JobState.FAILED, "j2": JobState.COMPLETED, "j3": JobState.CANCELLED}
        assert result.status is ThinkingStatus.LIMITED
        assert all("Result of j1" not in c["user"] for c in model.calls if c["role"] == "worker")

    def test_made_up_citations_are_rejected_without_asking_the_model(self):
        model = ScriptedModel({"worker": lambda u, n: "Long enough answer citing [G7]." if n == 0
                               else "Long enough answer citing [G1]."})
        result = orchestrator(model).think(request())
        assert result.retries.job_retries == 1
        assert model.count("job_critic") == 1  # only the second, valid answer reached the critic

    def test_an_unreadable_critique_accepts_and_says_so(self):
        model = ScriptedModel({"job_critic": lambda u, n: "not json"})
        result = orchestrator(model).think(request())
        assert result.status is ThinkingStatus.LIMITED
        assert any("could not be reviewed" in item for item in result.limitations)


class TestScenarioD_PlanCritique:
    def test_only_the_named_job_and_its_dependents_are_redone(self):
        model = ScriptedModel({
            "planner": three_step_plan,
            "plan_critic": lambda u, n: plan_revise("j2") if n == 0 else plan_accept(),
        })
        result = orchestrator(model).think(request(decomposition("Design it", Mode.DEVELOPMENT)))
        attempts = {j.job_id: j.attempts for j in result.jobs}
        assert attempts == {"j1": 1, "j2": 2, "j3": 2}
        assert result.retries.plan_revisions == 1
        assert result.retries.job_retries == 0
        assert result.status is ThinkingStatus.COMPLETED

    def test_plan_revisions_are_bounded_and_end_as_a_limitation(self):
        model = ScriptedModel({"planner": three_step_plan,
                               "plan_critic": lambda u, n: plan_revise("j1", missing="no numbers")})
        result = orchestrator(model).think(request(decomposition("Design it", Mode.DEVELOPMENT)))
        assert result.retries.plan_revisions == 1
        assert model.count("plan_critic") == 2
        assert result.status is ThinkingStatus.LIMITED
        assert result.retries.exhausted
        assert any("no numbers" in item for item in result.limitations)

    def test_a_revision_naming_no_job_is_not_a_blind_rerun(self):
        model = ScriptedModel({"planner": three_step_plan,
                               "plan_critic": lambda u, n: plan_revise(missing="unclear")})
        result = orchestrator(model).think(request(decomposition("Design it", Mode.DEVELOPMENT)))
        assert result.retries.plan_revisions == 0
        assert all(j.attempts == 1 for j in result.jobs)


class TestRuntimeFailures:
    def test_a_model_failure_is_retried_apart_from_reasoning_retries(self):
        def worker(user, n):
            if n == 0:
                raise InferenceError("llama decode failed")
            return "A detailed result citing [G1]."

        result = orchestrator(ScriptedModel({"worker": worker})).think(request())
        assert result.status is ThinkingStatus.COMPLETED
        assert result.retries.runtime_retries == 1
        assert result.retries.reasoning_retries == 0

    def test_a_prompt_too_large_is_not_retried(self):
        def worker(user, n):
            raise PromptTooLarge(40000, 32000)

        model = ScriptedModel({"worker": worker})
        result = orchestrator(model).think(request())
        assert result.status is ThinkingStatus.FAILED
        assert model.count("worker") == 1
        assert result.jobs[0].failure_kind is FailureKind.MODEL

    def test_the_deadline_stops_the_request(self):
        release = threading.Event()

        def worker(user, n):
            release.wait(5)
            return "A detailed result citing [G1]."

        clock = iter([0.0, 0.0, 0.0, 0.0] + [1000.0] * 100)
        result = ThinkingOrchestrator(
            ScriptedModel({"worker": worker}), PROMPTS,
            ThinkingSettings(deadline_seconds=10), clock=lambda: next(clock),
        ).think(request())
        release.set()
        assert result.status is ThinkingStatus.TIMED_OUT
        assert result.jobs[0].failure_kind is FailureKind.DEADLINE

    def test_cancellation(self):
        cancel = threading.Event()
        cancel.set()
        result = orchestrator(ScriptedModel()).think(request(), cancel=cancel)
        assert result.status is ThinkingStatus.CANCELLED


def test_independent_jobs_run_concurrently_but_never_twice():
    both_in_flight = threading.Barrier(2, timeout=5)

    def worker(user, n):
        if "relate them" not in user:
            both_in_flight.wait()  # j1 and j2 have no dependency, so both must be running
        return "A detailed result citing [G1]."

    model = ScriptedModel({"planner": three_step_plan, "worker": worker})
    result = orchestrator(model, max_workers=2).think(
        request(decomposition("Design it", Mode.DEVELOPMENT)))
    assert result.status is ThinkingStatus.COMPLETED
    assert model.max_inside >= 2
    assert [j.attempts for j in result.jobs] == [1, 1, 1]
    assert model.count("worker") == 3


def test_a_condensed_query_keeps_the_original_in_view():
    long_query = "Requirement: use PostgreSQL 16. " * 600
    dec = decomposition(long_query, working_query="Use PostgreSQL 16.")
    model = ScriptedModel()
    ThinkingOrchestrator(model, PROMPTS, ThinkingSettings(query_chars=4000)).think(request(dec))
    worker = next(c for c in model.calls if c["role"] == "worker")
    assert "condensed" in worker["user"] and "Use PostgreSQL 16." in worker["user"]
    assert "Requirement: use PostgreSQL 16." in worker["user"]
