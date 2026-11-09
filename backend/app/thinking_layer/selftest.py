"""A two-job plan run against a stub model, so a health check exercises every part in milliseconds."""

import json
from typing import Dict

from query_decomposition.models import Decomposition, Mode
from thinking_layer.models import Evidence, JobState, ThinkingRequest, ThinkingStatus
from thinking_layer.orchestrator import ThinkingOrchestrator
from thinking_layer.prompt_text import ThinkingPrompts
from thinking_layer.settings import ThinkingSettings

_PROMPTS = ThinkingPrompts("PLAN {max_jobs}", "WORK", "STITCH", "JOB_CRITIC", "PLAN_CRITIC")


class _Stub:
    def __init__(self) -> None:
        self.critiques = 0

    def infer(self, system, user, **options) -> str:
        if system.startswith("PLAN "):
            return json.dumps({"jobs": [
                {"id": "a", "task": "first", "expected_output": "x", "depends_on": []},
                {"id": "b", "task": "second", "expected_output": "y", "depends_on": ["a"]},
            ]})
        if system == "JOB_CRITIC":
            self.critiques += 1
            # The first review rejects, so the retry path runs too.
            verdict = "revise" if self.critiques == 1 else "accept"
            return json.dumps({"verdict": verdict, "reasons": ["self-test"], "missing": []})
        if system == "PLAN_CRITIC":
            return json.dumps({"verdict": "accept", "affected_jobs": [], "missing": [], "reasons": []})
        return "A self-test result long enough to pass review, citing [G1]."


def run_self_test() -> Dict[str, bool]:
    decomposition = Decomposition("selftest", "self test", "self test", Mode.DEVELOPMENT, (), True, ())
    request = ThinkingRequest("selftest", "selftest", decomposition,
                              (Evidence("G1", "global", "evidence", 1.0),))
    result = ThinkingOrchestrator(
        _Stub(), _PROMPTS, ThinkingSettings(max_workers=2, deadline_seconds=10)
    ).think(request)
    jobs = {job.job_id: job for job in result.jobs}
    ordered = [j.job_id for j in result.accepted] == ["j1", "j2"]
    return {
        "Planner": result.plan.source == "model" and len(result.plan.jobs) == 2,
        "Scheduler": ordered,
        "Job Pool": all(job.state is JobState.COMPLETED for job in jobs.values()),
        "Worker Pool": result.model_calls >= 5,
        "Accumulator": [e.ref for e in result.evidence] == ["G1"],
        "Critic": result.retries.job_retries == 1 and len(result.critiques) == 1,
        "Thinking Layer": result.status is ThinkingStatus.COMPLETED,
    }
