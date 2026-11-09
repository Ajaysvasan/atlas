"""The JSON the planner and critics reply with, and turning a reply into something typed."""

import json
from typing import Dict, List, Tuple

from thinking_layer.models import Critique, PlanCritique, PlanVerdict, Verdict


class UnusableReply(ValueError):
    def __init__(self, stage: str, reason: str) -> None:
        super().__init__(f"the {stage} reply could not be used: {reason}")
        self.stage = stage
        self.reason = reason


_STRINGS = {"type": "array", "items": {"type": "string"}}


def plan_schema(max_jobs: int) -> Dict:
    return {
        "type": "object",
        "properties": {
            "jobs": {
                "type": "array",
                "minItems": 1,
                "maxItems": max_jobs,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "task": {"type": "string"},
                        "expected_output": {"type": "string"},
                        "depends_on": _STRINGS,
                    },
                    "required": ["id", "task", "expected_output", "depends_on"],
                },
            }
        },
        "required": ["jobs"],
    }


JOB_CRITIQUE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": [verdict.value for verdict in Verdict]},
        "reasons": _STRINGS,
        "missing": _STRINGS,
    },
    "required": ["verdict", "reasons", "missing"],
}

PLAN_CRITIQUE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": [verdict.value for verdict in PlanVerdict]},
        "affected_jobs": _STRINGS,
        "missing": _STRINGS,
        "reasons": _STRINGS,
    },
    "required": ["verdict", "affected_jobs", "missing", "reasons"],
}


def _object(stage: str, reply: str) -> Dict:
    try:
        value = json.loads(reply)
    except (TypeError, ValueError) as error:
        raise UnusableReply(stage, f"not JSON ({error})") from None
    if not isinstance(value, dict):
        raise UnusableReply(stage, "not a JSON object")
    return value


def _strings(value, limit: int = 8) -> Tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())[:limit]


def read_plan(reply: str, max_jobs: int) -> List[Dict]:
    """The planned jobs, ids normalised to j1..jN and dependencies remapped to match."""
    jobs = _object("planner", reply).get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise UnusableReply("planner", "no jobs")
    if len(jobs) > max_jobs:
        raise UnusableReply("planner", f"{len(jobs)} jobs, more than {max_jobs}")
    renamed: Dict[str, str] = {}
    cleaned = []
    for index, job in enumerate(jobs, start=1):
        if not isinstance(job, dict):
            raise UnusableReply("planner", f"job {index} is not an object")
        task = job.get("task")
        if not isinstance(task, str) or not task.strip():
            raise UnusableReply("planner", f"job {index} has no task")
        original = str(job.get("id") or f"j{index}").strip()
        if original in renamed:
            raise UnusableReply("planner", f"duplicate job id {original!r}")
        renamed[original] = f"j{index}"
        expected = job.get("expected_output")
        cleaned.append(
            {
                "job_id": f"j{index}",
                "task": task.strip(),
                "expected_output": expected.strip() if isinstance(expected, str) and expected.strip() else task.strip(),
                "depends_on": _strings(job.get("depends_on"), limit=max_jobs),
            }
        )
    for job in cleaned:
        unknown = [dep for dep in job["depends_on"] if dep not in renamed]
        if unknown:
            raise UnusableReply("planner", f"{job['job_id']} depends on unknown {unknown}")
        job["depends_on"] = tuple(dict.fromkeys(renamed[dep] for dep in job["depends_on"]))
    return cleaned


def read_job_critique(reply: str) -> Critique:
    value = _object("job critic", reply)
    try:
        verdict = Verdict(str(value.get("verdict", "")).strip().lower())
    except ValueError:
        raise UnusableReply("job critic", f"verdict {value.get('verdict')!r}") from None
    return Critique(verdict, _strings(value.get("reasons")), _strings(value.get("missing")))


def read_plan_critique(reply: str, job_ids) -> PlanCritique:
    value = _object("plan critic", reply)
    try:
        verdict = PlanVerdict(str(value.get("verdict", "")).strip().lower())
    except ValueError:
        raise UnusableReply("plan critic", f"verdict {value.get('verdict')!r}") from None
    known = set(job_ids)
    affected = tuple(
        dict.fromkeys(job for job in _strings(value.get("affected_jobs")) if job in known)
    )
    return PlanCritique(
        verdict, affected, _strings(value.get("missing")), _strings(value.get("reasons"))
    )

