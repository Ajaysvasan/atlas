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

# A review's lists are bounded by item count, so a model that starts repeating
# cannot grow one until the token limit cuts the JSON off. Never by string
# length: maxLength compiles to a per-character grammar that breaks words apart
# (see pipeline/naming.py).
CRITIQUE_ITEMS = 3
_REVIEW_STRINGS = {"type": "array", "maxItems": CRITIQUE_ITEMS, "items": {"type": "string"}}

_DECODER = json.JSONDecoder()


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
        "reasons": _REVIEW_STRINGS,
        "missing": _REVIEW_STRINGS,
    },
    "required": ["verdict", "reasons", "missing"],
}


def plan_critique_schema(job_count: int) -> Dict:
    """The plan critic's reply; it can name at most the jobs it was shown."""
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": [verdict.value for verdict in PlanVerdict]},
            "affected_jobs": {"type": "array", "maxItems": max(1, job_count), "items": {"type": "string"}},
            "missing": _REVIEW_STRINGS,
            "reasons": _REVIEW_STRINGS,
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


def _skip(text: str, pos: int, chars: str = " \t\r\n,") -> int:
    while pos < len(text) and text[pos] in chars:
        pos += 1
    return pos


def _completed_fields(text: str) -> Dict:
    """From JSON cut off mid-object: every field finished before the cut, and the finished items of a cut list."""
    fields: Dict = {}
    pos = 1
    while True:
        try:
            key, pos = _DECODER.raw_decode(text, _skip(text, pos))
        except ValueError:
            return fields
        pos = _skip(text, pos, " \t\r\n")
        if not isinstance(key, str) or pos >= len(text) or text[pos] != ":":
            return fields
        pos = _skip(text, pos + 1, " \t\r\n")
        if pos < len(text) and text[pos] == "[":
            items: List = []
            fields[key] = items
            pos += 1
            while True:
                pos = _skip(text, pos)
                if pos < len(text) and text[pos] == "]":
                    pos += 1
                    break
                try:
                    item, pos = _DECODER.raw_decode(text, pos)
                except ValueError:
                    return fields
                items.append(item)
        else:
            try:
                fields[key], pos = _DECODER.raw_decode(text, pos)
            except ValueError:
                return fields


def _review_object(stage: str, reply: str) -> Tuple[Dict, bool]:
    """The reply's object, and whether it was cut off at the token limit.

    The grammar only lets the model write valid JSON, so JSON that starts but
    never closes was stopped by the token limit. The grammar also writes
    "verdict" first, so a cut-off review usually still holds its decision.
    """
    try:
        return _object(stage, reply), False
    except UnusableReply:
        text = (reply or "").strip()
        if not text.startswith("{"):
            raise
    fields = _completed_fields(text)
    if "verdict" not in fields:
        raise UnusableReply(stage, "cut off at its token limit before giving a verdict")
    return fields, True


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
    value, cut_off = _review_object("job critic", reply)
    try:
        verdict = Verdict(str(value.get("verdict", "")).strip().lower())
    except ValueError:
        raise UnusableReply("job critic", f"verdict {value.get('verdict')!r}") from None
    return Critique(verdict, _strings(value.get("reasons")), _strings(value.get("missing")), cut_off=cut_off)


def read_plan_critique(reply: str, job_ids) -> PlanCritique:
    value, cut_off = _review_object("plan critic", reply)
    try:
        verdict = PlanVerdict(str(value.get("verdict", "")).strip().lower())
    except ValueError:
        raise UnusableReply("plan critic", f"verdict {value.get('verdict')!r}") from None
    known = set(job_ids)
    affected = tuple(
        dict.fromkeys(job for job in _strings(value.get("affected_jobs")) if job in known)
    )
    return PlanCritique(
        verdict, affected, _strings(value.get("missing")), _strings(value.get("reasons")), cut_off=cut_off
    )

