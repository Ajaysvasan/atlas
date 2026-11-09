"""A scripted stand-in for the shared model, answering by purpose and by which prompt it was sent."""

import json
import threading
from typing import Callable, Dict, List

from query_decomposition.models import Decomposition, Mode, TopicPath
from thinking_layer.models import Evidence, ThinkingRequest
from thinking_layer.prompt_text import ThinkingPrompts

PROMPTS = ThinkingPrompts(
    planner="PLANNER at most {max_jobs} jobs",
    worker="WORKER",
    stitch="STITCH",
    job_critic="JOB_CRITIC",
    plan_critic="PLAN_CRITIC",
)


def accept() -> str:
    return json.dumps({"verdict": "accept", "reasons": [], "missing": []})


def revise(reason="incomplete") -> str:
    return json.dumps({"verdict": "revise", "reasons": [reason], "missing": ["more detail"]})


def plan_accept() -> str:
    return json.dumps({"verdict": "accept", "affected_jobs": [], "missing": [], "reasons": []})


def plan_revise(*jobs, missing="covers too little") -> str:
    return json.dumps({"verdict": "revise", "affected_jobs": list(jobs), "missing": [missing],
                       "reasons": []})


class ScriptedModel:
    """`handlers` maps a prompt role to a function of (user, call_number) returning the reply."""

    def __init__(self, handlers: Dict[str, Callable[[str, int], str]] | None = None) -> None:
        self.handlers = handlers or {}
        self.calls: List[dict] = []
        self.__lock = threading.Lock()
        self.inside = 0
        self.max_inside = 0

    def role(self, system: str) -> str:
        for role in ("PLANNER", "STITCH", "JOB_CRITIC", "PLAN_CRITIC", "WORKER"):
            if system.startswith(role):
                return role.lower()
        return "other"

    def count(self, role: str) -> int:
        return sum(1 for call in self.calls if call["role"] == role)

    def infer(self, system, user, *, schema=None, max_tokens=512, purpose="draft", temperature=0.0):
        role = self.role(system)
        with self.__lock:
            number = sum(1 for c in self.calls if c["role"] == role)
            self.calls.append({"role": role, "user": user, "purpose": purpose, "schema": schema})
            self.inside += 1
            self.max_inside = max(self.max_inside, self.inside)
        try:
            handler = self.handlers.get(role)
            if handler is not None:
                return handler(user, number)
            if role in ("job_critic",):
                return accept()
            if role == "plan_critic":
                return plan_accept()
            if role == "planner":
                return json.dumps({"jobs": [{"id": "j1", "task": "answer it",
                                             "expected_output": "an answer", "depends_on": []}]})
            return f"A detailed result for this job, citing [G1]. ({role} #{number})"
        finally:
            with self.__lock:
                self.inside -= 1


def decomposition(query="How does PostgreSQL MVCC work?", mode=Mode.DOUBT, sub_queries=(),
                  working_query=None) -> Decomposition:
    return Decomposition(
        exchange_id="x1",
        query=query,
        working_query=working_query or query,
        mode=mode,
        topics=(TopicPath("Databases", ("PostgreSQL",)),),
        sufficient=not sub_queries,
        sub_queries=tuple(sub_queries),
    )


EVIDENCE = (
    Evidence("G1", "global", "PostgreSQL keeps old row versions; readers see a snapshot.", 0.9, 1, "c1", "d1"),
    Evidence("G2", "global", "VACUUM removes dead tuples no snapshot can see.", 0.8, 2, "c2", "d1"),
)


def request(dec=None, evidence=EVIDENCE, limitations=()) -> ThinkingRequest:
    return ThinkingRequest("r1", "conv1", dec or decomposition(), tuple(evidence), limitations=limitations)
