"""The thinking layer's budgets, defaulted from Config."""

from typing import Dict, NamedTuple

from config import Config
from thinking_layer.models import ExecutionPath


class ThinkingSettings(NamedTuple):
    max_workers: int = Config.THINKING_MAX_WORKERS
    max_job_retries: int = Config.THINKING_MAX_JOB_RETRIES
    max_runtime_retries: int = Config.THINKING_MAX_RUNTIME_RETRIES
    max_plan_revisions: Dict[ExecutionPath, int] = {
        ExecutionPath.DIRECT: 0,
        ExecutionPath.PLANNED: Config.THINKING_MAX_PLAN_REVISIONS,
        ExecutionPath.EXPANDED: Config.THINKING_MAX_PLAN_REVISIONS,
    }
    max_planned_jobs: int = Config.THINKING_MAX_PLANNED_JOBS
    deadline_seconds: float = Config.THINKING_DEADLINE_SECONDS
    planner_max_tokens: int = 512
    job_max_tokens: int = 700
    stitch_max_tokens: int = 1000
    critic_max_tokens: int = 384
    query_chars: int = 12_000
    evidence_chars: int = 6_000
    dependency_chars: int = 5_000
    memory_chars: int = 2_000

    def validated(self) -> "ThinkingSettings":
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {self.max_workers}")
        for name in ("max_job_retries", "max_runtime_retries", "max_planned_jobs"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.max_planned_jobs < 1:
            raise ValueError("max_planned_jobs must be at least 1")
        if any(value < 0 for value in self.max_plan_revisions.values()):
            raise ValueError("max_plan_revisions cannot be negative")
        if self.deadline_seconds <= 0:
            raise ValueError("deadline_seconds must be positive")
        return self
