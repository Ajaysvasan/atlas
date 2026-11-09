"""Errors the thinking layer raises."""


class ThinkingError(Exception):
    """Base for every thinking-layer error."""


class InvalidPlan(ThinkingError):
    """A plan whose jobs cannot be scheduled: duplicate ids, unknown dependencies or a cycle."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"invalid plan: {reason}")
        self.reason = reason


class InvalidTransition(ThinkingError):
    def __init__(self, job_id: str, current, requested) -> None:
        super().__init__(
            f"job {job_id} cannot move from {current.value} to {requested.value}"
        )
        self.job_id = job_id
        self.current = current
        self.requested = requested


class UnknownJob(ThinkingError):
    def __init__(self, job_id: str) -> None:
        super().__init__(f"no job {job_id!r} in this plan")
        self.job_id = job_id
