"""Two levels of review: one job's result, then every accepted result against the request."""

from typing import Sequence, Tuple

from config import get_logger
from inference.model_port import CRITIC, TextModel
from thinking_layer.accumulator import trim, unknown_refs
from thinking_layer.models import (
    Critique,
    Evidence,
    JobSpec,
    PlanCritique,
    PlanVerdict,
    ThinkingRequest,
    Verdict,
)
from thinking_layer.prompt_text import (
    ThinkingPrompts,
    dependency_block,
    evidence_block,
    join_blocks,
    request_block,
)
from thinking_layer.replies import (
    JOB_CRITIQUE_SCHEMA,
    UnusableReply,
    plan_critique_schema,
    read_job_critique,
    read_plan_critique,
)
from thinking_layer.settings import ThinkingSettings

logger = get_logger(__name__)

MIN_OUTPUT_CHARS = 20


class JobCritic:
    def __init__(self, model: TextModel, prompts: ThinkingPrompts, settings: ThinkingSettings) -> None:
        self.model = model
        self.prompts = prompts
        self.settings = settings

    def review(
        self,
        request: ThinkingRequest,
        job: JobSpec,
        output: str,
        evidence: Sequence[Evidence],
    ) -> Critique:
        """Cheap rule checks first; the model is asked only about output that passes them."""
        text = (output or "").strip()
        if len(text) < MIN_OUTPUT_CHARS:
            return Critique(Verdict.REVISE, ("the result is empty or too short to use",),
                            (job.expected_output,))
        invented = unknown_refs(text, (item.ref for item in evidence))
        if invented:
            return Critique(
                Verdict.REVISE,
                (f"cites evidence that does not exist: {', '.join(invented)}",),
                ("cite only the evidence tags provided",),
            )
        user = join_blocks(
            request_block(request.decomposition, self.settings.query_chars // 2),
            f"<job>\n{job.description}\nExpected: {job.expected_output}\n</job>",
            evidence_block(evidence, self.settings.evidence_chars // 2),
            f"<result>\n{trim(text, self.settings.dependency_chars)}\n</result>",
        )
        limit = self.settings.critic_max_tokens
        try:
            critique = read_job_critique(
                self.model.infer(
                    self.prompts.job_critic,
                    user,
                    schema=JOB_CRITIQUE_SCHEMA,
                    max_tokens=limit,
                    purpose=CRITIC,
                )
            )
        except UnusableReply as error:
            # An unreadable review is not evidence the result is bad; it is
            # accepted and the missing review is reported as a limitation.
            logger.warning("%s (limit %d tokens); accepting job %s unreviewed", error, limit, job.job_id)
            return Critique(Verdict.ACCEPT, (str(error),), (), reviewed=False)
        if critique.cut_off:
            logger.warning(
                "The job critic reply for job %s hit its %d-token limit; using its verdict (%s) "
                "and the %d item(s) finished before the cut",
                job.job_id, limit, critique.verdict.value, len(critique.reasons) + len(critique.missing),
            )
        return critique


class PlanCritic:
    def __init__(self, model: TextModel, prompts: ThinkingPrompts, settings: ThinkingSettings) -> None:
        self.model = model
        self.prompts = prompts
        self.settings = settings

    def review(
        self,
        request: ThinkingRequest,
        results: Sequence[Tuple[str, str]],
        evidence: Sequence[Evidence],
    ) -> PlanCritique:
        if not results:
            return PlanCritique(PlanVerdict.LIMIT, (), ("no job produced an accepted result",), ())
        user = join_blocks(
            request_block(request.decomposition, self.settings.query_chars // 2),
            dependency_block(
                [(job_id, trim(text, self.settings.dependency_chars // len(results)))
                 for job_id, text in results]
            ),
            evidence_block(evidence, self.settings.evidence_chars // 2),
        )
        limit = self.settings.critic_max_tokens
        try:
            critique = read_plan_critique(
                self.model.infer(
                    self.prompts.plan_critic,
                    user,
                    schema=plan_critique_schema(len(results)),
                    max_tokens=limit,
                    purpose=CRITIC,
                ),
                [job_id for job_id, _ in results],
            )
        except UnusableReply as error:
            logger.warning("%s (limit %d tokens); accepting the plan unreviewed", error, limit)
            return PlanCritique(PlanVerdict.ACCEPT, (), (), (str(error),), reviewed=False)
        if critique.cut_off:
            logger.warning(
                "The plan critic reply hit its %d-token limit; using its verdict (%s) "
                "and the item(s) finished before the cut",
                limit, critique.verdict.value,
            )
        return critique
