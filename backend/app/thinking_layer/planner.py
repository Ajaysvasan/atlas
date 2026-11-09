"""Turns a decomposed request into the jobs that will answer it."""

import uuid
from typing import List

from config import get_logger
from inference.model_port import PLANNER, TextModel
from thinking_layer.job_pool import topological_order
from thinking_layer.models import (
    ExecutionPath,
    JobKind,
    JobSpec,
    Plan,
    ThinkingRequest,
)
from thinking_layer.prompt_text import (
    ThinkingPrompts,
    join_blocks,
    memory_block,
    request_block,
)
from thinking_layer.replies import UnusableReply, plan_schema, read_plan
from thinking_layer.settings import ThinkingSettings
from thinking_layer.thinking_exceptions import InvalidPlan

logger = get_logger(__name__)

STITCH_ID = "stitch"


def answer_job(description: str) -> JobSpec:
    return JobSpec("j1", JobKind.ANSWER, description, "A complete, correct answer to the request")


class Planner:
    """DIRECT and EXPANDED plans are built by rule; only PLANNED asks the model."""

    def __init__(self, model: TextModel, prompts: ThinkingPrompts, settings: ThinkingSettings) -> None:
        self.model = model
        self.prompts = prompts
        self.settings = settings

    def plan(self, request: ThinkingRequest, path: ExecutionPath) -> Plan:
        plan_id = uuid.uuid4().hex
        if path is ExecutionPath.DIRECT:
            return self.__plan(plan_id, request, path, (answer_job("Answer the request"),), "rule")
        if path is ExecutionPath.EXPANDED and request.decomposition.sub_queries:
            return self.__plan(plan_id, request, path, self.__expanded(request), "rule")
        return self.__planned(plan_id, request, path)

    def __plan(self, plan_id, request, path, jobs, source) -> Plan:
        plan = Plan(plan_id, request.request_id, path, tuple(jobs), source)
        topological_order(plan.jobs)
        logger.info("Planned %d job(s) for request %s (%s, %s)", len(jobs),
                    request.request_id, path.value, source)
        return plan

    @staticmethod
    def __expanded(request: ThinkingRequest) -> List[JobSpec]:
        workers = [
            JobSpec(
                f"w{index}",
                JobKind.WORKER,
                sub_query,
                f"A worked solution to: {sub_query}",
            )
            for index, sub_query in enumerate(request.decomposition.sub_queries, start=1)
        ]
        stitch = JobSpec(
            STITCH_ID,
            JobKind.STITCH,
            "Combine the sub-problem solutions into one solution to the whole request",
            "One coherent solution covering every requirement of the request",
            tuple(job.job_id for job in workers),
        )
        return workers + [stitch]

    def __planned(self, plan_id: str, request: ThinkingRequest, path: ExecutionPath) -> Plan:
        settings = self.settings
        system = self.prompts.planner.replace("{max_jobs}", str(settings.max_planned_jobs))
        user = join_blocks(
            request_block(request.decomposition, settings.query_chars),
            memory_block(request.memory, settings.memory_chars),
        )
        try:
            reply = self.model.infer(
                system,
                user,
                schema=plan_schema(settings.max_planned_jobs),
                max_tokens=settings.planner_max_tokens,
                purpose=PLANNER,
            )
            jobs = [
                JobSpec(
                    job["job_id"],
                    JobKind.STEP,
                    job["task"],
                    job["expected_output"],
                    job["depends_on"],
                )
                for job in read_plan(reply, settings.max_planned_jobs)
            ]
            if len(jobs) == 1:
                jobs = [jobs[0]._replace(kind=JobKind.ANSWER)]
            return self.__plan(plan_id, request, path, jobs, "model")
        except (UnusableReply, InvalidPlan) as error:
            # A plan the model got wrong costs one direct answer, not the request.
            logger.warning("%s; answering request %s as a single job", error, request.request_id)
            return self.__plan(plan_id, request, path, (answer_job("Answer the request"),), "fallback")
