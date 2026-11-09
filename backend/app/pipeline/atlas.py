"""One query from arrival to a persisted answer, through each layer's own public interface."""

import time
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Sequence, Tuple

from config import Config, get_logger, log_context, new_correlation_id
from generation_layer.context import (
    GenerationBudget,
    GenerationPrompts,
    KnowledgeStatus,
    assemble,
)
from generation_layer.generator import Generator
from knowledge_sufficiency.ksv_manager import KSVManager
from knowledge_sufficiency.verdict import Sufficiency
from memory.conversation_registry import ConversationRecord, ConversationRegistry
from memory.memory_database import MemoryDatabase
from memory.memory_manager import NO_TOPIC, MemoryManager, MemoryModels
from memory.memory_pool_exceptions import TopicAlreadyExists
from observability.telemetry import RequestRecord, StageClock, Telemetry
from pipeline.naming import Namer
from query_decomposition.models import Decomposition, Mode
from query_decomposition.query_decomposition_exceptions import QueryTooLarge
from retrieval_layer.models import Passage, RetrievalRequest
from storage.timestamps import utc_now
from thinking_layer.models import Evidence, MemoryContext, ThinkingRequest, ThinkingResult
from thinking_layer.orchestrator import ThinkingOrchestrator

logger = get_logger(__name__)

DEFAULT_TOPIC = "General"
MAX_GLOBAL_EVIDENCE = 10
THINKING_MEMORY_TURNS = 4


class AtlasError(Exception):
    category = "internal"


class ConversationNotFound(AtlasError):
    category = "conversation"

    def __init__(self, conversation_id: str) -> None:
        super().__init__(f"no conversation {conversation_id} for this user")


class AtlasResponse(NamedTuple):
    request_id: str
    conversation_id: str
    text: str | None
    status: str
    persisted: bool
    warnings: Tuple[str, ...]
    error: str | None
    record: RequestRecord


class Retrieved(NamedTuple):
    passages: List[Passage]
    limitations: List[str]


class RetrieverHandle:
    """The global retriever, built on first use and rebuilt if building failed before."""

    def __init__(self, build: Callable[[], object]) -> None:
        self.build = build
        self.retriever = None
        self.error: str | None = None

    def get(self):
        if self.retriever is None:
            try:
                self.retriever = self.build()
                self.error = None
            except Exception as error:
                self.error = str(error)
                raise
        return self.retriever

    def refresh(self) -> None:
        if self.retriever is not None:
            self.retriever.refresh()

    def close(self) -> None:
        if self.retriever is not None:
            self.retriever.close()
            self.retriever = None


def _category(error: BaseException) -> str:
    from inference.inference_exceptions import InferenceError

    if isinstance(error, AtlasError):
        return error.category
    if isinstance(error, (QueryTooLarge, ValueError)):
        return "input"
    if isinstance(error, InferenceError):
        return "model"
    module = type(error).__module__
    if "psycopg" in module or "sqlite3" in module:
        return "database"
    return "internal"


class AtlasPipeline:
    def __init__(
        self,
        *,
        decomposer,
        memory_models: MemoryModels,
        ksv: KSVManager,
        retriever: RetrieverHandle,
        thinking: ThinkingOrchestrator,
        generator: Generator,
        generation_prompts: GenerationPrompts,
        registry: ConversationRegistry,
        namer: Namer,
        database: MemoryDatabase | str | Path | None = None,
        acquirer=None,
        telemetry: Telemetry | None = None,
        provenance: Callable[[], Dict[str, str]] | None = None,
        memory_manager=MemoryManager,
        budget: GenerationBudget | None = None,
        clock: Callable[[], float] = time.monotonic,
        model_calls: Callable[[], Dict[str, int]] | None = None,
    ) -> None:
        self.decomposer = decomposer
        self.models = memory_models
        self.ksv = ksv
        self.retriever = retriever
        self.thinking = thinking
        self.generator = generator
        self.generation_prompts = generation_prompts
        self.registry = registry
        self.namer = namer
        self.database = MemoryDatabase.of(database)
        self.acquirer = acquirer
        self.acquisition_enabled = acquirer is not None
        self.telemetry = Telemetry() if telemetry is None else telemetry
        self.provenance = provenance or (lambda: {})
        self.memory_manager = memory_manager
        self.budget = GenerationBudget.for_tokens() if budget is None else budget
        self.clock = clock
        self.model_calls = model_calls

    def new_conversation(self, user_id: str, first_query: str) -> ConversationRecord:
        """A conversation named from its first query; the name is unique for this user."""
        return self.registry.create(user_id, self.namer.conversation_name(first_query))

    def ask(self, conversation_id: str, user_id: str, query: str,
            request_id: str | None = None) -> AtlasResponse:
        """One query; `request_id` is the caller's correlation id (gRPC), else a new one (CLI)."""
        request_id = request_id or new_correlation_id()
        record = RequestRecord(request_id, conversation_id, utc_now())
        stages = StageClock(record, self.clock)
        warnings: List[str] = []
        text = None
        error = None
        calls_before = self.model_calls() if self.model_calls else None
        with log_context(request_id=request_id, conversation_id=conversation_id):
            try:
                if self.registry.get(conversation_id, user_id) is None:
                    raise ConversationNotFound(conversation_id)
                text = self.__answer(conversation_id, user_id, query, record, stages, warnings)
                record.status = "completed_with_warnings" if warnings else "completed"
            except Exception as failure:
                error = str(failure)
                record.status = "failed"
                record.error_category = _category(failure)
                record.error = error
                logger.error("Request failed (%s): %s", record.error_category, failure,
                             exc_info=record.error_category == "internal")
            finally:
                stages.finish()
                if calls_before is not None:
                    after = self.model_calls()
                    record.model_calls.update({
                        purpose: count - calls_before.get(purpose, 0)
                        for purpose, count in after.items()
                        if count - calls_before.get(purpose, 0)
                    })
                self.telemetry.add(record)
        persisted = bool(record.persisted) and all(v is True for v in record.persisted.values())
        return AtlasResponse(request_id, conversation_id, text, record.status, persisted,
                             tuple(warnings), error, record)

    def __answer(self, conversation_id, user_id, query, record, stages, warnings) -> str:
        with stages.stage("decomposition"):
            decomposition: Decomposition = self.decomposer.decompose(query, conversation_id, user_id)
        record.mode = decomposition.mode.value
        topic = decomposition.primary_topic or DEFAULT_TOPIC

        limitations: List[str] = []
        with stages.stage("memory"):
            memory = self.memory_manager(
                decomposition.query, conversation_id, user_id, topic,
                database=self.database, models=self.models, ksv_manager=self.ksv,
            )
            route = memory.route
            context = memory.conversation_context(2 * Config.GENERATION_HISTORY_TURNS)
            memory_evidence = self.__recall(memory, limitations)
        record.retrieval["route"] = route.found_by
        record.retrieval["memory_sufficiency"] = None if memory.sufficiency is None else memory.sufficiency.value
        record.retrieval["memory_passages"] = len(memory_evidence)

        knowledge = KnowledgeStatus(memory=record.retrieval["memory_sufficiency"] or "nothing recalled")
        global_evidence: List[Evidence] = []
        if decomposition.mode is Mode.DOUBT and memory.sufficiency is Sufficiency.FULL:
            record.retrieval["global"] = "skipped: memory answers it"
        else:
            with stages.stage("retrieval"):
                retrieved = self.__retrieve(decomposition, user_id)
            with stages.stage("ksv"):
                sufficiency, global_evidence = self.__verify(decomposition, retrieved)
            record.retrieval["global"] = sufficiency
            if sufficiency in ("none", "empty") and self.acquisition_enabled:
                with stages.stage("acquisition"):
                    outcome = self.__acquire(decomposition, topic, record)
                if outcome is not None and outcome.stored_documents:
                    with stages.stage("retrieval"):
                        self.retriever.refresh()
                        retrieved = self.__retrieve(decomposition, user_id)
                    with stages.stage("ksv"):
                        sufficiency, global_evidence = self.__verify(decomposition, retrieved)
                    record.retrieval["global"] = sufficiency
            limitations.extend(retrieved.limitations)
            knowledge = knowledge._replace(
                global_retrieval=sufficiency,
                acquisition=record.retrieval.get("acquisition"),
            )
            if sufficiency in ("none", "empty") and not memory_evidence:
                limitations.append("No relevant material was found in the knowledge base or memory")
            elif sufficiency == "partial":
                limitations.append("The knowledge base covers this request only partially")
        record.retrieval["global_passages"] = len(global_evidence)

        evidence = self.__tag(memory_evidence, "M") + self.__tag(global_evidence, "G")
        thinking_request = ThinkingRequest(
            request_id=record.request_id,
            conversation_id=conversation_id,
            decomposition=decomposition,
            evidence=tuple(evidence),
            memory=MemoryContext(context.summary, tuple(context.turns[-THINKING_MEMORY_TURNS:])),
            limitations=tuple(limitations),
        )
        with stages.stage("thinking"):
            thought: ThinkingResult = self.thinking.think(thinking_request)
        self.__note_thinking(record, thought)

        generation_context = assemble(
            self.generation_prompts, thought, evidence,
            summary=context.summary,
            history=context.turns,
            history_offset=max(0, (context.total_turns - len(context.turns)) // 2),
            knowledge=knowledge,
            budget=self.budget,
        )
        with stages.stage("generation"):
            generated = self.generator.generate(conversation_id, user_id, generation_context)
        cache = generated.cache
        record.cache = {
            "outcome": cache.outcome, "reason": cache.reason,
            "prompt_tokens": cache.prompt_tokens, "evaluated_prompt_tokens": cache.evaluated_prompt_tokens,
            "reused_tokens": cache.reused_tokens, "restore_ms": round(cache.restore_ms, 1),
            "save_ms": round(cache.save_ms, 1), "saved_tokens": cache.saved_tokens,
            "saved_bytes": cache.saved_bytes, "save_error": cache.save_error,
        }
        if cache.save_error:
            warnings.append("The conversation's inference cache could not be saved; the next answer recomputes it")
        if generated.truncated:
            warnings.append("The answer reached its length limit and may be cut short")

        with stages.stage("persistence"):
            self.__persist(memory, route, decomposition, generated.text, record, warnings)
        return generated.text

    @staticmethod
    def __recall(memory, limitations: List[str]) -> List[Passage]:
        try:
            return memory.conversation_retrieval() or []
        except Exception as error:
            logger.warning("Memory recall failed: %s", error)
            limitations.append("Earlier conversations could not be searched")
            return []

    def __retrieve(self, decomposition: Decomposition, user_id: str) -> "Retrieved":
        queries = [decomposition.working_query, *decomposition.sub_queries]
        best: Dict[int, Passage] = {}
        try:
            retriever = self.retriever.get()
            for text in queries:
                for passage in retriever.retrieve(RetrievalRequest(text, viewer=user_id)).passages:
                    if passage.vector_id not in best or passage.score > best[passage.vector_id].score:
                        best[passage.vector_id] = passage
        except Exception as error:
            logger.warning("Global retrieval unavailable: %s", error)
            return Retrieved([], [f"The knowledge base could not be searched ({type(error).__name__})"])
        return Retrieved(list(best.values()), [])

    def __verify(self, decomposition: Decomposition, retrieved: "Retrieved") -> Tuple[str, List[Evidence]]:
        """KSV decides which retrieved passages are relevant enough to count as evidence."""
        passages = retrieved.passages
        if not passages:
            return "empty", []
        vectors = self.models.preparation.encode([decomposition.working_query] + [p.text for p in passages])
        verdict = self.ksv.sufficiency_verification(
            vectors[0], [(p.vector_id, vectors[i + 1]) for i, p in enumerate(passages)]
        )
        supporting = {vector_id: rank for rank, vector_id in enumerate(verdict.supporting)}
        kept = sorted((p for p in passages if p.vector_id in supporting),
                      key=lambda p: supporting[p.vector_id])[:MAX_GLOBAL_EVIDENCE]
        return verdict.sufficiency.value, kept

    def __acquire(self, decomposition, topic, record):
        try:
            sub_topics = decomposition.topics[0].sub_topics if decomposition.topics else ()
            outcome = self.acquirer.acquire(topic, decomposition.working_query, sub_topics)
        except Exception as error:
            logger.warning("Acquisition failed: %s", error)
            record.retrieval["acquisition"] = f"failed ({type(error).__name__})"
            return None
        record.retrieval["acquisition"] = (
            f"{outcome.sufficiency}, {outcome.stored_documents} document(s) stored"
            + (f", stopped: {outcome.stopped}" if outcome.stopped else "")
        )
        return outcome

    def __tag(self, passages: Sequence[Passage], prefix: str) -> List[Evidence]:
        origins = self.provenance() if prefix == "G" else {}
        return [
            Evidence(
                ref=f"{prefix}{index}",
                source="memory" if prefix == "M" else "global",
                text=passage.text,
                score=float(passage.score),
                vector_id=passage.vector_id,
                chunk_id=passage.chunk_id,
                document_id=passage.document_id,
                origin=origins.get(passage.document_id or "") if prefix == "G"
                else (f"earlier {passage.role} turn" if passage.role else "earlier conversation"),
            )
            for index, passage in enumerate(passages, start=1)
        ]

    @staticmethod
    def __note_thinking(record: RequestRecord, thought: ThinkingResult) -> None:
        record.path = thought.path.value
        record.thinking_status = thought.status.value
        record.jobs = len(thought.jobs)
        record.job_retries = thought.retries.job_retries
        record.plan_revisions = thought.retries.plan_revisions
        record.runtime_retries = thought.retries.runtime_retries
        record.retries_exhausted = thought.retries.exhausted
        record.retried_jobs = sorted({e.job_id for e in thought.retries.events if e.job_id})
        record.retry_reasons = [f"{e.level} {e.job_id or ''} ({e.kind.value}): {e.reason}".replace("  ", " ")
                                for e in thought.retries.events]
        record.model_calls["thinking"] = thought.model_calls
        for name, value in thought.timings.items():
            if name.endswith("_sum"):
                record.retrieval.setdefault("thinking_detail_ms", {})[name] = round(value, 1)
            elif name != "total":
                record.stages[f"thinking.{name}"] = value

    def __persist(self, memory, route, decomposition, text, record, warnings) -> None:
        """Memory first, then the exchange log; either failing is reported, never hidden."""
        try:
            if route.project_id is not None:
                memory.update(text)
            else:
                project_name = self.namer.project_name(decomposition.query)
                if route.found_by == NO_TOPIC:
                    try:
                        memory.store_new_topic(text, project_name)
                    except TopicAlreadyExists:
                        memory.store_new_project(text, project_name)
                else:
                    memory.store_new_project(text, project_name)
            record.persisted["memory"] = True
        except Exception as error:
            logger.exception("The answer was not stored in memory")
            record.persisted["memory"] = False
            warnings.append(f"The answer was NOT saved to conversation memory ({type(error).__name__})")
        try:
            self.decomposer.record_response(decomposition.exchange_id, text)
            record.persisted["exchange"] = True
        except Exception as error:
            logger.exception("The answer was not recorded against its query")
            record.persisted["exchange"] = False
            warnings.append(f"The answer was NOT recorded in the exchange log ({type(error).__name__})")

    def close(self) -> None:
        self.retriever.close()
        close = getattr(self.decomposer, "close", None)
        if close is not None:
            close()
