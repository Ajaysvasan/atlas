"""The local retrieval layer's only public entry point: a project's own conversations."""

import sqlite3
from pathlib import Path

from config import get_logger, log_context, new_correlation_id
from memory.identifiers import require_identifier
from memory.local_retrieval.turn_hydration import TurnHydration
from memory.local_retrieval.turn_keyword_search import TurnKeywordSearch
from memory.local_retrieval.turn_scope import CONVERSATION, TurnScope
from memory.local_retrieval.turn_vector_search import TurnVectorSearch
from memory.memory_database import MemoryDatabase
from memory.topic_pool.project_pool.conversation_pool.conversation_data_management.conversationVectorMetaManager import (
    ConversationVectorMetaDataRepository,
)
from retrieval_layer.assembly import assemble, deduplicate
from retrieval_layer.caching import ResultCache, cache_key
from retrieval_layer.diversity import maximal_marginal_relevance
from retrieval_layer.fusion import reciprocal_rank_fusion
from retrieval_layer.models import RetrievalRequest, RetrievalResult
from retrieval_layer.query_preparation import QueryPreparation
from retrieval_layer.reranking import Reranker
from retrieval_layer.retrieval_exceptions import EmptyQuery, InvalidRetrievalSetting
from retrieval_layer.settings import RetrievalSettings
from retrieval_layer.telemetry import StageRecorder

logger = get_logger(__name__)


class LocalRetriever:
    """Query in; the turns that answer it, in conversation order, within the token budget, out."""

    def __init__(
        self,
        project_id: str,
        project_name: str,
        conversation_id: str,
        settings: RetrievalSettings | None = None,
        *,
        database: MemoryDatabase | str | Path | None = None,
        preparation: QueryPreparation | None = None,
        vector_search: TurnVectorSearch | None = None,
        keyword_search: TurnKeywordSearch | None = None,
        hydration: TurnHydration | None = None,
        reranker: Reranker | None = None,
        cache: ResultCache | None = None,
    ) -> None:
        self.project_id = require_identifier(project_id, "project_id")
        self.conversation_id = require_identifier(conversation_id, "conversation_id")
        self.settings = RetrievalSettings() if settings is None else settings
        self.database = MemoryDatabase.of(database)
        self.__owned: list = []
        if vector_search is None:
            vector_search = TurnVectorSearch(self.project_id, project_name, self.database)
            self.__owned.append(vector_search)
        self.vector_search = vector_search
        self.keyword_search = (
            TurnKeywordSearch(self.database) if keyword_search is None else keyword_search
        )
        self.hydration = (
            TurnHydration(self.keyword_search, self.database)
            if hydration is None else hydration
        )
        self.preparation = QueryPreparation() if preparation is None else preparation
        self.reranker = (
            Reranker(self.settings.rerank_model, self.settings.rerank_batch)
            if reranker is None else reranker
        )
        self.cache = (
            ResultCache(self.settings.cache_size, self.settings.cache_ttl_seconds)
            if cache is None else cache
        )
        self.snapshots = ConversationVectorMetaDataRepository(
            self.project_id, self.conversation_id, self.database
        )

    def retrieve(
        self,
        request: RetrievalRequest | str,
        *,
        scope: str = CONVERSATION,
        before_sequence: int | None = None,
    ) -> RetrievalResult:
        """The turns that answer a query: this conversation's, or with scope="project" every conversation's in the project."""
        if isinstance(request, str):
            request = RetrievalRequest(query=request)
        request = request.with_defaults(self.settings)
        if request.top_k < 1:
            raise InvalidRetrievalSetting("top_k", request.top_k, "at least 1")
        if not isinstance(request.query, str) or not request.query.strip():
            raise EmptyQuery(request.query)
        turns = TurnScope.of(self.project_id, self.conversation_id, scope, before_sequence)

        recorder = StageRecorder()
        with log_context(retrieval_id=new_correlation_id()):
            with recorder.stage("index") as stage:
                stage.produced = self.__sync(turns)
            indexed = self.keyword_search.size(turns)
            if indexed == 0:
                recorder.log_breakdown()
                return RetrievalResult(passages=[], timings=recorder.timings)
            # The corpus grows with every turn and every snapshot, so the key
            # carries how far it had grown: an answer from before is a miss.
            key = (
                f"{turns.key}|{indexed}|{self.snapshots.get_highest_snapshot_seq()}"
                f"|{cache_key(request)}"
            )
            with recorder.stage("cache") as stage:
                hit = self.cache.get(key)
                stage.produced = 0 if hit is None else len(hit.passages)
            if hit is not None:
                recorder.log_breakdown()
                return hit._replace(
                    passages=list(hit.passages),
                    timings=recorder.timings,
                    cached=True,
                )
            result = self.__retrieve(request, turns, recorder)
            self.cache.put(key, result._replace(passages=list(result.passages)))
            recorder.log_breakdown()
        return result

    def __sync(self, turns: TurnScope) -> int:
        try:
            return self.keyword_search.sync(turns)
        except sqlite3.OperationalError as error:
            logger.warning(
                "Turns written since the last retrieval could not be indexed (%s); "
                "searching what is already indexed",
                error,
            )
            return 0

    def __retrieve(
        self, request: RetrievalRequest, turns: TurnScope, recorder: StageRecorder
    ) -> RetrievalResult:
        settings = self.settings
        pool = request.top_k * settings.candidate_multiplier

        with recorder.stage("prepare") as stage:
            plan = self.preparation.prepare(request.query)
            stage.produced = 1
        with recorder.stage("vector search") as stage:
            dense = self.vector_search.search(plan, pool, turns)
            stage.produced = len(dense)
        with recorder.stage("keyword search") as stage:
            sparse = self.keyword_search.search(plan, pool, turns)
            stage.produced = len(sparse)
        with recorder.stage("fusion") as stage:
            fused = reciprocal_rank_fusion([dense, sparse], settings.rrf_k)[:pool]
            stage.produced = len(fused)
        with recorder.stage("hydration") as stage:
            passages, unresolvable = self.hydration.hydrate(fused)
            passages, _ = deduplicate(passages)
            stage.produced = len(passages)
        if request.rerank:
            with recorder.stage("rerank") as stage:
                passages = self.reranker.rerank(plan, passages)
                stage.produced = len(passages)
        with recorder.stage("diversity") as stage:
            passages = maximal_marginal_relevance(
                plan,
                passages,
                request.top_k,
                settings.mmr_lambda,
                embed=self.preparation.encode,
                relevance=[passage.score for passage in passages],
            )
            stage.produced = len(passages)
        with recorder.stage("assembly") as stage:
            passages, tokens, _ = assemble(
                passages,
                settings.token_budget,
                settings.chars_per_token,
                reading_order=True,
            )
            stage.produced = len(passages)

        return RetrievalResult(
            passages=passages,
            timings=recorder.timings,
            candidates_considered=len(fused),
            dropped_unresolvable=unresolvable,
            tokens=tokens,
        )

    def close(self) -> None:
        """Release what this retriever opened. What was handed in stays open."""
        for resource in self.__owned:
            resource.close()
        self.__owned = []

    def __enter__(self) -> "LocalRetriever":
        return self

    def __exit__(self, *_) -> None:
        self.close()
