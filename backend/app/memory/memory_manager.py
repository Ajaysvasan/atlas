"""The memory layer's entry point: where a conversation belongs, what it remembers, and storing what it says."""

import threading
from pathlib import Path
from typing import Dict, List, NamedTuple, Sequence, Tuple

import numpy as np
from numpy import float32, ndarray

from config import get_logger, log_context
from knowledge_sufficiency.ksv_manager import KSVManager
from knowledge_sufficiency.verdict import Sufficiency
from memory.identifiers import require_identifier
from memory.local_retrieval.local_retriever import LocalRetriever
from memory.local_retrieval.turn_scope import PROJECT
from memory.memory_database import MemoryDatabase
from memory.memory_mapping_handler import MemoryMapping, MemoryMappingHandler
from memory.memory_pool_exceptions import (
    ConversationAlreadyRouted,
    ConversationNotRouted,
    EmptyQueryException,
    ProjectNotFound,
    TopicNotFound,
)
from memory.topic_pool.project_pool.conversation_pool.conversation_data_management.conversationVectorManager import (
    ConversationVectorManager,
)
from memory.topic_pool.project_pool.conversation_pool.conversation_pool_manager import (
    ConversationPoolManager,
)
from memory.topic_pool.project_pool.conversation_pool.full_conversation_bucket import (
    ROLE_ASSISTANT,
    ROLE_USER,
)
from memory.topic_pool.project_pool.project_data_repo.project_meta_data import (
    ProjectMetaData,
)
from memory.topic_pool.project_pool.project_manager import (
    HISTORICAL_SNAPSHOT,
    LATEST_SNAPSHOT,
    SIMILARITY_FLOOR,
    ProjectManager,
)
from memory.topic_pool.topic_manager import TopicManager
from retrieval_layer.caching import ResultCache
from retrieval_layer.models import Passage
from retrieval_layer.query_preparation import QueryPreparation, encoder_from
from retrieval_layer.reranking import Reranker
from retrieval_layer.settings import RetrievalSettings
from storage.timestamps import utc_now

logger = get_logger(__name__)

KNOWN = "known_conversation"
NO_TOPIC = "no_topic"
NO_PROJECT = "no_project"

__all__ = [
    "HISTORICAL_SNAPSHOT",
    "KNOWN",
    "LATEST_SNAPSHOT",
    "NO_PROJECT",
    "NO_TOPIC",
    "MemoryManager",
    "MemoryModels",
    "Route",
    "shared_models",
]


class Route(NamedTuple):
    """Where a conversation belongs, and how that was decided."""

    found_by: str
    topic_id: str | None
    project_id: str | None
    project_name: str | None
    latest_project_snapshot_id: int | None
    score: float | None


class MemoryModels:
    """The embedder, reranker and result cache a process loads once and every MemoryManager shares."""

    def __init__(
        self,
        embedder=None,
        reranker: Reranker | None = None,
        settings: RetrievalSettings | None = None,
        cache: ResultCache | None = None,
    ) -> None:
        self.settings = RetrievalSettings() if settings is None else settings
        self.__embedder = embedder
        self.__lock = threading.Lock()
        self.preparation = QueryPreparation(encode=self.__encode)
        self.reranker = (
            Reranker(self.settings.rerank_model, self.settings.rerank_batch)
            if reranker is None
            else reranker
        )
        self.cache = (
            ResultCache(self.settings.cache_size, self.settings.cache_ttl_seconds)
            if cache is None
            else cache
        )

    @property
    def embedder(self):
        """Loaded on first use; locked so concurrent first requests load it once."""
        if self.__embedder is None:
            with self.__lock:
                if self.__embedder is None:
                    from data_layer.ingestion.embedding.EmbeddingManager import (
                        EmbeddingManager,
                    )

                    self.__embedder = EmbeddingManager()
        return self.__embedder

    def __encode(self, texts: Sequence[str]) -> ndarray:
        return encoder_from(self.embedder)(texts)


_shared_models: MemoryModels | None = None
_shared_models_lock = threading.Lock()


def shared_models() -> MemoryModels:
    """The process-wide models, built on first use."""
    global _shared_models
    with _shared_models_lock:
        if _shared_models is None:
            _shared_models = MemoryModels()
        return _shared_models


class MemoryManager:
    """One query of one conversation: where it belongs, what memory holds for it, and storing the reply."""

    def __init__(
        self,
        query: str,
        conversation_id: str,
        user_id: str,
        topic: str,
        *,
        database: MemoryDatabase | str | Path | None = None,
        models: MemoryModels | None = None,
        ksv_manager: KSVManager | None = None,
        latest_floor: float = SIMILARITY_FLOOR,
        historical_floor: float = SIMILARITY_FLOOR,
    ) -> None:
        if not isinstance(query, str) or not query.strip():
            raise EmptyQueryException()
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("topic cannot be empty")
        self.query = query
        self.conversation_id = require_identifier(conversation_id, "conversation_id")
        self.user_id = require_identifier(user_id, "user_id")
        self.topic = topic
        self.database = MemoryDatabase.of(database)
        self.models = shared_models() if models is None else models
        self.ksv_manager = KSVManager() if ksv_manager is None else ksv_manager
        self.latest_floor = latest_floor
        self.historical_floor = historical_floor
        self.mapping = MemoryMappingHandler(self.database)
        self.__route: Route | None = None
        self.__sufficiency: Sufficiency | None = None

    @property
    def route(self) -> Route:
        """Where this conversation belongs, decided on first use and kept."""
        if self.__route is None:
            self.__route = self.__find_route()
        return self.__route

    @property
    def sufficiency(self) -> Sufficiency | None:
        """The last conversation_retrieval's verdict; None when it searched nothing."""
        return self.__sufficiency

    def __find_route(self) -> Route:
        mapping = self.mapping.search(self.conversation_id, self.user_id)
        if mapping is not None and mapping.project_id is not None:
            logger.debug(
                "Conversation %s is mapped to project %s",
                self.conversation_id,
                mapping.project_id,
            )
            return Route(
                KNOWN,
                mapping.topic_id,
                mapping.project_id,
                self.__project_name(mapping.topic_id, mapping.project_id),
                mapping.latest_project_snapshot_id,
                None,
            )

        with TopicManager(self.topic, database=self.database) as topics:
            try:
                topic_id = topics.get_topic_id()
            except TopicNotFound:
                logger.info("No topic %r to route conversation %s to", self.topic,
                            self.conversation_id)
                return Route(NO_TOPIC, None, None, None, None, None)

        with ProjectManager(
            topic_id,
            self.query,
            database=self.database,
            embedder=self.models.embedder,
            similarity_floor=self.latest_floor,
            historical_floor=self.historical_floor,
        ) as projects:
            match = projects.route_by_snapshots()
        if match is None:
            return Route(NO_PROJECT, topic_id, None, None, None, None)
        return Route(
            match.stage,
            topic_id,
            match.project_id,
            match.project_name,
            match.latest_project_snapshot_id,
            match.score,
        )

    def __project_name(self, topic_id: str, project_id: str) -> str:
        with ProjectMetaData(project_id, topic_id, database=self.database) as registry:
            project = registry.get_project()
        if project is None:
            raise ProjectNotFound(project_id)
        return project.project_name

    def conversation_retrieval(self) -> List[Passage] | None:
        """The remembered turns that answer the query, or None when memory cannot answer it."""
        with log_context(conversation_id=self.conversation_id):
            self.__sufficiency = None
            route = self.route
            if route.project_id is None:
                logger.info(
                    "Conversation %s has no project to recall from (%s)",
                    self.conversation_id,
                    route.found_by,
                )
                return None
            passages = self.__retrieve(route)
            if not passages:
                logger.info("Project %s holds no turns for this query", route.project_id)
                return None
            return self.__sufficient(route, passages)

    def __retrieve(self, route: Route) -> List[Passage]:
        with LocalRetriever(
            route.project_id,
            route.project_name,
            self.conversation_id,
            self.models.settings,
            database=self.database,
            preparation=self.models.preparation,
            reranker=self.models.reranker,
            cache=self.models.cache,
        ) as retriever:
            return retriever.retrieve(self.query, scope=PROJECT).passages

    def __sufficient(self, route: Route, passages: List[Passage]) -> List[Passage] | None:
        query_vector = self.models.preparation.encode([self.query])[0]
        vectors = self.__passage_vectors(route, passages)
        verdict = self.ksv_manager.sufficiency_verification(
            query_vector, [(passage.vector_id, vectors[passage.vector_id]) for passage in passages]
        )
        self.__sufficiency = verdict.sufficiency
        if verdict.sufficiency is Sufficiency.NONE:
            logger.info("Memory cannot answer this query (best %.3f)", verdict.best)
            return None
        supporting = set(verdict.supporting)
        kept = [passage for passage in passages if passage.vector_id in supporting]
        logger.info(
            "Memory is %s for this query: %d of %d passage(s), best %.3f",
            verdict.sufficiency.value,
            len(kept),
            len(passages),
            verdict.best,
        )
        return kept

    def __passage_vectors(
        self, route: Route, passages: List[Passage]
    ) -> Dict[int, ndarray]:
        """Each passage's stored vector; a turn no snapshot has covered yet is embedded now."""
        vector_ids = [passage.vector_id for passage in passages]
        try:
            store = ConversationVectorManager(route.project_name, route.project_id)
            try:
                vectors = store.vectors_for(vector_ids)
            finally:
                store.close()
        except Exception as error:
            logger.warning(
                "Turn vectors could not be read (%s); embedding the passages instead",
                error,
            )
            vectors = {}
        missing = [passage for passage in passages if passage.vector_id not in vectors]
        if missing:
            embedded = self.models.preparation.encode([passage.text for passage in missing])
            for passage, vector in zip(missing, embedded):
                vectors[passage.vector_id] = np.asarray(vector, dtype=float32)
        return vectors

    def store_new_topic(self, system_response: str, project_name: str) -> Route:
        """Start this conversation under a new topic, as the first project in it."""
        with log_context(conversation_id=self.conversation_id):
            return self.__store_new(system_response, project_name, create_topic=True)

    def store_new_project(self, system_response: str, project_name: str) -> Route:
        """Start this conversation as a new project under its existing topic."""
        with log_context(conversation_id=self.conversation_id):
            return self.__store_new(system_response, project_name, create_topic=False)

    def update(self, system_response: str) -> Route:
        """Store this exchange in the conversation's project, snapshotting when its context is full."""
        with log_context(conversation_id=self.conversation_id):
            self.__require_text(system_response, "system_response")
            route = self.route
            if route.project_id is None:
                raise ConversationNotRouted(self.conversation_id, self.user_id)
            pool = self.__pool(route.project_id, route.project_name)
            try:
                with self.database.writing():
                    if route.found_by != KNOWN:
                        mapping = self.mapping.search(self.conversation_id, self.user_id)
                        if mapping is not None and mapping.project_id not in (
                            None,
                            route.project_id,
                        ):
                            raise ConversationAlreadyRouted(
                                self.conversation_id, mapping.project_id
                            )
                        self.__attach(route.topic_id, route.project_id, mapping)
                    pool.add_turns(self.__exchange(system_response))
                logger.info(
                    "Stored an exchange in project %s (%s)", route.project_id, route.found_by
                )
                self.__snapshot_if_due(pool)
            finally:
                pool.close()
            return self.__rerouted()

    def __store_new(self, system_response: str, project_name: str, create_topic: bool) -> Route:
        self.__require_text(system_response, "system_response")
        self.__require_text(project_name, "project_name")
        # Loaded before the write lock is taken: a model load inside BEGIN
        # IMMEDIATE would stall every other writer for seconds.
        embedder = self.models.embedder
        project_id: str | None = None
        projects: ProjectManager | None = None
        pool: ConversationPoolManager | None = None
        try:
            with self.database.writing():
                mapping = self.mapping.search(self.conversation_id, self.user_id)
                if mapping is not None and mapping.project_id is not None:
                    raise ConversationAlreadyRouted(self.conversation_id, mapping.project_id)
                with TopicManager(self.topic, database=self.database) as topics:
                    topic_id = (
                        topics.create_new_topic() if create_topic else topics.get_topic_id()
                    )
                projects = ProjectManager(
                    topic_id, self.query, database=self.database, embedder=embedder
                )
                project_id = projects.create_project(
                    project_name, self.__opening_summary(system_response)
                )
                self.__attach(topic_id, project_id, mapping)
                pool = self.__pool(project_id, project_name)
                pool.add_turns(self.__exchange(system_response))
        except BaseException:
            # The rollback undoes the SQLite rows, not the vector in PostgreSQL.
            if project_id is not None:
                self.__forget_vector(projects, project_id)
            if pool is not None:
                pool.close()
            raise
        finally:
            if projects is not None:
                projects.close()
        logger.info(
            "Started conversation %s as project %s (%r) in topic %s",
            self.conversation_id,
            project_id,
            project_name,
            topic_id,
        )
        try:
            self.__snapshot_if_due(pool)
        finally:
            pool.close()
        return self.__rerouted()

    @staticmethod
    def __forget_vector(projects: ProjectManager, project_id: str) -> None:
        try:
            projects.vector_handler.delete_project_summary_vector(project_id)
        except Exception:
            logger.exception(
                "Could not remove the summary vector of project %s, whose rows rolled "
                "back; it is now unreachable",
                project_id,
            )

    def __attach(self, topic_id: str, project_id: str, mapping: MemoryMapping | None) -> None:
        if mapping is None:
            self.mapping.populate_conversation_id(self.conversation_id, self.user_id)
        self.mapping.insert_into_mapping_table(
            self.conversation_id, self.user_id, topic_id, project_id, utc_now()
        )

    def __pool(self, project_id: str, project_name: str) -> ConversationPoolManager:
        return ConversationPoolManager(
            project_id,
            project_name,
            self.conversation_id,
            database=self.database,
            embedder=self.models.embedder,
        )

    def __snapshot_if_due(self, pool: ConversationPoolManager) -> None:
        # The turns are committed; a failed snapshot is retried by the next update.
        try:
            pool.maybe_snapshot()
        except Exception:
            logger.exception(
                "The snapshot of conversation %s failed; its turns stay unsummarised "
                "until the next update",
                self.conversation_id,
            )

    def __rerouted(self) -> Route:
        """The route as stored now, with the latest project snapshot re-read."""
        self.__route = None
        return self.route

    def __exchange(self, system_response: str) -> List[Tuple[str, str]]:
        return [(ROLE_USER, self.query), (ROLE_ASSISTANT, system_response)]

    def __opening_summary(self, system_response: str) -> str:
        return f"User: {self.query}\nAssistant: {system_response}"

    @staticmethod
    def __require_text(value, name: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
