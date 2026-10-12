"""MemoryManager: routing a conversation, recalling from it, and storing what it says.

Nothing here reaches PostgreSQL or loads a model. The vector store is a dict per
project, the embedder maps chosen texts to chosen cosines against the query, and
the draft model returns a numbered summary.
"""

import functools
import hashlib
from typing import NamedTuple

import numpy as np
import pytest

from config import Config
from data_layer.datalayer_exceptions.datalayer_exceptions import VectorNotFoundEror
from data_layer.ingestion.embedding import EmbeddingManager as embedding_module
from data_layer.ingestion.embedding.vector_ids import vector_id_for
from knowledge_sufficiency.verdict import Sufficiency
from memory import memory_manager
from memory.memory_database import MemoryDatabase
from memory.memory_manager import (
    HISTORICAL_SNAPSHOT,
    KNOWN,
    LATEST_SNAPSHOT,
    NO_PROJECT,
    NO_TOPIC,
    MemoryManager,
    MemoryModels,
)
from memory.memory_mapping_handler import MemoryMappingHandler
from memory.memory_pool_exceptions import (
    ConversationAlreadyRouted,
    ConversationNotRouted,
    EmptyQueryException,
    InvalidIdentifier,
    TopicAlreadyExists,
    TopicNotFound,
)
from memory.snapshot import PROJECT, SnapShot
from memory.topic_pool.project_pool import project_manager
from memory.topic_pool.project_pool.conversation_pool.conversation_data_management import (
    conversationVectorManager,
)
from memory.topic_pool.project_pool.conversation_pool.conversation_pool_manager import (
    ConversationPoolManager,
)
from memory.topic_pool.project_pool.conversation_pool.conversation_summary_pipeline.conversation_summary import (
    ConversationSummary,
)
from memory.topic_pool.project_pool.project_data_repo import project_meta_data
from memory.topic_pool.project_pool.project_data_repo.project_meta_data import (
    list_topic_projects,
)
from memory.topic_pool.project_pool.project_data_repo.project_snapshot_repo import (
    ProjectSnapshotRepository,
)
from memory.topic_pool.project_pool.project_data_repo.project_vector_handler import (
    ProjectVectorHandler,
    summary_vector_id,
)
from memory.topic_pool.project_pool.project_manager import ProjectManager
from memory.topic_pool.topic_manager import TopicManager
from retrieval_layer.settings import RetrievalSettings

DIMENSIONS = Config.EMBEDDING_DIMENSIONS
TOPIC = "biology"
C1, C2 = "conv_one", "conv_two"
U1, U2 = "user_one", "user_two"

# Its words reach the keyword index only through turns that mention zebras.
ASK = "why do zebra have stripes"


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(DIMENSIONS).astype(np.float32)
    return v / np.linalg.norm(v)


QUERY = unit(1)


def toward(cosine: float, seed: int) -> np.ndarray:
    """A unit vector at an exact cosine to the query."""
    other = unit(seed)
    other = other - QUERY * float(other @ QUERY)
    other = other / np.linalg.norm(other)
    v = QUERY * cosine + other * np.sqrt(1 - cosine**2)
    return (v / np.linalg.norm(v)).astype(np.float32)


def seed_of(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


class Embedded(NamedTuple):
    vector: np.ndarray
    vector_id: int


class FakeEmbedder:
    """An exact text first, then the first marker it contains, else orthogonal to the query."""

    embedding_dimension = DIMENSIONS

    def __init__(self) -> None:
        self.exact = {ASK: QUERY}
        self.markers = []
        self.model = self

    def vector_of(self, text: str) -> np.ndarray:
        if text in self.exact:
            return self.exact[text]
        for marker, vector in self.markers:
            if marker in text:
                return vector
        return toward(0.0, seed_of(text))

    def encode(self, texts, truncate_dim=None):
        if isinstance(texts, str):
            return self.vector_of(texts)
        return np.array([self.vector_of(text) for text in texts], dtype=np.float32)

    def embed_text(self, text, chunk_id=None):
        chunk_id = chunk_id or hashlib.sha256(text.encode()).hexdigest()
        return Embedded(self.vector_of(text), vector_id_for(chunk_id))

    def embed_texts(self, texts, chunk_ids=None):
        chunk_ids = chunk_ids or [None] * len(texts)
        return [self.embed_text(text, chunk_id) for text, chunk_id in zip(texts, chunk_ids)]


class FakeVectorStore:
    """pgvector's `vectors` table: one dict per project id, shared by every connection."""

    stores: dict = {}
    down = False

    def __init__(self, project_id) -> None:
        self.rows = FakeVectorStore.stores.setdefault(project_id, {})

    @staticmethod
    def __reachable() -> None:
        if FakeVectorStore.down:
            raise ConnectionError("the vector store is down")

    def insert(self, vector_id, vector) -> None:
        self.__reachable()
        self.rows[int(vector_id)] = np.asarray(vector, dtype=np.float32).copy()

    def batch_insert(self, vector_ids, vectors) -> None:
        for vector_id, vector in zip(vector_ids, vectors):
            self.insert(vector_id, vector)

    def search(self, vector_id):
        if int(vector_id) not in self.rows:
            raise VectorNotFoundEror(vector_id)
        return self.rows[int(vector_id)]

    def vectors_for(self, vector_ids):
        self.__reachable()
        return {int(v): self.rows[int(v)] for v in vector_ids if int(v) in self.rows}

    def delete(self, vector_id) -> None:
        self.rows.pop(int(vector_id), None)

    def batch_delete(self, vector_ids) -> None:
        for vector_id in vector_ids:
            self.delete(vector_id)

    def close(self) -> None:
        pass


class FakeDraftModel:
    def __init__(self) -> None:
        self.calls = 0

    def create_chat_completion(self, messages, max_tokens, temperature):
        self.calls += 1
        return {"choices": [{"message": {"content": f"summary {self.calls}"}}]}


def no_second_embedder(*args, **kwargs):
    raise AssertionError("a component loaded an embedder of its own")


@pytest.fixture(autouse=True)
def vectors(monkeypatch):
    FakeVectorStore.stores = {}
    FakeVectorStore.down = False
    monkeypatch.setattr(conversationVectorManager, "VectorRepository", FakeVectorStore)
    monkeypatch.setattr(project_meta_data, "VectorRepository", FakeVectorStore)
    monkeypatch.setattr(
        project_manager,
        "ProjectVectorHandler",
        functools.partial(ProjectVectorHandler, repository_factory=FakeVectorStore),
    )
    monkeypatch.setattr(embedding_module, "EmbeddingManager", no_second_embedder)
    return FakeVectorStore.stores


@pytest.fixture
def draft(monkeypatch):
    model = FakeDraftModel()
    monkeypatch.setattr(ConversationSummary, "_ConversationSummary__load_model", lambda self: model)
    return model


@pytest.fixture
def database(tmp_path):
    return MemoryDatabase.of(tmp_path / "memory.db")


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def models(embedder):
    return MemoryModels(embedder=embedder, settings=RetrievalSettings(rerank=False))


@pytest.fixture
def manager(database, models):
    def build(query=ASK, conversation=C1, user=U1, topic=TOPIC, **kwargs):
        return MemoryManager(
            query, conversation, user, topic, database=database, models=models, **kwargs
        )

    return build


def new_topic(database, name=TOPIC) -> str:
    return TopicManager(name, database=database).create_new_topic()


def new_project(database, embedder, topic_id, name, summary=None) -> str:
    with ProjectManager(topic_id, "seed", database=database, embedder=embedder) as projects:
        return projects.create_project(name, summary or f"{name} summary")


def project_snapshot(database, project_id, vector, seq) -> int:
    snapshot = SnapShot(project_id, project_id, scope=PROJECT, database=database)
    try:
        return snapshot.add_project_snapshot(f"snapshot {seq}", seq, vector)
    finally:
        snapshot.close()


def pool(database, embedder, project_id, conversation=C1) -> ConversationPoolManager:
    return ConversationPoolManager(
        project_id, "alpha", conversation, database=database, embedder=embedder
    )


def texts(passages):
    return [passage.text for passage in passages]


def converse(manager, *exchanges, conversation=C1):
    """Start a conversation in a new topic, then add each later exchange to it."""
    (question, answer), *rest = exchanges
    route = manager(query=question, conversation=conversation).store_new_topic(answer, "alpha")
    for question, answer in rest:
        route = manager(query=question, conversation=conversation).update(answer)
    return route


class TestConstruction:
    @pytest.mark.parametrize("bad", ["", "   ", None])
    def test_an_empty_query_is_refused(self, manager, bad):
        with pytest.raises(EmptyQueryException):
            manager(query=bad)

    @pytest.mark.parametrize("field", ["conversation", "user"])
    def test_an_empty_identifier_is_refused(self, manager, field):
        with pytest.raises(InvalidIdentifier):
            manager(**{field: " "})

    def test_an_empty_topic_is_refused(self, manager):
        with pytest.raises(ValueError):
            manager(topic="")


class TestRouting:
    def test_an_unknown_topic_routes_nowhere(self, manager):
        m = manager()
        assert m.route.found_by == NO_TOPIC
        assert m.conversation_retrieval() is None
        assert m.sufficiency is None

    def test_a_topic_with_no_projects_routes_nowhere(self, database, manager):
        topic_id = new_topic(database)
        route = manager().route
        assert (route.found_by, route.topic_id, route.project_id) == (NO_PROJECT, topic_id, None)

    def test_the_latest_snapshot_decides_first(self, database, embedder, manager):
        topic_id = new_topic(database)
        alpha = new_project(database, embedder, topic_id, "alpha")
        beta = new_project(database, embedder, topic_id, "beta")
        project_snapshot(database, alpha, toward(0.2, 21), 1)
        alpha_latest = project_snapshot(database, alpha, toward(0.8, 22), 2)
        project_snapshot(database, beta, toward(0.5, 23), 3)

        route = manager().route
        assert route.found_by == LATEST_SNAPSHOT
        assert (route.project_id, route.project_name) == (alpha, "alpha")
        assert route.latest_project_snapshot_id == alpha_latest
        assert route.score == pytest.approx(0.8, abs=1e-5)

    def test_history_decides_when_no_latest_snapshot_clears_the_floor(
        self, database, embedder, manager
    ):
        topic_id = new_topic(database)
        alpha = new_project(database, embedder, topic_id, "alpha")
        beta = new_project(database, embedder, topic_id, "beta")
        project_snapshot(database, alpha, toward(0.7, 21), 1)
        alpha_latest = project_snapshot(database, alpha, toward(0.1, 22), 2)
        project_snapshot(database, beta, toward(0.2, 23), 3)

        route = manager().route
        assert (route.found_by, route.project_id) == (HISTORICAL_SNAPSHOT, alpha)
        assert route.score == pytest.approx(0.7, abs=1e-5)
        assert route.latest_project_snapshot_id == alpha_latest

    def test_a_latest_match_wins_over_a_stronger_old_one(self, database, embedder, manager):
        topic_id = new_topic(database)
        alpha = new_project(database, embedder, topic_id, "alpha")
        beta = new_project(database, embedder, topic_id, "beta")
        project_snapshot(database, alpha, toward(0.9, 21), 1)
        project_snapshot(database, alpha, toward(0.1, 22), 2)
        project_snapshot(database, beta, toward(0.5, 23), 3)

        route = manager().route
        assert (route.found_by, route.project_id) == (LATEST_SNAPSHOT, beta)

    def test_nothing_over_either_floor_routes_nowhere(self, database, embedder, manager):
        topic_id = new_topic(database)
        alpha = new_project(database, embedder, topic_id, "alpha")
        project_snapshot(database, alpha, toward(0.3, 21), 1)
        project_snapshot(database, alpha, toward(0.2, 22), 2)

        m = manager()
        assert m.route.found_by == NO_PROJECT
        assert m.conversation_retrieval() is None

    def test_the_floors_are_the_callers_to_set(self, database, embedder, manager):
        topic_id = new_topic(database)
        alpha = new_project(database, embedder, topic_id, "alpha")
        project_snapshot(database, alpha, toward(0.3, 21), 1)
        project_snapshot(database, alpha, toward(0.2, 22), 2)

        assert manager(latest_floor=0.15).route.found_by == LATEST_SNAPSHOT
        assert manager(historical_floor=0.25).route.found_by == HISTORICAL_SNAPSHOT

    def test_a_project_with_no_snapshot_is_routed_by_its_summary(
        self, database, embedder, manager
    ):
        embedder.exact["alpha summary"] = toward(0.8, 31)
        topic_id = new_topic(database)
        alpha = new_project(database, embedder, topic_id, "alpha")

        route = manager().route
        assert (route.found_by, route.project_id) == (LATEST_SNAPSHOT, alpha)
        assert route.latest_project_snapshot_id is None

    def test_a_known_conversation_is_not_routed_again(self, manager, monkeypatch):
        stored = manager(query="first question").store_new_topic("first answer", "alpha")

        def never(self):
            raise AssertionError("a mapped conversation was routed again")

        monkeypatch.setattr(ProjectManager, "route_by_snapshots", never)
        route = manager().route
        assert (route.found_by, route.project_id, route.project_name) == (
            KNOWN, stored.project_id, "alpha",
        )

    def test_an_opened_but_unrouted_conversation_is_routed(self, database, embedder, manager):
        embedder.exact["alpha summary"] = toward(0.8, 31)
        alpha = new_project(database, embedder, new_topic(database), "alpha")
        MemoryMappingHandler(database).populate_conversation_id(C1, U1)

        route = manager().route
        assert (route.found_by, route.project_id) == (LATEST_SNAPSHOT, alpha)

    def test_a_mapping_belongs_to_its_user(self, manager):
        manager(query="first question").store_new_topic("first answer", "alpha")
        assert manager(user=U2).route.found_by == NO_PROJECT


class TestRecall:
    def test_a_turn_that_answers_the_query_comes_back(self, embedder, vectors, manager):
        embedder.markers = [("zebra", toward(0.9, 41))]
        route = converse(
            manager,
            ("tell me about horses", "horses are fast"),
            ("and zebra stripes?", "zebra stripes confuse biting flies"),
        )

        m = manager()
        passages = m.conversation_retrieval()
        assert "zebra stripes confuse biting flies" in texts(passages)
        assert "horses are fast" not in texts(passages)
        assert m.sufficiency is Sufficiency.FULL
        # No snapshot has covered these turns, so their vectors were embedded
        # for the check rather than read from the store.
        assert all(
            passage.vector_id not in vectors.get(route.project_id, {}) for passage in passages
        )

    def test_a_weak_match_is_partial(self, embedder, manager):
        embedder.markers = [("zebra", toward(0.45, 41))]
        converse(manager, ("and zebra stripes?", "zebra stripes confuse biting flies"))

        m = manager()
        assert texts(m.conversation_retrieval())
        assert m.sufficiency is Sufficiency.PARTIAL

    def test_no_match_returns_none(self, embedder, manager):
        embedder.markers = [("zebra", toward(0.1, 41))]
        converse(manager, ("and zebra stripes?", "zebra stripes confuse biting flies"))

        m = manager()
        assert m.conversation_retrieval() is None
        assert m.sufficiency is Sufficiency.NONE

    def test_only_the_supporting_passages_come_back(self, embedder, manager):
        embedder.markers = [
            ("stripes confuse", toward(0.9, 41)),
            ("zebra", toward(0.1, 42)),
        ]
        converse(manager, ("and zebra stripes?", "zebra stripes confuse biting flies"))

        assert texts(manager().conversation_retrieval()) == [
            "zebra stripes confuse biting flies"
        ]

    def test_a_stored_turn_vector_is_used_rather_than_embedded_again(
        self, database, embedder, vectors, manager
    ):
        embedder.markers = [("zebra", toward(0.1, 41))]
        route = converse(manager, ("and zebra stripes?", "zebra stripes confuse biting flies"))
        answer = pool(database, embedder, route.project_id).history()[-1]
        vectors[route.project_id][vector_id_for(answer.chunk_id)] = toward(0.9, 51)

        m = manager()
        assert texts(m.conversation_retrieval()) == ["zebra stripes confuse biting flies"]
        assert m.sufficiency is Sufficiency.FULL

    def test_an_unreachable_vector_store_falls_back_to_embedding(self, embedder, manager):
        embedder.markers = [("zebra", toward(0.9, 41))]
        converse(manager, ("and zebra stripes?", "zebra stripes confuse biting flies"))
        FakeVectorStore.down = True

        m = manager()
        assert "zebra stripes confuse biting flies" in texts(m.conversation_retrieval())
        assert m.sufficiency is Sufficiency.FULL

    def test_a_routed_conversation_recalls_its_projects_other_conversations(
        self, embedder, manager
    ):
        embedder.markers = [("zebra", toward(0.9, 41))]
        stored = converse(manager, ("and zebra stripes?", "zebra stripes confuse biting flies"))

        m = manager(conversation=C2)
        passages = m.conversation_retrieval()
        assert (m.route.found_by, m.route.project_id) == (LATEST_SNAPSHOT, stored.project_id)
        assert "zebra stripes confuse biting flies" in texts(passages)
        assert {passage.document_id for passage in passages} == {C1}


class TestStoringANewTopic:
    def test_it_stores_the_topic_project_mapping_and_both_turns(
        self, database, embedder, vectors, manager
    ):
        route = manager(query="first question").store_new_topic("first answer", "alpha")

        assert route.found_by == KNOWN
        assert TopicManager(TOPIC, database=database).is_topic_exists()
        assert [p.project_name for p in list_topic_projects(route.topic_id, database)] == ["alpha"]
        mapped = MemoryMappingHandler(database).search(C1, U1)
        assert (mapped.topic_id, mapped.project_id) == (route.topic_id, route.project_id)
        turns = pool(database, embedder, route.project_id).history()
        assert [(t.role, t.text) for t in turns] == [
            ("user", "first question"), ("assistant", "first answer"),
        ]
        assert summary_vector_id(route.project_id) in vectors[route.project_id]

    def test_the_projects_first_summary_is_the_opening_exchange(self, database, manager):
        route = manager(query="first question").store_new_topic("first answer", "alpha")
        (project,) = list_topic_projects(route.topic_id, database)
        assert project.project_summary == "User: first question\nAssistant: first answer"

    def test_a_topic_that_exists_already_is_refused_and_nothing_is_written(
        self, database, vectors, manager
    ):
        new_topic(database)
        with pytest.raises(TopicAlreadyExists):
            manager().store_new_topic("an answer", "alpha")
        assert MemoryMappingHandler(database).search(C1, U1) is None
        assert not any(vectors.values())

    def test_a_failure_after_the_project_vector_rolls_everything_back(
        self, database, vectors, manager, monkeypatch
    ):
        def full_disk(self, turns):
            raise RuntimeError("disk full")

        monkeypatch.setattr(ConversationPoolManager, "add_turns", full_disk)
        with pytest.raises(RuntimeError, match="disk full"):
            manager().store_new_topic("an answer", "alpha")

        assert not TopicManager(TOPIC, database=database).is_topic_exists()
        assert MemoryMappingHandler(database).search(C1, U1) is None
        assert not any(vectors.values())

    @pytest.mark.parametrize("response, name", [("  ", "alpha"), ("an answer", "")])
    def test_empty_text_is_refused_before_anything_is_written(
        self, database, manager, response, name
    ):
        with pytest.raises(ValueError):
            manager().store_new_topic(response, name)
        assert not TopicManager(TOPIC, database=database).is_topic_exists()


class TestStoringANewProject:
    def test_it_joins_the_existing_topic(self, database, manager):
        first = manager(query="q1").store_new_topic("a1", "alpha")
        second = manager(query="q2", conversation=C2).store_new_project("a2", "beta")

        assert second.found_by == KNOWN
        assert second.topic_id == first.topic_id
        assert second.project_id != first.project_id
        names = [p.project_name for p in list_topic_projects(first.topic_id, database)]
        assert names == ["alpha", "beta"]

    def test_it_needs_its_topic(self, database, manager):
        with pytest.raises(TopicNotFound):
            manager().store_new_project("an answer", "alpha")
        assert MemoryMappingHandler(database).search(C1, U1) is None

    def test_a_routed_conversation_cannot_start_another_project(self, database, manager):
        manager(query="q1").store_new_topic("a1", "alpha")
        with pytest.raises(ConversationAlreadyRouted):
            manager(topic="geology").store_new_topic("a2", "beta")
        with pytest.raises(ConversationAlreadyRouted):
            manager().store_new_project("a2", "beta")
        assert not TopicManager("geology", database=database).is_topic_exists()


class TestUpdate:
    def test_it_appends_the_exchange_to_a_known_conversation(self, database, embedder, manager):
        first = manager(query="q1").store_new_topic("a1", "alpha")
        route = manager(query="q2").update("a2")

        assert (route.found_by, route.project_id) == (KNOWN, first.project_id)
        turns = pool(database, embedder, first.project_id).history()
        assert [(t.sequence_number, t.role, t.text) for t in turns] == [
            (1, "user", "q1"), (2, "assistant", "a1"),
            (3, "user", "q2"), (4, "assistant", "a2"),
        ]

    def test_it_attaches_a_routed_conversation_to_its_project(
        self, database, embedder, manager
    ):
        embedder.markers = [("zebra", toward(0.9, 41))]
        stored = manager(query="zebra question").store_new_topic("zebra answer", "alpha")

        m = manager(conversation=C2)
        assert m.route.found_by == LATEST_SNAPSHOT
        route = m.update("about zebras")

        assert (route.found_by, route.project_id) == (KNOWN, stored.project_id)
        assert MemoryMappingHandler(database).search(C2, U1).project_id == stored.project_id
        turns = pool(database, embedder, stored.project_id, C2).history()
        assert [t.text for t in turns] == [ASK, "about zebras"]

    def test_it_attaches_a_conversation_whose_mapping_row_is_open(
        self, database, embedder, manager
    ):
        embedder.exact["alpha summary"] = toward(0.8, 31)
        alpha = new_project(database, embedder, new_topic(database), "alpha")
        MemoryMappingHandler(database).populate_conversation_id(C1, U1)

        route = manager().update("an answer")
        assert (route.found_by, route.project_id) == (KNOWN, alpha)

    def test_a_conversation_with_nowhere_to_go_is_refused(self, manager):
        with pytest.raises(ConversationNotRouted):
            manager().update("an answer")

    def test_an_empty_response_is_refused(self, database, embedder, manager):
        first = manager(query="q1").store_new_topic("a1", "alpha")
        with pytest.raises(ValueError):
            manager(query="q2").update(" ")
        assert pool(database, embedder, first.project_id).size() == 2


class TestSnapshots:
    def test_a_full_turn_window_snapshots_the_conversation_and_the_project(
        self, draft, database, embedder, vectors, manager
    ):
        exchanges = [(f"q{i}", f"a{i}") for i in range(Config.SNAPSHOT_EVERY_N_TURNS // 2)]
        route = converse(manager, *exchanges)

        conversation = pool(database, embedder, route.project_id)
        assert conversation.summarised_upto() == Config.SNAPSHOT_EVERY_N_TURNS
        assert conversation.current_summary() == "summary 1"
        latest = ProjectSnapshotRepository(route.project_id, database).latest()
        assert latest.summary == "summary 2"
        assert route.latest_project_snapshot_id == latest.project_snapshot_id
        assert latest.project_snapshot_id in vectors[route.project_id]

    def test_each_conversation_snapshot_moves_the_project_snapshot_on(
        self, draft, database, manager
    ):
        exchanges = [(f"q{i}", f"a{i}") for i in range(Config.SNAPSHOT_EVERY_N_TURNS)]
        route = converse(manager, *exchanges)

        history = ProjectSnapshotRepository(route.project_id, database).history()
        assert len(history) == 2
        assert history[0].last_seq_included < history[1].last_seq_included

    def test_fewer_turns_than_the_window_take_no_snapshot(self, draft, manager):
        converse(manager, ("q0", "a0"), ("q1", "a1"), ("q2", "a2"))
        assert draft.calls == 0

    def test_a_context_over_the_token_window_snapshots_early(
        self, draft, database, embedder, manager
    ):
        long_answer = "word " * Config.MAIN_MODEL_CONTEXT_WINDOW_TOKENS
        route = manager(query="q0").store_new_topic(long_answer, "alpha")

        assert pool(database, embedder, route.project_id).summarised_upto() == 2

    def test_a_failed_snapshot_keeps_the_turns_and_the_next_update_retries(
        self, database, embedder, manager, monkeypatch
    ):
        attempts = []

        def missing(self):
            attempts.append(1)
            raise FileNotFoundError("no draft model")

        monkeypatch.setattr(ConversationSummary, "_ConversationSummary__load_model", missing)
        exchanges = [(f"q{i}", f"a{i}") for i in range(Config.SNAPSHOT_EVERY_N_TURNS // 2)]
        route = converse(manager, *exchanges)

        conversation = pool(database, embedder, route.project_id)
        assert attempts == [1]
        assert conversation.size() == Config.SNAPSHOT_EVERY_N_TURNS
        assert conversation.summarised_upto() == 0

        model = FakeDraftModel()
        monkeypatch.setattr(
            ConversationSummary, "_ConversationSummary__load_model", lambda self: model
        )
        manager(query="one more").update("and its answer")
        assert conversation.summarised_upto() == Config.SNAPSHOT_EVERY_N_TURNS + 2


class TestModels:
    def test_managers_share_the_process_models(self, database):
        first = MemoryManager("q", C1, U1, TOPIC, database=database)
        second = MemoryManager("q", C2, U2, TOPIC, database=database)
        assert first.models is second.models is memory_manager.shared_models()

    def test_the_embedder_loads_once_however_it_is_reached(self, monkeypatch):
        built = []

        class Counting(FakeEmbedder):
            def __init__(self):
                super().__init__()
                built.append(self)

        monkeypatch.setattr(embedding_module, "EmbeddingManager", Counting)
        models = MemoryModels(settings=RetrievalSettings(rerank=False))
        models.embedder
        models.preparation.encode(["one", "two"])
        models.embedder
        assert len(built) == 1
        assert models.embedder is built[0]
