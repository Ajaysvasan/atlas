"""The pipeline's own decisions: ownership, retrieval order, KSV gating, acquisition, persistence, telemetry."""

import pytest

from generation_layer.context import GenerationPrompts
from generation_layer.generator import Generator
from inference.inference_exceptions import InferenceError
from knowledge_sufficiency.ksv_manager import KSVManager
from knowledge_sufficiency.verdict import Sufficiency
from memory import conversation_registry as registry_module
from pipeline.atlas import AtlasPipeline
from pipeline.naming import Namer, NamingPrompts
from query_decomposition.models import Mode
from test.pipeline_testing.fakes import (
    FakeAcquirer,
    FakeDecomposer,
    FakeMemory,
    FakeRetriever,
    FakeSessions,
    Handle,
    Models,
    NamingModel,
    passage,
)
from test.thinking_testing.fakes import PROMPTS, ScriptedModel
from thinking_layer.orchestrator import ThinkingOrchestrator

USER, OTHER = "alice", "bob"


@pytest.fixture(autouse=True)
def fresh_memory():
    FakeMemory.store = {}
    FakeMemory.fail_update = False
    FakeMemory.recall = []
    FakeMemory.sufficiency = None


def build(tmp_path, *, retriever=None, decomposer=None, sessions=None, acquirer=None, names=None,
          thinking_model=None):
    pipeline = AtlasPipeline(
        decomposer=decomposer or FakeDecomposer(),
        memory_models=Models(),
        ksv=KSVManager(),
        retriever=Handle(retriever or FakeRetriever()),
        thinking=ThinkingOrchestrator(thinking_model or ScriptedModel(), PROMPTS),
        generator=Generator(sessions or FakeSessions()),
        generation_prompts=GenerationPrompts("ANSWER"),
        registry=registry_module.open_conversation_registry(),
        namer=Namer(names or NamingModel(), NamingPrompts("NAME CONVERSATION", "NAME PROJECT")),
        database=tmp_path / "memory.db",
        acquirer=acquirer,
        memory_manager=FakeMemory,
    )
    return pipeline


MVCC = [passage(1, "MVCC keeps row versions so readers never block writers."),
        passage(2, "Unrelated text about baking bread.")]


class TestConversations:
    def test_the_first_query_names_the_conversation(self, tmp_path):
        pipeline = build(tmp_path)
        record = pipeline.new_conversation(USER, "Explain PostgreSQL MVCC")
        assert record.conversation_name == "PostgreSQL MVCC"
        assert record.created_at.endswith("+00:00")

    def test_names_are_unique_per_user_whatever_the_model_proposes(self, tmp_path):
        pipeline = build(tmp_path, names=NamingModel(["PostgreSQL MVCC"]))
        names = [pipeline.new_conversation(USER, "q").conversation_name for _ in range(3)]
        assert names == ["PostgreSQL MVCC", "PostgreSQL MVCC (2)", "PostgreSQL MVCC (3)"]
        assert pipeline.new_conversation(OTHER, "q").conversation_name == "PostgreSQL MVCC"

    def test_a_naming_failure_falls_back_to_the_query_and_stays_unique(self, tmp_path):
        pipeline = build(tmp_path, names=NamingModel(fail=True))
        first = pipeline.new_conversation(USER, "How does Raft achieve leader election?")
        second = pipeline.new_conversation(USER, "How does Raft achieve leader election?")
        assert first.conversation_name == "Raft Achieve Leader Election"
        assert second.conversation_name == "Raft Achieve Leader Election (2)"

    def test_another_users_conversation_cannot_be_used(self, tmp_path):
        decomposer = FakeDecomposer()
        pipeline = build(tmp_path, decomposer=decomposer)
        conversation = pipeline.new_conversation(USER, "q")
        response = pipeline.ask(conversation.conversation_id, OTHER, "What is MVCC?")
        assert response.text is None and response.record.error_category == "conversation"
        assert decomposer.queries == [] and FakeMemory.store == {}


class TestRetrievalAndKSV:
    def test_an_empty_knowledge_base_still_answers_and_says_so(self, tmp_path):
        thinking = ScriptedModel()
        pipeline = build(tmp_path, thinking_model=thinking)
        conversation = pipeline.new_conversation(USER, "q")
        response = pipeline.ask(conversation.conversation_id, USER, "What is MVCC?")
        assert response.status == "completed" and response.persisted
        assert response.record.retrieval["global"] == "empty"
        worker = next(c for c in thinking.calls if c["role"] == "worker")
        assert "No relevant material" in worker["user"]

    def test_ksv_keeps_only_relevant_passages_as_evidence(self, tmp_path):
        sessions = FakeSessions()
        pipeline = build(tmp_path, retriever=FakeRetriever(MVCC), sessions=sessions)
        conversation = pipeline.new_conversation(USER, "q")
        response = pipeline.ask(conversation.conversation_id, USER, "What is MVCC?")
        assert response.record.retrieval["global"] == Sufficiency.FULL.value
        assert response.record.retrieval["global_passages"] == 1
        turn = sessions.calls[0]["turn"]["content"]
        assert "[G1]" in turn and "MVCC keeps row versions" in turn
        assert "baking bread" not in turn

    def test_a_doubt_memory_fully_answers_skips_global_retrieval(self, tmp_path):
        FakeMemory.sufficiency = Sufficiency.FULL
        FakeMemory.recall = [passage(9, "Earlier we said MVCC keeps versions.")]
        retriever = FakeRetriever(MVCC)
        pipeline = build(tmp_path, retriever=retriever)
        conversation = pipeline.new_conversation(USER, "q")
        response = pipeline.ask(conversation.conversation_id, USER, "What is MVCC?")
        assert retriever.queries == []
        assert response.record.retrieval["global"].startswith("skipped")

    def test_a_development_query_searches_globally_even_when_memory_suffices(self, tmp_path):
        FakeMemory.sufficiency = Sufficiency.FULL
        retriever = FakeRetriever(MVCC)
        pipeline = build(tmp_path, retriever=retriever,
                         decomposer=FakeDecomposer(Mode.DEVELOPMENT, ("MVCC internals",)))
        conversation = pipeline.new_conversation(USER, "q")
        pipeline.ask(conversation.conversation_id, USER, "Design MVCC cleanup")
        assert retriever.queries == ["Design MVCC cleanup", "MVCC internals"]
        # Global retrieval sees only public documents and this user's uploads.
        assert retriever.viewers == [USER, USER]

    def test_nothing_relevant_triggers_acquisition_then_one_more_search(self, tmp_path):
        retriever = FakeRetriever([passage(2, "Unrelated text about baking bread.")])
        acquirer = FakeAcquirer(on_acquire=lambda: retriever.passages.append(MVCC[0]))
        pipeline = build(tmp_path, retriever=retriever, acquirer=acquirer)
        conversation = pipeline.new_conversation(USER, "q")
        response = pipeline.ask(conversation.conversation_id, USER, "What is MVCC?")
        assert len(acquirer.calls) == 1 and retriever.refreshed == 1
        assert len(retriever.queries) == 2
        assert response.record.retrieval["global"] == "full"
        assert "1 document(s) stored" in response.record.retrieval["acquisition"]

    def test_no_acquisition_when_disabled(self, tmp_path):
        pipeline = build(tmp_path, retriever=FakeRetriever([]))
        conversation = pipeline.new_conversation(USER, "q")
        response = pipeline.ask(conversation.conversation_id, USER, "What is MVCC?")
        assert "acquisition" not in response.record.retrieval

    def test_an_unreachable_knowledge_base_is_a_limitation_not_a_failure(self, tmp_path):
        thinking = ScriptedModel()
        pipeline = build(tmp_path, retriever=FakeRetriever(fail=ConnectionError("pg down")),
                         thinking_model=thinking)
        conversation = pipeline.new_conversation(USER, "q")
        response = pipeline.ask(conversation.conversation_id, USER, "What is MVCC?")
        assert response.text and response.status == "completed"
        worker = next(c for c in thinking.calls if c["role"] == "worker")
        assert "could not be searched" in worker["user"]


class TestPersistence:
    def test_a_new_conversation_is_stored_as_a_new_topic_with_a_named_project(self, tmp_path):
        pipeline = build(tmp_path, names=NamingModel(["PostgreSQL MVCC", "MVCC Study"]))
        conversation = pipeline.new_conversation(USER, "q")
        response = pipeline.ask(conversation.conversation_id, USER, "What is MVCC?")
        stored = FakeMemory.store[conversation.conversation_id]
        assert stored["created"] == ("topic", "Databases", "MVCC Study")
        assert stored["turns"] == [("user", "What is MVCC?"), ("assistant", response.text)]
        assert response.record.persisted == {"memory": True, "exchange": True}

    def test_the_second_query_updates_and_sees_the_first(self, tmp_path):
        sessions = FakeSessions()
        pipeline = build(tmp_path, sessions=sessions)
        conversation = pipeline.new_conversation(USER, "q")
        pipeline.ask(conversation.conversation_id, USER, "What is MVCC?")
        pipeline.ask(conversation.conversation_id, USER, "And VACUUM?")
        assert len(FakeMemory.store[conversation.conversation_id]["turns"]) == 4
        prefix = sessions.calls[1]["prefix"]
        assert [m["role"] for m in prefix] == ["system", "user", "assistant"]
        assert prefix[1]["content"] == "What is MVCC?"

    def test_a_memory_write_failure_is_reported_never_hidden(self, tmp_path):
        FakeMemory.fail_update = True
        decomposer = FakeDecomposer()
        pipeline = build(tmp_path, decomposer=decomposer)
        conversation = pipeline.new_conversation(USER, "q")
        response = pipeline.ask(conversation.conversation_id, USER, "What is MVCC?")
        assert response.text is not None
        assert not response.persisted
        assert response.status == "completed_with_warnings"
        assert any("NOT saved" in warning for warning in response.warnings)

    def test_a_failed_generation_persists_nothing(self, tmp_path):
        decomposer = FakeDecomposer()
        pipeline = build(tmp_path, decomposer=decomposer,
                         sessions=FakeSessions(fail=InferenceError("llama crashed")))
        conversation = pipeline.new_conversation(USER, "q")
        response = pipeline.ask(conversation.conversation_id, USER, "What is MVCC?")
        assert response.text is None
        assert response.record.error_category == "model"
        assert FakeMemory.store == {} and decomposer.responses == {}


class TestTelemetry:
    def test_stages_and_total_are_recorded_on_a_monotonic_clock(self, tmp_path):
        pipeline = build(tmp_path, retriever=FakeRetriever(MVCC))
        conversation = pipeline.new_conversation(USER, "q")
        record = pipeline.ask(conversation.conversation_id, USER, "What is MVCC?").record
        top = [k for k in record.stages if "." not in k]
        assert top == ["decomposition", "memory", "retrieval", "ksv", "thinking", "generation",
                       "persistence"]
        assert record.total_ms >= sum(record.stages[k] for k in top) - 1
        assert pipeline.telemetry.last() is record

    def test_retries_are_recorded_apart_from_runtime_failures(self, tmp_path):
        from test.thinking_testing.fakes import accept, revise

        thinking = ScriptedModel({"job_critic": lambda u, n: revise("thin") if n == 0 else accept()})
        pipeline = build(tmp_path, thinking_model=thinking, retriever=FakeRetriever(MVCC))
        conversation = pipeline.new_conversation(USER, "q")
        record = pipeline.ask(conversation.conversation_id, USER, "What is MVCC?").record
        assert (record.job_retries, record.plan_revisions, record.runtime_retries) == (1, 0, 0)
        assert record.retried_jobs == ["j1"]
        assert "thin" in record.retry_reasons[0]
