"""The pipeline over the real MemoryManager: a conversation is routed once, then resumed with its history."""

from generation_layer.context import GenerationPrompts
from generation_layer.generator import Generator
from knowledge_sufficiency.ksv_manager import KSVManager
from memory import conversation_registry as registry_module
from memory.memory_manager import KNOWN, MemoryManager
from memory.memory_mapping_handler import MemoryMappingHandler
from pipeline.atlas import AtlasPipeline
from pipeline.naming import Namer, NamingPrompts
from test.memory_layer_testing.test_memory_manager import (  # noqa: F401  (fixtures)
    database,
    draft,
    embedder,
    models,
    vectors,
)
from test.pipeline_testing.fakes import FakeDecomposer, FakeRetriever, FakeSessions, Handle, NamingModel
from test.thinking_testing.fakes import PROMPTS, ScriptedModel
from thinking_layer.orchestrator import ThinkingOrchestrator

USER = "alice"


def pipeline_for(database, models, sessions, decomposer):
    return AtlasPipeline(
        decomposer=decomposer,
        memory_models=models,
        ksv=KSVManager(),
        retriever=Handle(FakeRetriever()),
        thinking=ThinkingOrchestrator(ScriptedModel(), PROMPTS),
        generator=Generator(sessions),
        generation_prompts=GenerationPrompts("ANSWER"),
        registry=registry_module.open_conversation_registry(),
        namer=Namer(NamingModel(["Indexing Basics", "Index Study"]), NamingPrompts("C", "P")),
        database=database,
        memory_manager=MemoryManager,
    )


def test_a_conversation_is_routed_once_then_resumed_with_its_history(database, models, draft):
    sessions = FakeSessions(text="A B-tree keeps keys sorted in pages.")
    decomposer = FakeDecomposer()
    pipeline = pipeline_for(database, models, sessions, decomposer)
    conversation = pipeline.new_conversation(USER, "What is a B-tree?")

    first = pipeline.ask(conversation.conversation_id, USER, "What is a B-tree?")
    assert first.persisted, first.warnings
    assert first.record.retrieval["route"] == "no_topic"
    mapping = MemoryMappingHandler(database).search(conversation.conversation_id, USER)
    assert mapping is not None and mapping.project_id is not None

    second = pipeline.ask(conversation.conversation_id, USER, "Why are they shallow?")
    assert second.persisted, second.warnings
    assert second.record.retrieval["route"] == KNOWN
    prefix = sessions.calls[1]["prefix"]
    assert [m["content"] for m in prefix[1:]] == ["What is a B-tree?", "A B-tree keeps keys sorted in pages."]
    assert set(decomposer.responses) == {"x1", "x2"}

    memory = MemoryManager("anything", conversation.conversation_id, USER, "Databases",
                           database=database, models=models)
    context = memory.conversation_context(10)
    assert context.total_turns == 4
    assert [role for role, _ in context.turns] == ["user", "assistant", "user", "assistant"]


def test_two_conversations_keep_separate_histories(database, models, draft):
    sessions = FakeSessions()
    pipeline = pipeline_for(database, models, sessions, FakeDecomposer())
    a = pipeline.new_conversation(USER, "first")
    b = pipeline.new_conversation(USER, "second")
    pipeline.ask(a.conversation_id, USER, "Question for A")
    pipeline.ask(b.conversation_id, USER, "Question for B")
    pipeline.ask(a.conversation_id, USER, "Follow-up for A")
    last_prefix = " ".join(m["content"] for m in sessions.calls[-1]["prefix"])
    assert "Question for A" in last_prefix and "Question for B" not in last_prefix
    assert a.conversation_name != b.conversation_name
