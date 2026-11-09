"""The pipeline with the real draft model: decomposition, thinking, naming, generation and KV state.

Retrieval is a fixed set of database passages and memory is a dict, so these
tests touch neither PostgreSQL nor the developer's data. Skipped unless the
model is installed. Slow on CPU: several minutes in all.
"""

import pytest

from generation_layer.context import load_generation_prompts
from generation_layer.generator import Generator
from inference import runtime
from inference.checkpoints import CheckpointStore
from inference.runtime import InferenceRuntime, ReasoningModel
from inference.sessions import KVSessionManager
from knowledge_sufficiency.ksv_manager import KSVManager
from memory import conversation_registry as registry_module
from pipeline.atlas import AtlasPipeline
from pipeline.naming import Namer
from query_decomposition.decompose import Decompose
from query_decomposition.models import Mode
from test.pipeline_testing.fakes import (
    Encoder,
    FakeDecomposer,
    FakeMemory,
    FakeRetriever,
    Handle,
    Models,
    passage,
)
from thinking_layer.models import ExecutionPath
from thinking_layer.orchestrator import ThinkingOrchestrator
from thinking_layer.settings import ThinkingSettings

pytestmark = pytest.mark.real_model

USER = "e2e-user"
CORPUS = [
    passage(1, "PostgreSQL MVCC: each transaction sees a snapshot; an UPDATE writes a new row "
               "version and marks the old one dead. VACUUM later reclaims dead tuples."),
    passage(2, "Write-ahead logging (WAL): changes are written to the log and flushed before "
               "the data pages, so after a crash the log is replayed from the last checkpoint."),
    passage(3, "A B+ tree stores all keys in leaf pages linked together; internal pages hold "
               "only separator keys, which keeps the tree shallow and range scans sequential."),
]


@pytest.fixture(autouse=True)
def fresh_memory():
    FakeMemory.store = {}
    FakeMemory.fail_update = False
    FakeMemory.recall = []
    FakeMemory.sufficiency = None


@pytest.fixture
def atlas(real_model, tmp_path, conversation_rows):
    shared = InferenceRuntime(real_model)
    runtime.install_runtime(shared)
    with real_model.exclusive() as llm:
        llm.reset()
    database = tmp_path / "memory.db"
    sessions = KVSessionManager(real_model, CheckpointStore(tmp_path / "kv", database=database))

    def build(decomposer=None, relevant="MVCC", settings=None):
        return AtlasPipeline(
            decomposer=decomposer or Decompose(database=database, exchanges=conversation_rows),
            memory_models=Models(Encoder(relevant)),
            ksv=KSVManager(),
            retriever=Handle(FakeRetriever(CORPUS)),
            thinking=ThinkingOrchestrator(ReasoningModel(shared), settings=settings),
            generator=Generator(sessions, max_tokens=300),
            generation_prompts=load_generation_prompts(),
            registry=registry_module.open_conversation_registry(),
            namer=Namer(real_model),
            database=database,
            memory_manager=FakeMemory,
        )

    build.sessions = sessions
    return build


def test_scenario_a_and_e_a_simple_question_on_a_new_named_conversation(atlas):
    pipeline = atlas()
    query = "What is MVCC in PostgreSQL and why does VACUUM matter?"
    conversation = pipeline.new_conversation(USER, query)
    assert 0 < len(conversation.conversation_name) <= 60
    assert conversation.conversation_name.lower() not in ("question", "help", "untitled conversation")

    response = pipeline.ask(conversation.conversation_id, USER, query)
    assert response.text, response.error
    assert response.persisted
    record = response.record
    assert record.mode in ("doubt", "development")
    if record.mode == "doubt":
        assert record.path == ExecutionPath.DIRECT.value and record.jobs == 1
    assert record.retrieval["global"] == "full"
    assert record.cache["outcome"] == "miss" and record.cache["saved_tokens"] > 0
    assert "vacuum" in response.text.lower() or "dead" in response.text.lower()


def test_scenario_b_a_complex_request_is_planned_executed_and_critiqued(atlas):
    pipeline = atlas(FakeDecomposer(Mode.DEVELOPMENT), relevant="B+ tree",
                     settings=ThinkingSettings())
    query = ("Design an index strategy for an orders table queried by customer and date range. "
             "Explain why a B+ tree suits the range scans, and list the trade-offs.")
    conversation = pipeline.new_conversation(USER, query)
    response = pipeline.ask(conversation.conversation_id, USER, query)
    assert response.text, response.error
    record = response.record
    assert record.path == ExecutionPath.PLANNED.value
    assert record.model_calls["thinking"] >= 3  # planner, at least one worker and its critique
    assert record.thinking_status in ("completed", "completed_with_limitations")
    assert record.reasoning_retries <= record.jobs * 1 + 1  # bounded by the configured budgets


def test_scenarios_g_and_h_switching_restores_each_conversation_and_survives_a_lost_cache(atlas):
    pipeline = atlas(FakeDecomposer(Mode.DOUBT))
    a = pipeline.new_conversation(USER, "MVCC")
    b = pipeline.new_conversation(USER, "WAL")
    first_a = pipeline.ask(a.conversation_id, USER, "What is MVCC?")
    second_a = pipeline.ask(a.conversation_id, USER, "How does it avoid read locks?")
    pipeline.ask(b.conversation_id, USER, "What is WAL?")
    third_a = pipeline.ask(a.conversation_id, USER, "And what does VACUUM do?")
    assert first_a.text and second_a.text and third_a.text
    # A's saved prefix holds its own history, which B's turn never touched.
    assert third_a.record.cache["outcome"] == "restored"
    assert third_a.record.cache["evaluated_prompt_tokens"] < third_a.record.cache["prompt_tokens"]

    for path in atlas.sessions.store.directory.iterdir():
        path.unlink()
    recovered = pipeline.ask(a.conversation_id, USER, "Summarise in one sentence.")
    assert recovered.text
    assert recovered.record.cache["outcome"] in ("invalid", "miss", "resident")
