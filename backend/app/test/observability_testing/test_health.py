"""status reports what is actually wrong, distinguishes empty from broken, and never leaks secrets."""

import os

import pytest

from inference.checkpoints import CheckpointStore
from inference.runtime import InferenceRuntime
from inference.sessions import KVSessionManager
from inference.shared_model import ModelSpec, SharedModel
from knowledge_sufficiency.ksv_manager import KSVManager
from memory import conversation_registry as registry_module
from observability import health
from observability.health import Health, check_all
from observability.telemetry import Telemetry
from pipeline.atlas import RetrieverHandle


class Components:
    def __init__(self, tmp_path, model_path) -> None:
        draft = SharedModel(ModelSpec(model_path))
        self.runtime = InferenceRuntime(draft)
        self.ksv = KSVManager()
        self.registry = registry_module.open_conversation_registry()
        self.checkpoints = CheckpointStore(tmp_path / "kv")
        self.sessions = KVSessionManager(draft, self.checkpoints)
        self.telemetry = Telemetry()
        self.retriever = RetrieverHandle(lambda: None)
        self.acquirer = None
        from generation_layer.generator import Generator

        self.generator = Generator(self.sessions)


@pytest.fixture
def pg_down(monkeypatch):
    secret = "s3cr3t-password"
    monkeypatch.setenv("DB_PASSWORD", secret)

    def refuse():
        raise ConnectionError(f"connection to server failed (password={secret})")

    monkeypatch.setattr(health, "_postgres", refuse)
    return secret


def by_name(results):
    return {r.name: r for r in results}


def test_a_missing_model_and_a_down_database_are_reported_as_such(tmp_path, pg_down):
    results = by_name(check_all(Components(tmp_path, tmp_path / "absent.gguf")))
    assert results["Draft model"].status in (Health.NOT_CONFIGURED, Health.UNAVAILABLE)
    assert results["PostgreSQL"].status is Health.UNAVAILABLE
    assert results["Vector storage (pgvector)"].status is Health.UNAVAILABLE
    assert results["Memory-Local Retrieval"].status is Health.DEGRADED
    assert results["Knowledge Acquisition"].status is Health.NOT_CONFIGURED
    assert all(pg_down not in r.detail for r in results.values())


def test_an_empty_corpus_is_healthy_not_broken(tmp_path, pg_down):
    results = by_name(check_all(Components(tmp_path, tmp_path / "absent.gguf")))
    assert results["Chunk storage"].status is Health.HEALTHY
    assert "empty" in results["Chunk storage"].detail
    assert results["Vector index (DiskANN)"].status is Health.HEALTHY


def test_the_thinking_layer_is_checked_by_running_it(tmp_path, pg_down):
    results = by_name(check_all(Components(tmp_path, tmp_path / "absent.gguf")))
    for part in ("Thinking Layer", "Planner", "Scheduler", "Job Pool", "Worker Pool", "Accumulator", "Critic"):
        assert results[part].status is Health.HEALTHY, part


def test_a_model_not_yet_loaded_is_not_initialized(tmp_path, pg_down):
    pytest.importorskip("llama_cpp")
    path = tmp_path / "model.gguf"
    path.write_bytes(b"x")
    results = by_name(check_all(Components(tmp_path, path)))
    assert results["Draft model"].status is Health.NOT_INITIALIZED
    assert results["Global Retrieval"].status is Health.NOT_INITIALIZED


def test_every_check_is_bounded(tmp_path, pg_down):
    results = check_all(Components(tmp_path, tmp_path / "absent.gguf"))
    assert len(results) >= 25
    assert sum(r.milliseconds for r in results) < 5000
    assert os.environ["DB_PASSWORD"] == pg_down
