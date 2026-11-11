"""The models go to the GPU when one is usable, fall back to the CPU when it is not, and status says which."""

import hashlib
import sys
import types
from pathlib import Path

import pytest

import config as config_module
from config import Config, parse_gpu_layers, parse_torch_device
from inference import devices
from inference import shared_model as shared_model_module
from inference.inference_exceptions import ModelUnavailable
from inference.runtime import InferenceRuntime
from inference.shared_model import (
    CHECKPOINT_FORMAT,
    ModelSpec,
    SharedModel,
    _runtime_version,
    describe_layers,
    file_fingerprint,
)
from observability import health
from observability.health import Health
from observability.service_state import service_facts
from observability.telemetry import Telemetry
from pipeline.atlas import RetrieverHandle


@pytest.fixture
def model_file(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"GGUF" + b"\0" * 64)
    return path


class FakeLlama:
    def tokenize(self, data):
        return list(data)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class TestSettings:
    @pytest.mark.parametrize("value, expected", [(-1, -1), (0, 0), (28, 28), ("-1", -1), (" 12 ", 12), ("0", 0)])
    def test_gpu_layers_accepts_all_none_or_a_count(self, value, expected):
        assert parse_gpu_layers(value) == expected

    @pytest.mark.parametrize("value", [-2, "all", "", "1.5", True, None])
    def test_gpu_layers_refuses_anything_else(self, value):
        with pytest.raises(ValueError, match="ATLAS_GPU_LAYERS"):
            parse_gpu_layers(value)

    @pytest.mark.parametrize("value, expected", [("auto", "auto"), ("CUDA", "cuda"), (" cpu ", "cpu")])
    def test_torch_device_accepts_auto_cuda_cpu(self, value, expected):
        assert parse_torch_device(value) == expected

    @pytest.mark.parametrize("value", ["gpu", "cuda:0", "mps", ""])
    def test_torch_device_refuses_anything_else(self, value):
        with pytest.raises(ValueError, match="ATLAS_TORCH_DEVICE"):
            parse_torch_device(value)

    def test_the_defaults_ask_for_the_gpu_when_there_is_one(self):
        assert config_module.Config.MODEL_GPU_LAYERS == parse_gpu_layers(
            __import__("os").environ.get("ATLAS_GPU_LAYERS", -1))
        assert config_module.Config.TORCH_DEVICE in config_module.TORCH_DEVICES


# ---------------------------------------------------------------------------
# Device choice
# ---------------------------------------------------------------------------

class TestDeviceChoice:
    def test_layers_are_requested_only_when_offload_is_possible(self):
        assert devices.gpu_layers(-1, supported=lambda: True) == -1
        assert devices.gpu_layers(20, supported=lambda: True) == 20
        assert devices.gpu_layers(-1, supported=lambda: False) == 0

    def test_zero_layers_never_asks_llama_cpp(self):
        def must_not_ask():
            raise AssertionError("asked")
        assert devices.gpu_layers(0, supported=must_not_ask) == 0

    def test_offload_is_off_when_llama_cpp_is_missing_or_cannot_answer(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "llama_cpp", None)
        assert devices.llama_gpu_offload() is False
        broken = types.SimpleNamespace(llama_supports_gpu_offload=lambda: (_ for _ in ()).throw(OSError("no lib")))
        monkeypatch.setitem(sys.modules, "llama_cpp", broken)
        assert devices.llama_gpu_offload() is False

    @pytest.mark.parametrize("configured, available, expected", [
        ("auto", True, "cuda"), ("auto", False, "cpu"),
        ("cuda", True, "cuda"), ("cuda", False, "cpu"),
        ("cpu", True, "cpu"), ("cpu", False, "cpu"),
    ])
    def test_torch_models_use_cuda_only_when_it_is_there_and_allowed(self, configured, available, expected):
        assert devices.torch_device(configured, available=lambda: available) == expected

    def test_a_gpu_load_that_fails_falls_back_to_the_cpu(self):
        tried = []

        def factory(device):
            tried.append(device)
            if device == "cuda":
                raise RuntimeError("CUDA out of memory")
            return f"model on {device}"

        assert devices.load_on_device(factory, "test model", device="cuda") == ("model on cpu", "cpu")
        assert tried == ["cuda", "cpu"]

    def test_the_cpu_is_tried_once(self):
        tried = []
        assert devices.load_on_device(lambda d: tried.append(d) or d, "m", device="cpu") == ("cpu", "cpu")
        assert tried == ["cpu"]


# ---------------------------------------------------------------------------
# The shared llama.cpp model
# ---------------------------------------------------------------------------

class TestSharedModelOffload:
    def test_the_loader_is_asked_for_the_configured_layers(self, model_file):
        seen = []
        model = SharedModel(ModelSpec(model_file, n_gpu_layers=-1), loader=lambda s: seen.append(s) or FakeLlama())
        model.ensure_loaded()
        assert seen[0].n_gpu_layers == -1
        assert model.gpu_layers == -1 and model.offload_error is None

    def test_unset_layers_are_decided_by_what_llama_cpp_supports(self, model_file, monkeypatch):
        monkeypatch.setattr(devices, "llama_gpu_offload", lambda: False)
        seen = []
        SharedModel(ModelSpec(model_file), loader=lambda s: seen.append(s) or FakeLlama()).ensure_loaded()
        assert seen[0].n_gpu_layers == 0

    def test_a_failed_gpu_load_is_retried_on_the_cpu_and_remembered(self, model_file):
        seen = []

        def loader(spec):
            seen.append(spec.n_gpu_layers)
            if spec.n_gpu_layers:
                raise RuntimeError("cudaMalloc failed: out of memory\nmore detail")
            return FakeLlama()

        model = SharedModel(ModelSpec(model_file, n_gpu_layers=-1), loader=loader)
        model.ensure_loaded()
        assert seen == [-1, 0]
        assert model.loaded and model.gpu_layers == 0
        assert model.offload_error == "cudaMalloc failed: out of memory"
        assert model.load_error is None

    def test_when_the_cpu_fails_too_the_model_is_unavailable(self, model_file):
        def loader(spec):
            raise RuntimeError(f"broken file ({spec.n_gpu_layers})")

        model = SharedModel(ModelSpec(model_file, n_gpu_layers=-1), loader=loader)
        with pytest.raises(ModelUnavailable, match=r"broken file \(0\)"):
            model.ensure_loaded()
        assert model.load_error == "broken file (0)" and not model.loaded

    def test_a_cpu_only_load_that_fails_is_not_retried(self, model_file):
        calls = []

        def loader(spec):
            calls.append(spec.n_gpu_layers)
            raise RuntimeError("bad")

        with pytest.raises(ModelUnavailable):
            SharedModel(ModelSpec(model_file, n_gpu_layers=0), loader=loader).ensure_loaded()
        assert calls == [0]

    def test_the_default_loader_passes_layers_only_when_there_are_some(self, model_file, monkeypatch):
        made = []
        fake = types.ModuleType("llama_cpp")
        fake.Llama = lambda **options: made.append(options) or FakeLlama()
        monkeypatch.setitem(sys.modules, "llama_cpp", fake)
        shared_model_module._default_loader(ModelSpec(model_file, n_gpu_layers=-1))
        shared_model_module._default_loader(ModelSpec(model_file, n_gpu_layers=0))
        assert made[0]["n_gpu_layers"] == -1
        assert "n_gpu_layers" not in made[1]

    @pytest.mark.parametrize("layers, text", [(0, "CPU"), (None, "CPU"), (-1, "GPU, every layer"), (12, "GPU, 12 layer(s)")])
    def test_describe_layers(self, layers, text):
        assert describe_layers(layers) == text


class TestCheckpointFingerprint:
    def test_cpu_fingerprints_are_what_they_were_so_saved_checkpoints_stay_valid(self, model_file):
        model = SharedModel(ModelSpec(model_file, n_ctx=4096, n_gpu_layers=0), loader=lambda s: FakeLlama())
        model.ensure_loaded()
        before = f"{file_fingerprint(model_file)}|n_ctx=4096|format={CHECKPOINT_FORMAT}|{_runtime_version()}"
        assert model.fingerprint == hashlib.sha256(before.encode()).hexdigest()

    def test_a_gpu_resident_model_has_its_own_fingerprint(self, model_file):
        cpu = SharedModel(ModelSpec(model_file, n_gpu_layers=0), loader=lambda s: FakeLlama())
        gpu = SharedModel(ModelSpec(model_file, n_gpu_layers=-1), loader=lambda s: FakeLlama())
        cpu.ensure_loaded()
        gpu.ensure_loaded()
        assert cpu.fingerprint != gpu.fingerprint

    def test_a_fallback_to_the_cpu_takes_the_cpu_fingerprint(self, model_file):
        def loader(spec):
            if spec.n_gpu_layers:
                raise RuntimeError("no GPU")
            return FakeLlama()

        cpu = SharedModel(ModelSpec(model_file, n_gpu_layers=0), loader=lambda s: FakeLlama())
        fell_back = SharedModel(ModelSpec(model_file, n_gpu_layers=-1), loader=loader)
        asked_early = fell_back.fingerprint  # before loading: assumes the GPU it was asked for
        fell_back.ensure_loaded()
        assert asked_early != fell_back.fingerprint == cpu.fingerprint


# ---------------------------------------------------------------------------
# Torch models
# ---------------------------------------------------------------------------

def test_the_embedder_is_built_on_the_chosen_device(monkeypatch):
    from data_layer.ingestion.embedding import EmbeddingManager as module

    made = []
    monkeypatch.setattr(module, "SentenceTransformer", lambda name, device: made.append(device) or object())
    monkeypatch.setattr(devices, "torch_device", lambda configured=None, available=None: "cuda")
    manager = module.EmbeddingManager()
    assert made == ["cuda"] and manager.device == "cuda"


def test_the_reranker_is_built_on_the_chosen_device_and_falls_back(monkeypatch):
    import sentence_transformers

    from retrieval_layer.reranking import Reranker

    made = []

    def cross_encoder(name, device):
        made.append(device)
        if device == "cuda":
            raise RuntimeError("CUDA error: no kernel image is available")
        return types.SimpleNamespace(predict=lambda pairs, batch_size: [0.0] * len(pairs))

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", cross_encoder)
    monkeypatch.setattr(devices, "torch_device", lambda configured=None, available=None: "cuda")
    reranker = Reranker("some/model")
    assert reranker.device is None
    reranker.scorer  # noqa: B018 - loading is the point
    assert made == ["cuda", "cpu"] and reranker.device == "cpu"


# ---------------------------------------------------------------------------
# What status says
# ---------------------------------------------------------------------------

class _Sessions:
    restores = misses = invalidations = save_failures = 0


class _Components:
    def __init__(self, model, embedder_device=None, reranker_device=None) -> None:
        self.runtime = InferenceRuntime(model)
        self.retriever = RetrieverHandle(lambda: None)
        self.sessions = _Sessions()
        self.acquirer = None
        self.telemetry = Telemetry()
        embedder = None if embedder_device is None else types.SimpleNamespace(device=embedder_device)
        self.models = types.SimpleNamespace(_MemoryModels__embedder=embedder,
                                            reranker=types.SimpleNamespace(device=reranker_device))


@pytest.fixture
def pg_down(monkeypatch):
    def refuse():
        raise ConnectionError("connection refused")
    monkeypatch.setattr(health, "_postgres", refuse)


def _draft_status(facts):
    return {r.name: r for r in health.check(facts)}["Draft model"]


def test_the_service_publishes_where_each_model_landed(model_file, monkeypatch):
    monkeypatch.setattr(devices, "gpu_name", lambda available=None: "NVIDIA GeForce RTX 5050 Laptop GPU")
    model = SharedModel(ModelSpec(model_file, n_gpu_layers=-1), loader=lambda s: FakeLlama())
    model.ensure_loaded()
    facts = service_facts(_Components(model, embedder_device="cuda", reranker_device="cuda"))
    assert facts["draft"]["gpu_layers"] == -1 and facts["draft"]["offload_error"] is None
    assert facts["devices"] == {"embedder": "cuda", "reranker": "cuda", "gpu": "NVIDIA GeForce RTX 5050 Laptop GPU"}


def test_a_cpu_only_service_does_not_ask_torch_for_a_gpu(model_file, monkeypatch):
    def must_not_ask(available=None):
        raise AssertionError("asked for a GPU name")
    monkeypatch.setattr(devices, "gpu_name", must_not_ask)
    model = SharedModel(ModelSpec(model_file, n_gpu_layers=0), loader=lambda s: FakeLlama())
    model.ensure_loaded()
    facts = service_facts(_Components(model, embedder_device="cpu", reranker_device="cpu"))
    assert facts["devices"]["gpu"] is None


@pytest.mark.usefixtures("pg_down")
class TestStatus:
    @pytest.fixture(autouse=True)
    def _llama_cpp_present(self):
        pytest.importorskip("llama_cpp")

    def _facts(self, model_file, **draft):
        model = SharedModel(ModelSpec(model_file, n_gpu_layers=0), loader=lambda s: FakeLlama())
        model.ensure_loaded()
        facts = service_facts(_Components(model, embedder_device="cuda", reranker_device="cuda"))
        facts["draft"].update(draft)
        facts["devices"]["gpu"] = "RTX 5050"
        return facts

    def test_a_model_on_the_gpu_says_so(self, model_file):
        result = _draft_status(self._facts(model_file, gpu_layers=-1))
        assert result.status is Health.HEALTHY
        assert "resident on the GPU (every layer)" in result.detail

    def test_a_model_that_fell_back_to_the_cpu_is_degraded_with_the_reason(self, model_file):
        result = _draft_status(self._facts(model_file, gpu_layers=0, offload_error="out of memory"))
        assert result.status is Health.DEGRADED
        assert "resident on the CPU" in result.detail and "GPU load failed (out of memory)" in result.detail

    def test_a_cpu_model_while_the_gpu_was_wanted_says_why(self, model_file, monkeypatch):
        monkeypatch.setattr(Config, "MODEL_GPU_LAYERS", -1)
        result = _draft_status(self._facts(model_file, gpu_layers=0))
        assert result.status is Health.HEALTHY
        assert "no GPU offload (llama-cpp-python is a CPU build, or no GPU is visible)" in result.detail

    def test_a_deliberately_cpu_only_model_adds_no_note(self, model_file, monkeypatch):
        monkeypatch.setattr(Config, "MODEL_GPU_LAYERS", 0)
        result = _draft_status(self._facts(model_file, gpu_layers=0))
        assert "no GPU offload" not in result.detail

    def test_the_embedder_device_is_reported(self, model_file):
        results = {r.name: r for r in health.check(self._facts(model_file))}
        assert "embedder resident on the GPU (RTX 5050)" in results["Memory Manager"].detail


def test_monitors_still_import_neither_torch_nor_llama_cpp():
    """`status` runs health.check in a fresh process; the device code must not drag either library in."""
    import subprocess

    code = (
        "import sys; import observability.health, inference.devices, inference.shared_model; "
        "print('torch' in sys.modules, 'llama_cpp' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parents[2])
    assert out.stdout.split() == ["False", "False"], out.stderr
