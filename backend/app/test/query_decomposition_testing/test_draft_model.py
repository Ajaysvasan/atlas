import sys
import types

import pytest

from query_decomposition.draft_model import DraftModel
from query_decomposition.query_decomposition_exceptions import QueryTooLarge


class FakeLlama:
    instances = []

    def __init__(self, model_path, n_ctx, verbose):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.requests = []
        self.closed = False
        FakeLlama.instances.append(self)

    def tokenize(self, text: bytes):
        return text.split()

    def create_chat_completion(self, **request):
        self.requests.append(request)
        return {"choices": [{"message": {"content": "  reply  "}}]}

    def close(self):
        self.closed = True


@pytest.fixture
def model_file(tmp_path, monkeypatch):
    FakeLlama.instances = []
    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=FakeLlama))
    path = tmp_path / "model.gguf"
    path.write_bytes(b"")
    return path


class TestLifetime:
    def test_loaded_on_entry_with_its_own_context_window(self, model_file):
        with DraftModel(context_window=4096, model_file=model_file):
            (llama,) = FakeLlama.instances
            assert (llama.model_path, llama.n_ctx) == (str(model_file), 4096)

    def test_released_on_exit(self, model_file):
        with DraftModel(model_file=model_file):
            pass
        assert FakeLlama.instances[0].closed

    def test_released_when_the_block_raises(self, model_file):
        with pytest.raises(KeyError):
            with DraftModel(model_file=model_file):
                raise KeyError("boom")
        assert FakeLlama.instances[0].closed

    def test_a_missing_model_file_says_how_to_get_it(self, model_file, tmp_path):
        with pytest.raises(FileNotFoundError, match="download_draft_model.py"):
            with DraftModel(model_file=tmp_path / "absent.gguf"):
                pass

    def test_unusable_outside_the_block(self, model_file):
        model = DraftModel(model_file=model_file)
        with pytest.raises(RuntimeError):
            model.infer("system", "user")
        with model:
            pass
        with pytest.raises(RuntimeError):
            model.infer("system", "user")


class TestInference:
    def test_sends_both_messages_and_strips_the_reply(self, model_file):
        with DraftModel(model_file=model_file) as model:
            assert model.infer("be brief", "why?", max_tokens=8) == "reply"
        request = FakeLlama.instances[0].requests[0]
        assert request["messages"] == [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "why?"},
        ]
        assert request["max_tokens"] == 8
        assert "response_format" not in request

    def test_a_schema_constrains_the_reply_to_json(self, model_file):
        schema = {"type": "object"}
        with DraftModel(model_file=model_file) as model:
            model.infer("s", "u", schema=schema)
        assert FakeLlama.instances[0].requests[0]["response_format"] == {
            "type": "json_object",
            "schema": schema,
        }

    def test_a_prompt_the_window_cannot_hold_is_refused_before_inference(self, model_file):
        with DraftModel(context_window=100, model_file=model_file) as model:
            with pytest.raises(QueryTooLarge):
                model.infer("s", "word " * 40, max_tokens=8)
        assert FakeLlama.instances[0].requests == []
