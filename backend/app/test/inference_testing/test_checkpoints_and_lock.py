"""The checkpoint store and the model lock, without a real model."""

import os
import stat
import threading
import time

import numpy as np
import pytest

from inference.checkpoints import (
    HIT,
    INVALID,
    MISSING,
    CheckpointStore,
    decode,
    encode,
)
from inference.inference_exceptions import CheckpointError, ModelFileMissing, PromptTooLarge
from inference.shared_model import ModelSpec, SharedModel

FP = "f" * 64
TOKENS = np.arange(100, 164, dtype=np.intc)
STATE = b"state-bytes" * 100


@pytest.fixture
def store(tmp_path):
    return CheckpointStore(tmp_path / "kv", database=tmp_path / "memory.db", max_bytes=10_000_000)


def save(store, conversation="c1", user="u1", fingerprint=FP, tokens=TOKENS, state=STATE):
    return store.save(conversation, user, "main", fingerprint, 4096, tokens, state, 7)


class TestFormat:
    def test_round_trip(self):
        tokens, state, seed = decode(encode(TOKENS, STATE, 7))
        assert np.array_equal(tokens, TOKENS) and state == STATE and seed == 7

    @pytest.mark.parametrize("damage", [
        lambda b: b[:5],
        lambda b: b"NOTMAGIC" + b[8:],
        lambda b: b[:-1],
        lambda b: b + b"x",
        lambda b: b[:8] + (10**6).to_bytes(4, "little") + b[12:],
    ])
    def test_damaged_files_are_refused_without_trusting_their_lengths(self, damage):
        with pytest.raises(CheckpointError):
            decode(damage(encode(TOKENS, STATE, 7)))


class TestStore:
    def test_a_saved_checkpoint_loads(self, store):
        save(store)
        loaded = store.load("c1", "u1", "main", FP, 4096)
        assert loaded.status == HIT
        assert np.array_equal(loaded.checkpoint.tokens, TOKENS)

    def test_files_are_private(self, store):
        row = save(store)
        mode = stat.S_IMODE(os.stat(store.directory / row.file_name).st_mode)
        assert mode == 0o600
        assert stat.S_IMODE(os.stat(store.directory).st_mode) == 0o700

    def test_no_checkpoint_is_missing(self, store):
        assert store.load("c1", "u1", "main", FP, 4096).status == MISSING

    def test_conversations_and_users_do_not_share(self, store):
        save(store)
        assert store.load("c2", "u1", "main", FP, 4096).status == MISSING
        assert store.load("c1", "u2", "main", FP, 4096).status == MISSING

    @pytest.mark.parametrize("fingerprint, n_ctx, reason", [
        ("0" * 64, 4096, "model"),
        (FP, 8192, "context window"),
    ])
    def test_a_changed_model_or_window_invalidates(self, store, fingerprint, n_ctx, reason):
        row = save(store)
        loaded = store.load("c1", "u1", "main", fingerprint, n_ctx)
        assert loaded.status == INVALID and reason in loaded.reason
        assert store.row_for("c1", "u1", "main") is None
        assert not (store.directory / row.file_name).exists()

    def test_a_corrupted_file_is_invalidated_not_loaded(self, store):
        row = save(store)
        path = store.directory / row.file_name
        data = bytearray(path.read_bytes())
        data[-3] ^= 1
        path.write_bytes(bytes(data))
        assert store.load("c1", "u1", "main", FP, 4096).status == INVALID
        assert store.row_for("c1", "u1", "main") is None

    def test_a_missing_file_is_invalidated(self, store):
        row = save(store)
        (store.directory / row.file_name).unlink()
        assert store.load("c1", "u1", "main", FP, 4096).status == INVALID

    def test_a_new_save_replaces_the_old_file_only_after_it_is_recorded(self, store):
        first = save(store)
        second = save(store, tokens=TOKENS[:40])
        assert first.file_name != second.file_name
        assert not (store.directory / first.file_name).exists()
        assert [r.file_name for r in store.rows()] == [second.file_name]

    def test_a_failed_write_records_nothing_and_leaves_no_file(self, store, monkeypatch):
        save(store)
        before = store.row_for("c1", "u1", "main")

        def fail(path, payload):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(store, "_write_file", fail)
        with pytest.raises(CheckpointError):
            save(store, tokens=TOKENS[:10])
        assert store.row_for("c1", "u1", "main") == before
        assert sorted(p.name for p in store.directory.iterdir()) == [before.file_name]

    def test_an_interrupted_write_leaves_only_a_temporary_file_that_sweep_removes(self, store):
        (store.directory / ".abc.kv.tmp").write_bytes(b"partial")
        (store.directory / "orphan.kv").write_bytes(b"x")
        save(store)
        assert store.sweep() == 2
        assert len(list(store.directory.iterdir())) == 1

    def test_eviction_keeps_the_cache_under_its_cap(self, tmp_path):
        store = CheckpointStore(tmp_path / "kv", database=tmp_path / "m.db", max_bytes=3000)
        save(store, "c1", state=b"x" * 1000)
        time.sleep(0.01)
        save(store, "c2", state=b"y" * 1000)
        time.sleep(0.01)
        save(store, "c3", state=b"z" * 1000)
        assert store.usage_bytes() <= 3000
        assert store.row_for("c1", "u1", "main") is None
        assert store.row_for("c3", "u1", "main") is not None

    def test_deleting_a_conversation_deletes_its_checkpoints(self, store):
        save(store, "c1")
        save(store, "c2")
        assert store.delete_conversation("c1") == 1
        assert store.row_for("c1", "u1", "main") is None
        assert len(list(store.directory.iterdir())) == 1

    def test_a_checkpoint_larger_than_the_cache_is_refused(self, tmp_path):
        store = CheckpointStore(tmp_path / "kv", database=tmp_path / "m.db", max_bytes=100)
        with pytest.raises(CheckpointError):
            save(store)


class FakeLlama:
    def __init__(self):
        self.inside = 0
        self.max_inside = 0
        self.lock = threading.Lock()

    def tokenize(self, text):
        return text.split()

    def create_chat_completion(self, **request):
        with self.lock:
            self.inside += 1
            self.max_inside = max(self.max_inside, self.inside)
        time.sleep(0.02)
        with self.lock:
            self.inside -= 1
        return {"choices": [{"message": {"content": " ok "}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1}}


class TestSharedModel:
    def test_concurrent_callers_never_overlap_inside_the_model(self, tmp_path):
        path = tmp_path / "m.gguf"
        path.write_bytes(b"x")
        llama = FakeLlama()
        loads = []
        model = SharedModel(ModelSpec(path, n_ctx=4096), loader=lambda s: loads.append(s) or llama)
        threads = [threading.Thread(target=model.infer, args=("s", "u")) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert llama.max_inside == 1
        assert len(loads) == 1
        assert model.stats().calls == {"draft": 8}

    def test_a_missing_model_file_is_named(self, tmp_path):
        model = SharedModel(ModelSpec(tmp_path / "absent.gguf"))
        with pytest.raises(ModelFileMissing, match="download_draft_model"):
            model.infer("s", "u")
        assert not model.loaded and model.load_error

    def test_a_prompt_that_cannot_fit_is_refused_before_inference(self, tmp_path):
        path = tmp_path / "m.gguf"
        path.write_bytes(b"x")
        llama = FakeLlama()
        model = SharedModel(ModelSpec(path, n_ctx=100), loader=lambda s: llama)
        with pytest.raises(PromptTooLarge):
            model.infer("s", "word " * 40, max_tokens=8)
        assert llama.max_inside == 0

    def test_the_fingerprint_changes_with_the_model_file(self, tmp_path):
        path = tmp_path / "m.gguf"
        path.write_bytes(b"one")
        first = SharedModel(ModelSpec(path)).fingerprint
        path.write_bytes(b"two")
        assert SharedModel(ModelSpec(path)).fingerprint != first
        assert SharedModel(ModelSpec(path, n_ctx=1024)).fingerprint != SharedModel(ModelSpec(path)).fingerprint
