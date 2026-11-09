"""KV checkpoints against the real model: reuse that actually happens, and isolation that holds.

Skipped unless llama-cpp-python is installed and the draft model is downloaded.
Greedy decoding throughout. llama.cpp's arithmetic depends on how a prompt is
split into batches, so a run is compared only with a run that split it the
same way; a cold recompute of the same prompt may differ in the last digits
and therefore in the text.
"""

import pytest

from inference.checkpoints import CheckpointStore
from inference.sessions import KVSessionManager, format_messages
from test.inference_testing.conftest import history

pytestmark = pytest.mark.real_model

U, A, B = "u1", "conv-a", "conv-b"


@pytest.fixture
def sessions(real_model, tmp_path):
    with real_model.exclusive() as llm:
        llm.reset()
    store = CheckpointStore(tmp_path / "kv", database=tmp_path / "memory.db")
    return KVSessionManager(real_model, store)


def ask(sessions, conversation, prefix, text, max_tokens=24):
    return sessions.generate(conversation, U, prefix, {"role": "user", "content": text},
                             max_tokens=max_tokens)


def clobber(model):
    """What the planner, critic or another conversation leaves in the context."""
    model.infer("You list things.", "Name three sorting algorithms.", max_tokens=12)


class TestReuse:
    def test_a_new_conversation_has_no_checkpoint_and_still_answers(self, sessions):
        out = ask(sessions, A, history(), "What is MVCC?")
        assert out.text
        assert out.cache.outcome == "miss"
        assert out.cache.saved_tokens > 0

    def test_restoring_skips_the_saved_prefix(self, sessions, real_model):
        first = ask(sessions, A, history(), "What is MVCC?")
        clobber(real_model)
        prefix = history(("What is MVCC?", first.text))
        second = ask(sessions, A, prefix, "How does VACUUM relate to it?")
        assert second.cache.outcome == "restored"
        assert second.cache.restored_prefix_tokens >= first.cache.saved_tokens - 1
        # Only the new turn was decoded, not the history before it.
        assert second.cache.evaluated_prompt_tokens < second.cache.prompt_tokens - first.cache.saved_tokens + 2

    def test_restored_generation_matches_uninterrupted_generation(self, sessions, real_model):
        first = ask(sessions, A, history(), "What is a B+ tree?")
        prefix = history(("What is a B+ tree?", first.text))
        uninterrupted = ask(sessions, A, prefix, "Why do databases use them?")

        ask(sessions, A, history(), "What is a B+ tree?")  # same state as before `uninterrupted`
        clobber(real_model)
        restored = ask(sessions, A, prefix, "Why do databases use them?")
        assert restored.cache.outcome == "restored"
        assert restored.text == uninterrupted.text


class TestIsolation:
    def test_a_b_a_switching_keeps_each_conversation_its_own_state(self, sessions):
        q1, q2 = "What is write-ahead logging?", "How does it help crash recovery?"
        a1 = ask(sessions, A, history(), q1)
        a2 = ask(sessions, A, history((q1, a1.text)), q2)
        b1 = ask(sessions, B, history(), "What is sharding?")
        a3 = ask(sessions, A, history((q1, a1.text), (q2, a2.text)), "What is a checkpoint?")
        # A's saved prefix (system + its first turn) is longer than what B left resident.
        assert a3.cache.outcome == "restored"
        assert a3.cache.restored_prefix_tokens > a3.cache.resident_prefix_tokens
        row_a = sessions.store.row_for(A, U, "main")
        row_b = sessions.store.row_for(B, U, "main")
        assert row_a.prefix_sha256 != row_b.prefix_sha256
        assert b1.text and a3.text

    def test_another_conversation_in_between_changes_nothing(self, sessions, real_model):
        """B's answer is identical whether A or C ran before it: no state leaks across."""
        ask(sessions, A, history(), "What is a page split?")
        after_a = ask(sessions, B, history(), "What is two-phase commit?")
        ask(sessions, "conv-c", history(), "What is a covering index?")
        after_c = ask(sessions, B, history(), "What is two-phase commit?")
        assert after_a.text == after_c.text

    def test_planner_and_critic_prompts_do_not_reach_the_conversation(self, sessions, real_model):
        first = ask(sessions, A, history(), "What is snapshot isolation?")
        real_model.infer("You plan.", "Plan an answer about snapshot isolation.", max_tokens=16,
                         purpose="planner")
        real_model.infer("You critique.", "Is this answer complete?", max_tokens=16, purpose="critic")
        second = ask(sessions, A, history(("What is snapshot isolation?", first.text)), "Give an example.")
        assert second.cache.outcome == "restored"


class TestRecovery:
    def test_a_corrupted_checkpoint_is_discarded_and_recomputed(self, sessions):
        first = ask(sessions, A, history(), "What is a heap file?")
        row = sessions.store.row_for(A, U, "main")
        path = sessions.store.directory / row.file_name
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 0xFF
        path.write_bytes(bytes(data))
        out = ask(sessions, A, history(("What is a heap file?", first.text)), "And a B-tree?")
        assert out.cache.outcome == "invalid"
        assert out.text
        assert sessions.store.row_for(A, U, "main").file_name != row.file_name

    def test_a_missing_checkpoint_file_is_recomputed(self, sessions, real_model):
        first = ask(sessions, A, history(), "What is a heap file?")
        row = sessions.store.row_for(A, U, "main")
        (sessions.store.directory / row.file_name).unlink()
        clobber(real_model)
        out = ask(sessions, A, history(("What is a heap file?", first.text)), "And a B-tree?")
        assert out.cache.outcome == "invalid"
        assert out.text

    def test_a_changed_prefix_is_not_restored(self, sessions, real_model):
        ask(sessions, A, history(), "What is a heap file?")
        clobber(real_model)
        other = [{"role": "system", "content": "You are terse."}]
        out = ask(sessions, A, other, "What is a heap file?")
        assert out.cache.outcome in ("stale", "miss")
        assert out.cache.restored_prefix_tokens == 0

    def test_a_failed_checkpoint_write_leaves_the_answer_and_no_row(self, sessions, monkeypatch):
        def refuse(path, payload):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(sessions.store, "_write_file", refuse)
        out = ask(sessions, A, history(), "What is a heap file?")
        assert out.text
        assert out.cache.save_error
        assert sessions.store.row_for(A, U, "main") is None
        assert list(sessions.store.directory.iterdir()) == []


def test_our_prompt_tokens_match_create_chat_completion(real_model):
    """Main generation tokenizes the prompt itself; it must be what the library would build."""
    messages = history(("a", "b")) + [{"role": "user", "content": "What is MVCC?"}]
    with real_model.exclusive() as llm:
        tokens = format_messages(llm, messages, True).tokens
        llm.reset()
        usage = llm.create_chat_completion(messages=messages, max_tokens=1)["usage"]
    assert usage["prompt_tokens"] == len(tokens)
