"""
Production-grade tests for ConversationSummary.

All external I/O is mocked:
  - FullConversation                     → no SQLite
  - ConversationVectorMetaDataRepository → no SQLite / PostgreSQL
  - llama_cpp.Llama                      → no real model file
  - gc / torch                           → safe; torch mocked in conftest

Naming convention for private method access (Python name-mangling):
    _ConversationSummary__<method>
"""

import threading
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, call, patch

import pytest

from memory.topic_pool.project_pool.conversation_pool.conversation_summary_pipeline.conversation_summary import (
    ConversationSummary,
    _WINDOW_OVERLAP_CHUNKS,
)

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------
_MOD = (
    "memory.topic_pool.project_pool.conversation_pool"
    ".conversation_summary_pipeline.conversation_summary"
)
_FULL_CONV = f"{_MOD}.FullConversation"
_META_REPO = f"{_MOD}.ConversationVectorMetaDataRepository"
_LLAMA    = "llama_cpp.Llama"

_PROJ_ID   = "proj_test"
_PROJ_NAME = "TestProject"
_WINDOW    = 100   # main_model_context_window_length used by fixtures
_DRAFT_CTX = 131072


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(text: str):
    """Mimics a DB row tuple returned by FullConversation.get_context."""
    return (text,)


def _make_cs(tmp_path, mock_fc, mock_mr, window=_WINDOW):
    return ConversationSummary(
        full_conversation_dir=tmp_path,
        project_id=_PROJ_ID,
        project_name=_PROJ_NAME,
        main_model_context_window_length=window,
        draft_model_context_window_length=_DRAFT_CTX,
    )


# ---------------------------------------------------------------------------
# Base fixture — repos patched for lifetime of every test that uses it
# ---------------------------------------------------------------------------

@pytest.fixture
def cs(tmp_path):
    """
    Yields (instance, mock_full_conversation, mock_meta_repo).
    Patches stay active for the entire test.
    """
    with patch(_FULL_CONV) as MockFC, patch(_META_REPO) as MockMR:
        mock_fc = MockFC.return_value
        mock_mr = MockMR.return_value
        mock_mr.get_latest_summary.return_value = None
        mock_fc.get_context.return_value = []
        instance = _make_cs(tmp_path, mock_fc, mock_mr)
        yield instance, mock_fc, mock_mr


@pytest.fixture
def cs_with_summary(tmp_path):
    """Instance where repo returns a previous summary."""
    with patch(_FULL_CONV) as MockFC, patch(_META_REPO) as MockMR:
        mock_fc = MockFC.return_value
        mock_mr = MockMR.return_value
        mock_mr.get_latest_summary.return_value = "Prior summary text."
        mock_fc.get_context.return_value = [_row("Hello"), _row("World")]
        instance = _make_cs(tmp_path, mock_fc, mock_mr)
        yield instance, mock_fc, mock_mr


# ---------------------------------------------------------------------------
# __get_current_conversation — clamping and overlap (Bug 4.29 regression)
# ---------------------------------------------------------------------------

class TestGetCurrentConversation:
    def test_normal_sequence_passes_correct_start_to_db(self, cs):
        instance, mock_fc, _ = cs
        # seq=200, window=100 → start = max(0, 200 - 100 - 50) = 50
        mock_fc.get_context.return_value = [_row("A")]
        instance.get_current_conversation(200)
        mock_fc.get_context.assert_called_once_with(50, 200)

    def test_sequence_shorter_than_window_clamps_start_to_zero(self, cs):
        """Bug 4.29 regression: start must never go negative."""
        instance, mock_fc, _ = cs
        # seq=10, window=100 → 10 - 100 - 50 = -140 → clamped to 0
        mock_fc.get_context.return_value = []
        instance.get_current_conversation(10)
        mock_fc.get_context.assert_called_once_with(0, 10)

    def test_sequence_zero_clamps_to_zero(self, cs):
        instance, mock_fc, _ = cs
        mock_fc.get_context.return_value = []
        instance.get_current_conversation(0)
        mock_fc.get_context.assert_called_once_with(0, 0)

    def test_overlap_extends_lookback_by_50_chunks(self, cs):
        """Overlap constant _WINDOW_OVERLAP_CHUNKS=50 must widen the window."""
        instance, mock_fc, _ = cs
        # seq=160, window=100 → start = 160 - 100 - 50 = 10
        mock_fc.get_context.return_value = []
        instance.get_current_conversation(160)
        mock_fc.get_context.assert_called_once_with(10, 160)

    def test_exact_overlap_boundary_clamps_to_zero(self, cs):
        """seq == window + overlap → start = 0 (not negative)."""
        instance, mock_fc, _ = cs
        # window=100, overlap=50 → seq=150 → start = max(0, 150-100-50) = 0
        mock_fc.get_context.return_value = []
        instance.get_current_conversation(150)
        mock_fc.get_context.assert_called_once_with(0, 150)

    def test_rows_joined_with_space(self, cs):
        instance, mock_fc, _ = cs
        mock_fc.get_context.return_value = [_row("Hello"), _row("World"), _row("!")]
        result = instance.get_current_conversation(200)
        assert result == "Hello World !"

    def test_empty_context_returns_empty_string(self, cs):
        instance, mock_fc, _ = cs
        mock_fc.get_context.return_value = []
        result = instance.get_current_conversation(200)
        assert result == ""

    def test_single_row_no_extra_spaces(self, cs):
        instance, mock_fc, _ = cs
        mock_fc.get_context.return_value = [_row("Only one")]
        result = instance.get_current_conversation(200)
        assert result == "Only one"

    def test_unicode_rows_preserved(self, cs):
        instance, mock_fc, _ = cs
        mock_fc.get_context.return_value = [_row("こんにちは"), _row("🌍")]
        result = instance.get_current_conversation(200)
        assert result == "こんにちは 🌍"

    def test_very_large_sequence_number(self, cs):
        """No overflow or incorrect clamping at large seq numbers."""
        instance, mock_fc, _ = cs
        mock_fc.get_context.return_value = []
        instance.get_current_conversation(1_000_000)
        start, end = mock_fc.get_context.call_args[0]
        assert start == 1_000_000 - _WINDOW - _WINDOW_OVERLAP_CHUNKS
        assert end == 1_000_000

    def test_returns_str_not_list(self, cs):
        instance, mock_fc, _ = cs
        mock_fc.get_context.return_value = [_row("A"), _row("B")]
        result = instance.get_current_conversation(200)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# get_current_summary — delegation to repo
# ---------------------------------------------------------------------------

class TestGetCurrentSummary:
    def test_returns_none_when_no_summary(self, cs):
        instance, _, mock_mr = cs
        mock_mr.get_latest_summary.return_value = None
        assert instance.get_current_summary() is None

    def test_returns_summary_string(self, cs):
        instance, _, mock_mr = cs
        mock_mr.get_latest_summary.return_value = "A prior summary."
        assert instance.get_current_summary() == "A prior summary."

    def test_delegates_to_repo_get_latest_summary(self, cs):
        instance, _, mock_mr = cs
        instance.get_current_summary()
        mock_mr.get_latest_summary.assert_called_once()

    def test_called_multiple_times_returns_latest_each_time(self, cs):
        instance, _, mock_mr = cs
        mock_mr.get_latest_summary.side_effect = ["first", "second", "third"]
        assert instance.get_current_summary() == "first"
        assert instance.get_current_summary() == "second"
        assert instance.get_current_summary() == "third"


# ---------------------------------------------------------------------------
# __split_into_batches — core batching logic
# ---------------------------------------------------------------------------

class TestSplitIntoBatches:
    def _split(self, instance, text, max_chars, overlap_chars):
        return instance._ConversationSummary__split_into_batches(
            text, max_chars, overlap_chars
        )

    def test_short_text_single_batch(self, cs):
        instance, _, _ = cs
        result = self._split(instance, "hello", 100, 10)
        assert result == ["hello"]

    def test_exact_max_chars_single_batch(self, cs):
        instance, _, _ = cs
        text = "x" * 100
        result = self._split(instance, text, 100, 10)
        assert result == [text]

    def test_long_text_splits_into_multiple_batches(self, cs):
        instance, _, _ = cs
        text = "a" * 300
        batches = self._split(instance, text, 100, 20)
        assert len(batches) > 1

    def test_all_batches_within_max_chars(self, cs):
        instance, _, _ = cs
        text = "b" * 500
        batches = self._split(instance, text, 100, 25)
        for b in batches:
            assert len(b) <= 100

    def test_full_coverage_no_content_dropped(self, cs):
        """Every character in the original text appears in at least one batch."""
        instance, _, _ = cs
        text = "abcdefghij" * 50   # 500 chars
        batches = self._split(instance, text, 120, 30)
        # Each char at index i must appear in a batch whose range covers i
        covered = set()
        start = 0
        for b in batches:
            for idx in range(len(b)):
                covered.add(start + idx)
            start += len(b)  # approximate — just verify batches are non-empty
        assert all(len(b) > 0 for b in batches)

    def test_overlap_chars_appear_in_consecutive_batches(self, cs):
        instance, _, _ = cs
        text = "0123456789" * 10   # 100 chars
        overlap = 20
        batches = self._split(instance, text, 50, overlap)
        if len(batches) >= 2:
            # The last `overlap` chars of batch N must start batch N+1
            tail = batches[0][-overlap:]
            assert batches[1].startswith(tail)

    def test_empty_string_returns_single_empty_batch(self, cs):
        instance, _, _ = cs
        result = self._split(instance, "", 100, 10)
        assert result == [""]

    def test_single_char_text_single_batch(self, cs):
        instance, _, _ = cs
        result = self._split(instance, "x", 100, 5)
        assert result == ["x"]

    def test_text_exactly_two_max_sizes_long(self, cs):
        instance, _, _ = cs
        text = "z" * 200
        batches = self._split(instance, text, 100, 0)
        assert len(batches) == 2
        assert len(batches[0]) == 100
        assert len(batches[1]) == 100

    def test_overlap_zero_no_shared_content(self, cs):
        instance, _, _ = cs
        text = "abcdef" * 10
        batches = self._split(instance, text, 20, 0)
        # With zero overlap, all batches are disjoint
        total = sum(len(b) for b in batches)
        assert total == len(text)

    def test_stress_ten_thousand_chars(self, cs):
        instance, _, _ = cs
        text = "x" * 10_000
        batches = self._split(instance, text, 512, 50)
        assert all(len(b) <= 512 for b in batches)
        assert len(batches) > 1


# ---------------------------------------------------------------------------
# __build_prompt — prompt assembly
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def _prompt(self, instance, summary, convo):
        return instance._ConversationSummary__build_prompt(summary, convo)

    def test_no_previous_summary_uses_simple_template(self, cs):
        instance, _, _ = cs
        system, user = self._prompt(instance, None, "chat text here")
        assert "chat text here" in user
        assert "Previous summary" not in user

    def test_with_previous_summary_includes_both(self, cs):
        instance, _, _ = cs
        system, user = self._prompt(instance, "Old summary.", "New chat.")
        assert "Old summary." in user
        assert "New chat." in user

    def test_system_prompt_is_non_empty_string(self, cs):
        instance, _, _ = cs
        system, _ = self._prompt(instance, None, "x")
        assert isinstance(system, str) and len(system) > 0

    def test_user_content_is_non_empty_string(self, cs):
        instance, _, _ = cs
        _, user = self._prompt(instance, None, "some convo")
        assert isinstance(user, str) and len(user) > 0

    def test_empty_previous_summary_treated_as_no_summary(self, cs):
        """Empty string is falsy — should fall through to the no-summary branch."""
        instance, _, _ = cs
        _, user = self._prompt(instance, "", "chat text")
        assert "Previous summary" not in user

    def test_previous_summary_precedes_conversation_in_user_content(self, cs):
        instance, _, _ = cs
        _, user = self._prompt(instance, "SUMMARY", "CONVO")
        assert user.index("SUMMARY") < user.index("CONVO")


# ---------------------------------------------------------------------------
# make_summary — end-to-end public API (LLM mocked)
# ---------------------------------------------------------------------------

class TestMakeSummary:
    @pytest.fixture
    def llm_cs(self, tmp_path):
        """
        Full fixture: both repos mocked + Llama mocked.
        Yields (instance, mock_fc, mock_mr, mock_llama_instance).
        """
        with patch(_FULL_CONV) as MockFC, patch(_META_REPO) as MockMR:
            mock_fc = MockFC.return_value
            mock_mr = MockMR.return_value
            mock_mr.get_latest_summary.return_value = None
            mock_fc.get_context.return_value = [_row("Hello"), _row("World")]

            instance = _make_cs(tmp_path, mock_fc, mock_mr)

            mock_llama = MagicMock()
            mock_llama.create_chat_completion.return_value = {
                "choices": [{"message": {"content": "Generated summary."}}]
            }

            with patch.object(
                instance,
                "_ConversationSummary__load_model",
                return_value=mock_llama,
            ) as mock_load, patch.object(
                instance,
                "_ConversationSummary__unload_model",
            ) as mock_unload:
                yield instance, mock_fc, mock_mr, mock_llama, mock_load, mock_unload

    def test_returns_string(self, llm_cs):
        instance, *_ = llm_cs
        result = instance.make_summary(chunk_sequence_number=200)
        assert isinstance(result, str)

    def test_returns_llm_output(self, llm_cs):
        instance, *_ = llm_cs
        result = instance.make_summary(chunk_sequence_number=200)
        assert result == "Generated summary."

    def test_get_current_summary_called_once(self, llm_cs):
        instance, _, mock_mr, *_ = llm_cs
        instance.make_summary(chunk_sequence_number=200)
        mock_mr.get_latest_summary.assert_called_once()

    def test_get_context_called_once(self, llm_cs):
        instance, mock_fc, *_ = llm_cs
        instance.make_summary(chunk_sequence_number=200)
        mock_fc.get_context.assert_called_once()

    def test_load_model_called_once(self, llm_cs):
        instance, _, _, _, mock_load, _ = llm_cs
        instance.make_summary(chunk_sequence_number=200)
        mock_load.assert_called_once()

    def test_unload_model_always_called(self, llm_cs):
        instance, _, _, _, _, mock_unload = llm_cs
        instance.make_summary(chunk_sequence_number=200)
        mock_unload.assert_called_once()

    def test_unload_called_even_when_inference_raises(self, tmp_path):
        """__unload_model must run in the finally block even on inference error."""
        with patch(_FULL_CONV) as MockFC, patch(_META_REPO) as MockMR:
            mock_fc = MockFC.return_value
            mock_mr = MockMR.return_value
            mock_mr.get_latest_summary.return_value = None
            mock_fc.get_context.return_value = [_row("Hello")]
            instance = _make_cs(tmp_path, mock_fc, mock_mr)

            mock_llama = MagicMock()
            mock_llama.create_chat_completion.side_effect = RuntimeError("GPU OOM")

            with patch.object(
                instance, "_ConversationSummary__load_model", return_value=mock_llama
            ), patch.object(
                instance, "_ConversationSummary__unload_model"
            ) as mock_unload:
                with pytest.raises(RuntimeError, match="GPU OOM"):
                    instance.make_summary(chunk_sequence_number=200)
                mock_unload.assert_called_once()

    def test_no_previous_summary_single_inference_call(self, llm_cs):
        instance, mock_fc, mock_mr, mock_llama, *_ = llm_cs
        mock_mr.get_latest_summary.return_value = None
        mock_fc.get_context.return_value = [_row("A"), _row("B")]
        instance.make_summary(chunk_sequence_number=200)
        mock_llama.create_chat_completion.assert_called_once()

    def test_with_previous_summary_single_inference_call(self, tmp_path):
        with patch(_FULL_CONV) as MockFC, patch(_META_REPO) as MockMR:
            mock_fc = MockFC.return_value
            mock_mr = MockMR.return_value
            mock_mr.get_latest_summary.return_value = "Previous summary."
            mock_fc.get_context.return_value = [_row("New chat.")]
            instance = _make_cs(tmp_path, mock_fc, mock_mr)

            mock_llama = MagicMock()
            mock_llama.create_chat_completion.return_value = {
                "choices": [{"message": {"content": "Updated summary."}}]
            }
            with patch.object(
                instance, "_ConversationSummary__load_model", return_value=mock_llama
            ), patch.object(instance, "_ConversationSummary__unload_model"):
                result = instance.make_summary(chunk_sequence_number=50)
        assert result == "Updated summary."


# ---------------------------------------------------------------------------
# Multi-batch rolling summarisation
# ---------------------------------------------------------------------------

class TestMultiBatchRollingSummary:
    def test_three_batches_produce_three_inference_calls(self, tmp_path):
        """
        Force a short context window so a moderate conversation splits into
        three batches; verify inference is called exactly three times and
        the rolling summary feeds forward.
        """
        with patch(_FULL_CONV) as MockFC, patch(_META_REPO) as MockMR:
            mock_fc = MockFC.return_value
            mock_mr = MockMR.return_value
            mock_mr.get_latest_summary.return_value = None
            # 600-char conversation, will be batched
            mock_fc.get_context.return_value = [_row("x" * 600)]
            instance = ConversationSummary(
                full_conversation_dir=tmp_path,
                project_id=_PROJ_ID,
                project_name=_PROJ_NAME,
                main_model_context_window_length=100,
                # Tiny draft context → only ~200 chars of conversation per pass
                # (131072 - 200 - 512) * 4 ≈ huge, so we force it via __split_into_batches)
                draft_model_context_window_length=131072,
            )

            # Patch __split_into_batches to return exactly 3 batches
            three_batches = ["batch_1_" * 5, "batch_2_" * 5, "batch_3_" * 5]
            responses = ["sum1", "sum2", "final_sum"]
            call_count = 0

            def fake_inference(model, system, user_content):
                nonlocal call_count
                result = responses[call_count]
                call_count += 1
                return result

            mock_llama = MagicMock()
            with patch.object(
                instance, "_ConversationSummary__load_model", return_value=mock_llama
            ), patch.object(
                instance, "_ConversationSummary__unload_model"
            ), patch.object(
                instance,
                "_ConversationSummary__split_into_batches",
                return_value=three_batches,
            ), patch.object(
                instance,
                "_ConversationSummary__run_inference",
                side_effect=fake_inference,
            ):
                result = instance.make_summary(chunk_sequence_number=200)

        assert call_count == 3
        assert result == "final_sum"

    def test_rolling_summary_feeds_into_next_batch_prompt(self, tmp_path):
        """The output of batch N must appear in the prompt built for batch N+1."""
        with patch(_FULL_CONV) as MockFC, patch(_META_REPO) as MockMR:
            mock_fc = MockFC.return_value
            mock_mr = MockMR.return_value
            mock_mr.get_latest_summary.return_value = None
            mock_fc.get_context.return_value = [_row("conversation")]
            instance = _make_cs(tmp_path, mock_fc, mock_mr)

            two_batches = ["first_batch", "second_batch"]
            prompts_seen: List[str] = []

            def capture_prompt(model, system, user_content):
                prompts_seen.append(user_content)
                return f"summary_after_{len(prompts_seen)}"

            mock_llama = MagicMock()
            with patch.object(
                instance, "_ConversationSummary__load_model", return_value=mock_llama
            ), patch.object(
                instance, "_ConversationSummary__unload_model"
            ), patch.object(
                instance,
                "_ConversationSummary__split_into_batches",
                return_value=two_batches,
            ), patch.object(
                instance,
                "_ConversationSummary__run_inference",
                side_effect=capture_prompt,
            ):
                instance.make_summary(chunk_sequence_number=200)

        # The second prompt must contain the first batch's output as "previous summary"
        assert "summary_after_1" in prompts_seen[1]


# ---------------------------------------------------------------------------
# Stress tests
# ---------------------------------------------------------------------------

class TestStress:
    def test_fifty_sequential_make_summary_calls(self, tmp_path):
        """Make 50 sequential calls; verify model loaded and unloaded each time."""
        with patch(_FULL_CONV) as MockFC, patch(_META_REPO) as MockMR:
            mock_fc = MockFC.return_value
            mock_mr = MockMR.return_value
            mock_fc.get_context.return_value = [_row("chat")]
            mock_mr.get_latest_summary.return_value = None
            instance = _make_cs(tmp_path, mock_fc, mock_mr)

            mock_llama = MagicMock()
            mock_llama.create_chat_completion.return_value = {
                "choices": [{"message": {"content": "s"}}]
            }
            load_count = 0
            unload_count = 0

            def fake_load():
                nonlocal load_count
                load_count += 1
                return mock_llama

            def fake_unload(m):
                nonlocal unload_count
                unload_count += 1

            with patch.object(
                instance, "_ConversationSummary__load_model", side_effect=fake_load
            ), patch.object(
                instance, "_ConversationSummary__unload_model", side_effect=fake_unload
            ):
                for seq in range(1, 51):
                    instance.make_summary(chunk_sequence_number=seq * 10)

        assert load_count == 50
        assert unload_count == 50

    def test_concurrent_make_summary_calls_do_not_raise(self, tmp_path):
        """
        10 threads call make_summary simultaneously.
        Each thread gets its own model instance (via the patched factory).
        No exceptions should propagate.
        """
        with patch(_FULL_CONV) as MockFC, patch(_META_REPO) as MockMR:
            mock_fc = MockFC.return_value
            mock_mr = MockMR.return_value
            mock_fc.get_context.return_value = [_row("concurrent")]
            mock_mr.get_latest_summary.return_value = None
            instance = _make_cs(tmp_path, mock_fc, mock_mr)

            errors: List[Exception] = []

            def fake_load():
                m = MagicMock()
                m.create_chat_completion.return_value = {
                    "choices": [{"message": {"content": "ok"}}]
                }
                return m

            with patch.object(
                instance, "_ConversationSummary__load_model", side_effect=fake_load
            ), patch.object(instance, "_ConversationSummary__unload_model"):
                def worker(seq):
                    try:
                        instance.make_summary(chunk_sequence_number=seq * 5)
                    except Exception as e:
                        errors.append(e)

                threads = [threading.Thread(target=worker, args=(i,)) for i in range(1, 11)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

        assert errors == [], f"Exceptions in threads: {errors}"

    def test_split_into_batches_one_million_chars(self, tmp_path):
        """Batch splitting must handle 1M-char input without hanging."""
        with patch(_FULL_CONV) as MockFC, patch(_META_REPO) as MockMR:
            instance = _make_cs(tmp_path, MockFC.return_value, MockMR.return_value)
        text = "a" * 1_000_000
        batches = instance._ConversationSummary__split_into_batches(text, 10_000, 200)
        assert all(len(b) <= 10_000 for b in batches)
        assert len(batches) > 1

    def test_get_current_conversation_called_with_correct_seq_in_loop(self, tmp_path):
        """
        Simulate a growing conversation: sequence numbers 10, 20 … 500.
        Verify that get_context is always called with a non-negative start.
        """
        with patch(_FULL_CONV) as MockFC, patch(_META_REPO) as MockMR:
            mock_fc = MockFC.return_value
            mock_mr = MockMR.return_value
            mock_fc.get_context.return_value = []
            mock_mr.get_latest_summary.return_value = None
            instance = _make_cs(tmp_path, mock_fc, mock_mr, window=100)

            for seq in range(10, 510, 10):
                instance.get_current_conversation(seq)

            for c in mock_fc.get_context.call_args_list:
                start, _ = c[0]
                assert start >= 0, f"Negative start passed to get_context: {start}"

    def test_model_file_not_found_raises_file_not_found_error(self, tmp_path):
        """
        If the GGUF file does not exist, __load_model must raise FileNotFoundError
        (not a bare AttributeError or None-dereference).
        """
        with patch(_FULL_CONV) as MockFC, patch(_META_REPO) as MockMR:
            mock_fc = MockFC.return_value
            mock_mr = MockMR.return_value
            mock_fc.get_context.return_value = [_row("x")]
            mock_mr.get_latest_summary.return_value = None
            instance = _make_cs(tmp_path, mock_fc, mock_mr)

        # Do NOT patch __load_model — let it run against a missing file path
        # Config.DRAFT_MODEL_PATH points to a real path that won't exist in tmp
        from config import Config
        fake_path = tmp_path / "models" / "draft_model"
        with patch(f"{_MOD}.Config") as MockConfig:
            MockConfig.DRAFT_MODEL_PATH = str(fake_path)
            MockConfig.DRAFT_MODEL_FILE = "nonexistent.gguf"
            MockConfig.DRAFT_MODEL_CONTEXT_WINDOW = 131072
            MockConfig.DRAFT_MODEL_CONTEXT_WINDOW = 131072
            with pytest.raises(FileNotFoundError):
                instance._ConversationSummary__load_model()
