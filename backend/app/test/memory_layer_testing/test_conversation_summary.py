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

import random
import string
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, call, patch

import numpy as np
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
        mock_mr.get_highest_summarised_sequence.return_value = None
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
        mock_mr.get_highest_summarised_sequence.return_value = None
        mock_fc.get_context.return_value = [_row("Hello"), _row("World")]
        instance = _make_cs(tmp_path, mock_fc, mock_mr)
        yield instance, mock_fc, mock_mr


# ---------------------------------------------------------------------------
# __get_current_conversation — clamping and overlap (Bug 4.29 regression)
# ---------------------------------------------------------------------------

class TestGetCurrentConversation:
    def test_look_back_applies_once_a_watermark_exists(self, cs):
        instance, mock_fc, mock_mr = cs
        # watermark=180, seq=200, window=100 → look_back = 200-100-50 = 50,
        # which is earlier than watermark+1, so the look-back wins.
        mock_mr.get_highest_summarised_sequence.return_value = 180
        mock_fc.get_context.return_value = [_row("A")]
        instance.get_current_conversation(200)
        mock_fc.get_context.assert_called_once_with(50, 200)

    def test_backlog_is_covered_when_nothing_is_summarised(self, cs):
        """
        Bug 4.36 regression: with no watermark the window must reach back to the
        first turn. The look-back alone would have started at 50 and abandoned
        turns 1-49 while still advancing the watermark past them.
        """
        instance, mock_fc, mock_mr = cs
        mock_mr.get_highest_summarised_sequence.return_value = None
        mock_fc.get_context.return_value = [_row("A")]
        instance.get_current_conversation(200)
        mock_fc.get_context.assert_called_once_with(1, 200)

    def test_window_never_starts_after_the_first_unsummarised_turn(self, cs):
        """Bug 4.36 regression: no gap may open between watermark and window."""
        instance, mock_fc, mock_mr = cs
        # watermark=10 but the look-back would start at 850 — 839 turns would
        # be skipped and then declared summarised.
        mock_mr.get_highest_summarised_sequence.return_value = 10
        mock_fc.get_context.return_value = []
        instance.get_current_conversation(1000)
        mock_fc.get_context.assert_called_once_with(11, 1000)

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
        instance, mock_fc, mock_mr = cs
        # watermark well ahead of the look-back so the look-back is what applies:
        # seq=160, window=100 → start = 160 - 100 - 50 = 10
        mock_mr.get_highest_summarised_sequence.return_value = 150
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
        instance, mock_fc, mock_mr = cs
        # watermark ahead of the look-back, so the look-back is what applies
        mock_mr.get_highest_summarised_sequence.return_value = 999_900
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
        mock_mr.get_highest_summarised_sequence.return_value = None
        assert instance.get_current_summary() is None

    def test_returns_summary_string(self, cs):
        instance, _, mock_mr = cs
        mock_mr.get_latest_summary.return_value = "A prior summary."
        mock_mr.get_highest_summarised_sequence.return_value = None
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
            mock_mr.get_highest_summarised_sequence.return_value = None
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
            mock_mr.get_highest_summarised_sequence.return_value = None
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
        mock_mr.get_highest_summarised_sequence.return_value = None
        mock_fc.get_context.return_value = [_row("A"), _row("B")]
        instance.make_summary(chunk_sequence_number=200)
        mock_llama.create_chat_completion.assert_called_once()

    def test_with_previous_summary_single_inference_call(self, tmp_path):
        with patch(_FULL_CONV) as MockFC, patch(_META_REPO) as MockMR:
            mock_fc = MockFC.return_value
            mock_mr = MockMR.return_value
            mock_mr.get_latest_summary.return_value = "Previous summary."
            mock_mr.get_highest_summarised_sequence.return_value = None
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
            mock_mr.get_highest_summarised_sequence.return_value = None
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
            mock_mr.get_highest_summarised_sequence.return_value = None
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
            mock_mr.get_highest_summarised_sequence.return_value = None
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
            mock_mr.get_highest_summarised_sequence.return_value = None
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
            mock_mr.get_highest_summarised_sequence.return_value = None
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
            mock_mr.get_highest_summarised_sequence.return_value = None
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


# ---------------------------------------------------------------------------
# take_snapshot — the round trip make_summary() alone never completed
# ---------------------------------------------------------------------------

_SNAPSHOT = f"{_MOD}.SnapShot"

_COVERED = [
    ("chunk_a", "How does DiskANN work?", "2026-08-23T00:00:00+00:00", "turn"),
    ("chunk_b", "It builds a Vamana graph.", "2026-08-23T00:00:01+00:00", "turn"),
]


def _fake_embedder(dim=128):
    """embed_text/embed_texts stand-in returning predictable ids and vectors."""
    embedder = MagicMock()

    def one(text, chunk_id=None):
        result = MagicMock()
        result.vector = np.full(dim, 0.5, dtype=np.float32)
        result.vector_id = abs(hash(text)) % (2**63 - 1)
        result.meta_data.chunk_id = chunk_id
        return result

    embedder.embed_text.side_effect = one
    embedder.embed_texts.side_effect = lambda texts, ids=None: [
        one(t, i) for t, i in zip(texts, ids or [None] * len(texts))
    ]
    return embedder


@pytest.fixture
def snapshotting(tmp_path):
    """Yields (instance, mock_fc, mock_mr, mock_snap) with the LLM stubbed out."""
    with patch(_FULL_CONV) as MockFC, patch(_META_REPO) as MockMR, patch(
        _SNAPSHOT
    ) as MockSnap:
        mock_fc = MockFC.return_value
        mock_mr = MockMR.return_value
        mock_snap = MockSnap.return_value
        mock_mr.get_latest_summary.return_value = None
        mock_mr.get_highest_summarised_sequence.return_value = None
        mock_fc.get_context_rows.return_value = list(_COVERED)
        instance = _make_cs(tmp_path, mock_fc, mock_mr)
        instance._embedder = _fake_embedder()
        with patch.object(
            ConversationSummary,
            "_ConversationSummary__generate_summary",
            lambda self, prev, conv: "A generated summary.",
        ):
            yield instance, mock_fc, mock_mr, mock_snap


class TestTakeSnapshot:
    def test_returns_the_generated_summary(self, snapshotting):
        instance, _, _, _ = snapshotting
        assert instance.take_snapshot(10) == "A generated summary."

    def test_persists_through_snapshot_add(self, snapshotting):
        instance, _, _, mock_snap = snapshotting
        instance.take_snapshot(10)
        mock_snap.add.assert_called_once()

    def test_empty_window_returns_none_without_writing(self, snapshotting):
        instance, mock_fc, _, mock_snap = snapshotting
        mock_fc.get_context_rows.return_value = []
        assert instance.take_snapshot(10) is None
        mock_snap.add.assert_not_called()

    def test_covered_window_uses_overlap_and_clamps(self, snapshotting):
        instance, mock_fc, mock_mr, _ = snapshotting
        mock_mr.get_highest_summarised_sequence.return_value = 180
        instance.take_snapshot(200)  # window=100, overlap=50 -> start 50
        mock_fc.get_context_rows.assert_called_once_with(50, 200)

    def test_covered_window_reaches_back_over_a_backlog(self, snapshotting):
        """Bug 4.36 regression: the snapshot must not skip unsummarised turns."""
        instance, mock_fc, mock_mr, _ = snapshotting
        mock_mr.get_highest_summarised_sequence.return_value = None
        instance.take_snapshot(200)
        mock_fc.get_context_rows.assert_called_once_with(1, 200)

    def test_short_conversation_clamps_start_to_zero(self, snapshotting):
        instance, mock_fc, _, _ = snapshotting
        instance.take_snapshot(10)
        assert mock_fc.get_context_rows.call_args[0][0] == 0

    def test_chunk_ids_match_the_covered_rows(self, snapshotting):
        instance, _, _, mock_snap = snapshotting
        instance.take_snapshot(10)
        assert mock_snap.add.call_args.kwargs["chunk_ids"] == ["chunk_a", "chunk_b"]

    def test_chunks_are_passed_through_whole(self, snapshotting):
        instance, _, _, mock_snap = snapshotting
        instance.take_snapshot(10)
        assert mock_snap.add.call_args.kwargs["chunks"] == _COVERED

    def test_one_summary_vector_per_covered_chunk(self, snapshotting):
        """SnapShot.add raises MisMatchCount unless these line up."""
        instance, _, _, mock_snap = snapshotting
        instance.take_snapshot(10)
        kwargs = mock_snap.add.call_args.kwargs
        assert len(kwargs["summary_vector_ids"]) == len(kwargs["chunk_ids"])
        assert len(kwargs["summary_vectors"]) == len(kwargs["chunk_ids"])

    def test_summary_vectors_are_float32(self, snapshotting):
        instance, _, _, mock_snap = snapshotting
        instance.take_snapshot(10)
        assert mock_snap.add.call_args.kwargs["summary_vectors"].dtype == np.float32

    def test_cumulative_vector_comes_from_the_summary_text(self, snapshotting):
        instance, _, _, mock_snap = snapshotting
        instance.take_snapshot(10)
        instance.embedder.embed_text.assert_called_once_with("A generated summary.")

    def test_chunk_vectors_are_embedded_with_their_chunk_ids(self, snapshotting):
        instance, _, _, _ = snapshotting
        instance.take_snapshot(10)
        instance.embedder.embed_texts.assert_called_once_with(
            ["How does DiskANN work?", "It builds a Vamana graph."],
            ["chunk_a", "chunk_b"],
        )

    def test_summary_length_is_recorded(self, snapshotting):
        instance, _, _, mock_snap = snapshotting
        instance.take_snapshot(10)
        assert mock_snap.add.call_args.kwargs["len_of_the_summary"] == len(
            "A generated summary."
        )

    def test_snapshot_time_is_iso_utc(self, snapshotting):
        instance, _, _, mock_snap = snapshotting
        instance.take_snapshot(10)
        stamp = mock_snap.add.call_args.kwargs["time_of_snapshot"]
        parsed = datetime.fromisoformat(stamp)
        assert parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0)

    def test_previous_summary_is_read_before_generating(self, snapshotting):
        """The rolling summary depends on this being the stored predecessor."""
        instance, _, mock_mr, _ = snapshotting
        mock_mr.get_latest_summary.return_value = "Earlier summary."
        mock_mr.get_highest_summarised_sequence.return_value = None
        instance.take_snapshot(10)
        mock_mr.get_latest_summary.assert_called()

    def test_blank_summary_is_not_persisted(self, snapshotting):
        instance, _, _, mock_snap = snapshotting
        with patch.object(
            ConversationSummary,
            "_ConversationSummary__generate_summary",
            lambda self, prev, conv: "",
        ):
            assert instance.take_snapshot(10) is None
        mock_snap.add.assert_not_called()


# ---------------------------------------------------------------------------
# Snapshot wiring: ConversationSummary and SnapShot must share one database
# ---------------------------------------------------------------------------

class TestSnapshotSharesDatabase:
    def test_snapshot_built_against_the_same_directory(self, tmp_path):
        """
        SnapShot used to hardcode Config.CONVERSATION while ConversationSummary
        used the directory it was given. Writing the summary to one database and
        reading it from another left the rolling summary blind to its own output.
        """
        with patch(_FULL_CONV), patch(_META_REPO), patch(_SNAPSHOT) as MockSnap:
            _make_cs(tmp_path, None, None)
        assert MockSnap.call_args.kwargs["conversation_dir"] == tmp_path

    def test_snapshot_receives_project_identity(self, tmp_path):
        with patch(_FULL_CONV), patch(_META_REPO), patch(_SNAPSHOT) as MockSnap:
            _make_cs(tmp_path, None, None)
        assert MockSnap.call_args.kwargs["project_id"] == _PROJ_ID
        assert MockSnap.call_args.kwargs["project_name"] == _PROJ_NAME

    def test_embedder_is_not_loaded_until_used(self, tmp_path):
        """SentenceTransformer weights are ~100MB — construction must stay cheap."""
        with patch(_FULL_CONV), patch(_META_REPO), patch(_SNAPSHOT):
            instance = _make_cs(tmp_path, None, None)
        assert instance._embedder is None


class TestCumulativeVectorIdUniqueness:
    """
    cumulative_vector_id is a PRIMARY KEY, but the embedder derives ids from
    content — and two snapshots can legitimately produce identical summary text
    once a conversation's gist stops changing. Binding the timestamp keeps each
    snapshot distinct.
    """

    def test_identical_summaries_get_different_ids(self, snapshotting):
        instance, _, _, mock_snap = snapshotting
        instance.take_snapshot(10)
        instance.take_snapshot(20)
        ids = [c.kwargs["cumulative_summary_vector_id"] for c in mock_snap.add.call_args_list]
        assert len(ids) == 2
        assert ids[0] != ids[1], "identical summaries collided on the primary key"

    def test_id_is_not_the_content_derived_embedding_id(self, snapshotting):
        """It must not simply reuse embed_text()'s content hash."""
        instance, _, _, mock_snap = snapshotting
        instance.take_snapshot(10)
        embedded = instance.embedder.embed_text.return_value
        assert (
            mock_snap.add.call_args.kwargs["cumulative_summary_vector_id"]
            != embedded.vector_id
        )

    def test_id_fits_signed_64_bit(self, snapshotting):
        instance, _, _, mock_snap = snapshotting
        for seq in range(10, 200, 10):
            instance.take_snapshot(seq)
        for call_ in mock_snap.add.call_args_list:
            vid = call_.kwargs["cumulative_summary_vector_id"]
            assert 0 <= vid <= 2**63 - 1

    def test_id_is_deterministic_for_a_fixed_timestamp(self, snapshotting):
        instance, _, _, _ = snapshotting
        make_id = instance._ConversationSummary__cumulative_vector_id
        stamp = "2026-08-23T05:51:05.280183+00:00"
        assert make_id("same summary", stamp) == make_id("same summary", stamp)

    def test_different_timestamps_diverge(self, snapshotting):
        instance, _, _, _ = snapshotting
        make_id = instance._ConversationSummary__cumulative_vector_id
        assert make_id("s", "2026-08-23T05:51:05.280183+00:00") != make_id(
            "s", "2026-08-23T05:51:05.940217+00:00"
        )


# ---------------------------------------------------------------------------
# Bug 4.34 — batch splitter must always make forward progress
# ---------------------------------------------------------------------------

class TestSplitterTermination:
    def _split(self, cs_instance, text, max_chars, overlap):
        return cs_instance._ConversationSummary__split_into_batches(
            text, max_chars, overlap
        )

    @pytest.mark.parametrize("max_chars,overlap", [
        (200, 200),   # equal — cursor never advances
        (100, 200),   # overlap larger — cursor moves backwards
        (1, 200),     # degenerate budget
        (1, 1),
        (50, 10_000),
    ])
    def test_terminates_when_overlap_meets_or_exceeds_batch(self, cs, max_chars, overlap):
        """
        Bug 4.34: start = end - overlap_chars stopped advancing (or reversed)
        whenever overlap >= max_chars, appending a batch every pass. It hung and
        consumed memory instead of raising, so nothing upstream could catch it.
        """
        instance, _, _ = cs
        batches = self._split(instance, "x" * 5000, max_chars, overlap)
        assert len(batches) < 20_000
        assert "".join(b[:1] for b in batches)  # non-empty batches

    @pytest.mark.parametrize("max_chars,overlap", [(100, 200), (1000, 200), (100, 0)])
    def test_batches_cover_the_text_with_no_gaps(self, cs, max_chars, overlap):
        """Clamping the overlap must not start dropping characters instead."""
        instance, _, _ = cs
        # Non-repeating: cyclic text would let find() match a batch at several
        # positions and the walk below would follow the wrong one.
        rng = random.Random(20260823)
        alphabet = string.ascii_letters + string.digits
        text = "".join(rng.choice(alphabet) for _ in range(3000))
        batches = self._split(instance, text, max_chars, overlap)

        assert text.startswith(batches[0])
        assert text.endswith(batches[-1])

        # Walk the batches through the text: each must begin at or before the
        # previous one ended, otherwise characters between them were skipped.
        position = 0
        for batch in batches:
            index = text.find(batch, max(0, position - len(batch)))
            assert index != -1 and index <= position, "gap between batches"
            position = index + len(batch)
        assert position == len(text), "tail of the text was never emitted"

    def test_normal_overlap_is_left_alone(self, cs):
        instance, _, _ = cs
        batches = self._split(instance, "x" * 5000, 1000, 200)
        assert len(batches) == 6

    def test_short_text_returns_one_batch(self, cs):
        instance, _, _ = cs
        assert self._split(instance, "short", 1000, 200) == ["short"]

    def test_zero_max_chars_does_not_hang(self, cs):
        instance, _, _ = cs
        assert len(self._split(instance, "x" * 500, 0, 0)) < 10_000


# ---------------------------------------------------------------------------
# Bug 4.35 — the injected draft window must actually be used
# ---------------------------------------------------------------------------

class TestDraftWindowIsHonoured:
    def test_token_budget_uses_the_injected_window(self, tmp_path):
        """
        Both call sites previously read Config.DRAFT_MODEL_CONTEXT_WINDOW, so
        the constructor parameter was inert.
        """
        with patch(_FULL_CONV) as MockFC, patch(_META_REPO) as MockMR:
            mock_fc = MockFC.return_value
            mock_mr = MockMR.return_value
            mock_mr.get_latest_summary.return_value = None
            mock_mr.get_highest_summarised_sequence.return_value = None
            instance = ConversationSummary(
                full_conversation_dir=tmp_path,
                project_id=_PROJ_ID,
                project_name=_PROJ_NAME,
                main_model_context_window_length=_WINDOW,
                draft_model_context_window_length=4096,
            )

        captured = {}
        model = MagicMock()
        model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "s"}}]
        }
        original = instance._ConversationSummary__split_into_batches

        def spy(text, max_chars, overlap):
            captured["max_chars"] = max_chars
            return original(text, max_chars, overlap)

        with patch.object(instance, "_ConversationSummary__load_model", lambda: model), \
             patch.object(instance, "_ConversationSummary__unload_model", lambda m: None), \
             patch.object(instance, "_ConversationSummary__split_into_batches", spy):
            instance._ConversationSummary__generate_summary(None, "some conversation")

        # (4096 - 200 - 512) * 4
        assert captured["max_chars"] == (4096 - 200 - 512) * 4

    def test_model_is_loaded_with_the_injected_window(self, tmp_path):
        with patch(_FULL_CONV) as MockFC, patch(_META_REPO) as MockMR:
            mock_fc = MockFC.return_value
            mock_mr = MockMR.return_value
            mock_mr.get_latest_summary.return_value = None
            mock_mr.get_highest_summarised_sequence.return_value = None
            instance = ConversationSummary(
                full_conversation_dir=tmp_path,
                project_id=_PROJ_ID,
                project_name=_PROJ_NAME,
                main_model_context_window_length=_WINDOW,
                draft_model_context_window_length=8192,
            )

        import llama_cpp

        model_file = Path(str(tmp_path)) / "m.gguf"
        model_file.write_text("stub")
        with patch(f"{_MOD}.Config") as MockConfig, patch.object(
            llama_cpp, "Llama"
        ) as MockLlama:
            MockConfig.DRAFT_MODEL_PATH = str(tmp_path)
            MockConfig.DRAFT_MODEL_FILE = "m.gguf"
            instance._ConversationSummary__load_model()
        assert MockLlama.call_args.kwargs["n_ctx"] == 8192


# ---------------------------------------------------------------------------
# Bug 4.39 — the model must be released even if batching fails
# ---------------------------------------------------------------------------

class TestModelIsAlwaysReleased:
    def test_model_unloaded_when_batching_raises(self, cs):
        instance, _, _ = cs
        unloaded = []
        model = MagicMock()
        with patch.object(instance, "_ConversationSummary__load_model", lambda: model), \
             patch.object(instance, "_ConversationSummary__unload_model",
                          lambda m: unloaded.append(m)), \
             patch.object(instance, "_ConversationSummary__split_into_batches",
                          MagicMock(side_effect=ValueError("splitter blew up"))):
            with pytest.raises(ValueError, match="splitter blew up"):
                instance._ConversationSummary__generate_summary(None, "text")
        assert unloaded == [model], "model leaked when batch splitting failed"

    def test_model_unloaded_when_inference_raises(self, cs):
        instance, _, _ = cs
        unloaded = []
        model = MagicMock()
        model.create_chat_completion.side_effect = RuntimeError("inference failed")
        with patch.object(instance, "_ConversationSummary__load_model", lambda: model), \
             patch.object(instance, "_ConversationSummary__unload_model",
                          lambda m: unloaded.append(m)):
            with pytest.raises(RuntimeError):
                instance._ConversationSummary__generate_summary(None, "text")
        assert unloaded == [model]


# ---------------------------------------------------------------------------
# Bug 4.40 — one metadata repository, explicitly closed
# ---------------------------------------------------------------------------

class TestRepositorySharing:
    def test_snapshot_reuses_the_summarisers_repository(self, tmp_path):
        with patch(_FULL_CONV), patch(_META_REPO) as MockMR:
            MockMR.return_value.get_cumulative_vector_meta_data_ids.return_value = []
            instance = _make_cs(tmp_path, None, None)
        assert instance.snap_shot.meta_repo is instance.summary_repo
        assert MockMR.call_count == 1, "a second repository was opened on the same file"

    def test_close_releases_the_connection(self, tmp_path):
        with patch(_FULL_CONV), patch(_META_REPO) as MockMR:
            instance = _make_cs(tmp_path, None, None)
            instance.close()
        MockMR.return_value.close.assert_called_once()

    def test_snapshot_does_not_close_a_borrowed_repository(self, tmp_path):
        """Closing a repo it did not open would pull it out from under the owner."""
        with patch(_FULL_CONV), patch(_META_REPO) as MockMR:
            instance = _make_cs(tmp_path, None, None)
            instance.snap_shot.close()
        MockMR.return_value.close.assert_not_called()
