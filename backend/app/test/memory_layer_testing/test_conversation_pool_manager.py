"""
Tests for ConversationPoolManager.

Two layers:
  - unit tests with FullConversation / ConversationSummary patched, covering
    wiring, delegation and the snapshot trigger arithmetic;
  - integration tests against real SQLite with only the LLM and the pgvector
    repository stubbed, covering the wiring that mocks cannot catch.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from memory.topic_pool.project_pool.conversation_pool.conversation_pool_manager import (
    ConversationPoolManager,
)

_MOD = "memory.topic_pool.project_pool.conversation_pool.conversation_pool_manager"
_FULL_CONV = f"{_MOD}.FullConversation"
_SUMMARISER = f"{_MOD}.ConversationSummary"
_VEC_REPO = (
    "memory.topic_pool.project_pool.conversation_pool.conversation_data_management"
    ".conversationVectorManager.VectorRepository"
)

_PID = "proj_pool"
_PNAME = "PoolProject"


# ---------------------------------------------------------------------------
# Unit level
# ---------------------------------------------------------------------------

@pytest.fixture
def mgr(tmp_path):
    """Yields (manager, mock_full_conversation, mock_summariser)."""
    with patch(_FULL_CONV) as MockFC, patch(_SUMMARISER) as MockCS:
        mock_fc = MockFC.return_value
        mock_cs = MockCS.return_value
        mock_cs.summary_repo.get_cumulative_vector_meta_data_ids.return_value = []
        mock_fc.next_sequence_number.return_value = 1
        manager = ConversationPoolManager(
            conversation_dir=tmp_path,
            project_id=_PID,
            project_name=_PNAME,
            snapshot_every_n_turns=5,
        )
        yield manager, mock_fc, mock_cs


class TestWiring:
    def test_shares_the_summarisers_snapshot(self, mgr):
        """A second SnapShot would keep its own cursors and drift apart."""
        manager, _, mock_cs = mgr
        assert manager.snap_shot is mock_cs.snap_shot

    def test_shares_the_summarisers_meta_repo(self, mgr):
        manager, _, mock_cs = mgr
        assert manager.meta_repo is mock_cs.summary_repo

    def test_summariser_gets_the_same_directory(self, tmp_path):
        with patch(_FULL_CONV), patch(_SUMMARISER) as MockCS:
            MockCS.return_value.summary_repo.get_cumulative_vector_meta_data_ids.return_value = []
            ConversationPoolManager(tmp_path, _PID, _PNAME)
        assert MockCS.call_args.kwargs["full_conversation_dir"] == tmp_path

    def test_rejects_a_zero_threshold(self, tmp_path):
        with patch(_FULL_CONV), patch(_SUMMARISER):
            with pytest.raises(ValueError):
                ConversationPoolManager(tmp_path, _PID, _PNAME, snapshot_every_n_turns=0)

    def test_rejects_a_negative_threshold(self, tmp_path):
        with patch(_FULL_CONV), patch(_SUMMARISER):
            with pytest.raises(ValueError):
                ConversationPoolManager(tmp_path, _PID, _PNAME, snapshot_every_n_turns=-3)


class TestCursorRestoration:
    def test_cursors_restored_when_snapshots_exist(self, tmp_path):
        """
        Cursors live in memory only, so reopening a project must re-point them
        at stored history or search() scans an empty range.
        """
        with patch(_FULL_CONV), patch(_SUMMARISER) as MockCS:
            mock_cs = MockCS.return_value
            mock_cs.summary_repo.get_cumulative_vector_meta_data_ids.return_value = [
                (1,),
                (2,),
            ]
            manager = ConversationPoolManager(tmp_path, _PID, _PNAME)
        manager.snap_shot.sync_cursors.assert_called_once()

    def test_cursors_not_touched_when_no_snapshots(self, tmp_path):
        with patch(_FULL_CONV), patch(_SUMMARISER) as MockCS:
            mock_cs = MockCS.return_value
            mock_cs.summary_repo.get_cumulative_vector_meta_data_ids.return_value = []
            manager = ConversationPoolManager(tmp_path, _PID, _PNAME)
        manager.snap_shot.sync_cursors.assert_not_called()


class TestWriting:
    def test_add_turn_delegates(self, mgr):
        manager, mock_fc, _ = mgr
        mock_fc.append_turn.return_value = 7
        assert manager.add_turn("user", "hi") == 7
        mock_fc.append_turn.assert_called_once_with("user", "hi")

    def test_add_turns_delegates(self, mgr):
        manager, mock_fc, _ = mgr
        mock_fc.append_turns.return_value = [1, 2]
        assert manager.add_turns([("user", "a"), ("assistant", "b")]) == [1, 2]


class TestTriggerArithmetic:
    def test_latest_sequence_is_zero_when_empty(self, mgr):
        manager, mock_fc, _ = mgr
        mock_fc.next_sequence_number.return_value = 1
        assert manager.latest_sequence() == 0

    def test_latest_sequence_tracks_appends(self, mgr):
        manager, mock_fc, _ = mgr
        mock_fc.next_sequence_number.return_value = 12
        assert manager.latest_sequence() == 11

    def test_summarised_upto_is_zero_when_nothing_summarised(self, mgr):
        manager, _, mock_cs = mgr
        mock_cs.summary_repo.get_highest_summarised_sequence.return_value = None
        assert manager.summarised_upto() == 0

    def test_pending_is_the_gap_to_the_watermark(self, mgr):
        manager, mock_fc, mock_cs = mgr
        mock_fc.next_sequence_number.return_value = 21  # latest = 20
        mock_cs.summary_repo.get_highest_summarised_sequence.return_value = 15
        assert manager.turns_since_last_snapshot() == 5

    def test_pending_never_goes_negative(self, mgr):
        """A watermark ahead of the conversation must not produce a negative."""
        manager, mock_fc, mock_cs = mgr
        mock_fc.next_sequence_number.return_value = 6  # latest = 5
        mock_cs.summary_repo.get_highest_summarised_sequence.return_value = 9
        assert manager.turns_since_last_snapshot() == 0

    @pytest.mark.parametrize(
        "latest,watermark,expected", [(4, 0, False), (5, 0, True), (6, 0, True),
                                      (19, 15, False), (20, 15, True)]
    )
    def test_threshold_boundary(self, mgr, latest, watermark, expected):
        manager, mock_fc, mock_cs = mgr
        mock_fc.next_sequence_number.return_value = latest + 1
        mock_cs.summary_repo.get_highest_summarised_sequence.return_value = watermark
        assert manager.should_snapshot() is expected


class TestSnapshotting:
    def test_snapshot_now_on_empty_conversation_returns_none(self, mgr):
        manager, mock_fc, mock_cs = mgr
        mock_fc.next_sequence_number.return_value = 1  # latest = 0
        assert manager.snapshot_now() is None
        mock_cs.take_snapshot.assert_not_called()

    def test_snapshot_now_summarises_up_to_the_latest_turn(self, mgr):
        manager, mock_fc, mock_cs = mgr
        mock_fc.next_sequence_number.return_value = 31
        mock_cs.take_snapshot.return_value = "a summary"
        assert manager.snapshot_now() == "a summary"
        mock_cs.take_snapshot.assert_called_once_with(30)

    def test_maybe_snapshot_below_threshold_does_nothing(self, mgr):
        manager, mock_fc, mock_cs = mgr
        mock_fc.next_sequence_number.return_value = 4  # latest 3, threshold 5
        mock_cs.summary_repo.get_highest_summarised_sequence.return_value = 0
        assert manager.maybe_snapshot() is None
        mock_cs.take_snapshot.assert_not_called()

    def test_maybe_snapshot_at_threshold_fires(self, mgr):
        manager, mock_fc, mock_cs = mgr
        mock_fc.next_sequence_number.return_value = 6  # latest 5, threshold 5
        mock_cs.summary_repo.get_highest_summarised_sequence.return_value = 0
        mock_cs.take_snapshot.return_value = "fired"
        assert manager.maybe_snapshot() == "fired"

    def test_record_turn_returns_sequence_and_no_summary(self, mgr):
        manager, mock_fc, mock_cs = mgr
        mock_fc.append_turn.return_value = 3
        mock_fc.next_sequence_number.return_value = 4
        mock_cs.summary_repo.get_highest_summarised_sequence.return_value = 0
        assert manager.record_turn("user", "hello") == (3, None)

    def test_record_turn_returns_the_summary_when_due(self, mgr):
        manager, mock_fc, mock_cs = mgr
        mock_fc.append_turn.return_value = 5
        mock_fc.next_sequence_number.return_value = 6
        mock_cs.summary_repo.get_highest_summarised_sequence.return_value = 0
        mock_cs.take_snapshot.return_value = "rolled up"
        assert manager.record_turn("user", "hello") == (5, "rolled up")


class TestSearch:
    def test_search_embeds_the_query_then_delegates(self, mgr):
        manager, _, mock_cs = mgr
        vector = np.ones(128, dtype=np.float32)
        mock_cs.embedder.embed_text.return_value.vector = vector
        manager.search("how does it work?")
        mock_cs.embedder.embed_text.assert_called_once_with("how does it work?")
        manager.snap_shot.search.assert_called_once()

    def test_search_vector_skips_embedding(self, mgr):
        manager, _, mock_cs = mgr
        manager.search_vector(np.ones(128, dtype=np.float32))
        mock_cs.embedder.embed_text.assert_not_called()

    def test_advance_and_prev_delegate(self, mgr):
        manager, _, _ = mgr
        manager.advance()
        manager.prev()
        manager.snap_shot.advance.assert_called_once()
        manager.snap_shot.prev.assert_called_once()


# ---------------------------------------------------------------------------
# Integration — real SQLite, stubbed LLM and pgvector
# ---------------------------------------------------------------------------

class _FakeVectorRepository:
    def __init__(self, project_id):
        self.project_id = project_id
        self.store = _FakeVectorRepository._shared.setdefault(project_id, {})

    _shared: dict = {}

    def insert(self, vector_id, vector):
        assert -(2**63) <= int(vector_id) <= 2**63 - 1
        self.store[int(vector_id)] = np.asarray(vector, dtype=np.float32)

    def batch_insert(self, vector_ids, vectors):
        for vid, vec in zip(vector_ids, vectors):
            self.insert(vid, vec)

    def search(self, vector_id):
        return self.store[int(vector_id)]

    def batch_search(self, vector_ids):
        return np.array([self.search(v) for v in vector_ids])


def _fake_llm():
    model = MagicMock()
    model.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "A rolled-up summary."}}]
    }
    return model


@pytest.fixture
def live(tmp_path):
    """A manager over real SQLite with the LLM and pgvector stubbed."""
    from memory.topic_pool.project_pool.conversation_pool.conversation_summary_pipeline.conversation_summary import (
        ConversationSummary,
    )

    _FakeVectorRepository._shared.clear()
    embedder = MagicMock()

    def embed_one(text, chunk_id=None):
        result = MagicMock()
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        result.vector = rng.random(128).astype(np.float32)
        result.vector_id = abs(hash(text)) % (2**63 - 1)
        return result

    embedder.embed_text.side_effect = embed_one
    embedder.embed_texts.side_effect = lambda texts, ids=None: [
        embed_one(t, i) for t, i in zip(texts, ids or [None] * len(texts))
    ]

    with patch(_VEC_REPO, _FakeVectorRepository), patch.multiple(
        ConversationSummary,
        _ConversationSummary__load_model=lambda self: _fake_llm(),
        _ConversationSummary__unload_model=lambda self, m: None,
    ):
        manager = ConversationPoolManager(
            conversation_dir=tmp_path,
            project_id=_PID,
            project_name=_PNAME,
            main_model_context_window_length=50,
            snapshot_every_n_turns=5,
        )
        manager.summariser._embedder = embedder
        yield manager, tmp_path


class TestIntegration:
    def test_empty_manager_state(self, live):
        manager, _ = live
        assert manager.size() == 0
        assert manager.latest_sequence() == 0
        assert manager.summarised_upto() == 0
        assert manager.current_summary() is None
        assert manager.should_snapshot() is False

    def test_turns_accumulate_without_firing(self, live):
        manager, _ = live
        for i in range(4):
            _, summary = manager.record_turn("user", f"turn {i}")
            assert summary is None
        assert manager.turns_since_last_snapshot() == 4

    def test_threshold_fires_a_snapshot(self, live):
        manager, _ = live
        summaries = [manager.record_turn("user", f"turn {i}")[1] for i in range(5)]
        assert summaries[:4] == [None] * 4
        assert summaries[4] == "A rolled-up summary."

    def test_watermark_advances_after_a_snapshot(self, live):
        manager, _ = live
        for i in range(5):
            manager.record_turn("user", f"turn {i}")
        assert manager.summarised_upto() == 5
        assert manager.turns_since_last_snapshot() == 0

    def test_summary_is_persisted_and_readable(self, live):
        manager, _ = live
        for i in range(5):
            manager.record_turn("user", f"turn {i}")
        assert manager.current_summary() == "A rolled-up summary."

    def test_snapshots_fire_at_a_steady_cadence(self, live):
        manager, _ = live
        fired = [manager.record_turn("user", f"turn {i}")[1] for i in range(15)]
        assert sum(s is not None for s in fired) == 3

    def test_turns_are_stored_in_order(self, live):
        manager, _ = live
        for i in range(4):
            manager.add_turn("user", f"turn {i}")
        assert [r[0] for r in manager.history()] == [f"turn {i}" for i in range(4)]

    def test_snapshot_now_ignores_the_threshold(self, live):
        manager, _ = live
        manager.add_turn("user", "only one turn")
        assert manager.should_snapshot() is False
        assert manager.snapshot_now() == "A rolled-up summary."

    def test_search_finds_a_stored_snapshot(self, live):
        manager, _ = live
        for i in range(5):
            manager.record_turn("user", f"turn {i}")
        assert manager.search("turn") is not None

    def test_reopened_manager_sees_stored_state(self, live, tmp_path):
        """
        The reopen path: a new manager over the same directory must restore the
        summary, the watermark and the snapshot cursors.
        """
        manager, _ = live
        for i in range(5):
            manager.record_turn("user", f"turn {i}")

        reopened = ConversationPoolManager(
            conversation_dir=tmp_path,
            project_id=_PID,
            project_name=_PNAME,
            main_model_context_window_length=50,
            snapshot_every_n_turns=5,
        )
        reopened.summariser._embedder = manager.summariser._embedder
        assert reopened.size() == 5
        assert reopened.summarised_upto() == 5
        assert reopened.current_summary() == "A rolled-up summary."
        assert reopened.search("turn") is not None
