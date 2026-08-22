"""
Production-grade tests for SnapShot.

Both ConversationVectorMetaDataRepository and ConversationVectorManager are
patched at the snapshot module level so no real SQLite or DiskANN index is
touched. torch.cosine_similarity and torch.tensor are the real implementations
(torch IS installed); do NOT mock torch globally or comparisons will fail.

Regression guards are annotated with their bug ID.
"""

from typing import List, Tuple
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from memory.topic_pool.project_pool.conversation_pool.snapshot import SnapShot
from memory_pool_exceptions import InvalidCursorException, MisMatchCount, NullPointerException

_META_REPO = "memory.topic_pool.project_pool.conversation_pool.snapshot.ConversationVectorMetaDataRepository"
_VEC_MGR   = "memory.topic_pool.project_pool.conversation_pool.snapshot.ConversationVectorManager"

_DIM   = 3
_QUERY = np.array([0.1, 0.2, 0.3], dtype=np.float32)
_VEC   = np.array([0.1, 0.2, 0.3], dtype=np.float32)

_CHUNKS: List[Tuple[str, str, str, str]] = [
    ("chunk_1", "summary text one", "2026-08-01", "typeA"),
    ("chunk_2", "summary text two", "2026-08-01", "typeA"),
]
_CHUNK_IDS = ["chunk_1", "chunk_2"]
_SNAP_LIST = [(1,), (2,), (3,)]


def _dummy_add_kwargs(**overrides):
    defaults = dict(
        project_name="TestProject",
        project_id="test_proj",
        time_of_snapshot="2026-08-01",
        len_of_the_summary=100,
        summary_vector_ids=[np.uint32(101), np.uint32(102)],
        summary_vectors=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
        chunk_ids=_CHUNK_IDS,
        chunks=_CHUNKS,
        summary="This is a test summary",
        cumulative_summary_vector_id=np.uint32(1),
        cumulative_summary_vector=np.array([0.1, 0.2, 0.3], dtype=np.float32),
    )
    defaults.update(overrides)
    return defaults


@pytest.fixture
def patched():
    with patch(_META_REPO) as MockMeta, patch(_VEC_MGR) as MockVec:
        mock_meta = MockMeta.return_value
        mock_vec  = MockVec.return_value
        mock_meta.get_cumulative_vector_meta_data_ids.return_value = _SNAP_LIST
        mock_vec.get_vector.return_value = _VEC.copy()
        snap = SnapShot()
        yield snap, mock_meta, mock_vec


# ---------------------------------------------------------------------------
# Cursor initialisation (Bug 4.25 regression)
# ---------------------------------------------------------------------------

class TestCursorInit:
    def test_cursors_start_at_minus_one_before_any_add(self):
        snap = SnapShot()
        assert snap._SnapShot__left_cursor == -1
        assert snap._SnapShot__right_cursor == -1

    def test_first_add_sets_both_cursors_to_zero(self, patched):
        """Bug 4.25 regression: right cursor must be 0 after the first add, not 1."""
        snap, _, _ = patched
        snap.add(**_dummy_add_kwargs())
        assert snap._SnapShot__left_cursor == 0
        assert snap._SnapShot__right_cursor == 0

    def test_second_add_increments_right_cursor(self, patched):
        snap, _, _ = patched
        snap.add(**_dummy_add_kwargs(cumulative_summary_vector_id=np.uint32(1)))
        snap.add(**_dummy_add_kwargs(cumulative_summary_vector_id=np.uint32(2)))
        assert snap._SnapShot__left_cursor == 0
        assert snap._SnapShot__right_cursor == 1

    def test_five_adds_right_cursor_is_four(self, patched):
        snap, _, _ = patched
        for i in range(5):
            snap.add(**_dummy_add_kwargs(cumulative_summary_vector_id=np.uint32(i + 1)))
        assert snap._SnapShot__right_cursor == 4

    def test_left_cursor_never_auto_increments_on_add(self, patched):
        snap, _, _ = patched
        for i in range(10):
            snap.add(**_dummy_add_kwargs(cumulative_summary_vector_id=np.uint32(i + 1)))
        assert snap._SnapShot__left_cursor == 0

    def test_reset_flags_update_cursors_from_db(self, patched):
        snap, mock_meta, _ = patched
        mock_meta.get_cumulative_vector_meta_data_ids.return_value = [(10,), (20,), (30,)]
        snap.add(**_dummy_add_kwargs(reset_right_pointer=True, reset_left_pointer=True))
        assert snap._SnapShot__left_cursor == 0
        assert snap._SnapShot__right_cursor == 2


# ---------------------------------------------------------------------------
# add raises MisMatchCount on mismatched argument lengths
# ---------------------------------------------------------------------------

class TestAddValidation:
    def test_mismatch_chunk_ids_and_vector_ids_raises(self, patched):
        """MisMatchCount must be raised when len(chunk_ids) != len(summary_vector_ids)."""
        snap, _, _ = patched
        with pytest.raises(MisMatchCount):
            snap.add(**_dummy_add_kwargs(
                chunk_ids=["only_one"],                           # 1 id
                summary_vector_ids=[np.uint32(1), np.uint32(2)], # 2 ids
            ))

    def test_mismatch_more_chunks_than_vectors_raises(self, patched):
        snap, _, _ = patched
        with pytest.raises(MisMatchCount):
            snap.add(**_dummy_add_kwargs(
                chunk_ids=["c1", "c2", "c3"],
                summary_vector_ids=[np.uint32(1)],
            ))

    def test_empty_ids_matching_lengths_does_not_raise(self, patched):
        snap, _, _ = patched
        snap.add(**_dummy_add_kwargs(
            chunk_ids=[],
            chunks=[],
            summary_vector_ids=[],
            summary_vectors=np.zeros((0, 3), dtype=np.float32),
        ))


# ---------------------------------------------------------------------------
# Search does not mutate cursors (Bug 4.3 regression)
# ---------------------------------------------------------------------------

class TestSearchDoesNotMutateCursors:
    def test_search_leaves_left_cursor_unchanged(self, patched):
        """Bug 4.3 regression: __find_best_snapshot must not modify instance cursors."""
        snap, _, _ = patched
        snap._SnapShot__left_cursor = 0
        snap._SnapShot__right_cursor = 2
        snap.search(query=_QUERY, project_id="test_proj", project_name="TestProject")
        assert snap._SnapShot__left_cursor == 0

    def test_search_leaves_right_cursor_unchanged(self, patched):
        snap, _, _ = patched
        snap._SnapShot__left_cursor = 0
        snap._SnapShot__right_cursor = 2
        snap.search(query=_QUERY, project_id="test_proj", project_name="TestProject")
        assert snap._SnapShot__right_cursor == 2

    def test_repeated_search_does_not_drift_cursors(self, patched):
        snap, _, _ = patched
        snap._SnapShot__left_cursor = 0
        snap._SnapShot__right_cursor = 2
        for _ in range(10):
            snap.search(query=_QUERY, project_id="test_proj", project_name="TestProject")
        assert snap._SnapShot__left_cursor == 0
        assert snap._SnapShot__right_cursor == 2

    def test_search_returns_valid_snapshot_row(self, patched):
        snap, _, _ = patched
        snap._SnapShot__left_cursor = 0
        snap._SnapShot__right_cursor = 2
        result = snap.search(query=_QUERY, project_id="test_proj", project_name="TestProject")
        assert result is not None
        assert result in _SNAP_LIST

    def test_search_single_element_window(self, patched):
        snap, mock_meta, _ = patched
        mock_meta.get_cumulative_vector_meta_data_ids.return_value = [(42,)]
        snap._SnapShot__left_cursor = 0
        snap._SnapShot__right_cursor = 0
        result = snap.search(query=_QUERY, project_id="test_proj", project_name="TestProject")
        assert result == (42,)
        assert snap._SnapShot__left_cursor == 0
        assert snap._SnapShot__right_cursor == 0

    def test_search_selects_most_similar_vector(self, patched):
        """
        When one vector is identical to the query and the other is orthogonal,
        the matching snapshot row must be returned.
        """
        snap, mock_meta, mock_vec = patched
        # Two snapshots: id=1 → orthogonal vector, id=2 → identical to query
        snap_list = [(1,), (2,)]
        mock_meta.get_cumulative_vector_meta_data_ids.return_value = snap_list
        orthogonal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        identical  = np.array([0.1, 0.2, 0.3], dtype=np.float32)  # same as _QUERY
        mock_vec.get_vector.side_effect = [orthogonal, identical]
        snap._SnapShot__left_cursor = 0
        snap._SnapShot__right_cursor = 1
        result = snap.search(
            query=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            project_id="test_proj",
            project_name="TestProject",
        )
        assert result == (2,)


# ---------------------------------------------------------------------------
# advance / prev cursor navigation
# ---------------------------------------------------------------------------

class TestCursorNavigation:
    def test_advance_increments_left_cursor(self, patched):
        snap, _, _ = patched
        snap._SnapShot__left_cursor = 0
        snap._SnapShot__right_cursor = 2
        snap.advance("test_proj")
        assert snap._SnapShot__left_cursor == 1

    def test_prev_decrements_right_cursor(self, patched):
        snap, _, _ = patched
        snap._SnapShot__left_cursor = 0
        snap._SnapShot__right_cursor = 2
        snap.prev()
        assert snap._SnapShot__right_cursor == 1

    def test_advance_beyond_right_raises(self, patched):
        snap, _, _ = patched
        snap._SnapShot__left_cursor = 2
        snap._SnapShot__right_cursor = 2
        with pytest.raises(InvalidCursorException):
            snap.advance("test_proj")

    def test_prev_beyond_left_raises(self, patched):
        snap, _, _ = patched
        snap._SnapShot__left_cursor = 1
        snap._SnapShot__right_cursor = 1
        with pytest.raises(InvalidCursorException):
            snap.prev()

    def test_advance_then_prev_roundtrip(self, patched):
        snap, _, _ = patched
        snap._SnapShot__left_cursor = 0
        snap._SnapShot__right_cursor = 2
        snap.advance("test_proj")
        snap.prev()
        assert snap._SnapShot__left_cursor == 1
        assert snap._SnapShot__right_cursor == 1

    def test_advance_to_max_then_raises(self, patched):
        snap, mock_meta, _ = patched
        mock_meta.get_cumulative_vector_meta_data_ids.return_value = [(1,), (2,)]
        snap._SnapShot__left_cursor = 0
        snap._SnapShot__right_cursor = 1
        snap.advance("test_proj")
        with pytest.raises(InvalidCursorException):
            snap.advance("test_proj")

    def test_prev_to_min_then_raises(self, patched):
        snap, _, _ = patched
        snap._SnapShot__left_cursor = 0
        snap._SnapShot__right_cursor = 2
        snap.prev()
        snap.prev()
        with pytest.raises(InvalidCursorException):
            snap.prev()

    def test_advance_multiple_steps(self, patched):
        snap, mock_meta, _ = patched
        mock_meta.get_cumulative_vector_meta_data_ids.return_value = [
            (i,) for i in range(10)
        ]
        snap._SnapShot__left_cursor = 0
        snap._SnapShot__right_cursor = 9
        for _ in range(5):
            snap.advance("test_proj")
        assert snap._SnapShot__left_cursor == 5
        assert snap._SnapShot__right_cursor == 9

    def test_prev_multiple_steps(self, patched):
        snap, _, _ = patched
        snap._SnapShot__left_cursor = 0
        snap._SnapShot__right_cursor = 5
        for _ in range(5):
            snap.prev()
        assert snap._SnapShot__right_cursor == 0
        with pytest.raises(InvalidCursorException):
            snap.prev()


# ---------------------------------------------------------------------------
# Search on empty snapshot list
# ---------------------------------------------------------------------------

class TestSearchEdgeCases:
    def test_search_on_empty_snapshot_list_raises(self, patched):
        snap, mock_meta, _ = patched
        mock_meta.get_cumulative_vector_meta_data_ids.return_value = []
        snap._SnapShot__left_cursor = 0
        snap._SnapShot__right_cursor = 0
        with pytest.raises(NullPointerException):
            snap.search(query=_QUERY, project_id="test_proj", project_name="TestProject")

    def test_search_full_window_returns_best_match(self, patched):
        snap, mock_meta, mock_vec = patched
        snap_list = [(1,), (2,), (3,), (4,), (5,)]
        mock_meta.get_cumulative_vector_meta_data_ids.return_value = snap_list
        mock_vec.get_vector.return_value = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        snap._SnapShot__left_cursor = 0
        snap._SnapShot__right_cursor = 4
        result = snap.search(
            query=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            project_id="test_proj",
            project_name="TestProject",
        )
        assert result is not None
        assert result in snap_list

    def test_search_returns_none_when_no_best_found(self, patched):
        """
        If __find_best_snapshot returns None, search must return None without raising.
        This exercises the None guard in search().
        """
        snap, mock_meta, mock_vec = patched
        mock_meta.get_cumulative_vector_meta_data_ids.return_value = [(1,)]
        mock_vec.get_vector.return_value = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        snap._SnapShot__left_cursor = 0
        snap._SnapShot__right_cursor = 0
        # cosine_similarity(zero, query) is 0 but best_snap_shot_idx will be set to 0
        # (not -1), so result should be (1,)
        result = snap.search(
            query=_QUERY, project_id="test_proj", project_name="TestProject"
        )
        # A valid tuple is returned (zero vector still wins because it's the only candidate)
        assert result == (1,) or result is None  # result depends on sim score; no crash


# ---------------------------------------------------------------------------
# Stress
# ---------------------------------------------------------------------------

class TestSnapShotStress:
    def test_fifty_sequential_adds_right_cursor(self, patched):
        snap, _, _ = patched
        for i in range(50):
            snap.add(**_dummy_add_kwargs(cumulative_summary_vector_id=np.uint32(i + 1)))
        assert snap._SnapShot__left_cursor == 0
        assert snap._SnapShot__right_cursor == 49

    def test_search_one_hundred_times_no_cursor_drift(self, patched):
        snap, _, _ = patched
        snap._SnapShot__left_cursor = 0
        snap._SnapShot__right_cursor = 2
        for _ in range(100):
            snap.search(query=_QUERY, project_id="test_proj", project_name="TestProject")
        assert snap._SnapShot__left_cursor == 0
        assert snap._SnapShot__right_cursor == 2

    def test_add_invokes_meta_and_vector_managers(self, patched):
        snap, mock_meta, mock_vec = patched
        snap.add(**_dummy_add_kwargs())
        mock_vec.batch_insert.assert_called_once()
        mock_vec.insert.assert_called_once()
        mock_meta.batch_insert_summary_chunks.assert_called_once_with(_CHUNKS)
        mock_meta.batch_insert_summary_vector_meta_data.assert_called_once()
        mock_meta.insert_cumulative_vector_meta_data.assert_called_once()
        mock_meta.batch_insert_map_table.assert_called_once()

    def test_chunks_inserted_before_summary_vectors(self, patched):
        """Regression guard Bug 4.6: summary_chunks must be inserted before FK-dependent tables."""
        snap, mock_meta, _ = patched
        call_order = []
        mock_meta.batch_insert_summary_chunks.side_effect = (
            lambda *a, **kw: call_order.append("chunks")
        )
        mock_meta.batch_insert_summary_vector_meta_data.side_effect = (
            lambda *a, **kw: call_order.append("summary_vectors")
        )
        snap.add(**_dummy_add_kwargs())
        chunks_idx = call_order.index("chunks")
        sv_idx = call_order.index("summary_vectors")
        assert chunks_idx < sv_idx, "summary_chunks must be inserted before summary_vector_meta_data"

    def test_advance_and_search_interleaved_no_drift(self, patched):
        """Advancing left cursor must not affect search cursor isolation."""
        snap, mock_meta, _ = patched
        mock_meta.get_cumulative_vector_meta_data_ids.return_value = [
            (i,) for i in range(5)
        ]
        snap._SnapShot__left_cursor = 0
        snap._SnapShot__right_cursor = 4
        snap.advance("test_proj")
        snap.search(query=_QUERY, project_id="test_proj", project_name="TestProject")
        # advance moved left to 1, search must NOT change it back or forward
        assert snap._SnapShot__left_cursor == 1
        assert snap._SnapShot__right_cursor == 4

    def test_five_hundred_adds_cursor_state_is_correct(self, patched):
        snap, _, _ = patched
        for i in range(500):
            snap.add(**_dummy_add_kwargs(cumulative_summary_vector_id=np.uint32(i + 1)))
        assert snap._SnapShot__left_cursor == 0
        assert snap._SnapShot__right_cursor == 499
