"""Snapshot navigation over a project's cumulative summary history.

See README.md in this directory.
"""

from pathlib import Path
from typing import List, Tuple

from config import get_logger
from memory.memory_pool_exceptions import (
    InvalidCursorException,
    MisMatchCount,
    NullPointerException,
)
from numpy import ndarray, uint32
from torch import cosine_similarity, tensor

from memory.topic_pool.project_pool.conversation_pool.conversation_data_management.conversationVectorManager import (
    ConversationVectorManager,
)
from memory.topic_pool.project_pool.conversation_pool.conversation_data_management.conversationVectorMetaManager import (
    ConversationVectorMetaDataRepository,
)

logger = get_logger(__name__)


class SnapShot:
    def __init__(
        self,
        conversation_dir: str | Path,
        project_id: str,
        project_name: str,
        meta_repo: ConversationVectorMetaDataRepository | None = None,
    ) -> None:
        self.conversation_dir = Path(conversation_dir)
        self.project_id = project_id
        self.project_name = project_name

        self._owns_meta_repo = meta_repo is None
        self.meta_repo = meta_repo or ConversationVectorMetaDataRepository(
            self.conversation_dir, project_id
        )
        self._vector_manager: ConversationVectorManager | None = None

        self.__left_cursor: int = -1
        self.__right_cursor: int = -1

    @property
    def vector_manager(self) -> ConversationVectorManager:
        if self._vector_manager is None:
            self._vector_manager = ConversationVectorManager(
                self.project_name, self.project_id
            )
        return self._vector_manager

    def __get_snap_shot(self):
        return self.meta_repo.get_cumulative_vector_meta_data_ids()

    def __add_snap_shot(
        self,
        time_of_snapshot: str,
        len_of_the_summary: int,
        summary_vector_ids: List,
        summary_vectors: ndarray,
        chunk_ids: List[str],
        chunks: List[Tuple[str, str, str, str]],
        summary: str,
        cumulative_summary_vector_id: uint32,
        cumulative_summary_vector: ndarray,
    ) -> None:
        if len(chunk_ids) != len(summary_vector_ids):
            raise MisMatchCount("Mis matched arguments received")

        summary_vector_tuple_list = [
            (summary_vector_ids[i], chunk_ids[i], self.project_id)
            for i in range(len(chunk_ids))
        ]
        map_list = [
            (cumulative_summary_vector_id, summary_vector_ids[i])
            for i in range(len(summary_vector_ids))
        ]

        self.vector_manager.batch_insert(summary_vector_ids, summary_vectors)
        self.vector_manager.insert(
            cumulative_summary_vector_id, cumulative_summary_vector
        )

        try:
            self.meta_repo.insert_snapshot(
                chunks=chunks,
                cumulative_row=(
                    int(cumulative_summary_vector_id),
                    summary,
                    time_of_snapshot,
                    self.project_id,
                    len_of_the_summary,
                ),
                summary_vector_rows=summary_vector_tuple_list,
                map_rows=map_list,
            )
        except Exception:
            # Compensating delete: the metadata rolled back, so these vectors
            # are now unreachable.
            orphans = list(summary_vector_ids) + [cumulative_summary_vector_id]
            logger.warning(
                "Snapshot metadata failed for project %s; removing %d vector(s)",
                self.project_id,
                len(orphans),
            )
            try:
                self.vector_manager.batch_delete(orphans)
            except Exception:
                logger.exception(
                    "Compensating delete failed for project %s, vector ids %s. "
                    "These vectors are now unreachable from summary_snapshot_map.",
                    self.project_id,
                    orphans,
                )
            raise

    def add(
        self,
        time_of_snapshot: str,
        len_of_the_summary: int,
        summary_vector_ids: List,
        summary_vectors: ndarray,
        chunk_ids: List[str],
        chunks: List[Tuple[str, str, str, str]],
        summary: str,
        cumulative_summary_vector_id: uint32,
        cumulative_summary_vector: ndarray,
        reset_right_pointer: bool = False,
        reset_left_pointer: bool = False,
    ):
        self.__add_snap_shot(
            time_of_snapshot,
            len_of_the_summary,
            summary_vector_ids,
            summary_vectors,
            chunk_ids,
            chunks,
            summary,
            cumulative_summary_vector_id,
            cumulative_summary_vector,
        )
        if self.__left_cursor == -1 and self.__right_cursor == -1:
            self.__left_cursor = 0
            self.__right_cursor = 0
        else:
            self.__right_cursor += 1

        logger.debug(
            "Snapshot added for project %s covering %d chunk(s); cursors now %d..%d",
            self.project_id,
            len(chunk_ids),
            self.__left_cursor,
            self.__right_cursor,
        )

        if reset_right_pointer:
            self.__reset_right_pointer()

        if reset_left_pointer:
            self.__reset_left_pointer()

    def advance(self) -> None:
        """Makes left cursor move"""
        snap_shot_list = self.__get_snap_shot()
        if (
            self.__left_cursor + 1 < len(snap_shot_list)
            and self.__left_cursor < self.__right_cursor
        ):
            self.__left_cursor += 1
            return
        raise InvalidCursorException("left", self.__left_cursor + 1)

    def prev(self) -> None:
        """Makes the right curosr move"""
        if self.__right_cursor - 1 >= 0 and self.__right_cursor > self.__left_cursor:
            self.__right_cursor -= 1
            return

        raise InvalidCursorException("right", self.__right_cursor - 1)

    def __reset_left_pointer(self) -> None:
        snap_shot_list = self.__get_snap_shot()
        if len(snap_shot_list) != 0:
            self.__left_cursor = 0
            return
        raise NullPointerException("No snap shots found")

    def __reset_right_pointer(self) -> None:
        snap_shot_list = self.__get_snap_shot()
        if len(snap_shot_list) != 0:
            self.__right_cursor = len(snap_shot_list) - 1
            return
        raise NullPointerException("No snap shots found")

    def sync_cursors(self) -> None:
        """Point the cursors at the full stored history."""
        self.__reset_left_pointer()
        self.__reset_right_pointer()

    def __ensure_cursors(self, snap_shot_list) -> None:
        """Open the cursors onto stored history if they were never set."""
        if self.__left_cursor < 0 or self.__right_cursor < 0:
            if len(snap_shot_list) == 0:
                raise NullPointerException("No snap shots found")
            self.__left_cursor = 0
            self.__right_cursor = len(snap_shot_list) - 1

    def __find_best_snapshot(self, query: ndarray, snap_shot_list) -> int | None:
        if len(snap_shot_list) == 0:
            raise NullPointerException("No snap shots found")

        self.__ensure_cursors(snap_shot_list)

        best_snap_shot_idx = -1
        best_similarity = float("-inf")
        vector_manager = self.vector_manager
        left = self.__left_cursor
        right = self.__right_cursor
        while left <= right:
            if left == right:
                snap = snap_shot_list[left]
                vec = tensor(vector_manager.get_vector(snap[0]))
                sim = cosine_similarity(vec, tensor(query), dim=0)
                if sim > best_similarity:
                    best_similarity = sim
                    best_snap_shot_idx = left
                break

            left_snap = snap_shot_list[left]
            right_snap = snap_shot_list[right]

            left_snap_vector_cumulative = tensor(vector_manager.get_vector(left_snap[0]))
            right_snap_vector_cumulative = tensor(
                vector_manager.get_vector(right_snap[0])
            )

            left_sim = cosine_similarity(
                left_snap_vector_cumulative, tensor(query), dim=0
            )
            right_sim = cosine_similarity(
                right_snap_vector_cumulative, tensor(query), dim=0
            )

            if left_sim > best_similarity:
                best_similarity = left_sim
                best_snap_shot_idx = left

            if right_sim > best_similarity:
                best_similarity = right_sim
                best_snap_shot_idx = right

            left += 1
            right -= 1

        return best_snap_shot_idx if best_snap_shot_idx > -1 else None

    def search(self, query: ndarray):
        # Fetched once and passed down (Bug 4.24): two reads let a concurrent
        # insert shift the list between them.
        snap_shot_list = self.__get_snap_shot()
        best_snap_shot_idx: int | None = self.__find_best_snapshot(
            query, snap_shot_list
        )
        if best_snap_shot_idx is None:
            logger.info(
                "No snapshot matched the query across %d candidate(s)",
                len(snap_shot_list),
            )
            return None
        logger.debug(
            "Best snapshot is index %d of %d", best_snap_shot_idx, len(snap_shot_list)
        )
        return snap_shot_list[best_snap_shot_idx]

    def close(self) -> None:
        """Release every connection this object opened."""
        if self._owns_meta_repo:
            self.meta_repo.close()
