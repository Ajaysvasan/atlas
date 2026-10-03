"""Snapshot navigation over a project's cumulative summary history.

See README.md in this directory.
"""

from pathlib import Path
from typing import List, Tuple

from config import get_logger
from memory.memory_pool_exceptions import (
    InvalidCursorException,
    InvalidSnapshotScope,
    MisMatchCount,
    NullPointerException,
    WrongSnapshotScope,
)
from numpy import ndarray, uint32
from torch import cosine_similarity, tensor

from memory.topic_pool.project_pool.conversation_pool.conversation_data_management.conversationVectorManager import (
    ConversationVectorManager,
)
from memory.topic_pool.project_pool.conversation_pool.conversation_data_management.conversationVectorMetaManager import (
    ConversationVectorMetaDataRepository,
)

from memory.identifiers import require_identifier
from memory.timestamps import utc_now
from memory.topic_pool.project_pool.project_data_repo.project_snapshot_repo import (
    ProjectSnapshotRepository,
    project_snapshot_id,
)

logger = get_logger(__name__)

CONVERSATION = "conversation"
PROJECT = "project"
SCOPES = frozenset({CONVERSATION, PROJECT})


class SnapShot:
    def __init__(
        self,
        conversation_dir: str | Path,
        project_id: str,
        project_name: str,
        conversation_id: str = "",
        meta_repo: ConversationVectorMetaDataRepository | None = None,
        scope: str = CONVERSATION,
        snapshot_repo: ProjectSnapshotRepository | None = None,
        project_db_path: str | Path | None = None,
    ) -> None:
        if scope not in SCOPES:
            raise InvalidSnapshotScope(scope, SCOPES)
        self.scope = scope
        self.conversation_dir = Path(conversation_dir)
        self.project_id = project_id
        self.project_name = project_name

        # The two scopes keep their history in different stores, so each builds
        # only the one it reads: a conversation snapshot walks this project's
        # cumulative summaries in the conversation database, a project snapshot
        # walks the project registry. conversation_id is meaningless to the
        # second, which is why it is only required for the first.
        self._owns_meta_repo = False
        self._owns_snapshot_repo = False
        self.meta_repo: ConversationVectorMetaDataRepository | None = None
        self.snapshot_repo: ProjectSnapshotRepository | None = None

        if scope == CONVERSATION:
            self.conversation_id = require_identifier(
                conversation_id, "conversation_id"
            )
            self._owns_meta_repo = meta_repo is None
            self.meta_repo = meta_repo or ConversationVectorMetaDataRepository(
                self.conversation_dir, project_id, self.conversation_id
            )
        else:
            self.conversation_id = conversation_id
            self._owns_snapshot_repo = snapshot_repo is None
            self.snapshot_repo = snapshot_repo or ProjectSnapshotRepository(
                project_id, db_path=project_db_path
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
        # The ids the cursors index into and the search ranks. Both scopes store
        # their vector under the id returned here, so everything downstream --
        # cursors, advance/prev, search -- is the same code for both.
        if self.scope == PROJECT:
            return [(row.project_snapshot_id,) for row in self.snapshot_repo.history()]
        return self.meta_repo.get_cumulative_vector_meta_data_ids()

    def __require_scope(self, required: str, operation: str) -> None:
        if self.scope != required:
            raise WrongSnapshotScope(operation, required, self.scope)

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
        self.__require_scope(CONVERSATION, "add()")
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

    def add_project_snapshot(
        self,
        summary: str,
        last_seq_included: int,
        summary_vector: ndarray,
        created_at: str | None = None,
    ) -> int:
        """Store one project snapshot: its vector, then its metadata.

        Vectors first, metadata second, with a compensating delete — the same
        order as a conversation snapshot, for the same reason: a failure then
        leaves an unreachable vector rather than a row pointing at nothing.
        """
        self.__require_scope(PROJECT, "add_project_snapshot()")
        created_at = created_at or utc_now()
        snapshot_id = project_snapshot_id(self.project_id, created_at, summary)

        self.vector_manager.insert(snapshot_id, summary_vector)
        try:
            stored = self.snapshot_repo.add_snapshot(
                summary=summary,
                last_seq_included=last_seq_included,
                created_at=created_at,
            )
        except Exception:
            logger.warning(
                "Project snapshot metadata failed for %s; removing vector %s",
                self.project_id, snapshot_id,
            )
            try:
                self.vector_manager.batch_delete([snapshot_id])
            except Exception:
                logger.exception(
                    "Compensating delete failed for project %s, vector id %s. "
                    "That vector is now unreachable from project_snapshot.",
                    self.project_id, snapshot_id,
                )
            raise
        return stored

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
        if self._vector_manager is not None:
            self._vector_manager.close()
            self._vector_manager = None
        if self._owns_meta_repo and self.meta_repo is not None:
            self.meta_repo.close()
        if self._owns_snapshot_repo and self.snapshot_repo is not None:
            self.snapshot_repo.close()
