"""
Snapshot navigation over a project's cumulative summary history.

A snapshot is one cumulative summary vector (the whole summary) plus one
summary vector per conversation chunk it covers, linked through
summary_snapshot_map. search() locates the best snapshot by cumulative vector;
the map then allows drilling into the chunks that snapshot summarised.

The conversation directory is supplied by the caller. It used to be read from
Config.CONVERSATION, a single global path, which meant SnapShot wrote to a
different database than ConversationSummary read from — the rolling summary
never saw its own previous output.
"""

from pathlib import Path
from typing import List, Tuple

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

        # Held for the lifetime of the object rather than rebuilt per call. A
        # caller that already has a repository for this database should pass it
        # in: building a second one opens a redundant connection to the same
        # file, and connections then scale with the number of live objects.
        self._owns_meta_repo = meta_repo is None
        self.meta_repo = meta_repo or ConversationVectorMetaDataRepository(
            self.conversation_dir, project_id
        )
        # Built on first use: this one opens a PostgreSQL connection, and cursor
        # navigation never needs it.
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

        # Vectors first, metadata second. The reverse order would let a vector
        # failure leave metadata pointing at vectors that do not exist, which
        # breaks search(); this way a failure leaves only unreachable vectors,
        # and those are then deleted below.
        self.vector_manager.batch_insert(summary_vector_ids, summary_vectors)
        self.vector_manager.insert(
            cumulative_summary_vector_id, cumulative_summary_vector
        )

        try:
            # One transaction for the whole metadata side. Previously these were
            # four independently committing calls, so a failure partway through
            # left a snapshot that half-existed.
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
            # Compensating delete: the metadata rolled back, so the vectors it
            # would have pointed at are unreachable garbage. Failing to remove
            # them would accumulate on every retry.
            try:
                self.vector_manager.batch_delete(
                    list(summary_vector_ids) + [cumulative_summary_vector_id]
                )
            except Exception:
                # The original metadata failure is the one worth reporting.
                pass
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
        """Point the cursors at the full stored history.

        Cursors live in memory only, so a freshly constructed SnapShot starts at
        -1/-1 and search() would scan nothing even with snapshots on disk. Call
        this after constructing one against an existing project.
        """
        self.__reset_left_pointer()
        self.__reset_right_pointer()

    def __ensure_cursors(self, snap_shot_list) -> None:
        """Open the cursors onto stored history if they were never set.

        Cursors live in memory only, so an object built against an existing
        project starts at -1/-1. Left alone, __find_best_snapshot would evaluate
        `snap_shot_list[-1]` — Python negative indexing silently picking the
        LAST snapshot instead of signalling an empty range — do pointless work
        against it, and then return None regardless.
        """
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
        # Fetched once and passed down (Bug 4.24): querying separately here and
        # inside __find_best_snapshot allowed a concurrent insert to shift the
        # list between the two reads, applying an index to the wrong rows.
        snap_shot_list = self.__get_snap_shot()
        best_snap_shot_idx: int | None = self.__find_best_snapshot(
            query, snap_shot_list
        )
        return (
            snap_shot_list[best_snap_shot_idx]
            if best_snap_shot_idx is not None
            else None
        )

    def close(self) -> None:
        """Release the metadata connection, but only if this object opened it."""
        if self._owns_meta_repo:
            self.meta_repo.close()
