"""

I need to maintain the snapshot that is taken recently
and then I also need to store the ids for each projects in a file and load them
A python dict would do I belive
{
    project_id : [snapshot_one_id , snapshot_two_id , ... , snapshot_n_id]
    each snapshot is a summary vector
}

"""

from typing import List

from memory_pool_exceptions import InvalidCursorException, NullPointerException
from numpy import uint32
from snapShotNodes import SnapShotNode
from torch import cosine_similarity, tensor


class SnapShot:
    def __init__(
        self,
    ):
        self.__left_cursor: int = -1
        self.__right_cursor: int = -1
        self.__snap_shot_list: List[SnapShotNode] = []

    def __add_snap_shot(
        self,
        snapshot_id,
        time_of_snapshot: str,
        size_of_the_summary: int,
        len_of_the_summary: int,
        summary_vector_ids: List,
        conversation_id: str,
        cumulative_summary_vector_id: uint32,
        project_id: str,
    ) -> None:
        snap_shot = SnapShotNode(
            snapshot_id,
            time_of_snapshot,
            size_of_the_summary,
            len_of_the_summary,
            summary_vector_ids,
            conversation_id,
            cumulative_summary_vector_id,
            project_id,
        )
        self.__snap_shot_list.append(snap_shot)

    def add(
        self,
        snapshot_id: str,
        time_of_snapshot: str,
        size_of_the_summary: int,
        len_of_the_summary: int,
        summary_vector_ids: List,
        conversation_id: str,
        project_id: str,
        cumulative_summary_vector_id: uint32,
        reset_right_pointer: bool = True,
        reset_left_pointer: bool = True,
    ):
        self.__add_snap_shot(
            snapshot_id,
            time_of_snapshot,
            size_of_the_summary,
            len_of_the_summary,
            summary_vector_ids,
            conversation_id,
            cumulative_summary_vector_id,
            project_id,
        )
        if self.__left_cursor == -1 and self.__right_cursor == -1:
            self.__left_cursor = self.__right_cursor = 0

        if reset_right_pointer:
            self.__reset_right_pointer()

        if reset_left_pointer:
            self.__reset_left_pointer()

    def advance(self) -> None:
        """Makes left cursor move"""
        if self.__left_cursor + 1 < len(self.__snap_shot_list):
            self.__left_cursor += 1
            return
        raise InvalidCursorException("left", self.__left_cursor + 1)

    def prev(self) -> None:
        """Makes the right curosr move"""
        if self.__right_cursor - 1 >= 0:
            self.__right_cursor -= 1
            return

        raise InvalidCursorException("right", self.__right_cursor - 1)

    def __reset_left_pointer(self) -> None:
        if len(self.__snap_shot_list) != 0:
            self.__left_cursor = 0
            return
        raise NullPointerException("No snap shots found")

    def __reset_right_pointer(self) -> None:
        if len(self.__snap_shot_list) != 0:
            self.__right_cursor = len(self.__snap_shot_list) - 1
            return
        raise NullPointerException("No snap shots found")

    def __find_best_snapshot(self, query: List) -> int:
        if len(self.__snap_shot_list) == 0:
            raise NullPointerException("No snap shots found")

        best_snap_shot_idx = -1
        best_similarity = float("-inf")
        try:
            while self.__left_cursor <= self.__right_cursor:
                left_snap = self.__snap_shot_list[self.__left_cursor]
                right_snap = self.__snap_shot_list[self.__right_cursor]
                # some logics
                right_snap_vector_cumulative = tensor([])
                left_snap_vector_cumulative = tensor([])
                left_sim = cosine_similarity(left_snap_vector_cumulative, tensor(query))
                right_sim = cosine_similarity(
                    right_snap_vector_cumulative, tensor(query)
                )

                if left_sim > best_similarity:
                    best_similarity = left_sim
                    best_snap_shot_idx = self.__left_cursor

                if right_sim > best_similarity:
                    best_similarity = right_sim
                    best_snap_shot_idx = self.__right_cursor
                self.__left_cursor += 1
                self.__right_cursor -= 1

        finally:
            self.__reset_right_pointer()
            self.__reset_left_pointer()
        return best_snap_shot_idx

    def search(self, query: List):
        return self.__snap_shot_list[self.__find_best_snapshot(query)]
