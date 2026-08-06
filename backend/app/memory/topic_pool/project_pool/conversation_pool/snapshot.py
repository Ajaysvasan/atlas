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

from memory.topic_pool.project_pool.conversation_pool.conversation_data_management.conversationVectorManager import (
    ConversationVectorManager,
)
from memory.topic_pool.project_pool.conversation_pool.conversation_data_management.conversationVectorMetaManager import (
    ConversationVectorMetaDataRepository,
)
from memory_pool_exceptions import (
    InvalidCursorException,
    MisMatchCount,
    NullPointerException,
)
from numpy import ndarray, uint32
from torch import cosine_similarity, tensor

from config import Config


class SnapShot:
    def __init__(
        self,
    ):
        self.__left_cursor: int = -1
        self.__right_cursor: int = -1

    def __get_snap_shot(self, project_id: str):

        conversationVectorMetaDataRepository = ConversationVectorMetaDataRepository(
            Config.FULL_CONVERSATION, Config.CONVERSATION_SNAPSHOT_DB, project_id
        )
        return (
            conversationVectorMetaDataRepository.get_cumulative_vector_meta_data_ids()
        )

    def __add_snap_shot(
        self,
        project_name: str,
        project_id: str,
        time_of_snapshot: str,
        len_of_the_summary: int,
        summary_vector_ids: List,
        summary_vectors: ndarray,
        chunk_ids: str,
        summary: str,
        cumulative_summary_vector_id: uint32,
        cumulative_summary_vector: ndarray,
    ) -> None:
        if len(chunk_ids) != len(summary_vector_ids):
            raise MisMatchCount("Mis matched arguments recieved")
        conversationVectorManager = ConversationVectorManager(project_name, project_id)
        conversationVectorMetaDataRepository = ConversationVectorMetaDataRepository(
            Config.FULL_CONVERSATION, Config.CONVERSATION_SNAPSHOT_DB, project_id
        )
        conversationVectorManager.batch_insert(summary_vector_ids, summary_vectors)
        conversationVectorManager.insert(
            cumulative_summary_vector_id, cumulative_summary_vector
        )
        summary_vector_tuple_list = []
        for i in range(len(chunk_ids)):
            summary_vector_tuple_list.append(
                (summary_vector_ids[i], chunk_ids[i], project_id)
            )
        conversationVectorMetaDataRepository.insert_cumulative_vector_meta_data(
            cumulative_summary_vector_id,
            summary,
            time_of_snapshot,
            project_id,
            len_of_the_summary,
        )
        conversationVectorMetaDataRepository.batch_insert_summary_vector_meta_data(
            summary_vector_tuple_list
        )

        map_list = []
        for i in range(len(summary_vector_ids)):
            map_list.append((cumulative_summary_vector_id, summary_vector_ids[i]))

        conversationVectorMetaDataRepository.batch_insert_map_table(map_list)

    def add(
        self,
        project_name: str,
        project_id: str,
        time_of_snapshot: str,
        len_of_the_summary: int,
        summary_vector_ids: List,
        summary_vectors: ndarray,
        chunk_ids: str,
        summary: str,
        cumulative_summary_vector_id: uint32,
        cumulative_summary_vector: ndarray,
        reset_right_pointer: bool = False,
        reset_left_pointer: bool = False,
    ):

        self.__add_snap_shot(
            project_name,
            project_id,
            time_of_snapshot,
            len_of_the_summary,
            summary_vector_ids,
            summary_vectors,
            chunk_ids,
            summary,
            cumulative_summary_vector_id,
            cumulative_summary_vector,
        )

        if self.__left_cursor == -1 and self.__right_cursor == -1:
            self.__left_cursor = self.__right_cursor = 0

        if reset_right_pointer:
            self.__reset_right_pointer(project_id)

        if reset_left_pointer:
            self.__reset_left_pointer(project_id)

    def advance(self, project_id: str) -> None:
        """Makes left cursor move"""
        snap_shot_list = self.__get_snap_shot(project_id)
        if self.__left_cursor + 1 < len(snap_shot_list):
            self.__left_cursor += 1
            return
        raise InvalidCursorException("left", self.__left_cursor + 1)

    def prev(self) -> None:
        """Makes the right curosr move"""
        if self.__right_cursor - 1 >= 0:
            self.__right_cursor -= 1
            return

        raise InvalidCursorException("right", self.__right_cursor - 1)

    def __reset_left_pointer(self, project_id: str) -> None:

        snap_shot_list = self.__get_snap_shot(project_id)
        if len(snap_shot_list) != 0:
            self.__left_cursor = 0
            return
        raise NullPointerException("No snap shots found")

    def __reset_right_pointer(self, project_id: str) -> None:
        snap_shot_list = self.__get_snap_shot(project_id)
        if len(snap_shot_list) != 0:
            self.__right_cursor = len(snap_shot_list) - 1
            return
        raise NullPointerException("No snap shots found")

    def __find_best_snapshot(
        self, query: ndarray, project_id: str, project_name: str
    ) -> int:
        snap_shot_list = self.__get_snap_shot(project_id)
        if len(snap_shot_list) == 0:
            raise NullPointerException("No snap shots found")

        best_snap_shot_idx = -1
        best_similarity = float("-inf")
        conversationVectorManager = ConversationVectorManager(project_name, project_id)
        try:
            while self.__left_cursor <= self.__right_cursor:
                if self.__left_cursor == self.__right_cursor:
                    snap = snap_shot_list[self.__left_cursor]
                    vec = tensor(conversationVectorManager.get_vector(snap))
                    sim = cosine_similarity(vec, tensor(query), dim=0)
                    if sim > best_similarity:
                        best_similarity = sim
                        best_snap_shot_idx = self.__left_cursor
                    break

                left_snap = snap_shot_list[self.__left_cursor]
                right_snap = snap_shot_list[self.__right_cursor]
                
                left_snap_vector_cumulative = tensor(
                    conversationVectorManager.get_vector(left_snap)
                )
                right_snap_vector_cumulative = tensor(
                    conversationVectorManager.get_vector(right_snap)
                )
                
                left_sim = cosine_similarity(left_snap_vector_cumulative, tensor(query), dim=0)
                right_sim = cosine_similarity(
                    right_snap_vector_cumulative, tensor(query), dim=0
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
            self.__reset_right_pointer(project_id)
            self.__reset_left_pointer(project_id)
        return best_snap_shot_idx

    def search(self, query: ndarray, project_id: str, project_name: str):
        snap_shot_list = self.__get_snap_shot(project_id)
        return snap_shot_list[
            self.__find_best_snapshot(query, project_id, project_name)
        ]
