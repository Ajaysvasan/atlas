import hashlib
from typing import List, Union

import numpy as np
from numpy import uint32

from data_layer.vector_db_manager.repository.vectorRepository import VectorRepository


class ConversationVectorManager:
    def __init__(self, project_name: str, project_id: str):
        self.project_name = project_name
        self.project_id = project_id
        self.repository = VectorRepository()

    def insert(self, vector_id: uint32, vector: np.ndarray) -> np.uint32:
        """
        Inserts a single vector by generating the vector_id automatically.
        """
        self.repository.insert(vector_id, vector)
        return vector_id

    def batch_insert(
        self, vector_ids: List[np.uint32], vectors: np.ndarray
    ) -> List[np.uint32]:
        """
        Inserts a batch of vectors by generating vector_ids automatically.
        """
        self.repository.batch_insert(vector_ids, vectors)
        return vector_ids

    def get_vector(self, vector_id: uint32) -> np.ndarray:
        """
        Retrieves a single vector from the repository by vector_id.
        """
        return self.repository.search(vector_id)

    def get_vectors(self, vector_ids: List[uint32]) -> np.ndarray:
        """
        Retrieves a batch of vectors from the repository by vector_ids.
        """
        return self.repository.batch_search(vector_ids)
