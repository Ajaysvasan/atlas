import hashlib
from typing import List, Union
import numpy as np

from data_layer.vector_db_manager.repository.vectorRepository import VectorRepository


class ConversationVectorManager:
    def __init__(self, project_name: str, project_id: str):
        self.project_name = project_name
        self.project_id = project_id
        # Note: VectorRepository requires environment variables (DBNAME, USER, PASSWORD, HOST, PORT)
        # to connect to PostgreSQL.
        self.repository = VectorRepository()

    def generate_vector_id(self, chunk_id: str) -> int:
        """
        Generate a unique unsigned integer vector_id based on project_name, project_id, and chunk_id.
        Since vectorRepository expects uint32 for IDs, we take the first 4 bytes of MD5.
        """
        unique_string = f"{self.project_name}_{self.project_id}_{chunk_id}"
        hash_bytes = hashlib.md5(unique_string.encode('utf-8')).digest()
        # Using 4 bytes for uint32
        return int.from_bytes(hash_bytes[:4], byteorder='little', signed=False)

    def insert(self, chunk_id: str, vector: np.ndarray) -> int:
        """
        Inserts a single vector by generating the vector_id automatically.
        """
        vector_id = self.generate_vector_id(chunk_id)
        self.repository.insert(vector_id, vector)
        return vector_id

    def batch_insert(self, chunk_ids: List[str], vectors: np.ndarray) -> List[int]:
        """
        Inserts a batch of vectors by generating vector_ids automatically.
        """
        vector_ids = [self.generate_vector_id(cid) for cid in chunk_ids]
        self.repository.batch_insert(vector_ids, vectors)
        return vector_ids

    def get_vector(self, vector_id: int) -> np.ndarray:
        """
        Retrieves a single vector from the repository by vector_id.
        """
        return self.repository.search(vector_id)

    def get_vectors(self, vector_ids: List[int]) -> np.ndarray:
        """
        Retrieves a batch of vectors from the repository by vector_ids.
        """
        return self.repository.batch_search(vector_ids)
