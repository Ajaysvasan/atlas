import os
from pathlib import Path
from typing import Tuple

import numpy as np
from memory_pool_exceptions import InvalidVectorDimension

from config import Config


class ConversationVectorManager:
    def __init__(
        self,
        vector_dimension=Config.VECTOR_DIMENSIONS,
        vector_dtype=Config.VECTOR_DTYPE,
        summary_path=Config.SUMMARY_PATH,
        cummulative_vector_path=Config.CUMMULATIVE_VECTOR_PATH,
    ) -> None:
        self.summary_path = summary_path
        self.cummulative_vector_path = cummulative_vector_path
        self.vector_dimension = vector_dimension

        self.vector_dtype = np.dtype(vector_dtype)

        Path(self.summary_path).mkdir(parents=True, exist_ok=True)
        Path(self.cummulative_vector_path).mkdir(parents=True, exist_ok=True)

    def __validate_vectors(self, vector: np.ndarray) -> bool:
        if vector.shape[1] != self.vector_dimension:
            return False
        return True

    def _get_file_path(self, base_path: str, project_id: str) -> str:
        """Helper to ensure paths are joined correctly and use the .bin extension."""
        return os.path.join(base_path, f"{project_id}.bin")

    def _read_vectors(
        self, base_path: str, project_id: str, start_idx: int, end_idx: int
    ) -> np.ndarray:
        """Private helper to handle the boilerplate of memory-mapping a disk read."""
        file_path = self._get_file_path(base_path, project_id)

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"The vector file doesn't exist for the given project ID : {project_id}.\nPath: {file_path}"
            )

        file_size = os.path.getsize(file_path)
        bytes_per_vector = self.vector_dimension * self.vector_dtype.itemsize
        total_vectors = file_size // bytes_per_vector

        if start_idx < 0 or end_idx > total_vectors or start_idx >= end_idx:
            raise IndexError(
                f"Invalid indices {start_idx}:{end_idx}. Total vectors in file: {total_vectors}"
            )

        mmap_array = np.memmap(
            file_path,
            dtype=self.vector_dtype,
            mode="r",
            shape=(total_vectors, self.vector_dimension),
        )

        return mmap_array[start_idx:end_idx].copy()

    def add_cummulative_summary_vector(
        self, project_id: str, vector: np.ndarray
    ) -> Tuple[int, int]:
        """This function is used to add cummulative conversation summary vector"""
        pass

    def add_summary_vectors(
        self, project_id: str, vectors: np.ndarray
    ) -> Tuple[int, int]:
        """This function is used to add the summary vectors for that project"""
        vectors = np.atleast_2d(vectors).astype(self.vector_dtype)

        if not self.__validate_vectors(vectors):
            raise InvalidVectorDimension(vectors.shape[1], self.vector_dimension)

        file_path = self._get_file_path(self.cummulative_vector_path, project_id)
        num_new_vectors = vectors.shape[0]

        # Determine the logical starting index
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            bytes_per_vector = self.vector_dimension * self.vector_dtype.itemsize
            start_idx = file_size // bytes_per_vector
        else:
            start_idx = 0

        end_idx = start_idx + num_new_vectors

        with open(file_path, "ab") as f:
            vectors.tofile(f)

        return start_idx, end_idx

    def get_cummulative_summary_vector(
        self, start_idx: int, end_idx: int, project_id: str
    ) -> np.ndarray:
        return self._read_vectors(self.summary_path, project_id, start_idx, end_idx)

    def get_summary_vector(
        self, start_idx: int, end_idx: int, project_id: str
    ) -> np.ndarray:
        return self._read_vectors(
            self.cummulative_vector_path, project_id, start_idx, end_idx
        )
