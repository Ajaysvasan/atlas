"""The project summary vector, addressed by project id.

One vector per project: "the summary of project X". Every method here takes a
project_id and no vector id, so the id has to be derived from the project —
summary_vector_id() is that mapping, and it is what makes add/update/get/delete
agree on which row they are talking about without a lookup table.

This is the project-level counterpart to ConversationVectorManager, which wraps
the same VectorRepository for conversation snapshots. The difference is the
addressing: a conversation holds many vectors and the caller supplies their ids,
while a project holds exactly one summary vector and its id is its identity.

Vectors are validated here, before a connection is touched, so a malformed one
is rejected by the memory layer's InvalidVectorDimension rather than surfacing
later as the data layer's exception of the same name.
"""

import hashlib
from typing import Callable, Dict, List

import numpy as np
from numpy import float32, ndarray
from numpy.typing import NDArray

from config import Config, get_logger
from data_layer.vector_db_manager.repository.vectorRepository import VectorRepository
from memory.memory_pool_exceptions import InvalidVectorDimension

logger = get_logger(__name__)


def summary_vector_id(project_id: str) -> int:
    """The id under which a project's summary vector is stored.

    Derived from the project id alone: it is an identity, not content, so the
    same project always resolves to the same row and a re-embedded summary
    replaces its predecessor instead of accumulating beside it. Masked into the
    non-negative signed 64-bit range both storage backends accept — see
    Config.VECTOR_ID_MASK.
    """
    digest = hashlib.sha256(project_id.encode("utf-8")).digest()
    packed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return packed & Config.VECTOR_ID_MASK


class ProjectVectorHandler:
    def __init__(
        self, repository_factory: Callable[[str], VectorRepository] = VectorRepository
    ) -> None:
        self.__repository_factory = repository_factory
        self.__repositories: Dict[str, VectorRepository] = {}

    def __repository(self, project_id: str) -> VectorRepository:
        """One repository per project, held open for the handler's lifetime.

        VectorRepository scopes itself to a project and opens its PostgreSQL
        connection in the constructor, so building one per call would open and
        drop a connection on every add, read or delete.
        """
        repository = self.__repositories.get(project_id)
        if repository is None:
            logger.debug("Opening the vector store for project %s", project_id)
            repository = self.__repository_factory(project_id)
            self.__repositories[project_id] = repository
        return repository

    @staticmethod
    def __validate_project_id(project_id) -> str:
        if not isinstance(project_id, str) or not project_id.strip():
            raise ValueError("project_id must be a non-empty string")
        return project_id

    @staticmethod
    def __as_vector(project_summary_vector) -> NDArray[float32]:
        vector = np.asarray(project_summary_vector, dtype=float32)
        if vector.ndim != 1:
            raise InvalidVectorDimension(vector.shape, Config.EMBEDDING_DIMENSIONS)
        if vector.shape[0] != Config.EMBEDDING_DIMENSIONS:
            raise InvalidVectorDimension(
                vector.shape[0], Config.EMBEDDING_DIMENSIONS
            )
        return vector

    # Adds new project summary vector
    def __add_project_summary_vector(
        self, project_id: str, project_summary_vector: List[float32] | ndarray
    ) -> None:
        checked_id = self.__validate_project_id(project_id)
        vector = self.__as_vector(project_summary_vector)
        self.__repository(checked_id).insert(summary_vector_id(checked_id), vector)

    # This method returns the project id after updating the project summary vector
    def __update_project_summary_vector(
        self, project_id: str, project_summary_vector: List[float32] | ndarray
    ) -> str:
        checked_id = self.__validate_project_id(project_id)
        vector = self.__as_vector(project_summary_vector)
        self.__repository(checked_id).update(summary_vector_id(checked_id), vector)
        return checked_id

    def __delete_project_summary_vector(self, project_id: str) -> None:
        checked_id = self.__validate_project_id(project_id)
        self.__repository(checked_id).delete(summary_vector_id(checked_id))

    def __get_project_summary_vector(self, project_id: str) -> NDArray[float32]:
        checked_id = self.__validate_project_id(project_id)
        return self.__repository(checked_id).search(summary_vector_id(checked_id))

    def add_project_summary_vector(
        self, project_id: str, project_summary_vector: List[float32] | ndarray
    ) -> None:
        """Store this project's summary vector.

        Raises DuplicateVectorException if the project already has one — the
        slot is taken, and replacing it is update_project_summary_vector's job
        rather than something an "add" should do silently.
        """
        self.__add_project_summary_vector(project_id, project_summary_vector)

    def update_project_summary_vector(
        self, project_id: str, project_summary_vector: List[float32] | ndarray
    ) -> str:
        """Replace this project's summary vector in place. Returns the project id.

        A single UPDATE rather than delete-then-insert: the pair is two commits,
        and a failure between them loses the vector entirely. Raises
        VectorNotFoundEror if the project has no summary vector yet.
        """
        return self.__update_project_summary_vector(project_id, project_summary_vector)

    def delete_project_summary_vector(self, project_id: str) -> None:
        """Remove this project's summary vector.

        Idempotent: a project that has none is not an error, because the caller
        that wants it gone does not care whether it was there.
        """
        self.__delete_project_summary_vector(project_id)

    def get_project_summary_vector(self, project_id: str) -> ndarray:
        """This project's summary vector. Raises VectorNotFoundEror if absent."""
        return self.__get_project_summary_vector(project_id)

    def close(self) -> None:
        """Release every connection this handler opened.

        One per project touched, so without this they are only reclaimed
        whenever the garbage collector happens to run and they scale with the
        number of projects a process has seen.
        """
        for repository in self.__repositories.values():
            repository.close()
        self.__repositories.clear()

    def __enter__(self) -> "ProjectVectorHandler":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
