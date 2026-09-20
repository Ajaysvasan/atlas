"""A project's vectors in pgvector: its summary and one per description.

See README.md in this directory.
"""

import hashlib
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from numpy import float32, ndarray
from numpy.typing import NDArray

from config import Config, get_logger
from data_layer.vector_db_manager.repository.vectorRepository import VectorRepository
from memory.memory_pool_exceptions import InvalidVectorDimension, MisMatchCount

logger = get_logger(__name__)

SUMMARY = "summary"
DESCRIPTION = "description"


def _derive(kind: str, project_id: str, source_id: str = "") -> int:
    """The id a vector of this kind, for this source, is stored under."""
    payload = f"{kind}\x00{project_id}\x00{source_id}".encode("utf-8")
    packed = int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="little")
    return packed & Config.VECTOR_ID_MASK


def summary_vector_id(project_id: str) -> int:
    """The id of this project's summary vector."""
    return _derive(SUMMARY, project_id)


def description_vector_id(project_id: str, description_id: str) -> int:
    """The id of the vector for one of this project's descriptions."""
    return _derive(DESCRIPTION, project_id, description_id)


class ProjectVectorHandler:
    def __init__(
        self, repository_factory: Callable[[str], VectorRepository] = VectorRepository
    ) -> None:
        self.__repository_factory = repository_factory
        self.__repositories: Dict[str, VectorRepository] = {}

    def __repository(self, project_id: str) -> VectorRepository:
        """One repository per project, held open for this object's lifetime."""
        repository = self.__repositories.get(project_id)
        if repository is None:
            logger.debug("Opening the vector store for project %s", project_id)
            repository = self.__repository_factory(project_id)
            self.__repositories[project_id] = repository
        return repository

    def repository_for(self, project_id: str) -> VectorRepository:
        """The connection this handler holds for a project."""
        return self.__repository(self.__validate_id(project_id, "project_id"))

    @staticmethod
    def __validate_id(value, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value

    @staticmethod
    def __as_vector(vector) -> NDArray[float32]:
        checked = np.asarray(vector, dtype=float32)
        if checked.ndim != 1:
            raise InvalidVectorDimension(checked.shape, Config.EMBEDDING_DIMENSIONS)
        if checked.shape[0] != Config.EMBEDDING_DIMENSIONS:
            raise InvalidVectorDimension(
                checked.shape[0], Config.EMBEDDING_DIMENSIONS
            )
        return checked

    def __add_vector(self, project_id: str, vector_id: int, vector) -> None:
        # On its own line: in `self.__repository(id).insert(..., self.__as_vector(v))`
        # Python resolves the call target first, so a bad vector would have
        # opened a connection before being rejected.
        checked = self.__as_vector(vector)
        self.__repository(project_id).insert(vector_id, checked)

    def __update_vector(self, project_id: str, vector_id: int, vector) -> str:
        checked = self.__as_vector(vector)
        self.__repository(project_id).update(vector_id, checked)
        return project_id

    def __delete_vector(self, project_id: str, vector_id: int) -> None:
        self.__repository(project_id).delete(vector_id)

    def __get_vector(self, project_id: str, vector_id: int) -> NDArray[float32]:
        return self.__repository(project_id).search(vector_id)

    # -- the project's summary ----------------------------------------------

    def add_project_summary_vector(
        self, project_id: str, project_summary_vector: List[float32] | ndarray
    ) -> None:
        """Store this project's summary vector. Raises DuplicateVectorException if it has one."""
        checked = self.__validate_id(project_id, "project_id")
        self.__add_vector(checked, summary_vector_id(checked), project_summary_vector)

    def update_project_summary_vector(
        self, project_id: str, project_summary_vector: List[float32] | ndarray
    ) -> str:
        """Replace this project's summary vector in place. Returns the project id."""
        checked = self.__validate_id(project_id, "project_id")
        return self.__update_vector(
            checked, summary_vector_id(checked), project_summary_vector
        )

    def delete_project_summary_vector(self, project_id: str) -> None:
        """Remove this project's summary vector. Idempotent."""
        checked = self.__validate_id(project_id, "project_id")
        self.__delete_vector(checked, summary_vector_id(checked))

    def get_project_summary_vector(self, project_id: str) -> ndarray:
        """This project's summary vector. Raises VectorNotFoundEror if absent."""
        checked = self.__validate_id(project_id, "project_id")
        return self.__get_vector(checked, summary_vector_id(checked))

    # -- the project's descriptions -----------------------------------------

    def add_project_description_vector(
        self,
        project_id: str,
        description_id: str,
        description_vector: List[float32] | ndarray,
    ) -> None:
        """Store the vector for one of this project's descriptions."""
        checked = self.__validate_id(project_id, "project_id")
        source = self.__validate_id(description_id, "description_id")
        self.__add_vector(
            checked, description_vector_id(checked, source), description_vector
        )

    def add_project_description_vectors(
        self,
        project_id: str,
        descriptions: Sequence[Tuple[str, List[float32] | ndarray]],
    ) -> None:
        """Store several (description_id, vector) pairs. Idempotent; an empty sequence is a no-op."""
        checked = self.__validate_id(project_id, "project_id")
        rows = [
            (
                description_vector_id(
                    checked, self.__validate_id(description_id, "description_id")
                ),
                self.__as_vector(vector),
            )
            for description_id, vector in descriptions
        ]
        if not rows:
            return
        identifiers = [vector_id for vector_id, _ in rows]
        if len(set(identifiers)) != len(identifiers):
            raise MisMatchCount(
                "The batch repeats a description id. Each id may appear at most once."
            )
        self.__repository(checked).batch_insert(
            identifiers, np.array([vector for _, vector in rows], dtype=float32)
        )

    def update_project_description_vector(
        self,
        project_id: str,
        description_id: str,
        description_vector: List[float32] | ndarray,
    ) -> str:
        """Replace one description's vector in place. Returns the project id."""
        checked = self.__validate_id(project_id, "project_id")
        source = self.__validate_id(description_id, "description_id")
        return self.__update_vector(
            checked, description_vector_id(checked, source), description_vector
        )

    def delete_project_description_vector(
        self, project_id: str, description_id: str
    ) -> None:
        """Remove one description's vector. Idempotent."""
        checked = self.__validate_id(project_id, "project_id")
        source = self.__validate_id(description_id, "description_id")
        self.__delete_vector(checked, description_vector_id(checked, source))

    def get_project_description_vector(
        self, project_id: str, description_id: str
    ) -> ndarray:
        """One description's vector. Raises VectorNotFoundEror if absent."""
        checked = self.__validate_id(project_id, "project_id")
        source = self.__validate_id(description_id, "description_id")
        return self.__get_vector(checked, description_vector_id(checked, source))

    # -- reading a whole project's vectors ----------------------------------

    def get_project_vectors(
        self, project_id: str, vector_ids: Sequence[int]
    ) -> NDArray[float32]:
        """The embeddings behind these ids, in the order given."""
        checked = self.__validate_id(project_id, "project_id")
        if not vector_ids:
            return np.empty((0, Config.EMBEDDING_DIMENSIONS), dtype=float32)
        return self.__repository(checked).batch_search(list(vector_ids))

    def close(self) -> None:
        """Release every connection this object opened."""
        for repository in self.__repositories.values():
            repository.close()
        self.__repositories.clear()

    def __enter__(self) -> "ProjectVectorHandler":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
