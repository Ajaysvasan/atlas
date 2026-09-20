"""Decides whether a query belongs to a project that already exists.

See README.md in this directory for the architecture and the reasoning.
"""

import uuid
from pathlib import Path
from typing import List, NamedTuple, Sequence, Tuple

import numpy as np
from numpy import float32, ndarray
from numpy.typing import NDArray

from config import Config, get_logger
from data_layer.ingestion.nodes.nodes import EmbeddedChunk
from memory.memory_pool_exceptions import EmptyQueryException

from .project_data_repo.project_meta_data import (
    ProjectMetaData,
    TopicProject,
    list_topic_projects,
    list_topic_vector_ids,
)
from .project_data_repo.project_vector_handler import (
    ProjectVectorHandler,
    summary_vector_id,
)

logger = get_logger(__name__)

SIMILARITY_FLOOR = 0.35
AMBIGUITY_MARGIN = 0.05


class ScoredProject(NamedTuple):
    project_id: str
    project_name: str
    score: float


class ProjectMatch(NamedTuple):
    """The answer to "Exists?", and how confident it is."""

    exists: bool
    project_id: str | None
    project_name: str | None
    score: float
    margin: float
    ambiguous: bool
    candidates: Tuple[ScoredProject, ...]


def cosine_scores(query: ndarray, matrix: NDArray[float32]) -> NDArray[float32]:
    """Cosine of the query against each row. A zero row scores -1."""
    if matrix.size == 0:
        return np.empty(0, dtype=float32)
    scale = np.linalg.norm(matrix, axis=1) * float(np.linalg.norm(query))
    # Guarded twice over: a zero row has no direction, and dividing by its norm
    # would make every score nan, which then propagates through max().
    scores = matrix @ query / np.where(scale == 0, 1.0, scale)
    return np.where(scale == 0, -1.0, scores).astype(float32)


class ProjectManager:
    def __init__(
        self,
        topic_id: str,
        query: str,
        db_path: str | Path | None = None,
        embedder=None,
        vector_handler: ProjectVectorHandler | None = None,
        similarity_floor: float = SIMILARITY_FLOOR,
        ambiguity_margin: float = AMBIGUITY_MARGIN,
    ):
        if not isinstance(topic_id, str) or not topic_id.strip():
            raise ValueError("topic_id must be a non-empty string")
        if not isinstance(query, str) or not query.strip():
            raise EmptyQueryException()

        self.topic_id = topic_id
        self.query = query
        self.db_path = db_path
        self.similarity_floor = similarity_floor
        self.ambiguity_margin = ambiguity_margin

        self.__embedder = embedder
        self.__vector_handler = vector_handler
        self.__owns_vector_handler = vector_handler is None
        self.__query_vector: ndarray | None = None

    @property
    def embedder(self):
        """Built on first use, and once."""
        if self.__embedder is None:
            from data_layer.ingestion.embedding.EmbeddingManager import (
                EmbeddingManager,
            )

            logger.debug("Loading the embedder for topic %s", self.topic_id)
            self.__embedder = EmbeddingManager()
        return self.__embedder

    @property
    def vector_handler(self) -> ProjectVectorHandler:
        if self.__vector_handler is None:
            self.__vector_handler = ProjectVectorHandler()
        return self.__vector_handler

    def __query_embedder(self) -> EmbeddedChunk:
        return self.embedder.embed_text(self.query)

    def query_vector(self) -> ndarray:
        """The query as a vector, embedded once per manager."""
        if self.__query_vector is None:
            self.__query_vector = self.__query_embedder().vector
        return self.__query_vector

    def __meta(self, project_id: str) -> ProjectMetaData:
        """A registry handle for one project, sharing this manager's connection."""
        return ProjectMetaData(
            project_id,
            self.topic_id,
            db_path=self.db_path,
            vector_repository=self.vector_handler.repository_for(project_id),
        )

    def projects(self) -> List[TopicProject]:
        """The project list this layer decides against."""
        return list_topic_projects(self.topic_id, self.db_path)

    def __vector_ids_by_project(self) -> List[Tuple[str, List[int]]]:
        grouped: dict[str, List[int]] = {}
        for project_id, vector_id in list_topic_vector_ids(self.topic_id, self.db_path):
            grouped.setdefault(project_id, []).append(vector_id)
        return list(grouped.items())

    def score_projects(self) -> List[ScoredProject]:
        """Every project in the topic, best-matching vector first."""
        names = {project.project_id: project.project_name for project in self.projects()}
        query = self.query_vector()

        scored: List[ScoredProject] = []
        for project_id, vector_ids in self.__vector_ids_by_project():
            vectors = self.vector_handler.get_project_vectors(project_id, vector_ids)
            scores = cosine_scores(query, vectors)
            if scores.size == 0:
                continue
            scored.append(
                ScoredProject(
                    project_id, names.get(project_id, project_id), float(scores.max())
                )
            )
        scored.sort(key=lambda project: project.score, reverse=True)
        return scored

    def resolve(self) -> ProjectMatch:
        """Does this query belong to a project that already exists?"""
        candidates = tuple(self.score_projects())
        if not candidates:
            logger.info("Topic %s holds no project vectors yet", self.topic_id)
            return ProjectMatch(False, None, None, 0.0, 0.0, False, ())

        best = candidates[0]
        runner_up = candidates[1].score if len(candidates) > 1 else float("-inf")
        margin = best.score - runner_up if len(candidates) > 1 else best.score

        if best.score < self.similarity_floor:
            logger.info(
                "No project in topic %s matched (best %s at %.3f, floor %.3f)",
                self.topic_id,
                best.project_id,
                best.score,
                self.similarity_floor,
            )
            return ProjectMatch(False, None, None, best.score, margin, False, candidates)

        ambiguous = margin < self.ambiguity_margin
        if ambiguous:
            logger.info(
                "Project match in topic %s is ambiguous: %s at %.3f, next at %.3f",
                self.topic_id,
                best.project_id,
                best.score,
                candidates[1].score,
            )
        return ProjectMatch(
            True, best.project_id, best.project_name, best.score, margin, ambiguous,
            candidates,
        )

    def route(self) -> str | None:
        """The project this query belongs to, or None if there is no such project."""
        match = self.resolve()
        if match.exists:
            return match.project_id
        # PENDING: the no branch hands off to the thinking layer, which names
        # and summarises the new project before create_project() stores it.
        # That layer does not exist yet.
        pass

    def create_project(
        self,
        project_name: str,
        project_summary: str,
        project_id: str | None = None,
    ) -> str:
        """Add a project to this topic. Returns its id."""
        if not isinstance(project_name, str) or not project_name.strip():
            raise ValueError("project_name must be a non-empty string")
        if not isinstance(project_summary, str) or not project_summary.strip():
            raise ValueError("project_summary must be a non-empty string")

        new_id = project_id or uuid.uuid4().hex
        vector = self.embedder.embed_text(project_summary).vector
        meta = self.__meta(new_id)
        try:
            meta.add_project_vector(
                vector, summary_vector_id(new_id), project_name, project_summary
            )
        finally:
            meta.close()
        logger.info(
            "Created project %s (%r) under topic %s", new_id, project_name, self.topic_id
        )
        return new_id

    def update_project_summary(self, project_id: str, project_summary: str) -> None:
        """Replace a project's summary and the vector that represents it."""
        if not isinstance(project_summary, str) or not project_summary.strip():
            raise ValueError("project_summary must be a non-empty string")

        meta = self.__meta(project_id)
        try:
            if meta.get_project() is None:
                raise ValueError(f"No project {project_id!r} in topic {self.topic_id!r}")
            vector = self.embedder.embed_text(project_summary).vector
            meta.update_summary_vector(summary_vector_id(project_id), vector)
            meta.set_project_summary(project_summary)
        finally:
            meta.close()
        logger.info("Updated the summary of project %s", project_id)

    def close(self) -> None:
        if self.__owns_vector_handler and self.__vector_handler is not None:
            self.__vector_handler.close()
            self.__vector_handler = None

    def __enter__(self) -> "ProjectManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
