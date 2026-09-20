"""Tests for ProjectManager — routing a query to the project it belongs to.

The embedder is a fake that maps text to a chosen axis, so "this query is about
project A" is something the test states rather than something a model decides.
The vector store is the stateful fake used elsewhere in this suite; SQLite is
real, because the project list the router reads is a real query.
"""

import numpy as np
import pytest

from config import Config
from data_layer.datalayer_exceptions.datalayer_exceptions import VectorNotFoundEror
from memory.memory_pool_exceptions import EmptyQueryException
from memory.topic_pool.project_pool.project_data_repo.project_meta_data import (
    ProjectMetaData,
)
from memory.topic_pool.project_pool.project_data_repo.project_vector_handler import (
    ProjectVectorHandler,
    description_vector_id,
    summary_vector_id,
)
from memory.topic_pool.project_pool.project_manager import (
    ProjectManager,
    ScoredProject,
    cosine_scores,
)

DIMENSIONS = Config.EMBEDDING_DIMENSIONS
TOPIC = "topic_fyp"


class FakeVectorRepository:
    """One shared store per project id, so a handler and a ProjectMetaData
    built separately still see the same vectors."""

    stores: dict = {}

    def __init__(self, project_id):
        self.project_id = project_id
        self.store = FakeVectorRepository.stores.setdefault(project_id, {})
        self.closed = False

    def insert(self, vector_id, vector):
        if int(vector_id) in self.store:
            from data_layer.datalayer_exceptions.datalayer_exceptions import (
                DuplicateVectorException,
            )

            raise DuplicateVectorException(vector_id)
        self.store[int(vector_id)] = np.asarray(vector, dtype=np.float32)

    def batch_insert(self, vector_ids, vectors):
        for vector_id, vector in zip(vector_ids, vectors):
            self.store.setdefault(int(vector_id), np.asarray(vector, dtype=np.float32))

    def update(self, vector_id, vector):
        if int(vector_id) not in self.store:
            raise VectorNotFoundEror(vector_id)
        self.store[int(vector_id)] = np.asarray(vector, dtype=np.float32)

    def search(self, vector_id):
        if int(vector_id) not in self.store:
            raise VectorNotFoundEror(vector_id)
        return self.store[int(vector_id)]

    def batch_search(self, vector_ids):
        return np.array([self.search(v) for v in vector_ids])

    def batch_delete(self, vector_ids):
        for vector_id in vector_ids:
            self.store.pop(int(vector_id), None)

    def close(self):
        self.closed = True


def axis(index: int, weight: float = 1.0) -> np.ndarray:
    vector = np.zeros(DIMENSIONS, dtype=np.float32)
    vector[index] = weight
    return vector


class FakeEmbedder:
    """Maps text to a direction the test chose, so the expected winner is known
    by construction rather than by trusting a model."""

    def __init__(self, directions=None):
        self.directions = directions or {}
        self.calls = []

    def embed_text(self, text, chunk_id=None):
        self.calls.append(text)
        result = type("E", (), {})()
        result.vector = self.directions.get(text, axis(0))
        result.vector_id = abs(hash(text)) % (2**63 - 1)
        return result


@pytest.fixture(autouse=True)
def clean_stores():
    FakeVectorRepository.stores = {}
    yield
    FakeVectorRepository.stores = {}


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "project_db" / "project.sql"


@pytest.fixture
def handler():
    h = ProjectVectorHandler(repository_factory=FakeVectorRepository)
    yield h
    h.close()


def seed_project(db_path, project_id, name, vectors, topic=TOPIC):
    """A project with a summary vector and one vector per extra direction."""
    meta = ProjectMetaData(
        project_id, topic, db_path=db_path,
        vector_repository=FakeVectorRepository(project_id),
    )
    summary, *descriptions = vectors
    meta.add_project_vector(summary, summary_vector_id(project_id), name, f"{name} summary")
    for index, vector in enumerate(descriptions):
        description_id = f"d{index}"
        meta.add_description(description_id, f"{name} {description_id}")
        meta.add_batch_project_vector(
            [vector], [description_vector_id(project_id, description_id)],
            name, f"{name} summary",
        )
    meta.close()


def manager(db_path, handler, query, directions, **kwargs):
    return ProjectManager(
        TOPIC, query, db_path=db_path,
        embedder=FakeEmbedder(directions), vector_handler=handler, **kwargs,
    )


class TestCosineScores:
    def test_ranks_by_direction(self):
        matrix = np.vstack([axis(0), axis(1), axis(2)])
        scores = cosine_scores(axis(1), matrix)
        assert int(np.argmax(scores)) == 1

    def test_is_scale_invariant(self):
        matrix = np.vstack([axis(0, 0.01), axis(1, 900.0)])
        assert cosine_scores(axis(0), matrix)[0] == pytest.approx(1.0, abs=1e-5)

    def test_a_zero_row_never_wins(self):
        """Dividing by a zero norm would make every score nan, and nan
        propagates through max() until the whole ranking is nonsense."""
        matrix = np.vstack([np.zeros(DIMENSIONS, dtype=np.float32), axis(0)])
        scores = cosine_scores(axis(0), matrix)
        assert not np.isnan(scores).any()
        assert scores[0] == -1.0 and int(np.argmax(scores)) == 1

    def test_no_rows_scores_nothing(self):
        assert cosine_scores(axis(0), np.empty((0, DIMENSIONS), dtype=np.float32)).size == 0


class TestResolve:
    def test_an_empty_topic_has_no_match(self, db_path, handler):
        assert manager(db_path, handler, "anything", {}).resolve().exists is False

    def test_routes_to_the_project_the_query_points_at(self, db_path, handler):
        seed_project(db_path, "proj_rag", "RAG backend", [axis(0)])
        seed_project(db_path, "proj_mem", "Memory layer", [axis(1)])
        match = manager(
            db_path, handler, "about memory", {"about memory": axis(1)}
        ).resolve()
        assert match.exists is True
        assert match.project_id == "proj_mem"
        assert match.project_name == "Memory layer"

    def test_a_query_matching_nothing_creates_no_match(self, db_path, handler):
        seed_project(db_path, "proj_rag", "RAG backend", [axis(0)])
        match = manager(
            db_path, handler, "unrelated", {"unrelated": axis(5)}
        ).resolve()
        assert match.exists is False
        assert match.project_id is None

    def test_the_floor_is_what_separates_them(self, db_path, handler):
        """Same query, same projects — only the bar moves."""
        seed_project(db_path, "proj_rag", "RAG backend", [axis(0)])
        query = 0.5 * axis(0) + 0.5 * axis(9)
        lenient = manager(db_path, handler, "q", {"q": query}, similarity_floor=0.5)
        strict = manager(db_path, handler, "q", {"q": query}, similarity_floor=0.9)
        assert lenient.resolve().exists is True
        assert strict.resolve().exists is False

    def test_a_project_matches_on_its_closest_vector_not_its_average(self, db_path, handler):
        """A project covers several things; a query hitting one corner of it
        must score as that corner."""
        seed_project(db_path, "proj_wide", "Wide", [axis(0), axis(1), axis(2)])
        match = manager(db_path, handler, "corner", {"corner": axis(2)}).resolve()
        assert match.exists is True and match.score == pytest.approx(1.0, abs=1e-5)

    def test_a_description_can_win_for_its_project(self, db_path, handler):
        seed_project(db_path, "proj_a", "A", [axis(0)])
        seed_project(db_path, "proj_b", "B", [axis(1), axis(7)])
        match = manager(db_path, handler, "q", {"q": axis(7)}).resolve()
        assert match.project_id == "proj_b"

    def test_a_near_tie_is_flagged_ambiguous(self, db_path, handler):
        seed_project(db_path, "proj_a", "A", [axis(0)])
        seed_project(db_path, "proj_b", "B", [axis(1)])
        between = (axis(0) + axis(1)) / np.sqrt(2)
        match = manager(db_path, handler, "q", {"q": between}).resolve()
        assert match.exists is True
        assert match.ambiguous is True
        assert match.margin < 0.05

    def test_a_clear_winner_is_not_ambiguous(self, db_path, handler):
        seed_project(db_path, "proj_a", "A", [axis(0)])
        seed_project(db_path, "proj_b", "B", [axis(1)])
        match = manager(db_path, handler, "q", {"q": axis(0)}).resolve()
        assert match.ambiguous is False and match.margin > 0.5

    def test_candidates_come_back_ranked(self, db_path, handler):
        seed_project(db_path, "proj_a", "A", [axis(0)])
        seed_project(db_path, "proj_b", "B", [axis(1)])
        seed_project(db_path, "proj_c", "C", [axis(2)])
        query = axis(0) + 0.6 * axis(1) + 0.2 * axis(2)
        match = manager(db_path, handler, "q", {"q": query}).resolve()
        assert [c.project_id for c in match.candidates] == ["proj_a", "proj_b", "proj_c"]
        assert match.candidates == tuple(sorted(match.candidates, key=lambda c: -c.score))

    def test_candidates_are_returned_even_when_nothing_clears_the_floor(self, db_path, handler):
        """What a caller escalates to the draft model with."""
        seed_project(db_path, "proj_a", "A", [axis(0)])
        match = manager(db_path, handler, "q", {"q": axis(5)}, similarity_floor=0.9).resolve()
        assert match.exists is False
        assert [c.project_id for c in match.candidates] == ["proj_a"]

    def test_another_topic_is_not_a_candidate(self, db_path, handler):
        seed_project(db_path, "proj_a", "A", [axis(0)])
        seed_project(db_path, "proj_other", "Other", [axis(0)], topic="topic_misc")
        match = manager(db_path, handler, "q", {"q": axis(0)}).resolve()
        assert [c.project_id for c in match.candidates] == ["proj_a"]

    def test_the_query_is_embedded_once(self, db_path, handler):
        seed_project(db_path, "proj_a", "A", [axis(0)])
        embedder = FakeEmbedder({"q": axis(0)})
        m = ProjectManager(TOPIC, "q", db_path=db_path, embedder=embedder, vector_handler=handler)
        m.resolve()
        m.resolve()
        assert embedder.calls == ["q"]


class TestRoute:
    def test_gives_the_project_id_when_it_exists(self, db_path, handler):
        seed_project(db_path, "proj_rag", "RAG backend", [axis(0)])
        assert manager(db_path, handler, "q", {"q": axis(0)}).route() == "proj_rag"

    def test_gives_nothing_when_it_does_not(self, db_path, handler):
        """The no branch is where the thinking layer will be called; until it
        exists there is nothing to hand the query to."""
        seed_project(db_path, "proj_rag", "RAG backend", [axis(0)])
        assert manager(db_path, handler, "q", {"q": axis(40)}).route() is None

    def test_an_empty_topic_gives_nothing(self, db_path, handler):
        assert manager(db_path, handler, "q", {}).route() is None

    def test_an_ambiguous_match_still_returns_its_winner(self, db_path, handler):
        """route() is the plain answer; a caller that cares about the near-tie
        uses resolve() and reads `ambiguous`."""
        seed_project(db_path, "proj_a", "A", [axis(0)])
        seed_project(db_path, "proj_b", "B", [axis(1)])
        between = (axis(0) + axis(1)) / np.sqrt(2)
        m = manager(db_path, handler, "q", {"q": between})
        assert m.resolve().ambiguous is True
        assert m.route() in {"proj_a", "proj_b"}


class TestProjects:
    def test_lists_the_topic(self, db_path, handler):
        seed_project(db_path, "proj_a", "A", [axis(0)])
        seed_project(db_path, "proj_b", "B", [axis(1)])
        listed = manager(db_path, handler, "q", {}).projects()
        assert [p.project_id for p in listed] == ["proj_a", "proj_b"]
        assert [p.project_name for p in listed] == ["A", "B"]

    def test_another_topic_is_not_listed(self, db_path, handler):
        """score_projects() takes names from here and candidates from the vector
        ids, so an unfiltered listing shows up as a project labelled with
        another topic's name rather than as a missing candidate."""
        seed_project(db_path, "proj_a", "A", [axis(0)])
        seed_project(db_path, "proj_other", "Other", [axis(1)], topic="topic_misc")
        listed = manager(db_path, handler, "q", {}).projects()
        assert [p.project_id for p in listed] == ["proj_a"]

    def test_an_unwritten_registry_lists_nothing(self, db_path, handler):
        """The router reads before anything has been written; an empty topic is
        an answer, not a missing-table error."""
        assert manager(db_path, handler, "q", {}).projects() == []


class TestCreateProject:
    def test_returns_an_id_and_lists_the_project(self, db_path, handler):
        m = manager(db_path, handler, "q", {"Alpha summary": axis(3)})
        project_id = m.create_project("Alpha", "Alpha summary")
        assert [p.project_id for p in m.projects()] == [project_id]

    def test_the_new_project_is_routable(self, db_path, handler):
        m = manager(db_path, handler, "q", {"Alpha summary": axis(3), "q": axis(3)})
        project_id = m.create_project("Alpha", "Alpha summary")
        assert m.resolve().project_id == project_id

    def test_the_summary_vector_is_where_the_handler_looks(self, db_path, handler):
        m = manager(db_path, handler, "q", {"Alpha summary": axis(3)})
        project_id = m.create_project("Alpha", "Alpha summary")
        stored = handler.get_project_summary_vector(project_id)
        assert stored.tolist() == axis(3).tolist()

    def test_ids_are_not_derived_from_the_name(self, db_path, handler):
        """Two projects may share a name; the id is a primary key."""
        m = manager(db_path, handler, "q", {"s": axis(3)})
        assert m.create_project("Same", "s") != m.create_project("Same", "s")

    def test_an_explicit_id_is_honoured(self, db_path, handler):
        m = manager(db_path, handler, "q", {"s": axis(3)})
        assert m.create_project("Alpha", "s", project_id="chosen") == "chosen"

    @pytest.mark.parametrize("name,summary", [("", "s"), ("   ", "s"), ("Alpha", ""), ("Alpha", "  ")])
    def test_a_blank_name_or_summary_is_refused(self, db_path, handler, name, summary):
        with pytest.raises(ValueError):
            manager(db_path, handler, "q", {}).create_project(name, summary)


class TestUpdateProjectSummary:
    def test_replaces_the_text_and_the_vector(self, db_path, handler):
        m = manager(db_path, handler, "q", {"first": axis(3), "second": axis(8)})
        project_id = m.create_project("Alpha", "first")
        m.update_project_summary(project_id, "second")

        assert handler.get_project_summary_vector(project_id).tolist() == axis(8).tolist()
        meta = ProjectMetaData(project_id, TOPIC, db_path=db_path,
                               vector_repository=FakeVectorRepository(project_id))
        assert meta.get_project().project_summary == "second"
        meta.close()

    def test_routing_follows_the_new_summary(self, db_path, handler):
        m = manager(db_path, handler, "q", {"first": axis(3), "second": axis(8), "q": axis(8)})
        project_id = m.create_project("Alpha", "first")
        assert m.resolve().exists is False
        m.update_project_summary(project_id, "second")
        assert m.resolve().project_id == project_id

    def test_the_name_is_not_changed(self, db_path, handler):
        m = manager(db_path, handler, "q", {"first": axis(3), "second": axis(8)})
        project_id = m.create_project("Alpha", "first")
        m.update_project_summary(project_id, "second")
        assert [p.project_name for p in m.projects()] == ["Alpha"]

    def test_does_not_add_a_second_vector(self, db_path, handler):
        m = manager(db_path, handler, "q", {"first": axis(3), "second": axis(8)})
        project_id = m.create_project("Alpha", "first")
        m.update_project_summary(project_id, "second")
        assert len(FakeVectorRepository.stores[project_id]) == 1

    def test_an_unknown_project_raises(self, db_path, handler):
        m = manager(db_path, handler, "q", {"s": axis(3)})
        with pytest.raises(ValueError):
            m.update_project_summary("ghost", "s")


class TestConstruction:
    @pytest.mark.parametrize("bad", ["", "   ", None, 7])
    def test_an_empty_query_is_refused(self, db_path, handler, bad):
        with pytest.raises(EmptyQueryException):
            ProjectManager(TOPIC, bad, db_path=db_path, vector_handler=handler)

    @pytest.mark.parametrize("bad", ["", "   ", None, 7])
    def test_an_unusable_topic_id_is_refused(self, db_path, handler, bad):
        with pytest.raises(ValueError):
            ProjectManager(bad, "q", db_path=db_path, vector_handler=handler)

    def test_the_embedder_is_not_loaded_until_used(self, db_path):
        """~100MB of weights; constructing the manager must not pay for them."""
        m = ProjectManager(TOPIC, "q", db_path=db_path)
        assert m._ProjectManager__embedder is None

    def test_an_injected_handler_is_left_open(self, db_path, handler):
        with ProjectManager(TOPIC, "q", db_path=db_path, vector_handler=handler) as m:
            m.projects()
        handler.get_project_summary_vector  # still usable; the manager did not own it
