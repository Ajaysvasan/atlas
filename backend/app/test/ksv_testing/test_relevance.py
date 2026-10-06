"""The gate that decides what is on-subject enough to keep.

Without it, every chunk of every fetched page is embedded and accumulated — a
page about something else contributes thousands of chunks that answered nothing
and are then searched forever. These tests are about what gets thrown away.
"""

import numpy as np
import pytest

from knowledge_sufficiency.relevance import (
    MAX_KEPT_CHUNKS,
    TOPIC_FLOOR,
    filter_relevant,
)
from knowledge_sufficiency.target import AcquisitionTarget, build_target

DIMENSIONS = 128


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(DIMENSIONS).astype(np.float32)
    return v / np.linalg.norm(v)


def at_cosine(reference: np.ndarray, target: float) -> np.ndarray:
    orthogonal = unit(999)
    orthogonal = orthogonal - reference * float(orthogonal @ reference)
    orthogonal = orthogonal / np.linalg.norm(orthogonal)
    built = reference * target + orthogonal * np.sqrt(max(0.0, 1.0 - target**2))
    return (built / np.linalg.norm(built)).astype(np.float32)


TOPIC = unit(1)
QUERY = unit(2)
SUBTOPIC = unit(3)


def target(subtopics=()):
    return AcquisitionTarget(
        topic="databases", query="how does WAL work",
        topic_vector=TOPIC, query_vector=QUERY,
        subtopics=tuple(f"sub{i}" for i in range(len(subtopics))),
        subtopic_vectors=tuple(subtopics),
    )


class TestTheTopicGate:
    def test_on_topic_material_is_kept(self):
        result = filter_relevant(target(), [("a", TOPIC.copy())])
        assert [i for i, _ in result.kept] == ["a"]

    def test_off_topic_material_is_dropped(self):
        """The whole point: unrelated material never reaches the store."""
        result = filter_relevant(target(), [("a", at_cosine(TOPIC, 0.05))])
        assert result.kept == []
        assert result.dropped == 1

    def test_a_mixed_page_keeps_only_its_relevant_part(self):
        result = filter_relevant(target(), [
            ("on", TOPIC.copy()),
            ("off", at_cosine(TOPIC, 0.0)),
            ("on2", at_cosine(TOPIC, TOPIC_FLOOR + 0.1)),
        ])
        assert set(i for i, _ in result.kept) == {"on", "on2"}
        assert result.dropped == 1

    def test_exactly_on_the_floor_is_kept(self):
        result = filter_relevant(target(), [("a", at_cosine(TOPIC, TOPIC_FLOOR))])
        assert len(result.kept) == 1

    def test_the_floor_can_be_raised(self):
        candidates = [("a", at_cosine(TOPIC, 0.5))]
        assert len(filter_relevant(target(), candidates, topic_floor=0.4).kept) == 1
        assert len(filter_relevant(target(), candidates, topic_floor=0.9).kept) == 0

    def test_nothing_in_nothing_out(self):
        result = filter_relevant(target(), [])
        assert result.kept == [] and result.dropped == 0

    def test_relevance_is_judged_on_the_topic_not_the_query(self):
        """A chunk can be on-subject without answering the question yet."""
        on_topic_only = at_cosine(TOPIC, 0.9)
        result = filter_relevant(target(), [("a", on_topic_only)])
        assert len(result.kept) == 1


class TestSubtopicsNarrow:
    def test_a_chunk_must_match_the_topic_and_a_subtopic(self):
        on_both = (TOPIC + SUBTOPIC) / np.linalg.norm(TOPIC + SUBTOPIC)
        result = filter_relevant(target([SUBTOPIC]), [("a", on_both.astype(np.float32))])
        assert len(result.kept) == 1

    def test_topic_alone_is_not_enough_when_subtopics_are_given(self):
        """The narrowing case: on-subject, but not this specialisation."""
        result = filter_relevant(target([SUBTOPIC]), [("a", TOPIC.copy())])
        assert result.kept == []
        assert result.dropped == 1

    def test_a_subtopic_alone_is_not_enough_either(self):
        result = filter_relevant(target([SUBTOPIC]), [("a", SUBTOPIC.copy())])
        assert result.kept == []

    def test_any_one_of_several_subtopics_will_do(self):
        second = unit(4)
        on_second = (TOPIC + second) / np.linalg.norm(TOPIC + second)
        result = filter_relevant(
            target([SUBTOPIC, second]), [("a", on_second.astype(np.float32))]
        )
        assert len(result.kept) == 1

    def test_no_subtopics_means_the_topic_alone_decides(self):
        result = filter_relevant(target(), [("a", TOPIC.copy())])
        assert len(result.kept) == 1


class TestTheCap:
    def test_more_than_the_cap_is_trimmed(self):
        candidates = [(f"c{i}", TOPIC.copy()) for i in range(10)]
        result = filter_relevant(target(), candidates, max_kept=4)
        assert len(result.kept) == 4
        assert result.capped == 6

    def test_the_cap_keeps_what_is_closest_to_the_query(self):
        """When there is more on-subject material than room, the question decides."""
        near_query = (TOPIC + QUERY) / np.linalg.norm(TOPIC + QUERY)
        candidates = [(f"filler{i}", TOPIC.copy()) for i in range(10)]
        candidates.append(("useful", near_query.astype(np.float32)))

        result = filter_relevant(target(), candidates, max_kept=1)
        assert [i for i, _ in result.kept] == ["useful"]

    def test_under_the_cap_nothing_is_trimmed(self):
        result = filter_relevant(target(), [("a", TOPIC.copy())], max_kept=10)
        assert result.capped == 0

    def test_the_default_cap_is_finite(self):
        assert 0 < MAX_KEPT_CHUNKS < 10_000


class TestBuildingATarget:
    def test_it_embeds_every_part(self):
        built = build_target("databases", "how does WAL work", unit_for, ["WAL"])
        assert built.topic_vector is not None
        assert len(built.subtopic_vectors) == 1

    def test_blank_subtopics_are_dropped(self):
        built = build_target("databases", "q", unit_for, ["WAL", "", "   "])
        assert built.subtopics == ("WAL",)

    def test_a_target_needs_a_topic(self):
        """It is what bounds collection, so there is no sensible default."""
        with pytest.raises(ValueError, match="topic"):
            build_target("", "a query", unit_for)

    def test_a_target_needs_a_query(self):
        with pytest.raises(ValueError, match="query"):
            build_target("databases", "   ", unit_for)

    def test_the_search_terms_join_every_part(self):
        built = build_target("databases", "how does WAL work", unit_for, ["WAL"])
        assert "databases" in built.terms
        assert "WAL" in built.terms
        assert "how does WAL work" in built.terms


def unit_for(text: str) -> np.ndarray:
    return unit(abs(hash(text)) % 10_000)
