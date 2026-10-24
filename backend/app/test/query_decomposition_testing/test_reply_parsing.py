import json

import pytest

from query_decomposition.models import Mode, TopicPath
from query_decomposition.parsing import (
    MAX_SUB_TOPICS,
    MAX_TOPICS,
    read_analysis,
    read_mode,
    read_sub_queries,
)
from query_decomposition.query_decomposition_exceptions import UnusableReply


class TestMode:
    @pytest.mark.parametrize("value, mode", [
        ("development", Mode.DEVELOPMENT),
        ("doubt", Mode.DOUBT),
        (" Doubt ", Mode.DOUBT),
    ])
    def test_reads_the_mode(self, value, mode):
        assert read_mode(json.dumps({"mode": value})) is mode

    def test_a_fenced_reply_is_read(self):
        assert read_mode('```json\n{"mode": "doubt"}\n```') is Mode.DOUBT

    @pytest.mark.parametrize("reply", [
        "", "doubt", "[]", '{"mode": "M1"}', '{"mode": 1}', "{}", None,
    ])
    def test_anything_else_is_unusable(self, reply):
        with pytest.raises(UnusableReply):
            read_mode(reply)


def analysis(topics, sufficient=True):
    return json.dumps({"topics": topics, "sufficient": sufficient})


class TestAnalysis:
    def test_reads_topics_in_order_and_sufficiency(self):
        result = read_analysis(analysis([
            {"topic": "Physics", "sub_topics": ["Collision detection"]},
            {"topic": "C programming", "sub_topics": []},
        ], sufficient=False), [])

        assert result.topics == (
            TopicPath("Physics", ("Collision detection",)),
            TopicPath("C programming", ()),
        )
        assert result.sufficient is False

    def test_an_existing_topic_takes_its_stored_name(self):
        result = read_analysis(
            analysis([{"topic": "machine LEARNING", "sub_topics": []}]),
            ["Machine Learning", "Physics"],
        )
        assert result.topics[0].topic == "Machine Learning"

    def test_a_topic_named_twice_is_kept_once(self):
        result = read_analysis(analysis([
            {"topic": "Physics", "sub_topics": ["a"]},
            {"topic": "physics", "sub_topics": ["b"]},
        ]), [])
        assert result.topics == (TopicPath("Physics", ("a",)),)

    def test_malformed_entries_are_dropped_not_fatal(self):
        result = read_analysis(analysis([
            "Physics", {"topic": ""}, {"sub_topics": ["x"]}, {"topic": 3},
            {"topic": "Maths", "sub_topics": ["Algebra", "", 7, "algebra"]},
        ]), [])
        assert result.topics == (TopicPath("Maths", ("Algebra",)),)

    def test_counts_are_capped(self):
        result = read_analysis(analysis([
            {"topic": f"T{i}", "sub_topics": [f"s{j}" for j in range(9)]}
            for i in range(6)
        ]), [])
        assert len(result.topics) == MAX_TOPICS
        assert all(len(path.sub_topics) == MAX_SUB_TOPICS for path in result.topics)

    def test_whitespace_inside_a_name_is_collapsed(self):
        result = read_analysis(analysis([{"topic": "  Web \n development ",
                                          "sub_topics": []}]), [])
        assert result.topics[0].topic == "Web development"

    @pytest.mark.parametrize("reply", [
        '{"topics": [], "sufficient": "yes"}',
        '{"topics": []}',
        '{"topics": {}, "sufficient": true}',
        '{"sufficient": true}',
        "not json",
    ])
    def test_without_a_boolean_and_a_list_it_is_unusable(self, reply):
        with pytest.raises(UnusableReply):
            read_analysis(reply, [])


class TestSubQueries:
    def test_reads_them_in_order(self):
        reply = json.dumps({"sub_queries": ["maths", "physics", "code"]})
        assert read_sub_queries(reply, 4) == ("maths", "physics", "code")

    def test_blanks_duplicates_and_non_strings_go(self):
        reply = json.dumps({"sub_queries": ["a", " ", "A", 1, None, "b"]})
        assert read_sub_queries(reply, 4) == ("a", "b")

    def test_capped_at_the_limit(self):
        reply = json.dumps({"sub_queries": [str(i) for i in range(9)]})
        assert read_sub_queries(reply, 4) == ("0", "1", "2", "3")

    @pytest.mark.parametrize("reply", ['{"sub_queries": "a"}', "{}", "[]"])
    def test_without_a_list_it_is_unusable(self, reply):
        with pytest.raises(UnusableReply):
            read_sub_queries(reply, 4)
