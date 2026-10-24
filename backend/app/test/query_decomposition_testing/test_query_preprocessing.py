import pytest

from query_decomposition.preprocessing import clean, estimate_tokens, require_within
from query_decomposition.query_decomposition_exceptions import EmptyQuery, QueryTooLarge


class TestClean:
    @pytest.mark.parametrize("query", ["", "   ", "\n\t\r\n", "\x00\x07", None, 42])
    def test_nothing_to_decompose_is_refused(self, query):
        with pytest.raises(EmptyQuery):
            clean(query)

    def test_ends_are_trimmed(self):
        assert clean("  why vectors?  \n") == "why vectors?"

    def test_line_endings_become_newlines(self):
        assert clean("a\r\nb\rc") == "a\nb\nc"

    def test_indentation_inside_the_query_survives(self):
        query = "fix this:\ndef f():\n    return 1"
        assert clean(query) == query

    def test_trailing_spaces_on_each_line_go(self):
        assert clean("a   \nb\t\nc") == "a\nb\nc"

    def test_blank_runs_collapse_to_one_blank_line(self):
        assert clean("a\n\n\n\n\nb") == "a\n\nb"

    def test_control_characters_go_but_tabs_stay(self):
        assert clean("a\x00b\tc") == "ab\tc"

    def test_unicode_is_composed(self):
        assert clean("café") == "café"


class TestSize:
    def test_estimate_rounds_up(self):
        assert estimate_tokens("abcde") == 2
        assert estimate_tokens("abcd") == 1

    def test_within_the_limit_returns_the_estimate(self):
        assert require_within("a" * 40, 10) == 10

    def test_past_the_limit_is_refused(self):
        with pytest.raises(QueryTooLarge) as raised:
            require_within("a" * 41, 10)
        assert raised.value.estimated_tokens == 11
        assert raised.value.limit == 10
