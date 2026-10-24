import pytest

from query_decomposition.prompts import SystemPrompts, load_prompts
from query_decomposition.query_decomposition_exceptions import SystemPromptMissing


def write(tmp_path, text):
    path = tmp_path / "prompts.yml"
    path.write_text(text, encoding="utf-8")
    return path


FULL = """
query_decomposition:
  condense: c
  complexity: m
  analysis: a
  expansion: e
"""


class TestTheShippedFile:
    def test_every_prompt_is_present_and_non_empty(self):
        prompts = load_prompts()
        assert all(isinstance(value, str) and value for value in prompts)

    def test_the_json_prompts_name_the_keys_the_parser_reads(self):
        prompts = load_prompts()
        assert '"development"' in prompts.complexity and '"doubt"' in prompts.complexity
        assert '"topics"' in prompts.analysis and '"sufficient"' in prompts.analysis
        assert '"sub_queries"' in prompts.expansion


class TestLoading:
    def test_reads_each_prompt_stripped(self, tmp_path):
        assert load_prompts(write(tmp_path, FULL)) == SystemPrompts("c", "m", "a", "e")

    def test_a_missing_file_names_the_file(self, tmp_path):
        with pytest.raises(SystemPromptMissing) as raised:
            load_prompts(tmp_path / "absent.yml")
        assert raised.value.key == "query_decomposition"

    @pytest.mark.parametrize("text", ["", "- a list\n", "other_layer:\n  x: y\n"])
    def test_a_file_without_the_section_is_refused(self, tmp_path, text):
        with pytest.raises(SystemPromptMissing):
            load_prompts(write(tmp_path, text))

    @pytest.mark.parametrize("broken", ["  expansion: ''\n", "  expansion: 3\n", ""])
    def test_a_missing_or_empty_prompt_is_named(self, tmp_path, broken):
        text = FULL.replace("  expansion: e\n", broken)
        with pytest.raises(SystemPromptMissing) as raised:
            load_prompts(write(tmp_path, text))
        assert raised.value.key == "query_decomposition.expansion"
