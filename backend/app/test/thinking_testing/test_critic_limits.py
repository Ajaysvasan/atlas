"""A review reply cannot run away: its lists are bounded, and one cut off at the token limit still keeps its verdict."""

import logging

import pytest

from test.thinking_testing.fakes import EVIDENCE, PROMPTS, request
from thinking_layer.critic import JobCritic, PlanCritic
from thinking_layer.models import JobKind, JobSpec, PlanVerdict, Verdict
from thinking_layer.replies import (
    CRITIQUE_ITEMS,
    JOB_CRITIQUE_SCHEMA,
    UnusableReply,
    plan_critique_schema,
    read_job_critique,
    read_plan_critique,
)
from thinking_layer.settings import ThinkingSettings

JOB = JobSpec("j1", JobKind.STEP, "Explain MVCC snapshots", "A short explanation")
RESULT = "PostgreSQL gives each transaction a snapshot of committed rows [G1]."


def _walk(schema):
    yield schema
    for value in schema.values():
        if isinstance(value, dict):
            yield from _walk(value)


class FixedReply:
    """A model that answers every call with one reply, and records what it was asked."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = []

    def infer(self, system, user, *, schema=None, max_tokens=512, purpose="draft", temperature=0.0):
        self.calls.append({"schema": schema, "max_tokens": max_tokens, "purpose": purpose})
        return self.reply


# ---------------------------------------------------------------------------
# The schemas
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_job_review_lists_are_bounded(self):
        for key in ("reasons", "missing"):
            assert JOB_CRITIQUE_SCHEMA["properties"][key]["maxItems"] == CRITIQUE_ITEMS == 3

    def test_plan_review_lists_are_bounded_and_affected_jobs_by_the_jobs_shown(self):
        schema = plan_critique_schema(2)
        assert schema["properties"]["affected_jobs"]["maxItems"] == 2
        assert schema["properties"]["missing"]["maxItems"] == schema["properties"]["reasons"]["maxItems"] == 3
        assert plan_critique_schema(0)["properties"]["affected_jobs"]["maxItems"] == 1

    @pytest.mark.parametrize("schema", [JOB_CRITIQUE_SCHEMA, plan_critique_schema(3)])
    def test_no_string_length_bound_which_breaks_words_apart(self, schema):
        assert not any("maxLength" in node or "minLength" in node for node in _walk(schema))

    @pytest.mark.parametrize("schema", [JOB_CRITIQUE_SCHEMA, plan_critique_schema(3)])
    def test_the_verdict_comes_first_so_a_cut_off_reply_still_has_it(self, schema):
        assert next(iter(schema["properties"])) == "verdict"

    @pytest.mark.parametrize("schema", [JOB_CRITIQUE_SCHEMA, plan_critique_schema(3)])
    def test_llama_cpp_compiles_the_schema(self, schema):
        llama_grammar = pytest.importorskip("llama_cpp.llama_grammar")
        grammar = llama_grammar.json_schema_to_gbnf(__import__("json").dumps(schema))
        assert "verdict" in grammar

    def test_the_critic_token_limit_leaves_room_for_full_lists(self):
        assert ThinkingSettings().critic_max_tokens == 384


# ---------------------------------------------------------------------------
# Reading a reply cut off at the token limit
# ---------------------------------------------------------------------------

class TestCutOffReplies:
    def test_a_complete_reply_is_not_cut_off(self):
        critique = read_job_critique('{"verdict": "revise", "reasons": ["a"], "missing": ["b"]}')
        assert critique == (Verdict.REVISE, ("a",), ("b",), True, False)

    def test_a_reply_cut_inside_a_list_keeps_the_verdict_and_the_finished_items(self):
        reply = '{"verdict": "revise", "reasons": ["too vague", "no example", "and then it kept going and go'
        critique = read_job_critique(reply)
        assert critique.verdict is Verdict.REVISE and critique.cut_off and critique.reviewed
        assert critique.reasons == ("too vague", "no example") and critique.missing == ()

    def test_a_reply_cut_in_the_second_list_keeps_the_first_whole(self):
        reply = '{"verdict": "fail", "reasons": ["off topic"], "missing": ["the interval", "a ci'
        critique = read_job_critique(reply)
        assert critique == (Verdict.FAIL, ("off topic",), ("the interval",), True, True)

    @pytest.mark.parametrize("reply", ['{', '{"verd', '{"verdict": "rev', '{"verdict":'])
    def test_a_reply_cut_before_its_verdict_is_unusable(self, reply):
        with pytest.raises(UnusableReply, match="cut off at its token limit before giving a verdict"):
            read_job_critique(reply)

    def test_a_cut_off_reply_with_an_unknown_verdict_is_unusable(self):
        with pytest.raises(UnusableReply, match="verdict 'maybe'"):
            read_job_critique('{"verdict": "maybe", "reasons": ["x", "y')

    def test_text_that_is_not_json_at_all_is_still_reported_as_such(self):
        with pytest.raises(UnusableReply, match="not JSON"):
            read_job_critique("I think this result is fine.")

    def test_a_cut_off_plan_review_keeps_only_known_jobs(self):
        reply = '{"verdict": "revise", "affected_jobs": ["j2", "j9", "j1"], "missing": ["the cost", "the ri'
        critique = read_plan_critique(reply, ["j1", "j2"])
        assert critique.verdict is PlanVerdict.REVISE and critique.cut_off
        assert critique.affected_jobs == ("j2", "j1") and critique.missing == ("the cost",)

    def test_escaped_quotes_inside_finished_items_survive(self):
        critique = read_job_critique('{"verdict": "accept", "reasons": ["says \\"snapshot\\" right", "cut')
        assert critique.reasons == ('says "snapshot" right',)


# ---------------------------------------------------------------------------
# The critics
# ---------------------------------------------------------------------------

class TestCritics:
    def test_the_job_critic_asks_for_the_bounded_schema_at_the_new_limit(self):
        model = FixedReply('{"verdict": "accept", "reasons": [], "missing": []}')
        JobCritic(model, PROMPTS, ThinkingSettings()).review(request(), JOB, RESULT, EVIDENCE)
        assert model.calls[0]["schema"] is JOB_CRITIQUE_SCHEMA
        assert model.calls[0]["max_tokens"] == 384 and model.calls[0]["purpose"] == "critic"

    def test_a_cut_off_job_review_is_used_and_logged_as_hitting_the_limit(self, caplog):
        model = FixedReply('{"verdict": "revise", "reasons": ["no snapshot detail"], "missing": ["vacuu')
        with caplog.at_level(logging.WARNING, logger="thinking_layer.critic"):
            critique = JobCritic(model, PROMPTS, ThinkingSettings()).review(request(), JOB, RESULT, EVIDENCE)
        assert critique.verdict is Verdict.REVISE and critique.reviewed and critique.cut_off
        assert critique.reasons == ("no snapshot detail",)
        message = caplog.records[-1].getMessage()
        assert "hit its 384-token limit" in message and "using its verdict (revise)" in message
        assert "not JSON" not in message

    def test_a_job_review_cut_before_its_verdict_is_accepted_unreviewed_and_says_why(self, caplog):
        with caplog.at_level(logging.WARNING, logger="thinking_layer.critic"):
            critique = JobCritic(FixedReply('{"verd'), PROMPTS, ThinkingSettings()).review(
                request(), JOB, RESULT, EVIDENCE)
        assert critique.verdict is Verdict.ACCEPT and not critique.reviewed
        message = caplog.records[-1].getMessage()
        assert "cut off at its token limit before giving a verdict (limit 384 tokens)" in message

    def test_the_plan_critic_bounds_affected_jobs_by_the_results_it_shows(self):
        model = FixedReply('{"verdict": "accept", "affected_jobs": [], "missing": [], "reasons": []}')
        PlanCritic(model, PROMPTS, ThinkingSettings()).review(
            request(), [("j1", RESULT), ("j2", RESULT)], EVIDENCE)
        assert model.calls[0]["schema"] == plan_critique_schema(2)
        assert model.calls[0]["max_tokens"] == 384

    def test_a_cut_off_plan_review_is_used(self, caplog):
        model = FixedReply('{"verdict": "limit", "affected_jobs": [], "missing": ["no source has it", "an')
        with caplog.at_level(logging.WARNING, logger="thinking_layer.critic"):
            critique = PlanCritic(model, PROMPTS, ThinkingSettings()).review(
                request(), [("j1", RESULT)], EVIDENCE)
        assert critique.verdict is PlanVerdict.LIMIT and critique.reviewed and critique.cut_off
        assert critique.missing == ("no source has it",)
        assert "plan critic reply hit its 384-token limit" in caplog.records[-1].getMessage()
