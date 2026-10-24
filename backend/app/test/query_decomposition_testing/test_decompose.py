import json

import pytest

from memory.memory_pool_exceptions import InvalidIdentifier
from memory.topic_pool.topic_manager import TopicManager
from query_decomposition.decompose import Decompose
from query_decomposition.models import Mode, TopicPath
from query_decomposition.prompts import SystemPrompts
from query_decomposition.query_decomposition_exceptions import (
    EmptyQuery,
    ExchangeNotFound,
    QueryTooLarge,
    ResponseAlreadyRecorded,
)

PROMPTS = SystemPrompts("CONDENSE", "COMPLEXITY", "ANALYSIS", "EXPANSION")
STAGES = {prompt: name for name, prompt in PROMPTS._asdict().items()}

DEVELOPMENT = json.dumps({"mode": "development"})
DOUBT = json.dumps({"mode": "doubt"})


def analysed(sufficient, *topics):
    return json.dumps({
        "topics": [{"topic": t, "sub_topics": [f"{t} basics"]} for t in topics],
        "sufficient": sufficient,
    })


def expanded(*sub_queries):
    return json.dumps({"sub_queries": list(sub_queries)})


class FakeModel:
    """A draft model whose replies are set per stage, recording what it was asked."""

    def __init__(self, **replies) -> None:
        self.replies = replies
        self.calls = []
        self.loads = 0
        self.unloads = 0
        self.resident = False

    def __call__(self):
        return self

    def __enter__(self):
        self.loads += 1
        self.resident = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self.unloads += 1
        self.resident = False

    def infer(self, system, user, schema=None, max_tokens=512):
        assert self.resident, "inference outside the with block"
        stage = STAGES[system]
        self.calls.append((stage, user, schema))
        reply = self.replies[stage]
        if isinstance(reply, Exception):
            raise reply
        return reply

    @property
    def stages(self):
        return [stage for stage, _, _ in self.calls]


@pytest.fixture
def decompose_with(conversation_rows):
    def build(model, **options):
        return Decompose(model=model, prompts=PROMPTS, exchanges=conversation_rows, **options)

    return build


class TestTheFlow:
    def test_a_doubt_query_is_never_expanded(self, decompose_with):
        model = FakeModel(complexity=DOUBT, analysis=analysed(False, "Programming"))
        result = decompose_with(model).decompose("Why vectors here?", "c1", "u1")

        assert result.mode is Mode.DOUBT
        assert result.sub_queries == ()
        assert not result.expanded
        assert model.stages == ["complexity", "analysis"]

    def test_a_sufficient_development_query_is_not_expanded(self, decompose_with):
        model = FakeModel(complexity=DEVELOPMENT, analysis=analysed(True, "Physics"))
        result = decompose_with(model).decompose("Build X in C with Y", "c1", "u1")

        assert result.mode is Mode.DEVELOPMENT
        assert result.sufficient is True
        assert result.sub_queries == ()
        assert model.stages == ["complexity", "analysis"]

    def test_an_insufficient_development_query_is_expanded(self, decompose_with):
        model = FakeModel(
            complexity=DEVELOPMENT,
            analysis=analysed(False, "Physics", "C programming"),
            expansion=expanded("maths", "physics", "data structures", "C code"),
        )
        result = decompose_with(model).decompose("Help me build a physics engine in C",
                                                 "c1", "u1")

        assert result.sub_queries == ("maths", "physics", "data structures", "C code")
        assert result.expanded
        assert result.primary_topic == "Physics"
        assert model.stages == ["complexity", "analysis", "expansion"]

    def test_expansion_is_told_the_topics(self, decompose_with):
        model = FakeModel(complexity=DEVELOPMENT, analysis=analysed(False, "Physics"),
                          expansion=expanded("a"))
        decompose_with(model).decompose("build it", "c1", "u1")

        _, user, _ = model.calls[-1]
        assert user.startswith("Topics: Physics (Physics basics)")
        assert user.endswith("Request:\nbuild it")

    def test_sub_queries_are_capped(self, decompose_with):
        model = FakeModel(complexity=DEVELOPMENT, analysis=analysed(False, "X"),
                          expansion=expanded("a", "b", "c"))
        result = decompose_with(model, max_sub_queries=2).decompose("q", "c1", "u1")

        assert result.sub_queries == ("a", "b")
        assert model.calls[-1][2]["properties"]["sub_queries"]["maxItems"] == 2

    def test_every_prompt_runs_on_one_load(self, decompose_with):
        model = FakeModel(complexity=DEVELOPMENT, analysis=analysed(False, "X"),
                          expansion=expanded("a"))
        decompose_with(model).decompose("q", "c1", "u1")

        assert (model.loads, model.unloads) == (1, 1)

    def test_json_stages_are_constrained_by_a_schema(self, decompose_with):
        model = FakeModel(complexity=DEVELOPMENT, analysis=analysed(False, "X"),
                          expansion=expanded("a"))
        decompose_with(model).decompose("q", "c1", "u1")

        assert all(schema is not None for _, _, schema in model.calls)


class TestTopics:
    def test_analysis_is_told_the_existing_topics(self, decompose_with):
        for name in ("Physics", "Web development"):
            with TopicManager(name) as topics:
                topics.create_new_topic()
        model = FakeModel(complexity=DOUBT, analysis=analysed(True, "physics"))
        result = decompose_with(model).decompose("q", "c1", "u1")

        _, user, _ = model.calls[1]
        assert "Physics" in user and "Web development" in user
        assert result.primary_topic == "Physics"

    def test_with_no_topics_yet_the_model_is_told_so(self, decompose_with):
        model = FakeModel(complexity=DOUBT, analysis=analysed(True, "Physics"))
        decompose_with(model).decompose("q", "c1", "u1")

        assert model.calls[1][1].startswith("Existing topics: none yet")


class TestLongQueries:
    def test_a_query_past_the_soft_limit_is_condensed_first(self, decompose_with):
        model = FakeModel(condense="short", complexity=DOUBT, analysis=analysed(True, "X"))
        result = decompose_with(model, soft_limit_tokens=10, hard_limit_tokens=100) \
            .decompose("word " * 20, "c1", "u1")

        assert model.stages == ["condense", "complexity", "analysis"]
        assert result.working_query == "short"
        assert result.query == ("word " * 20).strip()
        assert model.calls[1][1] == "short"

    def test_condensing_is_plain_text(self, decompose_with):
        model = FakeModel(condense="short", complexity=DOUBT, analysis=analysed(True, "X"))
        decompose_with(model, soft_limit_tokens=10, hard_limit_tokens=100) \
            .decompose("word " * 20, "c1", "u1")

        assert model.calls[0][2] is None

    @pytest.mark.parametrize("condensed", ["", "word " * 30])
    def test_a_condensing_that_does_not_shorten_keeps_the_query(self, decompose_with,
                                                                condensed):
        model = FakeModel(condense=condensed, complexity=DOUBT,
                          analysis=analysed(True, "X"))
        result = decompose_with(model, soft_limit_tokens=10, hard_limit_tokens=100) \
            .decompose("word " * 20, "c1", "u1")

        assert result.working_query == result.query

    def test_a_short_query_is_not_condensed(self, decompose_with):
        model = FakeModel(complexity=DOUBT, analysis=analysed(True, "X"))
        decompose_with(model, soft_limit_tokens=10, hard_limit_tokens=100) \
            .decompose("short", "c1", "u1")

        assert "condense" not in model.stages

    def test_past_the_hard_limit_nothing_is_recorded_or_loaded(self, decompose_with,
                                                               conversation_rows):
        model = FakeModel()
        with pytest.raises(QueryTooLarge):
            decompose_with(model, soft_limit_tokens=5, hard_limit_tokens=10) \
                .decompose("a" * 41, "c1", "u1")

        assert conversation_rows.rows == {}
        assert model.loads == 0


class TestFallbacks:
    def test_an_unusable_mode_is_treated_as_doubt(self, decompose_with):
        model = FakeModel(complexity="nonsense", analysis=analysed(False, "X"))
        result = decompose_with(model).decompose("q", "c1", "u1")

        assert result.mode is Mode.DOUBT
        assert result.sub_queries == ()

    def test_an_unusable_analysis_leaves_no_topics_and_no_expansion(self, decompose_with):
        model = FakeModel(complexity=DEVELOPMENT, analysis="{}")
        result = decompose_with(model).decompose("q", "c1", "u1")

        assert result.topics == ()
        assert result.primary_topic is None
        assert result.sufficient is True
        assert model.stages == ["complexity", "analysis"]

    def test_an_unusable_expansion_leaves_the_query_unexpanded(self, decompose_with):
        model = FakeModel(complexity=DEVELOPMENT, analysis=analysed(False, "X"),
                          expansion="[]")
        result = decompose_with(model).decompose("q", "c1", "u1")

        assert result.sub_queries == ()
        assert result.topics == (TopicPath("X", ("X basics",)),)

    def test_a_failing_model_is_still_unloaded(self, decompose_with, conversation_rows):
        model = FakeModel(complexity=RuntimeError("llama.cpp failed"))
        with pytest.raises(RuntimeError):
            decompose_with(model).decompose("q", "c1", "u1")

        assert model.unloads == 1
        assert len(conversation_rows.rows) == 1


class UnloadableModel(FakeModel):
    def __enter__(self):
        raise FileNotFoundError("no model file")


class TestTheConversationTable:
    def test_a_model_that_cannot_load_does_not_lose_the_query(self, decompose_with,
                                                              conversation_rows):
        with pytest.raises(FileNotFoundError):
            decompose_with(UnloadableModel()).decompose("why?", "c1", "u1")

        (row,) = conversation_rows.rows.values()
        assert row.query == "why?"

    def test_the_cleaned_query_is_recorded_awaiting_a_response(self, decompose_with,
                                                               conversation_rows):
        model = FakeModel(complexity=DOUBT, analysis=analysed(True, "X"))
        result = decompose_with(model).decompose("  why?  ", "c1", "u1")

        row = conversation_rows.get(result.exchange_id)
        assert (row.conversation_id, row.user_id, row.query) == ("c1", "u1", "why?")
        assert row.query_created_at is not None
        assert (row.system_response, row.system_response_time) == (None, None)

    def test_the_stored_query_is_the_original_not_the_condensed_one(
        self, decompose_with, conversation_rows
    ):
        model = FakeModel(condense="short", complexity=DOUBT, analysis=analysed(True, "X"))
        result = decompose_with(model, soft_limit_tokens=10, hard_limit_tokens=100) \
            .decompose("word " * 20, "c1", "u1")

        assert conversation_rows.get(result.exchange_id).query == result.query

    def test_the_response_is_recorded_later_with_its_own_time(self, decompose_with,
                                                              conversation_rows):
        decompose = decompose_with(FakeModel(complexity=DOUBT, analysis=analysed(True, "X")))
        result = decompose.decompose("why?", "c1", "u1")
        row = decompose.record_response(result.exchange_id, "because")

        assert row.system_response == "because"
        assert row.system_response_time >= row.query_created_at

    def test_a_response_is_recorded_once(self, decompose_with):
        decompose = decompose_with(FakeModel(complexity=DOUBT, analysis=analysed(True, "X")))
        result = decompose.decompose("why?", "c1", "u1")
        decompose.record_response(result.exchange_id, "because")

        with pytest.raises(ResponseAlreadyRecorded):
            decompose.record_response(result.exchange_id, "again")

    def test_a_response_to_an_unknown_exchange_is_refused(self, decompose_with):
        with pytest.raises(ExchangeNotFound):
            decompose_with(FakeModel()).record_response("missing", "because")

    @pytest.mark.parametrize("response", ["", "   ", None])
    def test_an_empty_response_is_refused(self, decompose_with, response):
        with pytest.raises(ValueError):
            decompose_with(FakeModel()).record_response("any", response)

    def test_without_a_table_it_opens_the_default_one(self, conversation_rows):
        model = FakeModel(complexity=DOUBT, analysis=analysed(True, "X"))
        with Decompose(model=model, prompts=PROMPTS) as decompose:
            result = decompose.decompose("why?", "c1", "u1")

        assert result.exchange_id in conversation_rows.rows
        assert conversation_rows.closed

    def test_a_table_it_was_given_is_left_open(self, decompose_with, conversation_rows):
        decompose_with(FakeModel()).close()
        assert not conversation_rows.closed


class TestValidation:
    @pytest.mark.parametrize("query", ["", "  \n ", None])
    def test_an_empty_query_records_nothing(self, decompose_with, conversation_rows, query):
        with pytest.raises(EmptyQuery):
            decompose_with(FakeModel()).decompose(query, "c1", "u1")
        assert conversation_rows.rows == {}

    @pytest.mark.parametrize("ids", [("", "u1"), ("c1", " "), (None, "u1")])
    def test_ids_are_required(self, decompose_with, ids):
        with pytest.raises(InvalidIdentifier):
            decompose_with(FakeModel()).decompose("q", *ids)

    @pytest.mark.parametrize("soft, hard", [(0, 10), (11, 10), (-1, 5)])
    def test_limits_must_be_ordered_and_positive(self, decompose_with, soft, hard):
        with pytest.raises(ValueError):
            decompose_with(FakeModel(), soft_limit_tokens=soft, hard_limit_tokens=hard)

    def test_at_least_one_sub_query(self, decompose_with):
        with pytest.raises(ValueError):
            decompose_with(FakeModel(), max_sub_queries=0)
