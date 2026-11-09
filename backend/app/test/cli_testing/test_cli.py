"""The development CLI: what counts as a command, and the session it drives."""

import pytest

from cli import render
from cli.cli_interface import CliSession
from cli.commands import Command, CommandError, parse
from memory import conversation_registry as registry_module
from observability.health import ComponentHealth, Health
from observability.telemetry import RequestRecord, Telemetry


class TestParsing:
    @pytest.mark.parametrize("line, command", [
        ("switch", Command("switch")),
        ("  SWITCH  ", Command("switch")),
        ("switch 2", Command("switch", ("2",))),
        ("switch 4f0a9c", Command("switch", ("4f0a9c",))),
        ("/switch", Command("switch")),
        ("status", Command("status")),
        ("status --time", Command("status", ("time",))),
        ("/status --retries", Command("status", ("retries",))),
        ("exit", Command("exit")),
        ("quit", Command("exit")),
        ("new", Command("new")),
        ("help", Command("help")),
        ("", Command("empty")),
    ])
    def test_commands(self, line, command):
        assert parse(line) == command

    @pytest.mark.parametrize("line", [
        "switch to PostgreSQL 16 from MySQL",
        "start a new design for the cache",
        "status of replication in Cassandra?",
        "How do I switch isolation levels?",
        "new features in PostgreSQL 17",
        "exit codes of pg_ctl",
        "status --verbose",
    ])
    def test_ordinary_queries_that_use_command_words_are_queries(self, line):
        assert parse(line) is None

    @pytest.mark.parametrize("line", ["/frobnicate", "/status --verbose", "/switch a b", "/"])
    def test_a_slash_line_must_be_a_valid_command(self, line):
        with pytest.raises(CommandError):
            parse(line)


class Pipeline:
    def __init__(self, registry) -> None:
        self.registry = registry
        self.asked = []
        self.telemetry = Telemetry()

    def new_conversation(self, user_id, query):
        return self.registry.create(user_id, f"About {query[:20]}")

    def ask(self, conversation_id, user_id, query):
        from pipeline.atlas import AtlasResponse

        self.asked.append((conversation_id, user_id, query))
        record = RequestRecord("req1", conversation_id, "now", status="completed", total_ms=12.0)
        record.stages.update(decomposition=5.0, generation=7.0)
        record.path, record.jobs, record.job_retries, record.plan_revisions = "planned", 3, 2, 1
        self.telemetry.add(record)
        return AtlasResponse("req1", conversation_id, f"answer to {query}", "completed", True, (), None, record)


class Sessions:
    def checkpoint_status(self, conversation_id, user_id):
        return "none"


class Components:
    def __init__(self, registry, telemetry) -> None:
        self.registry = registry
        self.telemetry = telemetry
        self.sessions = Sessions()


def session(tmp_path, lines, user="alice"):
    registry = registry_module.open_conversation_registry()
    pipeline = Pipeline(registry)
    out = []
    feed = iter(lines)

    def read(prompt):
        value = next(feed)
        if isinstance(value, BaseException):
            raise value
        return value

    cli = CliSession(pipeline, Components(registry, pipeline.telemetry), user, read, out.append,
                     health=lambda components: [ComponentHealth("PostgreSQL", Health.HEALTHY, "ok", 1.0)])
    return cli, pipeline, registry, out


class TestSession:
    def test_starting_creates_no_conversation_until_the_first_query(self, tmp_path):
        cli, pipeline, registry, out = session(tmp_path, ["exit"])
        cli.run()
        assert registry.list("alice") == []

    def test_the_first_query_creates_one_conversation_and_later_ones_reuse_it(self, tmp_path):
        cli, pipeline, registry, out = session(tmp_path, ["What is MVCC?", "And VACUUM?", EOFError()])
        cli.run()
        (conversation,) = registry.list("alice")
        assert [a[0] for a in pipeline.asked] == [conversation.conversation_id] * 2
        assert "answer to And VACUUM?" in out

    def test_switch_lists_only_this_users_conversations_and_switches_by_number(self, tmp_path):
        cli, pipeline, registry, out = session(tmp_path, ["switch", "2", "a follow-up", "exit"])
        first = registry.create("alice", "First")
        registry.create("alice", "Second")
        registry.create("bob", "Not yours")
        cli.run()
        listing = next(text for text in out if text.startswith("Available conversations"))
        assert "Not yours" not in listing and "First" in listing
        assert pipeline.asked[-1][0] == first.conversation_id  # newest first, so 2 is "First"

    def test_switch_by_id_prefix_and_never_into_another_users(self, tmp_path):
        cli, pipeline, registry, out = session(tmp_path, [])
        mine = registry.create("alice", "Mine", conversation_id="abcd1234")
        registry.create("bob", "Theirs", conversation_id="ffff9999")
        cli.switch("abcd")
        assert cli.active.conversation_id == mine.conversation_id
        cli.switch("ffff9999")
        assert cli.active.conversation_id == mine.conversation_id
        assert out[-1] == "No conversation 'ffff9999'."

    def test_status_variants_report_real_records(self, tmp_path):
        cli, pipeline, registry, out = session(
            tmp_path, ["status --time", "status --retries", "What is MVCC?", "status --time",
                       "status --retries", "status", "exit"])
        cli.run()
        assert out[1] == out[2] == "No query has run in this session yet."
        timing = next(text for text in out if text.startswith("Last query") and "Total" in text)
        assert "Query decomposition" in timing and "12 ms" in timing
        retries = next(text for text in out if "Planner/Critique retries" in text)
        assert "Job-level retries:     2" in retries and "Total reasoning retries: 3" in retries
        assert any(text.startswith("Atlas status") and "PostgreSQL" in text for text in out)

    def test_ctrl_c_at_the_prompt_does_not_exit_and_eof_does(self, tmp_path):
        cli, pipeline, registry, out = session(tmp_path, [KeyboardInterrupt(), "help", EOFError()])
        cli.run()
        assert any("Commands" in text for text in out)
        assert out[-1] == "Goodbye."

    def test_a_failing_query_is_reported_and_the_session_continues(self, tmp_path):
        cli, pipeline, registry, out = session(tmp_path, ["boom", "help", "exit"])

        def explode(*args):
            raise RuntimeError("database unreachable")

        pipeline.ask = explode
        cli.run()
        assert "Error: database unreachable" in out
        assert any("Commands" in text for text in out)

    def test_new_forgets_the_active_conversation(self, tmp_path):
        cli, pipeline, registry, out = session(tmp_path, ["one", "new", "two", "exit"])
        cli.run()
        assert len(registry.list("alice")) == 2


def test_durations_render_in_ms_or_seconds():
    assert render.duration(12.4) == "12 ms"
    assert render.duration(65_400) == "65.4 s"
    assert render.duration(None) == "-"
