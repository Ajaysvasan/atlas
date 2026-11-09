"""`main.py status` and `main.py conversations`: standalone, read-only, and never loading the model."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import main
from cli import monitor
from config import Config
from memory import conversation_registry as registry_module
from observability import health
from observability.service_state import ServiceLease
from test.observability_testing.test_service_state import Components, a_record

APP = Path(__file__).resolve().parents[2]


@pytest.fixture
def pg_down(monkeypatch):
    secret = "s3cr3t-password"
    monkeypatch.setenv("DB_PASSWORD", secret)

    def refuse():
        raise ConnectionError(f"connection to server failed (password={secret})")

    monkeypatch.setattr(health, "_postgres", refuse)
    return secret


def run(command, *args, **kwargs):
    out = []
    code = command(*args, write=out.append, **kwargs)
    return code, "\n".join(out)


class TestStatus:
    def test_never_started_exits_3_and_still_checks_what_it_can(self, pg_down):
        code, text = run(monitor.status)
        assert code == monitor.NOT_RUNNING
        assert "never been started" in text
        assert "PostgreSQL" in text and "UNAVAILABLE" in text
        assert "Atlas is not running" in text
        assert pg_down not in text

    def test_a_running_service_is_reported_from_what_it_published(self, pg_down, tmp_path):
        components = Components(tmp_path / "absent.gguf")
        with ServiceLease("start") as lease:
            lease.attach(components)
            code, text = run(monitor.status)
        assert code == monitor.RUNNING
        assert "Atlas service: running (start" in text and "acquisition on" in text
        assert "restores 2, misses 1" in text
        assert "Atlas is not running" not in text

    def test_a_service_still_starting_is_not_called_stopped(self, pg_down):
        with ServiceLease("start"):
            code, text = run(monitor.status)
        assert code == monitor.RUNNING
        assert "Atlas is still starting" in text and "Atlas is not running" not in text

    def test_time_and_retries_read_the_running_services_last_query(self, tmp_path):
        components = Components(tmp_path / "absent.gguf")
        with ServiceLease("start") as lease:
            lease.attach(components)
            assert "No query has run since Atlas started." in run(monitor.status, "time")[1]
            components.telemetry.add(a_record())
            code, timing = run(monitor.status, "time")
            _, retries = run(monitor.status, "retries")
        assert code == monitor.RUNNING
        assert "From the running service" in timing and "Query decomposition" in timing and "42 ms" in timing
        assert "Job-level retries:     1" in retries and "job 2 cited an unknown passage" in retries

    def test_after_a_stop_the_last_runs_query_is_shown_and_labelled(self, tmp_path):
        components = Components(tmp_path / "absent.gguf")
        with ServiceLease("start") as lease:
            lease.attach(components)
            components.telemetry.add(a_record("last-one"))
        code, text = run(monitor.status, "time")
        assert code == monitor.NOT_RUNNING
        assert "From the run that ended" in text and "last-one" in text

    def test_time_before_any_run(self):
        code, text = run(monitor.status, "retries")
        assert code == monitor.NOT_RUNNING and "Atlas has not run yet." in text

    def test_status_writes_nothing(self, pg_down):
        run(monitor.status)
        run(monitor.status, "time")
        for path in (Config.RUN_DIR, Config.INFERENCE_CACHE_DIR, Config.MEMORY_DB, Config.CORPUS_DIR):
            assert not Path(path).exists(), path


class TestConversations:
    def test_lists_only_the_users_conversations(self, conversation_registry):
        conversation_registry.create("alice", "Raft Leader Election")
        conversation_registry.create("bob", "Not yours")
        code, text = run(monitor.conversations, "alice")
        assert code == 0
        assert "Raft Leader Election" in text and "Not yours" not in text

    def test_none_yet(self, conversation_registry):
        code, text = run(monitor.conversations, "carol")
        assert code == 0 and text == "carol has no conversations yet."

    def test_opens_the_registry_read_only(self, monkeypatch, conversation_registry):
        seen = {}

        def open_registry(**options):
            seen.update(options)
            return conversation_registry

        monkeypatch.setattr(registry_module, "open_conversation_registry", open_registry)
        run(monitor.conversations, "alice")
        assert seen["read_only"] is True and seen["connect_timeout"]

    def test_an_unreachable_database_exits_1_without_the_password(self, monkeypatch):
        secret = "s3cr3t-password"
        monkeypatch.setenv("DB_PASSWORD", secret)

        def refuse(**options):
            raise ConnectionError(f"connection failed (password={secret})")

        monkeypatch.setattr(registry_module, "open_conversation_registry", refuse)
        code, text = run(monitor.conversations, "alice")
        assert code == monitor.FAILED and "Cannot reach PostgreSQL" in text and secret not in text


class TestArguments:
    @pytest.mark.parametrize("argv, command, detail", [
        (["status"], "status", None),
        (["status", "--time"], "status", "time"),
        (["status", "--retries"], "status", "retries"),
        (["conversations"], "conversations", None),
        (["--user", "alice", "conversations"], "conversations", None),
    ])
    def test_monitor_commands_parse(self, argv, command, detail):
        args = main.parse_arguments(argv)
        assert args.command == command and getattr(args, "detail", None) == detail

    @pytest.mark.parametrize("argv", [
        ["status", "--time", "--retries"],
        ["status", "--verbose-ish"],
        ["-q", "what is MVCC?", "status"],
        ["-q", "what is MVCC?", "serve"],
    ])
    def test_invalid_combinations_are_refused(self, argv):
        with pytest.raises(SystemExit):
            main.parse_arguments(argv)

    def test_monitor_commands_never_reach_the_service(self, monkeypatch):
        def no_service(args):
            raise AssertionError("a monitor started the service")

        monkeypatch.setattr(main, "run_service", no_service)
        monkeypatch.setattr(main, "configure_logging", lambda **options: None)
        seen = []
        monkeypatch.setattr(monitor, "status", lambda detail: seen.append(detail) or 3)
        monkeypatch.setattr(monitor, "conversations", lambda user: seen.append(user) or 0)
        assert main.main(["status", "--retries"]) == 3
        assert main.main(["--user", "alice", "conversations"]) == 0
        assert seen == ["retries", "alice"]


class TestSingleInstance:
    @pytest.mark.parametrize("argv", [["start"], ["serve"], ["-q", "What is MVCC?"], []])
    def test_a_second_service_is_refused_before_anything_loads(self, argv, monkeypatch, capsys):
        import cli.cli_interface

        def must_not_run(*args, **kwargs):
            raise AssertionError("loaded while another Atlas holds the lease")

        monkeypatch.setattr(cli.cli_interface, "cli_interface", must_not_run)
        monkeypatch.setattr(main, "single_query", must_not_run)
        monkeypatch.setenv("ATLAS_GRPC_SERVICE_TOKEN", "x" * 32)
        import grpc_server.server

        monkeypatch.setattr(grpc_server.server, "serve", must_not_run)
        with ServiceLease("start"):
            assert main.run_service(main.parse_arguments(argv)) == 1
        assert "Atlas is already running (start" in capsys.readouterr().err

    def test_the_lease_is_released_when_the_service_ends(self, monkeypatch):
        import cli.cli_interface

        held = []

        def fake_cli(user, acquisition=True, lease=None):
            held.append(lease.held)
            return 0

        monkeypatch.setattr(cli.cli_interface, "cli_interface", fake_cli)
        assert main.run_service(main.parse_arguments(["start"])) == 0
        assert held == [True]
        ServiceLease("start").acquire().release()


def test_status_in_a_fresh_process_imports_neither_the_model_runtime_nor_torch(tmp_path):
    """The monitor's whole cost: no llama.cpp, no torch, no sentence-transformers."""
    script = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(APP)!r})
        os.environ.update(DB_HOST="127.0.0.1", DB_PORT="1", DB_NAME="x", DB_USER="x", DB_PASSWORD="x")
        from pathlib import Path
        from config import Config
        base = Path({str(tmp_path)!r})
        for name in ("RUN_DIR", "INFERENCE_CACHE_DIR", "CORPUS_DIR", "DB_PATH", "INDEX_PATH"):
            setattr(Config, name, base / name.lower())
        Config.MEMORY_DB = base / "memory.db"
        from cli import monitor
        monitor.status(write=lambda text: None)
        monitor.status("time", write=lambda text: None)
        heavy = sorted(m for m in ("llama_cpp", "torch", "sentence_transformers", "diskannpy") if m in sys.modules)
        print(",".join(heavy) or "none")
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=APP, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "none"
    assert not any((tmp_path / name).exists() for name in ("run_dir", "inference_cache_dir", "memory.db"))
