"""Tests for logging_setup and the way the rest of the application uses it.

Two kinds of test live here. The first exercise the module directly. The second
are structural guards over the source tree — the same shape as
test_every_conversation_connection_goes_through_connect — because the defects
this module was written to fix (handlers per module, file I/O at import time,
eager f-strings in log calls) are the kind that creep back one file at a time
and never fail a behavioural test.
"""

import ast
import json
import logging
import os
import threading
from pathlib import Path

import pytest

import logging_setup
from logging_setup import (
    configure,
    current_context,
    get_logger,
    log_context,
    log_timing,
    new_correlation_id,
    reset,
)

APP_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRS = ("data_layer", "memory", "cli")
LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical"}


@pytest.fixture(autouse=True)
def isolated_logging():
    """Give every test the global logging state to itself and hand it back.

    configure() mutates the root logger, which is process-wide. Without this a
    test that raised the level would silently change what later tests — and the
    rest of the suite — record.
    """
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    saved_flag = logging_setup._configured
    for handler in list(root.handlers):
        root.removeHandler(handler)
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in saved_handlers:
        root.addHandler(handler)
    root.setLevel(saved_level)
    logging_setup._configured = saved_flag


def read(path):
    return Path(path).read_text(encoding="utf-8") if Path(path).exists() else ""


def configured_handlers():
    """The handlers configure() installed, ignoring anyone else's.

    pytest attaches its own capture handlers to the root logger around every
    test, so a bare count of root.handlers measures the test runner rather than
    this module. The context filter is what configure() puts on each of its
    handlers and nothing else does, so it identifies them exactly.
    """
    return [
        handler
        for handler in logging.getLogger().handlers
        if any(isinstance(f, logging_setup._ContextFilter) for f in handler.filters)
    ]


class TestGetLoggerIsInert:
    """get_logger must be safe to call at import time, from any module."""

    def test_returns_a_logger_with_no_handlers_of_its_own(self):
        assert get_logger("data_layer.anything").handlers == []

    def test_creates_no_file(self, tmp_path):
        target = tmp_path / "nothing"
        os.environ["LOG_FILE"] = str(target / "app.log")
        try:
            get_logger("memory.anything").info("dropped, nothing is configured")
        finally:
            del os.environ["LOG_FILE"]
        assert not target.exists()

    def test_the_same_name_returns_the_same_logger(self):
        assert get_logger("data_layer.x") is get_logger("data_layer.x")

    def test_importing_config_does_not_create_the_log_directory(self):
        """The regression that motivated the split.

        The previous helper opened a FileHandler as a side effect of being
        called at module scope, so importing config wrote to disk. Nothing in
        the import graph may do that now.
        """
        import config

        source = ast.parse(Path(config.__file__).read_text())
        calls = [
            node
            for node in ast.walk(source)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"mkdir", "makedirs", "FileHandler", "open"}
        ]
        assert calls == []


class TestConfigure:
    def test_installs_a_file_and_a_console_handler(self, tmp_path):
        configure(log_file=tmp_path / "a.log", force=True)
        kinds = {type(h).__name__ for h in configured_handlers()}
        assert kinds == {"RotatingFileHandler", "StreamHandler"}

    def test_writes_records_from_any_module_logger(self, tmp_path):
        path = tmp_path / "a.log"
        configure(log_file=path, console=False, force=True)
        get_logger("data_layer.ingestion.chunker").warning("from the data layer")
        get_logger("memory.topic_pool.snapshot").warning("from the memory layer")
        contents = read(path)
        assert "from the data layer" in contents
        assert "from the memory layer" in contents

    def test_repeat_calls_do_not_stack_handlers(self, tmp_path):
        path = tmp_path / "a.log"
        configure(log_file=path, console=False, force=True)
        for _ in range(5):
            configure(log_file=path, console=False)
        get_logger("dup").warning("say once")
        assert read(path).count("say once") == 1

    def test_a_repeat_call_leaves_the_existing_handlers_alone(self, tmp_path):
        """Ignored, not reapplied.

        Tearing the handlers down and rebuilding them would also avoid
        duplicate lines, but it drops every record logged in between and
        reopens the file — so the weaker "no duplicates" check alone would pass
        a configure() that quietly restarts logging on every call.
        """
        configure(log_file=tmp_path / "a.log", console=False, force=True)
        before = configured_handlers()
        configure(log_file=tmp_path / "a.log", console=False)
        assert configured_handlers() == before

    def test_force_replaces_the_previous_configuration(self, tmp_path):
        first, second = tmp_path / "a.log", tmp_path / "b.log"
        configure(log_file=first, console=False, force=True)
        configure(log_file=second, console=False, force=True)
        get_logger("moved").warning("only in the second")
        assert "only in the second" not in read(first)
        assert "only in the second" in read(second)

    def test_concurrent_configure_installs_one_set_of_handlers(self, tmp_path):
        path = tmp_path / "a.log"
        errors = []
        barrier = threading.Barrier(12)

        def worker():
            try:
                barrier.wait()
                configure(log_file=path, console=False)
            except Exception as error:  # pragma: no cover - failure detail
                errors.append(error)

        threads = [threading.Thread(target=worker) for _ in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        assert len(configured_handlers()) == 1
        get_logger("threaded").warning("exactly once")
        assert read(path).count("exactly once") == 1

    def test_no_file_is_written_until_something_is_logged(self, tmp_path):
        path = tmp_path / "a.log"
        configure(log_file=path, console=False, force=True)
        assert not path.exists()

    def test_reset_closes_the_handlers(self, tmp_path):
        configure(log_file=tmp_path / "a.log", console=False, force=True)
        handler = configured_handlers()[0]
        reset()
        assert configured_handlers() == []
        assert not logging_setup.is_configured()
        assert handler.stream is None or handler.stream.closed


class TestOneLevelReachesEveryModule:
    """The --verbose fix.

    Every module logger propagates to the root handlers, so raising the root
    level raises it everywhere. Under the arrangement this replaced — handlers
    on each module logger with propagate=False — main.py's --verbose reached
    only the logger it was applied to, and every other module stayed at INFO.
    """

    @pytest.mark.parametrize(
        "module",
        [
            "data_layer.ingestion.Chunker.chunker",
            "data_layer.ingestion.embedding.EmbeddingManager",
            "memory.topic_pool.project_pool.conversation_pool.snapshot",
            "cli.cli_interface",
        ],
    )
    def test_debug_reaches_module(self, tmp_path, module):
        path = tmp_path / "a.log"
        configure(log_file=path, level="DEBUG", console=False, force=True)
        get_logger(module).debug("verbose from %s", module)
        assert f"verbose from {module}" in read(path)

    def test_modules_keep_no_handlers_of_their_own(self, tmp_path):
        configure(log_file=tmp_path / "a.log", console=False, force=True)
        log = get_logger("data_layer.ingestion.normalizer.normalizer")
        log.info("routed to root")
        assert log.handlers == []
        assert log.propagate is True

    def test_set_level_reaches_the_handlers_too(self, tmp_path):
        path = tmp_path / "a.log"
        configure(log_file=path, level="INFO", console=False, force=True)
        logging_setup.set_level("DEBUG")
        get_logger("late").debug("raised after configure")
        assert "raised after configure" in read(path)

    def test_a_console_below_the_file_level_still_reaches_the_root_gate(
        self, tmp_path
    ):
        """LOG_LEVEL=WARNING with --verbose.

        The root logger gates before any handler is consulted, so it has to sit
        at the lower of the two levels. Pinned to the file level instead, a
        console asked for DEBUG would be handed nothing below WARNING and the
        flag would look broken again.
        """
        configure(
            log_file=tmp_path / "a.log",
            level="WARNING",
            console_level="DEBUG",
            force=True,
        )
        assert logging.getLogger().level == logging.DEBUG
        assert get_logger("data_layer.anything").isEnabledFor(logging.DEBUG)

    def test_console_and_file_can_hold_different_levels(self, tmp_path):
        path = tmp_path / "a.log"
        configure(log_file=path, level="DEBUG", console_level="ERROR", force=True)
        handlers = {type(h).__name__: h.level for h in configured_handlers()}
        assert handlers["RotatingFileHandler"] == logging.DEBUG
        assert handlers["StreamHandler"] == logging.ERROR
        # The root gate has to sit at the lower of the two or the file handler
        # never sees the debug records it is configured to accept.
        assert logging.getLogger().level == logging.DEBUG


class TestContext:
    def test_fields_appear_on_records_logged_inside_the_block(self, tmp_path):
        path = tmp_path / "a.log"
        configure(log_file=path, console=False, force=True)
        log = get_logger("ctx")
        with log_context(query_id="q1"):
            log.warning("inside")
        log.warning("outside")
        contents = read(path)
        assert "[query_id=q1] inside" in contents
        assert "outside" in contents
        assert "[query_id=q1] outside" not in contents

    def test_nesting_layers_fields(self, tmp_path):
        path = tmp_path / "a.log"
        configure(log_file=path, console=False, force=True)
        with log_context(query_id="q1"):
            with log_context(node="slave-2"):
                get_logger("ctx").warning("planning")
        line = read(path)
        assert "query_id=q1" in line and "node=slave-2" in line

    def test_the_block_restores_what_it_found(self):
        with log_context(a="1"):
            with log_context(a="2"):
                assert current_context()["a"] == "2"
            assert current_context()["a"] == "1"
        assert current_context() == {}

    def test_an_exception_still_unbinds(self):
        with pytest.raises(ValueError):
            with log_context(a="1"):
                raise ValueError
        assert current_context() == {}

    def test_worker_threads_are_named_in_the_output(self, tmp_path):
        path = tmp_path / "a.log"
        configure(log_file=path, console=False, force=True)
        thread = threading.Thread(
            target=lambda: get_logger("ctx").warning("in a worker"), name="Worker-7"
        )
        thread.start()
        thread.join()
        assert "thread=Worker-7" in read(path)

    def test_correlation_ids_differ(self):
        assert new_correlation_id() != new_correlation_id()


class TestJsonFormat:
    def test_every_line_is_one_json_object(self, tmp_path):
        path = tmp_path / "a.log"
        configure(log_file=path, console=False, json_lines=True, force=True)
        with log_context(query_id="q1"):
            get_logger("json").warning("planned %d step(s)", 3, extra={"steps": 3})
        record = json.loads(read(path).strip())
        assert record["message"] == "planned 3 step(s)"
        assert record["level"] == "WARNING"
        assert record["logger"] == "json"
        assert record["query_id"] == "q1"
        assert record["steps"] == 3

    def test_an_exception_is_carried(self, tmp_path):
        path = tmp_path / "a.log"
        configure(log_file=path, console=False, json_lines=True, force=True)
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            get_logger("json").exception("failed")
        record = json.loads(read(path).strip())
        assert "RuntimeError: boom" in record["exception"]

    def test_a_value_json_cannot_encode_does_not_lose_the_line(self, tmp_path):
        path = tmp_path / "a.log"
        configure(log_file=path, console=False, json_lines=True, force=True)
        get_logger("json").warning("odd", extra={"blob": object()})
        assert json.loads(read(path).strip())["message"] == "odd"


class TestLogTiming:
    def test_reports_a_duration_on_success(self, tmp_path):
        path = tmp_path / "a.log"
        configure(log_file=path, console=False, force=True)
        with log_timing(get_logger("t"), "embedding", level=logging.WARNING):
            pass
        assert "embedding took" in read(path)

    def test_reports_and_re_raises_on_failure(self, tmp_path):
        path = tmp_path / "a.log"
        configure(log_file=path, console=False, force=True)
        with pytest.raises(ValueError):
            with log_timing(get_logger("t"), "embedding"):
                raise ValueError("no")
        assert "embedding failed after" in read(path)

    def test_the_record_is_credited_to_the_caller(self, tmp_path):
        """Without stacklevel every timing line in the application would be
        credited to the one `logger.log` call inside logging_setup.py."""
        path = tmp_path / "a.log"
        configure(log_file=path, console=False, json_lines=True, force=True)
        with log_timing(get_logger("t"), "here", level=logging.WARNING):
            pass
        line = json.loads(read(path).strip())["line"]
        assert line.startswith("test_logging_setup.py:")

    def test_duration_is_structured_in_json(self, tmp_path):
        path = tmp_path / "a.log"
        configure(log_file=path, console=False, json_lines=True, force=True)
        with log_timing(get_logger("t"), "embedding", level=logging.WARNING):
            pass
        assert isinstance(json.loads(read(path).strip())["duration_ms"], float)


class TestRobustness:
    def test_rotation_caps_the_number_of_files(self, tmp_path):
        path = tmp_path / "a.log"
        configure(
            log_file=path, console=False, max_bytes=2048, backup_count=2, force=True
        )
        log = get_logger("rot")
        for index in range(300):
            log.warning("padding %03d %s", index, "x" * 80)
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "a.log",
            "a.log.1",
            "a.log.2",
        ]

    def test_an_unwritable_log_path_leaves_the_console_working(self, tmp_path):
        blocked = tmp_path / "blocked"
        blocked.mkdir()
        blocked.chmod(0o500)
        try:
            configure(log_file=blocked / "sub" / "a.log", force=True)
            kinds = [type(h).__name__ for h in configured_handlers()]
            assert kinds == ["StreamHandler"]
        finally:
            blocked.chmod(0o700)

    def test_an_unusable_level_falls_back_instead_of_raising(self, tmp_path):
        configure(log_file=tmp_path / "a.log", level="LOUDER", console=False, force=True)
        assert logging.getLogger().level == logging.INFO

    def test_a_percent_in_a_message_survives(self, tmp_path):
        path = tmp_path / "a.log"
        configure(log_file=path, console=False, force=True)
        get_logger("pct").warning("recall@10 improved 12% over the baseline")
        assert "12% over the baseline" in read(path)

    def test_noisy_libraries_stay_quiet_even_at_debug(self, tmp_path):
        configure(log_file=tmp_path / "a.log", level="DEBUG", console=False, force=True)
        for name in ("sentence_transformers", "transformers", "urllib3"):
            assert logging.getLogger(name).getEffectiveLevel() == logging.WARNING
        assert get_logger("data_layer.anything").isEnabledFor(logging.DEBUG)

    def test_environment_variables_override_the_arguments(self, tmp_path, monkeypatch):
        path = tmp_path / "from_env.log"
        monkeypatch.setenv("LOG_FILE", str(path))
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        configure(log_file=tmp_path / "ignored.log", level="ERROR", console=False, force=True)
        get_logger("env").debug("via the environment")
        assert "via the environment" in read(path)


class TestSourceTreeUsesTheModule:
    """Guards over how the rest of the application logs."""

    @staticmethod
    def source_files():
        for directory in SOURCE_DIRS:
            for path in (APP_ROOT / directory).rglob("*.py"):
                yield path
        for name in ("main.py", "config.py", "logging_setup.py"):
            yield APP_ROOT / name

    def test_no_module_configures_logging_for_itself(self):
        """basicConfig or an addHandler outside logging_setup takes a module out
        of the single configuration and back to the arrangement this replaced."""
        offenders = []
        for path in self.source_files():
            if path.name == "logging_setup.py":
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr in {"basicConfig", "addHandler", "setLevel"}:
                        offenders.append(f"{path.relative_to(APP_ROOT)}: {node.func.attr}")
        assert offenders == []

    def test_every_module_logger_is_named_after_its_module(self):
        """A hard-coded name breaks the hierarchy the levels are applied through."""
        offenders = []
        for path in self.source_files():
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                    continue
                if node.func.id != "get_logger" or not node.args:
                    continue
                argument = node.args[0]
                named_after_module = (
                    isinstance(argument, ast.Name) and argument.id == "__name__"
                )
                if not named_after_module and path.name != "main.py":
                    offenders.append(str(path.relative_to(APP_ROOT)))
        assert offenders == []

    def test_no_log_call_formats_its_message_eagerly(self):
        """An f-string is built whether or not the level is enabled.

        Several of these sat in per-file and per-vector loops, paying full
        formatting cost on every iteration with DEBUG switched off.
        """
        offenders = []
        for path in self.source_files():
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                    continue
                if node.func.attr not in LOG_METHODS:
                    continue
                if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "logger"):
                    continue
                for argument in node.args:
                    if isinstance(argument, ast.JoinedStr):
                        offenders.append(
                            f"{path.relative_to(APP_ROOT)}:{argument.lineno}"
                        )
        assert offenders == []

    def test_no_log_call_builds_its_message_by_calling_a_method(self):
        """A log line must not do work the caller did not ask for.

        snapshot_now() logged a count it fetched with a second query, so the
        message cost a database read on every snapshot regardless of level.
        """
        offenders = []
        for path in self.source_files():
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                    continue
                if node.func.attr not in LOG_METHODS:
                    continue
                if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "logger"):
                    continue
                for argument in node.args:
                    calls = [
                        inner
                        for inner in ast.walk(argument)
                        if isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Attribute)
                        and isinstance(inner.func.value, ast.Name)
                        and inner.func.value.id == "self"
                    ]
                    if calls:
                        offenders.append(
                            f"{path.relative_to(APP_ROOT)}:{argument.lineno}"
                        )
        assert offenders == []
