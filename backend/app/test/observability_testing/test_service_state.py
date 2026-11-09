"""The single-instance lock and the state a running Atlas publishes for monitors in other processes."""

import json
import os
import signal
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from config import Config
from inference.runtime import InferenceRuntime
from inference.shared_model import ModelSpec, SharedModel
from observability import service_state
from observability.service_state import (
    LOCK_FILE,
    STATE_FILE,
    AlreadyRunning,
    ServiceLease,
    read_state,
    record_from_dict,
    record_to_dict,
)
from observability.telemetry import RequestRecord, Telemetry
from pipeline.atlas import RetrieverHandle

APP = Path(__file__).resolve().parents[2]


class Sessions:
    restores, misses, invalidations, save_failures = 2, 1, 0, 0


class Components:
    def __init__(self, model_path) -> None:
        self.runtime = InferenceRuntime(SharedModel(ModelSpec(Path(model_path))))
        self.retriever = RetrieverHandle(lambda: None)
        self.sessions = Sessions()
        self.acquirer = object()
        self.telemetry = Telemetry()


def a_record(request_id="r1") -> RequestRecord:
    record = RequestRecord(request_id, "c1", "2026-11-09T10:00:00+00:00", status="completed", total_ms=42.0)
    record.stages.update(decomposition=5.0, generation=30.0)
    record.path, record.jobs, record.job_retries = "planned", 3, 1
    record.retry_reasons.append("job 2 cited an unknown passage")
    return record


@pytest.fixture
def run_dir(tmp_path):
    return tmp_path / "run"


def test_a_record_round_trips_through_json():
    record = a_record()
    again = record_from_dict(json.loads(json.dumps(record_to_dict(record))))
    assert again == record and list(again.stages) == ["decomposition", "generation"]


def test_nothing_published_reads_as_never_started_and_reading_creates_nothing(run_dir):
    assert read_state(run_dir) is None
    assert not run_dir.exists()


def test_a_held_lease_reads_as_running_and_its_files_are_private(run_dir):
    with ServiceLease("start", run_dir):
        state = read_state(run_dir)
        assert state.running and not state.crashed
        assert state.mode == "start" and state.pid == os.getpid()
        assert stat.S_IMODE(run_dir.stat().st_mode) == 0o700
        for name in (LOCK_FILE, STATE_FILE):
            assert stat.S_IMODE((run_dir / name).stat().st_mode) == 0o600


def test_a_second_service_is_refused_while_the_first_holds_the_lease(run_dir):
    with ServiceLease("serve", run_dir):
        with pytest.raises(AlreadyRunning, match=f"serve, pid {os.getpid()}"):
            ServiceLease("start", run_dir).acquire()


def test_a_clean_stop_is_recorded_and_frees_the_lease(run_dir):
    lease = ServiceLease("start", run_dir).acquire()
    lease.release()
    state = read_state(run_dir)
    assert not state.running and not state.crashed and state.stopped_at
    assert state.facts is None
    ServiceLease("start", run_dir).acquire().release()


def test_attached_components_and_every_finished_request_are_published(run_dir, tmp_path):
    components = Components(tmp_path / "absent.gguf")
    with ServiceLease("start", run_dir) as lease:
        lease.attach(components)
        components.telemetry.add(a_record("r1"))
        components.telemetry.add(a_record("r2"))
        state = read_state(run_dir)
    assert [r.request_id for r in state.records] == ["r1", "r2"]
    assert state.facts["draft"]["loaded"] is False
    assert state.facts["kv"]["restores"] == 2
    assert state.facts["retriever"] == {"built": False, "error": None, "generation": None}
    assert state.acquisition is True
    after = read_state(run_dir)
    assert not after.running and [r.request_id for r in after.records] == ["r1", "r2"]


def test_a_publish_that_fails_never_fails_the_request(run_dir, tmp_path, monkeypatch):
    components = Components(tmp_path / "absent.gguf")
    with ServiceLease("start", run_dir) as lease:
        lease.attach(components)

        def broken(_):
            raise RuntimeError("cannot describe")

        monkeypatch.setattr(service_state, "service_facts", broken)
        components.telemetry.add(a_record())
        assert components.telemetry.last().request_id == "r1"


def test_published_records_are_bounded(run_dir, tmp_path):
    components = Components(tmp_path / "absent.gguf")
    with ServiceLease("start", run_dir) as lease:
        lease.attach(components)
        for n in range(service_state.KEEP_RECORDS + 5):
            components.telemetry.add(a_record(f"r{n}"))
    assert len(read_state(run_dir).records) == service_state.KEEP_RECORDS


def test_a_lease_holder_starting_up_reads_as_starting(run_dir):
    with ServiceLease("start", run_dir):
        (run_dir / STATE_FILE).write_text(json.dumps({"format": 1, "running": False, "pid": 1}))
        state = read_state(run_dir)
    assert state.running and state.mode == "starting" and state.pid is None


def test_another_process_is_seen_running_and_a_killed_one_as_crashed(run_dir):
    holder = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str(APP)!r})
            from observability.service_state import ServiceLease
            ServiceLease("serve", {str(run_dir)!r}).acquire()
            print("held", flush=True)
            sys.stdin.read()
        """)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, cwd=APP,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        state = read_state(run_dir)
        assert state.running and state.pid == holder.pid
        with pytest.raises(AlreadyRunning):
            ServiceLease("start", run_dir).acquire()
    finally:
        holder.send_signal(signal.SIGKILL)
        holder.wait(10)
    state = read_state(run_dir)
    assert not state.running and state.crashed and state.pid == holder.pid
    ServiceLease("start", run_dir).acquire().release()


def test_the_default_directory_is_config_run_dir():
    assert ServiceLease("start").directory == Path(Config.RUN_DIR)
