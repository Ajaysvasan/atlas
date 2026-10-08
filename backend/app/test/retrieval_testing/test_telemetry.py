"""The record of where a retrieval's time went.

Timings are the only thing that tells a slow retrieval from a slow stage, so
what matters here is that a stage is recorded even when it fails, that the
timeline reads in the order the stages ran, and that the clock behind it
cannot run backwards.
"""

import time

import pytest

from retrieval_layer.telemetry import Stage, StageRecorder


class FakeClock:
    """A clock that only moves when a test says so."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestTimingAStage:
    def test_a_stage_records_its_name_duration_and_yield(self):
        clock = FakeClock()
        recorder = StageRecorder(clock=clock)

        with recorder.stage("vector search") as stage:
            clock.advance(0.012)
            stage.produced = 32

        timing, = recorder.timings
        assert timing.stage == "vector search"
        assert timing.milliseconds == pytest.approx(12.0)
        assert timing.produced == 32

    def test_a_stage_that_says_nothing_produced_nothing(self):
        recorder = StageRecorder(clock=FakeClock())
        with recorder.stage("fusion"):
            pass
        assert recorder.timings[0].produced == 0

    def test_duration_is_milliseconds_not_seconds(self):
        clock = FakeClock()
        recorder = StageRecorder(clock=clock)
        with recorder.stage("rerank"):
            clock.advance(1.5)
        assert recorder.timings[0].milliseconds == pytest.approx(1500.0)

    def test_a_real_retrieval_measures_something(self):
        recorder = StageRecorder()
        with recorder.stage("work"):
            sum(range(200_000))
        assert recorder.timings[0].milliseconds > 0


class TestWhenAStageFails:
    def test_the_exception_still_reaches_the_caller(self):
        recorder = StageRecorder()
        with pytest.raises(ZeroDivisionError):
            with recorder.stage("hydration"):
                1 / 0

    def test_a_failed_stage_is_still_timed(self):
        """A stage that took four seconds to fail is the interesting one."""
        clock = FakeClock()
        recorder = StageRecorder(clock=clock)
        with pytest.raises(RuntimeError):
            with recorder.stage("hydration"):
                clock.advance(4.0)
                raise RuntimeError("no chunk store")

        timing, = recorder.timings
        assert timing.stage == "hydration"
        assert timing.milliseconds == pytest.approx(4000.0)

    def test_a_failure_is_logged_as_a_warning(self, caplog):
        """A stage that raised is an event, not a measurement."""
        recorder = StageRecorder()
        with caplog.at_level("WARNING", logger="retrieval_layer.telemetry"):
            with pytest.raises(RuntimeError):
                with recorder.stage("keyword search"):
                    raise RuntimeError("no fts table")

        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert "keyword search" in warnings[0].getMessage()

    def test_what_a_failed_stage_had_produced_is_kept(self):
        recorder = StageRecorder(clock=FakeClock())
        with pytest.raises(RuntimeError):
            with recorder.stage("hydration") as stage:
                stage.produced = 3
                raise RuntimeError("halfway")
        assert recorder.timings[0].produced == 3


class TestTheTimeline:
    def test_stages_come_back_in_the_order_they_ran(self):
        recorder = StageRecorder(clock=FakeClock())
        for name in ("plan", "search", "fuse", "rerank"):
            with recorder.stage(name):
                pass
        assert [t.stage for t in recorder.timings] == [
            "plan", "search", "fuse", "rerank"
        ]

    def test_a_nested_stage_follows_the_one_it_sits_inside(self):
        """Ordered by when each stage started, not when it finished."""
        recorder = StageRecorder(clock=FakeClock())
        with recorder.stage("search"):
            with recorder.stage("vector"):
                pass
            with recorder.stage("keyword"):
                pass
        assert [t.stage for t in recorder.timings] == [
            "search", "vector", "keyword"
        ]

    def test_an_outer_stage_is_at_least_as_long_as_what_it_contains(self):
        clock = FakeClock()
        recorder = StageRecorder(clock=clock)
        with recorder.stage("search"):
            with recorder.stage("vector"):
                clock.advance(0.01)
            clock.advance(0.01)

        outer = next(t for t in recorder.timings if t.stage == "search")
        inner = next(t for t in recorder.timings if t.stage == "vector")
        assert outer.milliseconds >= inner.milliseconds

    def test_a_stage_still_running_is_not_reported_yet(self):
        recorder = StageRecorder(clock=FakeClock())
        with recorder.stage("rerank"):
            assert recorder.timings == []

    def test_nothing_ran_nothing_recorded(self):
        assert StageRecorder().timings == []


class TestTheClock:
    def test_the_default_clock_is_monotonic(self):
        """time.time() would let an NTP step report a negative duration."""
        assert StageRecorder().clock is time.perf_counter


class TestTheBreakdown:
    def test_it_names_every_stage_and_its_count(self, caplog):
        clock = FakeClock()
        recorder = StageRecorder(clock=clock)
        with recorder.stage("search"):
            clock.advance(0.02)
        with recorder.stage("rerank") as stage:
            clock.advance(0.05)
            stage.produced = 8

        with caplog.at_level("DEBUG", logger="retrieval_layer.telemetry"):
            recorder.log_breakdown()

        line = caplog.text
        assert "search" in line and "rerank" in line
        assert "8" in line
        assert "70 ms" in line

    def test_nothing_is_logged_when_nothing_ran(self, caplog):
        with caplog.at_level("DEBUG", logger="retrieval_layer.telemetry"):
            StageRecorder().log_breakdown()
        assert caplog.records == []


class TestTheHandle:
    def test_it_carries_only_a_name_and_a_count(self):
        """__slots__, because one is built per stage per query."""
        stage = Stage("fusion")
        with pytest.raises(AttributeError):
            stage.anything_else = 1
