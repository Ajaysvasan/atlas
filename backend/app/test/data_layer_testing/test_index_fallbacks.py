"""The worst cases: every one degrades the vector index, none takes the app down.

Memory that runs out, a container limit, a full disk, a corrupt generation, a
build the kernel kills, a PostgreSQL that is down. Linux overcommits, so a
MemoryError is never relied on — allocations are admitted before they are made.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from config import Config
from data_layer.ingestion.embedding.vector_ids import vector_id_for
from data_layer.vector_db_manager import index_generations, memory_guard
from data_layer.vector_db_manager.index_generations import IndexGenerations
from data_layer.vector_db_manager.memory_guard import MiB

APP = Path(__file__).resolve().parents[2]


def points(n, seed=0):
    return np.random.default_rng(seed).standard_normal((n, 128)).astype(np.float32)


@pytest.fixture
def stored(tmp_path, chunk_vectors):
    from data_layer.vector_db_manager.repository.vectorMetaDataRepository import (
        VectorMetaDataRepository,
    )
    from data_layer.vector_db_manager.stored_vectors import StoredVectors

    class Store:
        mapping = VectorMetaDataRepository(str(tmp_path / "chunks"))

        def add(self, data, prefix="c"):
            chunk_ids = [f"{prefix}{i}" for i in range(len(data))]
            ids = [vector_id_for(c) for c in chunk_ids]
            chunk_vectors.batch_insert(ids, data)
            return self.mapping.batch_insert(ids, chunk_ids)

        def source(self):
            return StoredVectors(self.mapping, chunk_vectors)

    made = Store()
    yield made
    made.mapping.close()


@pytest.fixture
def root(tmp_path):
    return tmp_path / "index"


def spare(monkeypatch, megabytes):
    monkeypatch.setattr(memory_guard, "spare_memory", lambda: megabytes * MiB)


def builder_that_runs(code):
    return lambda spec: [sys.executable, "-c", code]


class TestReadingMemory:
    def test_the_lower_of_the_system_and_the_cgroup_is_what_is_available(self, monkeypatch):
        monkeypatch.setattr(memory_guard, "_meminfo_available", lambda: 8 << 30)
        monkeypatch.setattr(memory_guard, "_cgroup_headroom", lambda: 1 << 30)
        assert memory_guard.available_memory() == 1 << 30

    def test_unknown_memory_admits_on_the_budget_alone(self, monkeypatch):
        monkeypatch.setattr(memory_guard, "_meminfo_available", lambda: None)
        monkeypatch.setattr(memory_guard, "_cgroup_headroom", lambda: None)
        assert memory_guard.available_memory() is None
        assert memory_guard.can_hold(1 << 40)

    def test_the_headroom_is_never_planned_away(self, monkeypatch):
        monkeypatch.setattr(memory_guard, "available_memory", lambda: 300 * MiB)
        assert memory_guard.spare_memory() == (300 - Config.MEMORY_HEADROOM_MB) * MiB
        assert not memory_guard.can_hold(100 * MiB)

    def test_this_machine_reports_something(self):
        assert memory_guard.available_memory() > 0

    def test_the_tightest_cgroup_limit_above_the_process_counts(self, tmp_path, monkeypatch):
        """A container's limit, two levels up, is the one that kills."""
        root = tmp_path / "cgroup"
        leaf = root / "outer" / "inner"
        leaf.mkdir(parents=True)
        (root / "outer" / "memory.max").write_text(str(1 << 30))
        (root / "outer" / "memory.current").write_text(str(900 * MiB))
        (leaf / "memory.max").write_text(str(4 << 30))
        (leaf / "memory.current").write_text(str(1 << 30))
        own = tmp_path / "own"
        own.write_text("0::/outer/inner\n")
        monkeypatch.setattr(memory_guard, "OWN_CGROUP", own)
        monkeypatch.setattr(memory_guard, "CGROUP_ROOT", root)
        assert memory_guard._cgroup_headroom() == (1 << 30) - 900 * MiB

    def test_an_unlimited_cgroup_is_unknown_rather_than_zero(self, tmp_path, monkeypatch):
        root = tmp_path / "cgroup"
        (root / "app").mkdir(parents=True)
        (root / "app" / "memory.max").write_text("max\n")
        own = tmp_path / "own"
        own.write_text("0::/app\n")
        monkeypatch.setattr(memory_guard, "OWN_CGROUP", own)
        monkeypatch.setattr(memory_guard, "CGROUP_ROOT", root)
        assert memory_guard._cgroup_headroom() is None

    def test_no_cgroup_v2_entry_is_unknown_rather_than_zero(self, tmp_path, monkeypatch):
        own = tmp_path / "own"
        own.write_text("1:name=systemd:/x\n")
        monkeypatch.setattr(memory_guard, "OWN_CGROUP", own)
        assert memory_guard._cgroup_headroom() is None

    def test_a_disk_build_is_bounded_by_what_is_spare(self, monkeypatch):
        spare(monkeypatch, 1024)
        assert memory_guard.disk_build_budget() == 1024 * MiB - memory_guard.BUILD_BASE
        spare(monkeypatch, 100)
        assert memory_guard.disk_build_budget() is None

    def test_free_disk_is_read_from_the_nearest_existing_directory(self, tmp_path):
        assert memory_guard.free_disk(tmp_path / "not" / "yet") > 0


class TestBuildingUnderPressure:
    def test_no_memory_for_any_build_leaves_the_current_one(self, real_diskann, stored, root, monkeypatch):
        stored.add(points(50))
        first = IndexGenerations(root).build(stored.source())
        spare(monkeypatch, 0)
        stored.add(points(10, seed=1), prefix="n")
        outcome = IndexGenerations(root).build(stored.source())
        assert not outcome.built and "too little for even a small disk build" in outcome.reason
        assert IndexGenerations(root).current() == first.generation

    def test_too_little_memory_for_a_memory_build_builds_for_disk(self, real_diskann, stored, root, monkeypatch):
        monkeypatch.setattr(memory_guard, "memory_build_bytes", lambda *a: 1 << 40)
        stored.add(points(50))
        outcome = IndexGenerations(root).build(stored.source())
        assert outcome.built and outcome.kind == "disk"

    def test_a_full_disk_builds_nothing(self, stored, root, monkeypatch):
        monkeypatch.setattr(memory_guard, "free_disk", lambda path: 0)
        stored.add(points(50))
        outcome = IndexGenerations(root).build(stored.source())
        assert not outcome.built and "of disk" in outcome.reason
        assert IndexGenerations(root).current() is None

    def test_a_budget_too_small_to_hold_any_index_builds_nothing(self, stored, root, monkeypatch):
        monkeypatch.setattr(Config, "VECTOR_INDEX_RAM_MB", 40)
        stored.add(points(50))
        outcome = IndexGenerations(root).build(stored.source())
        assert not outcome.built and "VECTOR_INDEX_RAM_MB" in outcome.reason

    def test_a_builder_the_kernel_kills_is_survived(self, real_diskann, stored, root, monkeypatch):
        """SIGKILL is what the OOM killer sends; the application sees a failed build."""
        stored.add(points(50))
        first = IndexGenerations(root).build(stored.source())
        monkeypatch.setattr(index_generations, "builder_command", builder_that_runs(
            "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"))
        stored.add(points(10, seed=1), prefix="n")
        outcome = IndexGenerations(root).build(stored.source())
        assert not outcome.built and "killed" in outcome.reason
        assert IndexGenerations(root).current() == first.generation
        assert not [p for p in root.iterdir() if p.name.endswith(".building")]
        assert IndexGenerations(root).open(512 * MiB) is not None

    def test_a_builder_that_raises_is_survived(self, stored, root, monkeypatch):
        monkeypatch.setattr(index_generations, "builder_command", builder_that_runs(
            "raise RuntimeError('diskann refused')"))
        stored.add(points(50))
        outcome = IndexGenerations(root).build(stored.source())
        assert not outcome.built and "diskann refused" in outcome.reason

    def test_the_builder_volunteers_for_the_oom_killer(self):
        done = subprocess.run(
            [sys.executable, "-c",
             "from data_layer.vector_db_manager.index_build import _yield_to_the_application\n"
             "_yield_to_the_application()\n"
             "print(open('/proc/self/oom_score_adj').read().strip())\n"],
            cwd=APP, capture_output=True, text=True, check=True, timeout=60,
        )
        assert done.stdout.strip() == "1000"

    def test_a_build_that_hangs_is_stopped(self, stored, root, monkeypatch):
        """diskannpy has hung before (a disk index searched with one thread)."""
        monkeypatch.setattr(index_generations, "SHORTEST_BUILD_WAIT", 1)
        monkeypatch.setattr(index_generations, "builder_command", builder_that_runs(
            "import time; time.sleep(60)"))
        stored.add(points(50))
        outcome = IndexGenerations(root).build(stored.source())
        assert not outcome.built and "did not finish" in outcome.reason

    def test_the_real_builder_runs_without_the_callers_main_script(self, stored, root):
        """`multiprocessing`'s spawn re-imports the caller's main script in the
        child, which re-runs any entry point without a __main__ guard."""
        command = index_generations.builder_command({})
        assert command[:3] == [sys.executable, "-m", "data_layer.vector_db_manager.index_build"]

    def test_a_second_build_while_one_runs_is_skipped(self, stored, root):
        import fcntl

        stored.add(points(50))
        root.mkdir()
        with open(root / ".build.lock", "w") as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            outcome = IndexGenerations(root).build(stored.source())
        assert not outcome.built and "another build is running" in outcome.reason

    def test_fewer_than_two_vectors_build_nothing(self, stored, root):
        stored.add(points(1))
        assert not IndexGenerations(root).build(stored.source()).built


class TestHousekeeping:
    def test_a_crashed_build_is_swept_by_the_next(self, real_diskann, stored, root):
        stored.add(points(50))
        root.mkdir()
        (root / "gen-000007.building").mkdir()
        outcome = IndexGenerations(root).build(stored.source())
        assert outcome.built
        assert not (root / "gen-000007.building").exists()

    def test_only_the_current_generation_and_the_one_before_are_kept(self, real_diskann, stored, root):
        names = []
        for round_ in range(3):
            stored.add(points(20, seed=round_), prefix=f"r{round_}-")
            names.append(IndexGenerations(root).build(stored.source()).generation)
        kept = sorted(p.name for p in root.iterdir() if p.name.startswith("gen-"))
        assert kept == names[1:]
        assert IndexGenerations(root).manifest(names[2])["previous"] == names[1]

    def test_current_is_replaced_whole(self, real_diskann, stored, root):
        stored.add(points(20))
        IndexGenerations(root).build(stored.source())
        assert not (root / "CURRENT.tmp").exists()


class TestOpeningWhatIsOnDisk:
    def build_two(self, stored, root):
        stored.add(points(30))
        first = IndexGenerations(root).build(stored.source()).generation
        stored.add(points(30, seed=1), prefix="n")
        second = IndexGenerations(root).build(stored.source()).generation
        return first, second

    def test_nothing_built_opens_nothing(self, root):
        assert IndexGenerations(root).open(512 * MiB) is None

    def test_a_corrupt_current_falls_back_to_the_one_before(self, real_diskann, stored, root, caplog):
        first, second = self.build_two(stored, root)
        with open(root / second / "ann.data", "r+b") as data:
            data.truncate(10)
        with caplog.at_level("WARNING"):
            base = IndexGenerations(root).open(512 * MiB)
        assert base.name == first and base.count == 30
        assert any("wrong size" in r.getMessage() for r in caplog.records)

    def test_a_corrupt_current_with_nothing_before_opens_nothing(self, real_diskann, stored, root):
        stored.add(points(30))
        name = IndexGenerations(root).build(stored.source()).generation
        (root / name / "labels.npy").write_bytes(b"garbage")
        assert IndexGenerations(root).open(512 * MiB) is None

    def test_a_current_naming_nothing_opens_nothing(self, root):
        root.mkdir()
        (root / "CURRENT").write_text("gen-000042")
        assert IndexGenerations(root).open(512 * MiB) is None

    def test_another_models_generation_is_refused(self, real_diskann, stored, root, monkeypatch):
        stored.add(points(30))
        IndexGenerations(root).build(stored.source())
        monkeypatch.setattr(Config, "EMBEDDING_MODEL", "some/other-model")
        assert IndexGenerations(root).open(512 * MiB) is None

    def test_a_memory_index_is_not_loaded_into_memory_that_is_not_there(self, real_diskann, stored, root, monkeypatch, caplog):
        stored.add(points(300))
        IndexGenerations(root).build(stored.source())
        spare(monkeypatch, 0)
        with caplog.at_level("WARNING"):
            assert IndexGenerations(root).open(512 * MiB) is None
        assert any("are free" in r.getMessage() for r in caplog.records)

    def test_an_index_larger_than_the_budget_allows_is_not_loaded(self, real_diskann, stored, root, caplog):
        stored.add(points(300))
        IndexGenerations(root).build(stored.source())
        with caplog.at_level("WARNING"):
            assert IndexGenerations(root).open(1024) is None
        assert any("the budget leaves" in r.getMessage() for r in caplog.records)

    def test_a_disk_index_under_pressure_opens_with_a_smaller_cache(self, real_diskann, stored, root, monkeypatch, caplog):
        """The cache is the dial: less memory, slower search, same answers."""
        monkeypatch.setattr(memory_guard, "memory_index_bytes", lambda *a: 1 << 40)
        data = points(300)
        labels = stored.add(data)
        name = IndexGenerations(root).build(stored.source()).generation
        manifest = IndexGenerations(root).manifest(name)
        pq = sum(manifest["files"][f] for f in index_generations.PQ_FILES) + index_generations.PQ_OVERHEAD
        monkeypatch.setattr(memory_guard, "spare_memory", lambda: pq + 1)
        with caplog.at_level("INFO"):
            base = IndexGenerations(root).open(512 * MiB)
        assert any(", 0 cached" in r.getMessage() for r in caplog.records)
        assert base.search(data[9], 1, 64)[0].tolist() == [labels[9]]


class TestVectorSearchDegrades:
    """The retrieval side: whatever is missing, the dense half shrinks and
    keyword search carries on."""

    def search(self, tmp_path):
        from retrieval_layer.vector_search import VectorSearch

        return VectorSearch(chunk_store_path=tmp_path / "chunks", index_path=tmp_path / "index")

    def plan(self, vector):
        from retrieval_layer.models import QueryPlan

        return QueryPlan("q", vector, "q")

    def test_no_memory_for_the_recent_index_turns_vector_search_off(self, stored, tmp_path, monkeypatch, caplog):
        stored.add(points(5))
        monkeypatch.setattr(memory_guard, "can_hold", lambda size: False)
        search = self.search(tmp_path)
        with caplog.at_level("WARNING"):
            assert search.search(self.plan(points(1)[0]), 3) == []
        assert any("Vector search is off" in r.getMessage() for r in caplog.records)

    def test_refresh_turns_it_back_on_once_memory_frees(self, real_diskann, stored, tmp_path, monkeypatch):
        data = points(5)
        labels = stored.add(data)
        search = self.search(tmp_path)
        with pytest.MonkeyPatch.context() as pressure:
            pressure.setattr(memory_guard, "can_hold", lambda size: False)
            assert search.search(self.plan(data[2]), 1) == []
        search.catch_up()
        assert search.search(self.plan(data[2]), 1)[0].vector_id == labels[2]

    def test_with_postgresql_down_the_built_index_still_answers(self, real_diskann, stored, tmp_path, chunk_vectors, caplog):
        """The built index is files on disk; only the recent vectors need PostgreSQL."""
        data = points(70)
        labels = stored.add(data[:60])
        IndexGenerations(tmp_path / "index").build(stored.source())
        stored.add(data[60:], prefix="later")
        chunk_vectors.down = True
        search = self.search(tmp_path)
        with caplog.at_level("WARNING"):
            hit = search.search(self.plan(data[40]), 1)
        assert hit[0].vector_id == labels[40]
        assert search.index.recent_count() == 0
        assert any("could not be loaded" in r.getMessage() for r in caplog.records)

    def test_with_no_store_at_all_the_built_index_still_answers(self, real_diskann, stored, tmp_path, monkeypatch):
        from data_layer.vector_db_manager import stored_vectors

        data = points(70)
        labels = stored.add(data[:60])
        IndexGenerations(tmp_path / "index").build(stored.source())
        stored.add(data[60:], prefix="later")

        def refuse():
            raise OSError("connection refused")

        monkeypatch.setattr(stored_vectors, "chunk_vector_store", refuse)
        hit = self.search(tmp_path).search(self.plan(data[40]), 1)
        assert hit[0].vector_id == labels[40]

    def test_vectors_ingested_while_postgresql_was_down_arrive_on_refresh(self, real_diskann, stored, tmp_path, chunk_vectors):
        data = points(70)
        stored.add(data[:60])
        IndexGenerations(tmp_path / "index").build(stored.source())
        later = stored.add(data[60:], prefix="later")
        chunk_vectors.down = True
        search = self.search(tmp_path)
        search.search(self.plan(data[0]), 1)
        assert search.index.recent_count() == 0
        chunk_vectors.down = False
        search.catch_up()
        assert search.search(self.plan(data[65]), 1)[0].vector_id == later[5]

    def test_a_new_generation_replaces_the_old_on_refresh(self, real_diskann, stored, tmp_path):
        data = points(80)
        stored.add(data[:50])
        root = tmp_path / "index"
        first = IndexGenerations(root).build(stored.source()).generation
        search = self.search(tmp_path)
        search.search(self.plan(data[0]), 1)
        assert search.generation == first

        later = stored.add(data[50:], prefix="later")
        search.catch_up()
        assert search.index.recent_count() == 30
        second = IndexGenerations(root).build(stored.source()).generation
        search.catch_up()
        assert search.generation == second
        assert search.index.recent_count() == 0
        assert search.search(self.plan(data[70]), 1)[0].vector_id == later[20]

    def test_a_generation_that_will_not_open_is_retried_on_refresh(self, real_diskann, stored, tmp_path, monkeypatch):
        data = points(60)
        labels = stored.add(data)
        IndexGenerations(tmp_path / "index").build(stored.source())
        search = self.search(tmp_path)
        with pytest.MonkeyPatch.context() as pressure:
            pressure.setattr(memory_guard, "LOADED_FACTOR", 1e9)
            search.search(self.plan(data[0]), 1)
        assert search.generation is None
        search.catch_up()
        assert search.generation is not None
        assert search.search(self.plan(data[33]), 1)[0].vector_id == labels[33]

    def test_a_fallback_generation_is_not_reopened_on_every_refresh(self, real_diskann, stored, tmp_path, monkeypatch):
        stored.add(points(30))
        root = tmp_path / "index"
        first = IndexGenerations(root).build(stored.source()).generation
        stored.add(points(30, seed=1), prefix="n")
        second = IndexGenerations(root).build(stored.source()).generation
        (root / second / "labels.npy").write_bytes(b"garbage")
        search = self.search(tmp_path)
        search.search(self.plan(points(1)[0]), 1)
        assert search.generation == first
        opened = []
        real_open = IndexGenerations.open
        monkeypatch.setattr(IndexGenerations, "open", lambda self, a: opened.append(1) or real_open(self, a))
        search.catch_up()
        assert opened == []
