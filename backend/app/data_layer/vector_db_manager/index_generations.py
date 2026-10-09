"""Built index generations on disk, and which one is in use."""

import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, NamedTuple, Optional, Tuple

import numpy as np
from numpy import float32, uint32

from config import Config, get_logger
from data_layer.datalayer_exceptions.datalayer_exceptions import IndexGenerationUnusable
from data_layer.vector_db_manager import memory_guard
from data_layer.vector_db_manager.memory_guard import GiB, MiB

logger = get_logger(__name__)

MEMORY, DISK = "memory", "disk"
PREFIX = "ann"
CURRENT = "CURRENT"
MANIFEST = "manifest.json"
LABELS = "labels.npy"
VECTORS = "vectors.bin"
BUILD_LOG = "build.log"
LOCK = ".build.lock"
BUILDING = ".building"
NAME = re.compile(r"^gen-(\d{6})$")
# Not needed to search, and each is a full copy of the vectors.
BUILD_ONLY = (VECTORS, f"{PREFIX}_mem.index.data", BUILD_LOG)
MEMORY_FILES = (PREFIX, f"{PREFIX}.data")
PQ_FILES = (f"{PREFIX}_pq_compressed.bin", f"{PREFIX}_pq_pivots.bin")
PQ_OVERHEAD = 32 * MiB
# Wider reads per hop: measured 2.45 ms against 2.63 ms at width 2, same recall.
BEAM_WIDTH = 4
APP_ROOT = Path(__file__).resolve().parents[2]
# A build that hangs is stopped rather than holding ingestion forever.
SHORTEST_BUILD_WAIT = 900
SECONDS_PER_VECTOR = 0.005


def builder_command(spec: dict) -> List[str]:
    """The process that builds; a module run fresh, never a copy of the caller."""
    return [sys.executable, "-m", "data_layer.vector_db_manager.index_build", json.dumps(spec)]


def search_threads() -> int:
    # A disk index searched with one thread never returns (README.md).
    return max(2, int(Config.NUM_THREADS))


class BuildOutcome(NamedTuple):
    built: bool
    reason: str
    generation: Optional[str] = None
    kind: Optional[str] = None
    count: int = 0


class BuiltIndex:
    """One generation, opened for search, answering in labels."""

    def __init__(self, name: str, kind: str, index, labels: np.ndarray, through: int) -> None:
        self.name = name
        self.kind = kind
        self.index = index
        self.labels = labels
        self.through = through

    @property
    def count(self) -> int:
        return len(self.labels)

    def search(self, query, k: int, complexity: int) -> Tuple[np.ndarray, np.ndarray]:
        # Never more than it holds: past that a disk index pads with position 0.
        k = min(int(k), self.count)
        if k < 1:
            return np.empty(0, uint32), np.empty(0, float32)
        if self.kind == DISK:
            found = self.index.search(query, k, max(complexity, k), beam_width=BEAM_WIDTH)
        else:
            found = self.index.search(query, k, max(complexity, k))
        positions = np.asarray(found.identifiers, dtype=np.int64)
        distances = np.asarray(found.distances, dtype=float32)
        valid = (positions >= 0) & (positions < self.count)
        return self.labels[positions[valid]], distances[valid]


class IndexGenerations:
    """`gen-NNNNNN/` directories, a `CURRENT` file naming one, and the builds that make them."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root if root is not None else Config.INDEX_PATH)

    # Reading

    def current(self) -> Optional[str]:
        try:
            name = (self.root / CURRENT).read_text().strip()
        except OSError:
            return None
        return name if NAME.match(name) else None

    def manifest(self, name: Optional[str]) -> Optional[dict]:
        if name is None:
            return None
        try:
            return json.loads((self.root / name / MANIFEST).read_text())
        except (OSError, ValueError):
            return None

    def through(self) -> int:
        """The highest label the current generation was built through; 0 without one."""
        manifest = self.manifest(self.current())
        return int(manifest["through"]) if manifest else 0

    def open(self, allowance: int) -> Optional[BuiltIndex]:
        """The current generation, else the one before it, within `allowance` bytes; None if neither."""
        name, tried = self.current(), set()
        while name is not None and name not in tried:
            tried.add(name)
            try:
                return self.__open(name, allowance)
            except IndexGenerationUnusable as error:
                logger.warning("%s", error)
                manifest = self.manifest(name)
                name = manifest.get("previous") if manifest else None
        return None

    def __open(self, name: str, allowance: int) -> BuiltIndex:
        directory = self.root / name
        manifest = self.manifest(name)
        if manifest is None:
            raise IndexGenerationUnusable(name, "its manifest is missing or unreadable")
        expected = {
            "model": Config.EMBEDDING_MODEL,
            "dimensions": Config.EMBEDDING_DIMENSIONS,
            "metric": Config.DISTANCE_METRIC,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise IndexGenerationUnusable(
                    name, f"built for {key} {manifest.get(key)!r}, not {value!r}"
                )
        for file, size in manifest["files"].items():
            path = directory / file
            if not path.is_file() or path.stat().st_size != size:
                raise IndexGenerationUnusable(name, f"{file} is missing or the wrong size")
        try:
            labels = np.load(directory / LABELS)
        except (OSError, ValueError) as error:
            raise IndexGenerationUnusable(name, f"its labels could not be read: {error}") from error
        if labels.dtype != uint32 or len(labels) != manifest["count"]:
            raise IndexGenerationUnusable(name, "its labels do not match its manifest")

        import diskannpy

        try:
            if manifest["kind"] == MEMORY:
                need = int(sum(manifest["files"][f] for f in MEMORY_FILES)
                           * memory_guard.LOADED_FACTOR)
                self.__admit(name, need, allowance)
                index = diskannpy.StaticMemoryIndex(
                    str(directory), num_threads=search_threads(),
                    initial_search_complexity=Config.COMPLEXITY, index_prefix=PREFIX,
                )
                nodes = None
            else:
                need = sum(manifest["files"][f] for f in PQ_FILES) + PQ_OVERHEAD
                self.__admit(name, need, allowance)
                room = allowance - need
                spare = memory_guard.spare_memory()
                if spare is not None:
                    room = min(room, spare - need)
                node = memory_guard.cache_node_bytes(Config.EMBEDDING_DIMENSIONS, manifest["degree"])
                nodes = int(min(max(room // node, 0), len(labels)))
                index = diskannpy.StaticDiskIndex(
                    str(directory), num_threads=search_threads(),
                    num_nodes_to_cache=nodes, index_prefix=PREFIX,
                )
        except IndexGenerationUnusable:
            raise
        except Exception as error:
            raise IndexGenerationUnusable(name, f"it could not be opened: {error}") from error
        logger.info(
            "Opened index %s (%s, %d vectors%s)",
            name, manifest["kind"], len(labels),
            "" if nodes is None else f", {nodes} cached",
        )
        return BuiltIndex(name, manifest["kind"], index, labels, int(manifest["through"]))

    @staticmethod
    def __admit(name: str, need: int, allowance: int) -> None:
        if need > allowance:
            raise IndexGenerationUnusable(
                name, f"it needs {need // MiB} MB and the budget leaves {allowance // MiB} MB"
            )
        if not memory_guard.can_hold(need):
            spare = memory_guard.spare_memory() or 0
            raise IndexGenerationUnusable(
                name, f"it needs {need // MiB} MB and only {spare // MiB} MB are free"
            )

    # Building

    def build(self, source) -> BuildOutcome:
        """Build a generation from `source` and make it current; never raises."""
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with self.__exclusive() as held:
                if not held:
                    return self.__skipped("another build is running")
                return self.__build(source)
        except Exception as error:
            return self.__skipped(f"{type(error).__name__}: {error}")

    @staticmethod
    def __skipped(reason: str) -> BuildOutcome:
        logger.warning("Index not rebuilt: %s. The previous index stays in use", reason)
        return BuildOutcome(False, reason)

    def __build(self, source) -> BuildOutcome:
        count = source.pending(0)
        if count < 2:
            return BuildOutcome(False, "fewer than two vectors; an index needs a graph")
        plan = self.__plan(count)
        if isinstance(plan, str):
            return self.__skipped(plan)
        kind, pq_gb, build_gb = plan
        dims, degree = Config.EMBEDDING_DIMENSIONS, Config.GRAPH_DEGREE
        needed = memory_guard.index_disk_bytes(count, dims, degree)
        free = memory_guard.free_disk(self.root)
        if free < needed:
            return self.__skipped(
                f"it needs {needed // MiB} MB of disk and {free // MiB} MB are free"
            )

        previous = self.current()
        self.__sweep(keep={previous, self.__previous_of(previous)})
        name = self.__next_name()
        building = self.root / f"{name}{BUILDING}"
        building.mkdir()
        try:
            written, through = self.__write_vectors(source, building)
            if written < 2:
                return self.__skipped(f"only {written} stored vector(s) to build from")
            reason = self.__run_builder(kind, pq_gb, build_gb, building, written)
            if reason is not None:
                return self.__skipped(reason)
            for file in BUILD_ONLY:
                (building / file).unlink(missing_ok=True)
            files = {f.name: f.stat().st_size for f in building.iterdir() if f.name != LABELS}
            manifest = {
                "kind": kind, "count": written, "through": through,
                "model": Config.EMBEDDING_MODEL, "dimensions": dims,
                "metric": Config.DISTANCE_METRIC, "degree": degree,
                "previous": previous, "files": files,
            }
            manifest["files"][LABELS] = (building / LABELS).stat().st_size
            self.__write_durably(building / MANIFEST, json.dumps(manifest, indent=1))
            for file in building.iterdir():
                self.__fsync(file)
            os.rename(building, self.root / name)
            self.__write_durably(self.root / CURRENT, name)
            self.__sweep(keep={name, previous})
        finally:
            shutil.rmtree(building, ignore_errors=True)
        logger.info("Built index %s (%s, %d vectors through label %d)", name, kind, written, through)
        return BuildOutcome(True, "built", name, kind, written)

    def __plan(self, count: int):
        """`(kind, pq_gb, build_gb)`, or why nothing can be built now."""
        dims, degree = Config.EMBEDDING_DIMENSIONS, Config.GRAPH_DEGREE
        allowance = memory_guard.budget() - memory_guard.recent_index_bytes(
            Config.RECENT_VECTOR_CAPACITY, dims, degree
        )
        if allowance <= PQ_OVERHEAD:
            return f"VECTOR_INDEX_RAM_MB leaves {max(allowance, 0) // MiB} MB for a built index"
        if memory_guard.memory_index_bytes(count, dims, degree) <= allowance and \
                memory_guard.can_hold(memory_guard.memory_build_bytes(count, dims, degree)):
            return MEMORY, 0.0, 0.0
        build = memory_guard.disk_build_budget()
        if build is None:
            spare = memory_guard.spare_memory() or 0
            return f"only {spare // MiB} MB are free, too little for even a small disk build"
        return DISK, allowance / 4 / GiB, build / GiB

    @staticmethod
    def __write_vectors(source, directory: Path) -> Tuple[int, int]:
        """DiskANN's vector file, written a page at a time, and the labels in the same order."""
        dims = Config.EMBEDDING_DIMENSIONS
        labels, written, through = [], 0, 0
        with open(directory / VECTORS, "wb") as out:
            out.write(np.array([0, dims], dtype=np.int32).tobytes())
            for page in source.pages(0):
                through = page.through
                if len(page.labels):
                    out.write(np.ascontiguousarray(page.vectors, dtype=float32).tobytes())
                    labels.append(page.labels)
                    written += len(page.labels)
            out.seek(0)
            out.write(np.array([written, dims], dtype=np.int32).tobytes())
        np.save(directory / LABELS, np.concatenate(labels) if labels else np.empty(0, uint32))
        if source.missing:
            logger.warning(
                "%d label(s) have no stored vector and are left out of the index "
                "until their documents are ingested again",
                source.missing,
            )
        return written, through

    @staticmethod
    def __run_builder(kind: str, pq_gb: float, build_gb: float, directory: Path, count: int) -> Optional[str]:
        """None once built, else why not. The build runs in a process the kernel kills first."""
        spec = {
            "kind": kind, "vectors": str(directory / VECTORS), "directory": str(directory),
            "metric": Config.DISTANCE_METRIC, "complexity": Config.COMPLEXITY,
            "degree": Config.GRAPH_DEGREE, "threads": search_threads(),
            "pq_gb": pq_gb, "build_gb": build_gb,
        }
        log = directory / BUILD_LOG
        allowed = max(SHORTEST_BUILD_WAIT, count * SECONDS_PER_VECTOR)
        with open(log, "wb") as out:
            try:
                done = subprocess.run(
                    builder_command(spec), cwd=APP_ROOT, stdin=subprocess.DEVNULL,
                    stdout=out, stderr=out, timeout=allowed,
                )
            except subprocess.TimeoutExpired:
                return f"the build did not finish within {allowed:.0f} s and was stopped"
            except OSError as error:
                return f"the build process could not be started: {error}"
        if done.returncode < 0:
            try:
                name = signal.Signals(-done.returncode).name
            except ValueError:
                name = f"signal {-done.returncode}"
            likely = ", most likely by the OOM killer" if name == "SIGKILL" else ""
            return f"the build process was killed by {name}{likely}"
        if done.returncode > 0:
            tail = log.read_text(errors="replace").strip()[-500:]
            return f"the build failed: {tail}"
        return None

    # Housekeeping

    @contextmanager
    def __exclusive(self) -> Iterator[bool]:
        with open(self.root / LOCK, "w") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def __previous_of(self, name: Optional[str]) -> Optional[str]:
        manifest = self.manifest(name)
        return manifest.get("previous") if manifest else None

    def __next_name(self) -> str:
        numbers = [
            int(match.group(1))
            for entry in self.root.iterdir()
            if (match := NAME.match(entry.name.removesuffix(BUILDING)))
        ]
        return f"gen-{max(numbers, default=0) + 1:06d}"

    def __sweep(self, keep: set) -> None:
        """Remove generations other than `keep`, and builds a crash left behind."""
        for entry in self.root.iterdir():
            stale_build = entry.name.endswith(BUILDING)
            old = NAME.match(entry.name) and entry.name not in keep
            if entry.is_dir() and (stale_build or old):
                shutil.rmtree(entry, ignore_errors=True)

    def __write_durably(self, path: Path, text: str) -> None:
        """Replace `path` atomically: a crash leaves the old contents or the new, never half."""
        temporary = path.with_name(path.name + ".tmp")
        with open(temporary, "w") as out:
            out.write(text)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
        self.__fsync(path.parent)

    @staticmethod
    def __fsync(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
