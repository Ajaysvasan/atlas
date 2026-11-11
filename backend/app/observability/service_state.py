"""The single-instance lock a running Atlas holds, and the state it publishes for monitors in other processes."""

import dataclasses
import fcntl
import json
import os
import threading
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple

from config import Config, get_logger
from observability.telemetry import RequestRecord
from storage.timestamps import utc_now

logger = get_logger(__name__)

FORMAT = 1
LOCK_FILE = "atlas.lock"
STATE_FILE = "atlas.json"
KEEP_RECORDS = 20


class ServiceState(NamedTuple):
    running: bool
    # The last run wrote "running" and nobody holds the lock: it was killed or crashed.
    crashed: bool
    mode: str | None
    pid: int | None
    started_at: str | None
    updated_at: str | None
    stopped_at: str | None
    acquisition: bool | None
    facts: Dict[str, object] | None
    records: List[RequestRecord]


class AlreadyRunning(RuntimeError):
    def __init__(self, state: ServiceState | None) -> None:
        self.state = state
        if state is None or state.pid is None:
            detail = "another process holds the lock"
        else:
            detail = f"{state.mode}, pid {state.pid}, since {_local(state.started_at)}"
        super().__init__(f"Atlas is already running ({detail})")


def _local(stamp: str | None) -> str:
    try:
        return datetime.fromisoformat(stamp).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    except (TypeError, ValueError):
        return str(stamp)


def _directory(directory) -> Path:
    return Path(Config.RUN_DIR if directory is None else directory)


def record_to_dict(record: RequestRecord) -> Dict[str, object]:
    return dataclasses.asdict(record)


def record_from_dict(data: Dict[str, object]) -> RequestRecord:
    known = {f.name for f in dataclasses.fields(RequestRecord)}
    values = {k: v for k, v in data.items() if k in known}
    values["stages"] = OrderedDict(values.get("stages") or {})
    return RequestRecord(**values)


def _model_facts(shared) -> Dict[str, object]:
    calls = failures = 0
    if shared.loaded:
        stats = shared.stats()
        calls, failures = sum(stats.calls.values()), stats.failures
    return {
        "path": str(shared.spec.path),
        "n_ctx": shared.spec.n_ctx,
        "loaded": shared.loaded,
        "load_error": shared.load_error,
        "gpu_layers": getattr(shared, "gpu_layers", None),
        "offload_error": getattr(shared, "offload_error", None),
        "calls": calls,
        "failures": failures,
    }


def _any_gpu(runtime, embedder, models) -> bool:
    """Whether anything resident is on the GPU, so its name is worth asking torch for."""
    layers = [getattr(runtime.draft, "gpu_layers", None), getattr(runtime.main, "gpu_layers", None)]
    torch_devices = [getattr(embedder, "device", None), getattr(getattr(models, "reranker", None), "device", None)]
    return any(isinstance(n, int) and n != 0 for n in layers) or any(d not in (None, "cpu") for d in torch_devices)


def service_facts(components) -> Dict[str, object]:
    """What only the running process knows: what is loaded, and its counters. Never prompt text."""
    from inference.devices import gpu_name

    runtime = components.runtime
    models = getattr(components, "models", None)
    handle = components.retriever
    retriever = handle.retriever
    sessions = components.sessions
    embedder = getattr(models, "_MemoryModels__embedder", None) if models is not None else None
    return {
        "draft": _model_facts(runtime.draft),
        "main": _model_facts(runtime.main) if runtime.separate_main else None,
        "embedder_resident": embedder is not None,
        "devices": {
            "embedder": getattr(embedder, "device", None),
            "reranker": getattr(getattr(models, "reranker", None), "device", None),
            "gpu": gpu_name() if _any_gpu(runtime, embedder, models) else None,
        },
        "retriever": {
            "built": retriever is not None,
            "error": handle.error,
            "generation": None if retriever is None else retriever.vector_search.generation,
        },
        "kv": {
            "restores": sessions.restores,
            "misses": sessions.misses,
            "invalidations": sessions.invalidations,
            "save_failures": sessions.save_failures,
        },
        "acquisition": components.acquirer is not None,
    }


def _write_atomically(path: Path, payload: Dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _lock_held(lock: Path) -> bool:
    """Whether some process holds the service lock. Opens read-only, so it never creates the file."""
    try:
        descriptor = os.open(lock, os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)


def read_state(directory: str | Path | None = None) -> ServiceState | None:
    """The service's published state, or None if Atlas has never run here. Writes nothing."""
    directory = _directory(directory)
    held = _lock_held(directory / LOCK_FILE)
    try:
        data = json.loads((directory / STATE_FILE).read_text(encoding="utf-8"))
    except FileNotFoundError:
        data = None
    except (OSError, ValueError) as error:
        logger.warning("Unreadable service state %s: %s", directory / STATE_FILE, error)
        data = None
    if data is None or data.get("format") != FORMAT:
        if held:
            return ServiceState(True, False, "starting", None, None, None, None, None, None, [])
        return None
    published_running = bool(data.get("running"))
    records = []
    for entry in data.get("records") or []:
        try:
            records.append(record_from_dict(entry))
        except (TypeError, ValueError):
            continue
    if held and not published_running:
        # A new run holds the lock but has not published yet; what is on disk is the last run's.
        return ServiceState(True, False, "starting", None, None, None, None, None, None, [])
    return ServiceState(
        running=held,
        crashed=published_running and not held,
        mode=data.get("mode"),
        pid=data.get("pid"),
        started_at=data.get("started_at"),
        updated_at=data.get("updated_at"),
        stopped_at=data.get("stopped_at"),
        acquisition=data.get("acquisition"),
        facts=data.get("facts") if held else None,
        records=records,
    )


class ServiceLease:
    """Held by `start`, `serve` and `-q` for their whole run: one Atlas per data directory."""

    def __init__(self, mode: str, directory: str | Path | None = None) -> None:
        self.mode = mode
        self.directory = _directory(directory)
        self.components = None
        self.__descriptor: int | None = None
        self.__started_at: str | None = None
        self.__lock = threading.Lock()

    @property
    def held(self) -> bool:
        return self.__descriptor is not None

    def acquire(self) -> "ServiceLease":
        self.directory.mkdir(parents=True, exist_ok=True)
        os.chmod(self.directory, 0o700)
        descriptor = os.open(self.directory / LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            raise AlreadyRunning(read_state(self.directory)) from None
        self.__descriptor = descriptor
        self.__started_at = utc_now()
        self.__write(running=True)
        return self

    def attach(self, components) -> None:
        """Publish this process's components, and again after every request they record."""
        self.components = components
        components.telemetry.subscribe(lambda record: self.publish())
        self.publish()

    def publish(self) -> None:
        """Never raises: a monitor that cannot be told must not fail a query."""
        if not self.held:
            return
        try:
            self.__write(running=True)
        except Exception as error:
            logger.warning("Could not publish service state: %s", error)

    def release(self) -> None:
        if not self.held:
            return
        try:
            self.__write(running=False)
        except Exception as error:
            logger.warning("Could not record that the service stopped: %s", error)
        finally:
            descriptor, self.__descriptor = self.__descriptor, None
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def __write(self, running: bool) -> None:
        components = self.components
        with self.__lock:
            now = utc_now()
            records = [] if components is None else components.telemetry.records()[-KEEP_RECORDS:]
            payload = {
                "format": FORMAT,
                "running": running,
                "mode": self.mode,
                "pid": os.getpid(),
                "started_at": self.__started_at,
                "updated_at": now,
                "stopped_at": None if running else now,
                "acquisition": None if components is None else components.acquirer is not None,
                "facts": service_facts(components) if components is not None and running else None,
                "records": [record_to_dict(r) for r in records],
            }
            _write_atomically(self.directory / STATE_FILE, payload)

    def __enter__(self) -> "ServiceLease":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()
