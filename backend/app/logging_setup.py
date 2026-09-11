"""Application-wide logging.

Two entry points, and the split between them is the whole design.

`get_logger(__name__)` is what every module calls. It returns a bare logger:
no handlers, no files opened, no global state touched. Importing a module
therefore stays free of side effects — the helper this replaces created `log/`
and opened a file handle at import time, so merely importing `config` in a test
wrote to disk.

`configure()` is what the process entry point calls, once. It installs the
handlers on the *root* logger and every module logger reaches them by
propagation. That is what makes a single `--verbose` flag work everywhere:
raising the root level raises it for data_layer, memory and cli at the same
time. The previous arrangement gave each module its own handlers with
`propagate = False`, so the flag reached only the one logger it was applied to
and every other module stayed at INFO.

Nothing here imports from this project. config.py re-exports `get_logger`, and
a module that imported config in order to log would otherwise close a cycle.
"""

from __future__ import annotations

import contextvars
import json
import logging
import logging.handlers
import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping

DEFAULT_LOG_FILE = "log/app.log"
DEFAULT_LEVEL = "INFO"

# The console is a different audience from the file. A person running the CLI
# wants their own output back, not a running commentary, so the terminal gets
# warnings and worse while the file keeps everything.
DEFAULT_CONSOLE_LEVEL = "WARNING"

DEFAULT_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 5

TEXT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d |%(context)s %(message)s"
TIME_FORMAT = "%Y-%m-%dT%H:%M:%S%z"

# log_timing logs from inside a context manager, so the frame the logging
# module would credit is this file. Three frames up is the caller's `with`.
_TIMING_STACKLEVEL = 3

# Model loading and HTTP clients narrate themselves at INFO and DEBUG. Left
# alone they bury this project's own output the moment --verbose is passed,
# which is exactly when someone is trying to read it.
QUIET_LIBRARIES = (
    "asyncio",
    "chardet",
    "charset_normalizer",
    "filelock",
    "fsspec",
    "httpcore",
    "httpx",
    "huggingface_hub",
    "matplotlib",
    "numexpr",
    "PIL",
    "sentence_transformers",
    "torch",
    "transformers",
    "urllib3",
)

_context: contextvars.ContextVar[Mapping[str, Any]] = contextvars.ContextVar(
    "log_context", default={}
)

_configure_lock = threading.RLock()
_configured = False

# Everything the logging module itself puts on a record. Anything outside this
# set arrived through `extra=` and belongs in the structured output.
_RESERVED = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", (), None))
) | {"asctime", "message", "context", "context_fields", "taskName"}


def get_logger(name: str) -> logging.Logger:
    """The logger for one module. Call it as `get_logger(__name__)`.

    Deliberately a thin wrapper: the dotted module name is what gives the log
    its hierarchy, and handlers belong to `configure()`, not here.
    """
    return logging.getLogger(name)


class _ContextFilter(logging.Filter):
    """Stamps the ambient context fields onto every record a handler sees.

    On the handler rather than on a logger, because logger-level filters do not
    run for records that arrive by propagation — and in this project every
    record arrives that way. Doing it in a filter rather than inside the
    formatter also keeps the fields correct if a handler ever defers formatting
    to another thread.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        fields: Dict[str, Any] = dict(_context.get())
        if record.threadName and record.threadName != "MainThread":
            fields.setdefault("thread", record.threadName)
        record.context_fields = fields
        pairs = " ".join(f"{key}={value}" for key, value in fields.items())
        record.context = f" [{pairs}]" if pairs else ""
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line, for when logs are shipped rather than read.

    Context fields and anything passed through `extra=` are merged in at the top
    level so a log search can filter on them directly.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "time": self.formatTime(record, TIME_FORMAT),
            "level": record.levelname,
            "logger": record.name,
            "line": f"{record.filename}:{record.lineno}",
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "context_fields", {}))
        payload.update(
            {
                key: value
                for key, value in vars(record).items()
                if key not in _RESERVED
            }
        )
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _env(name: str, fallback):
    value = os.getenv(name)
    return fallback if value is None else value


def _level(value, fallback: int) -> int:
    """Accept a level as either a name or a number; fall back rather than raise.

    A typo in LOG_LEVEL must not stop the process from starting — logging is
    diagnostics, and taking the application down to report a bad diagnostic
    setting trades a small problem for a total one.
    """
    if value is None:
        return fallback
    if isinstance(value, int):
        return value
    resolved = logging.getLevelName(str(value).strip().upper())
    return resolved if isinstance(resolved, int) else fallback


def configure(
    *,
    level=None,
    log_file=None,
    console: bool = True,
    console_level=None,
    json_lines: bool | None = None,
    max_bytes: int | None = None,
    backup_count: int | None = None,
    quiet_libraries: bool = True,
    capture_warnings: bool = True,
    force: bool = False,
) -> logging.Logger:
    """Install the handlers. Call once, from the process entry point.

    Repeat calls are ignored rather than stacking a second set of handlers,
    which is what duplicates every line in the log. Pass force=True to replace
    the existing configuration instead — tests do, entry points should not.

    Every argument falls back to an environment variable and then to a module
    default, so a deployment can retune logging without a code change:
    LOG_LEVEL, LOG_FILE, LOG_CONSOLE_LEVEL, LOG_FORMAT (text|json),
    LOG_MAX_BYTES, LOG_BACKUP_COUNT.
    """
    global _configured

    with _configure_lock:
        if _configured and not force:
            return logging.getLogger()
        if _configured:
            reset()

        file_level = _level(_env("LOG_LEVEL", level), _level(DEFAULT_LEVEL, logging.INFO))
        console_lvl = _level(
            _env("LOG_CONSOLE_LEVEL", console_level),
            _level(DEFAULT_CONSOLE_LEVEL, logging.WARNING),
        )
        if json_lines is None:
            json_lines = str(_env("LOG_FORMAT", "text")).lower() == "json"

        formatter: logging.Formatter = (
            JsonFormatter() if json_lines else logging.Formatter(TEXT_FORMAT, TIME_FORMAT)
        )

        root = logging.getLogger()
        # The root logger gates before any handler is consulted, so it has to
        # sit at the most permissive level of the two; each handler then filters
        # down to its own audience.
        root.setLevel(min(file_level, console_lvl) if console else file_level)

        path = Path(_env("LOG_FILE", log_file) or DEFAULT_LOG_FILE)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path

        handler = _file_handler(
            path,
            file_level,
            formatter,
            int(_env("LOG_MAX_BYTES", max_bytes) or DEFAULT_MAX_BYTES),
            int(_env("LOG_BACKUP_COUNT", backup_count) or DEFAULT_BACKUP_COUNT),
        )
        if handler is not None:
            root.addHandler(handler)

        if console:
            stream = logging.StreamHandler()
            stream.setLevel(console_lvl)
            stream.setFormatter(formatter)
            stream.addFilter(_ContextFilter())
            root.addHandler(stream)

        if quiet_libraries:
            for name in QUIET_LIBRARIES:
                logging.getLogger(name).setLevel(logging.WARNING)

        # warnings.warn() otherwise goes to stderr unformatted and never reaches
        # the file, so a DeprecationWarning from a long batch run is lost.
        logging.captureWarnings(capture_warnings)

        _configured = True
        return root


def _file_handler(path, level, formatter, max_bytes, backup_count):
    """A rotating file handler, or None if the file cannot be opened.

    Rotation because the previous configuration appended to one file forever.
    None rather than an exception because an unwritable log directory — a
    read-only mount, a container running as another uid — is not a reason for
    the application to refuse to start; the console handler still reports.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
            # Nothing is created until something is actually logged, so
            # configure() on its own leaves no file behind.
            delay=True,
        )
    except OSError as error:
        logging.getLogger(__name__).warning(
            "File logging disabled: cannot open %s (%s)", path, error
        )
        return None
    handler.setLevel(level)
    handler.setFormatter(formatter)
    handler.addFilter(_ContextFilter())
    return handler


def reset() -> None:
    """Remove and close the handlers configure() installed.

    Tests need it between configurations; leaving the handlers attached would
    leak a file descriptor per run and keep writing to a temporary directory
    that has already been deleted.
    """
    global _configured

    with _configure_lock:
        root = logging.getLogger()
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
        logging.captureWarnings(False)
        _configured = False


def is_configured() -> bool:
    return _configured


def set_level(level, console_level=None) -> None:
    """Retune levels after configure(), which is what --verbose does.

    Reaches the handlers as well as the root logger: the root level only decides
    what is offered to a handler, and a handler pinned at WARNING would drop
    debug records that the root had just started admitting.
    """
    resolved = _level(level, logging.INFO)
    root = logging.getLogger()
    root.setLevel(resolved)
    for handler in root.handlers:
        # RotatingFileHandler is itself a StreamHandler, so the file case has to
        # be excluded first or every handler looks like the console.
        is_console = not isinstance(handler, logging.FileHandler)
        if is_console and console_level is not None:
            handler.setLevel(_level(console_level, resolved))
        else:
            handler.setLevel(resolved)


@contextmanager
def log_context(**fields: Any) -> Iterator[Dict[str, Any]]:
    """Attach fields to every record logged inside the block.

    What makes a concurrent pipeline readable: bind a query id once at the top
    and every line from every stage carries it, so one request can be pulled out
    of an interleaved log without threading an id through each call signature.

    Nesting layers fields onto whatever is already bound. The binding is a
    contextvar, so it follows async tasks but *not* threads — a worker started
    with Thread or submitted to an executor begins empty, and has to bind its
    own context or be launched through `contextvars.copy_context().run(...)`.
    """
    merged = {**_context.get(), **fields}
    token = _context.set(merged)
    try:
        yield dict(merged)
    finally:
        _context.reset(token)


def current_context() -> Dict[str, Any]:
    return dict(_context.get())


def new_correlation_id() -> str:
    """A short id for one unit of work, when the caller has nothing better."""
    return uuid.uuid4().hex[:12]


@contextmanager
def log_timing(
    logger: logging.Logger, operation: str, level: int = logging.INFO, **fields: Any
) -> Iterator[None]:
    """Time a block and log how long it took, whether or not it succeeded.

    The duration also goes out as a `duration_ms` field, so the JSON handler
    emits something aggregatable rather than a number embedded in prose.
    """
    started = time.perf_counter()
    try:
        yield
    except BaseException:
        elapsed = (time.perf_counter() - started) * 1000
        logger.warning(
            "%s failed after %.0f ms",
            operation,
            elapsed,
            extra={"duration_ms": round(elapsed, 3), **fields},
            stacklevel=_TIMING_STACKLEVEL,
        )
        raise
    else:
        elapsed = (time.perf_counter() - started) * 1000
        logger.log(
            level,
            "%s took %.0f ms",
            operation,
            elapsed,
            extra={"duration_ms": round(elapsed, 3), **fields},
            stacklevel=_TIMING_STACKLEVEL,
        )


# The name config.py and main.py import it under. `configure` reads well inside
# this module and ambiguously outside it, where several things get configured.
configure_logging = configure
