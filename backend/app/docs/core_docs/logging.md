# Logging (`logging_setup.py`)

## Overview & Purpose

One module owns logging for the whole application. Every other module calls
`get_logger(__name__)` and nothing else; the process entry point calls
`configure_logging()` once. That split is the design, and everything below
follows from it.

`config.py` re-exports `get_logger`, `configure_logging`, `log_context`,
`log_timing` and `new_correlation_id`, so `from config import get_logger` —
which most modules already used — keeps working. The implementation cannot live
in `config.py`: `logging_setup` must not import `config`, or a module importing
`config` in order to log would close an import cycle. The dependency runs one
way only, `config -> logging_setup`.

---

## The two entry points

### `get_logger(name: str) -> logging.Logger`

Returns a bare `logging.Logger`. No handlers, no files opened, no global state
touched. It is a thin wrapper over `logging.getLogger`, and that is deliberate:
the dotted module name is what gives the log its hierarchy, and handlers belong
to `configure_logging()`.

```python
from config import get_logger

logger = get_logger(__name__)
```

Always `__name__`. A hard-coded name breaks the hierarchy that levels are
applied through, and a test enforces it.

### `configure_logging(**options) -> logging.Logger`

Installs the handlers on the **root** logger. Called once, from `main.py`.
Repeat calls are ignored rather than stacking a second set of handlers — which
is what makes every line in the log appear twice. `force=True` replaces the
configuration instead; tests use it, entry points should not.

| Option | Env var | Default | Meaning |
| :--- | :--- | :--- | :--- |
| `level` | `LOG_LEVEL` | `INFO` | Level for the file handler. |
| `log_file` | `LOG_FILE` | `log/app.log` | Relative paths resolve against `backend/app`. |
| `console` | — | `True` | Whether to attach a console handler at all. |
| `console_level` | `LOG_CONSOLE_LEVEL` | `WARNING` | Level for the console handler. |
| `json_lines` | `LOG_FORMAT=json` | `False` | One JSON object per line instead of text. |
| `max_bytes` | `LOG_MAX_BYTES` | 10 MB | Rotate the file at this size. |
| `backup_count` | `LOG_BACKUP_COUNT` | 5 | How many rotated files to keep. |
| `quiet_libraries` | — | `True` | Pin known-chatty third-party loggers to `WARNING`. |
| `capture_warnings` | — | `True` | Route `warnings.warn` into the log. |

The environment variable wins over the argument, so a deployment can retune
logging without a code change.

---

## Why handlers go on the root logger

Because that is what makes one `--verbose` flag work.

Module loggers have no handlers of their own; they propagate to the root, so
raising the root level raises it for `data_layer`, `memory` and `cli` at the
same time. The arrangement this replaced gave each module its own file and
stream handlers with `propagate = False`, and had three consequences:

- **`--verbose` did nothing.** `main.py` set `DEBUG` on the one logger it held.
  Every other module kept its own handlers pinned at `INFO`, so no debug output
  from the data or memory layers ever appeared. This is the defect the flag
  existed to provide and never did.
- **One open file descriptor per module.** Around thirty handles on the same
  file, all appending.
- **Importing a module wrote to disk.** `get_logger` was called at module scope
  and created `log/` and opened a `FileHandler` as a side effect, so importing
  `config` in a test left a log file behind whether or not anything was logged.

The root logger also gates records before any handler is consulted, so
`configure_logging` sets it to the *lower* of the file and console levels and
lets each handler filter down to its own audience. Pinned to the file level
instead, `LOG_LEVEL=WARNING` with `--verbose` would hand the console nothing
below `WARNING` and the flag would look broken again.

---

## Console and file are different audiences

The console defaults to `WARNING`, the file to `INFO`. Someone running the CLI
wants their own output back, not a running commentary interleaved with the
input prompt; the file keeps the detail for afterwards. `--verbose` lowers both
to `DEBUG`.

Model loading and HTTP clients narrate themselves at `INFO` and `DEBUG`, so
`QUIET_LIBRARIES` pins `sentence_transformers`, `transformers`, `torch`,
`urllib3`, `huggingface_hub` and others to `WARNING`. Without it, `--verbose`
buries the application's own output at exactly the moment someone is trying to
read it. Warnings and errors from those libraries still come through.

---

## Correlation: `log_context`

```python
from config import log_context, new_correlation_id

with log_context(query_id=new_correlation_id()):
    logger.info("Received user query: %r", query)
    ...
```

Every record logged inside the block carries the fields:

```
2026-09-09T21:35:28+0530 | INFO | cli.cli_interface:20 | [query_id=9f91dddf49a7] Received user query: 'what is the retrieval layer?'
```

This is what makes a concurrent pipeline readable. Bind a query id once at the
top and every stage's lines carry it, so one request can be pulled out of a log
that the planner's master, slave and critique nodes are all writing to — without
threading an id through every call signature. Nesting layers fields onto
whatever is already bound.

The binding is a `contextvars.ContextVar`, so it follows `async` tasks but
**not** threads. A worker started with `Thread` or submitted to a
`ThreadPoolExecutor` begins with an empty context and has to bind its own, or be
launched through `contextvars.copy_context().run(...)`. Records from a
non-main thread do carry `thread=<name>` automatically.

---

## Timing: `log_timing`

```python
with log_timing(logger, "embedding", chunks=len(chunks)):
    return self.embedder.embed(chunks)
```

Logs the duration whether the block succeeds or raises, and re-raises on
failure. The duration also goes out as a structured `duration_ms` field, so the
JSON handler emits something aggregatable rather than a number buried in prose.

`IngestionPipeline` wraps its five expensive stages in it, which is the only
place the cost of a whole ingestion run is visible in one log:

```
INFO | ...EmbeddingManager:31 | loading embedding model took 7439 ms
```

Records are credited to the caller's `with` statement, not to `logging_setup.py`
— without that every timing line in the application would point at the same
place.

---

## JSON output

`--log-json`, or `LOG_FORMAT=json`, emits one object per line. Context fields
and anything passed through `extra=` are merged in at the top level so a log
search can filter on them directly.

```json
{"time": "2026-09-09T21:35:28+0530", "level": "INFO", "logger": "backend_main", "line": "main.py:60", "message": "Executing single query: 'third query'", "query_id": "b3c5f4dd6cf9"}
```

Text is the default and is what a person reads. JSON is for when the logs are
shipped somewhere — the shape a FastAPI deployment will want.

---

## Conventions the tests enforce

`test/logging_testing/test_logging_setup.py` holds four structural guards over
the source tree, in the same spirit as
`test_every_conversation_connection_goes_through_connect`. They exist because
these defects come back one file at a time and never fail a behavioural test.

| Guard | Why |
| :--- | :--- |
| No module calls `basicConfig`, `addHandler` or `setLevel` | Takes that module out of the single configuration and back to the old arrangement. |
| Every logger is `get_logger(__name__)` | A hard-coded name breaks the hierarchy levels are applied through. |
| No log call uses an f-string | An f-string is built whether or not the level is enabled. Several of these sat in per-file and per-vector loops. Use `logger.info("took %s", n)`, never `logger.info(f"took {n}")`. |
| No log call builds its message by calling a method | `snapshot_now()` logged a count it fetched with a second query, so the message cost a database read on every snapshot regardless of level. A log line must not do work the caller did not ask for. |

---

## Failure behaviour

Logging is diagnostics. Taking the application down to report a bad diagnostic
setting trades a small problem for a total one, so:

- An unwritable log path (read-only mount, a container running as another uid)
  disables the file handler and leaves the console working. It does not raise.
- An unrecognised `LOG_LEVEL` falls back to the default rather than raising.
- A value the JSON encoder cannot handle is coerced rather than dropping the
  line.
- `enable_wal` failing to convert the conversation database now logs a warning.
  It used to return the fallback mode silently, so a database running in a
  journal mode where reads block writes said nothing about it.

---

## Files on disk

`log/app.log` rotates at 10 MB, keeping `app.log.1` through `app.log.5`. The
previous configuration appended to one file forever. Both patterns are
gitignored — note that `*.log` does not match the rotated `app.log.1`, so
`*.log.[0-9]` is listed separately in `backend/.gitignore`.

The file is not created until something is actually logged, so
`configure_logging()` on its own leaves nothing behind.
