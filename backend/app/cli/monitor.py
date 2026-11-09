"""`main.py status` and `main.py conversations`: read what a running Atlas published and exit. Never load a model."""

from typing import Callable

from cli import render
from config import get_logger

logger = get_logger(__name__)

# Exit codes, as service managers use them: 0 running, 3 not running.
RUNNING = 0
FAILED = 1
NOT_RUNNING = 3

PG_TIMEOUT_SECONDS = 3


def service_line(state) -> str:
    if state is None:
        return "Atlas service: not running (it has never been started here)."
    if state.running:
        if state.pid is None:
            return "Atlas service: starting."
        acquisition = {True: "on", False: "off"}.get(state.acquisition, "unknown")
        return (f"Atlas service: running ({state.mode}, pid {state.pid}, since "
                f"{render.timestamp(state.started_at)}; last update {render.timestamp(state.updated_at)}; "
                f"acquisition {acquisition}).")
    if state.crashed:
        return (f"Atlas service: not running. The last run ({state.mode}, pid {state.pid}) stopped "
                f"unexpectedly; its last update was {render.timestamp(state.updated_at)}.")
    return (f"Atlas service: not running. The last run ({state.mode}, pid {state.pid}) ended "
            f"{render.timestamp(state.stopped_at)}.")


def _source(state) -> str:
    if state.running:
        return f"From the running service (pid {state.pid}):"
    if state.crashed:
        return f"From the run that stopped unexpectedly (last update {render.timestamp(state.updated_at)}):"
    return f"From the run that ended {render.timestamp(state.stopped_at)}:"


def _no_query(state) -> str:
    if state is None:
        return "Atlas has not run yet."
    if state.running:
        return "No query has run since Atlas started."
    return "The last run answered no queries."


def status(flag: str | None = None, write: Callable[[str], None] = print, directory=None) -> int:
    """The service's state and every component's health; `flag` 'time' or 'retries' for the last query."""
    from observability.service_state import read_state

    state = read_state(directory)
    code = RUNNING if state is not None and state.running else NOT_RUNNING
    write(service_line(state))
    if flag in ("time", "retries"):
        record = state.records[-1] if state is not None and state.records else None
        write("")
        if record is None:
            write(_no_query(state))
            return code
        write(_source(state))
        write(render.time_report(record) if flag == "time" else render.retries_report(record))
        return code
    from observability.health import check

    facts = state.facts if code == RUNNING else None
    absent = "Atlas is still starting" if code == RUNNING and facts is None else "Atlas is not running"
    write("")
    write(render.health_report(check(facts, absent)))
    return code


def conversations(user_id: str, write: Callable[[str], None] = print, limit: int = 50) -> int:
    """This user's conversations, most recently active first, read from PostgreSQL without writing."""
    from memory import conversation_registry
    from observability.health import _safe

    try:
        registry = conversation_registry.open_conversation_registry(
            read_only=True, connect_timeout=PG_TIMEOUT_SECONDS)
    except Exception as error:
        logger.warning("conversations: PostgreSQL unreachable: %s", _safe(error))
        write(f"Cannot reach PostgreSQL: {type(error).__name__}: {_safe(error)}")
        return FAILED
    try:
        write(render.conversation_summaries(user_id, registry.summaries(user_id, limit=limit)))
        return RUNNING
    except Exception as error:
        logger.warning("conversations: listing failed: %s", _safe(error))
        write(f"Cannot list conversations: {type(error).__name__}: {_safe(error)}")
        return FAILED
    finally:
        registry.close()
