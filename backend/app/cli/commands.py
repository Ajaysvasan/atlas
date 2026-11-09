"""What a line typed at the prompt means: a command only if it is exactly one, otherwise a query."""

import shlex
from typing import NamedTuple, Tuple

HELP = """Commands (anything else is sent to Atlas as a query):
  switch               list your conversations and pick one
  switch <n | id>      switch by list number or by conversation id (a prefix is enough)
  new                  start a new conversation with your next query
  status               health of every component
  status --time        timings of the last query
  status --retries     planner/critic retries of the last query
  help                 this text
  exit                 leave (Ctrl+D works too)
Each command may also be written with a leading '/', e.g. /status.
From another terminal, without loading anything: main.py status [--time | --retries],
main.py conversations [--user NAME]."""


class Command(NamedTuple):
    name: str
    args: Tuple[str, ...] = ()


class CommandError(ValueError):
    pass


_EXACT = {
    "exit": "exit", "quit": "exit",
    "help": "help",
    "new": "new",
    "switch": "switch",
    "status": "status",
}
_STATUS_FLAGS = {"--time": "time", "--retries": "retries"}


def parse(line: str) -> Command | None:
    """A Command, or None when the line is a query.

    Bare words count only when the whole line is the command's syntax, so
    "switch to PostgreSQL 16?" is a query. A line starting with '/' must be a
    command, and an unknown one is an error rather than a query.
    """
    text = line.strip()
    if not text:
        return Command("empty")
    slashed = text.startswith("/")
    body = text[1:] if slashed else text
    try:
        words = shlex.split(body)
    except ValueError:
        words = body.split()
    if not words:
        if slashed:
            raise CommandError("empty command; type help")
        return None
    head = words[0].lower()
    name = _EXACT.get(head)
    if name is None:
        if slashed:
            raise CommandError(f"unknown command /{words[0]}; type help")
        return None
    rest = words[1:]
    if name in ("exit", "help", "new"):
        if rest:
            return _not_a_command(slashed, f"{name} takes no arguments")
        return Command(name)
    if name == "switch":
        if len(rest) > 1:
            return _not_a_command(slashed, "switch takes at most one argument: a number or an id")
        if rest and not _looks_like_selector(rest[0]):
            return _not_a_command(slashed, f"{rest[0]!r} is not a list number or conversation id")
        return Command("switch", tuple(rest))
    if name == "status":
        if not rest:
            return Command("status")
        if len(rest) == 1 and rest[0] in _STATUS_FLAGS:
            return Command("status", (_STATUS_FLAGS[rest[0]],))
        return _not_a_command(slashed, "status takes --time or --retries")
    return None


def _looks_like_selector(word: str) -> bool:
    return word.isdigit() or (len(word) >= 4 and all(c in "0123456789abcdef" for c in word.lower()))


def _not_a_command(slashed: bool, message: str) -> None:
    if slashed:
        raise CommandError(message)
    return None
