"""The interactive development CLI: queries, conversation switching and status."""

import sys
from typing import Callable

from cli import render
from cli.commands import HELP, Command, CommandError, parse
from config import get_logger

logger = get_logger(__name__)

PROMPT = "Atlas > "


class CliSession:
    """Owns the active conversation; knows nothing of models, caches or databases."""

    def __init__(
        self,
        pipeline,
        components,
        user_id: str,
        read: Callable[[str], str] = input,
        write: Callable[[str], None] = print,
        health=None,
    ) -> None:
        self.pipeline = pipeline
        self.components = components
        self.user_id = user_id
        self.read = read
        self.write = write
        self.active = None
        if health is None:
            from observability.health import check_all

            health = check_all
        self.health = health

    def run(self) -> None:
        self.write(f"Atlas development CLI. User: {self.user_id}. Type help for commands.")
        while True:
            try:
                line = self.read(self.__prompt())
            except EOFError:
                self.write("")
                break
            except KeyboardInterrupt:
                self.write("\n(Ctrl+C) Type exit or press Ctrl+D to leave.")
                continue
            try:
                command = parse(line)
            except CommandError as error:
                self.write(str(error))
                continue
            try:
                if command is None:
                    self.query(line.strip())
                elif not self.dispatch(command):
                    break
            except KeyboardInterrupt:
                # A model call cannot be interrupted midway; the request ends
                # when it returns, and its state stays consistent.
                self.write("\nInterrupted.")
            except Exception as error:
                logger.exception("CLI command failed")
                self.write(f"Error: {error}")
        self.write("Goodbye.")

    def __prompt(self) -> str:
        if self.active is None:
            return PROMPT
        name = self.active.conversation_name
        return f"[{name[:30]}] {PROMPT}"

    def dispatch(self, command: Command) -> bool:
        """Run a command; False means leave."""
        if command.name == "exit":
            return False
        if command.name == "empty":
            return True
        if command.name == "help":
            self.write(HELP)
        elif command.name == "new":
            self.active = None
            self.write("Your next query starts a new conversation.")
        elif command.name == "switch":
            self.switch(command.args[0] if command.args else None)
        elif command.name == "status":
            self.status(command.args[0] if command.args else None)
        return True

    def query(self, text: str) -> None:
        if self.active is None:
            self.active = self.pipeline.new_conversation(self.user_id, text)
            self.write(f"New conversation: {self.active.conversation_name}")
        self.write("Thinking…")
        response = self.pipeline.ask(self.active.conversation_id, self.user_id, text)
        if response.text is None:
            self.write(f"Atlas could not answer ({response.record.error_category}): {response.error}")
            return
        self.write("")
        self.write(response.text)
        self.write("")
        for warning in response.warnings:
            self.write(f"Warning: {warning}")
        self.write(f"({render.duration(response.record.total_ms)}; status --time for detail)")

    def switch(self, selector: str | None) -> None:
        registry = self.components.registry
        records = registry.list(self.user_id)
        active_id = self.active.conversation_id if self.active else None
        if selector is None:
            self.write(render.conversation_list(records, active_id))
            if not records:
                return
            try:
                selector = self.read("Select conversation (number or id, Enter to cancel): ").strip()
            except (EOFError, KeyboardInterrupt):
                selector = ""
            if not selector:
                self.write("Unchanged.")
                return
        chosen = None
        if selector.isdigit() and 1 <= int(selector) <= len(records):
            chosen = records[int(selector) - 1]
        else:
            matches = registry.find_by_prefix(self.user_id, selector)
            if len(matches) > 1:
                self.write(f"{len(matches)} conversations start with {selector!r}; give more of the id.")
                return
            chosen = matches[0] if matches else None
        if chosen is None:
            self.write(f"No conversation {selector!r}.")
            return
        self.active = chosen
        cache = self.components.sessions.checkpoint_status(chosen.conversation_id, self.user_id)
        self.write(f"Active conversation:\n{chosen.conversation_name}\n(inference cache: {cache})")

    def status(self, flag: str | None) -> None:
        telemetry = self.components.telemetry
        if flag == "time":
            self.write(render.time_report(telemetry.last()))
        elif flag == "retries":
            self.write(render.retries_report(telemetry.last()))
        else:
            self.write(render.health_report(self.health(self.components)))


def cli_interface(user_id: str, acquisition: bool = True, lease=None) -> int:
    """Start Atlas and run the interactive loop; returns the process exit code."""
    from pipeline.bootstrap import Components

    print("Starting Atlas…")
    components = Components(acquisition=acquisition)
    try:
        if lease is not None:
            lease.attach(components)
        try:
            pipeline = components.pipeline()
        except Exception as error:
            logger.exception("Atlas could not start")
            print(f"Atlas could not start: {type(error).__name__}: {error}", file=sys.stderr)
            print("Run `uv run python scripts/smoke.py` to see which dependency is missing.",
                  file=sys.stderr)
            return 1
        try:
            problems = components.warm_up()
        except Exception as error:
            print(f"Atlas could not load the draft model: {error}", file=sys.stderr)
            return 1
        if lease is not None:
            lease.publish()
        for problem in problems:
            print(f"Warning: {problem} (answers continue without it; see `status`)")
        CliSession(pipeline, components, user_id).run()
        return 0
    finally:
        components.close()
