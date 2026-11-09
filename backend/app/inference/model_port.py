"""The one method every layer calls a language model through."""

from typing import Dict, Protocol

DRAFT = "draft"
PLANNER = "planner"
WORKER = "worker"
CRITIC = "critic"
NAMING = "naming"
SUMMARY = "summary"
MAIN = "main"


class TextModel(Protocol):
    def infer(
        self,
        system: str,
        user: str,
        *,
        schema: Dict | None = None,
        max_tokens: int = 512,
        purpose: str = DRAFT,
        temperature: float = 0.0,
    ) -> str:
        """One chat completion; with a schema the reply is constrained to JSON matching it."""
        ...
