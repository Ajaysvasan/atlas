"""The system prompts this layer sends, read from Config.SYSTEM_PROMPTS_FILE."""

from pathlib import Path
from typing import NamedTuple

import yaml

from config import Config
from query_decomposition.query_decomposition_exceptions import SystemPromptMissing

SECTION = "query_decomposition"


class SystemPrompts(NamedTuple):
    condense: str
    complexity: str
    analysis: str
    expansion: str


def load_prompts(path: str | Path | None = None) -> SystemPrompts:
    """Every prompt this layer needs, each a non-empty string."""
    path = Path(Config.SYSTEM_PROMPTS_FILE if path is None else path)
    try:
        with open(path, encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except FileNotFoundError as error:
        raise SystemPromptMissing(path, SECTION) from error

    section = document.get(SECTION) if isinstance(document, dict) else None
    if not isinstance(section, dict):
        raise SystemPromptMissing(path, SECTION)
    prompts = {}
    for key in SystemPrompts._fields:
        value = section.get(key)
        if not isinstance(value, str) or not value.strip():
            raise SystemPromptMissing(path, f"{SECTION}.{key}")
        prompts[key] = value.strip()
    return SystemPrompts(**prompts)
