"""Named system prompts read from Config.SYSTEM_PROMPTS_FILE, one section per layer."""

from pathlib import Path
from typing import Dict, Sequence

import yaml

from config import Config


class PromptMissing(LookupError):
    """A section or prompt the caller needs is absent or empty."""

    def __init__(self, path: Path, key: str) -> None:
        super().__init__(f"{path} has no non-empty prompt at {key!r}")
        self.path = path
        self.key = key


def load_section(
    section: str, keys: Sequence[str], path: str | Path | None = None
) -> Dict[str, str]:
    """Every prompt `keys` names under `section`, each a non-empty string."""
    path = Path(Config.SYSTEM_PROMPTS_FILE if path is None else path)
    try:
        with open(path, encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except FileNotFoundError as error:
        raise PromptMissing(path, section) from error
    block = document.get(section) if isinstance(document, dict) else None
    if not isinstance(block, dict):
        raise PromptMissing(path, section)
    prompts = {}
    for key in keys:
        value = block.get(key)
        if not isinstance(value, str) or not value.strip():
            raise PromptMissing(path, f"{section}.{key}")
        prompts[key] = value.strip()
    return prompts
