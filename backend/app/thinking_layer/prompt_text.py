"""The user-message blocks the thinking layer's prompts are built from."""

from typing import NamedTuple, Sequence, Tuple

from prompt_registry import load_section
from query_decomposition.models import Decomposition
from thinking_layer.accumulator import trim
from thinking_layer.models import Evidence, MemoryContext

SECTION = "thinking_layer"


class ThinkingPrompts(NamedTuple):
    planner: str
    worker: str
    stitch: str
    job_critic: str
    plan_critic: str


def load_thinking_prompts(path=None) -> ThinkingPrompts:
    return ThinkingPrompts(**load_section(SECTION, ThinkingPrompts._fields, path))


def request_block(decomposition: Decomposition, budget_chars: int) -> str:
    """The user's request as written; the condensed form only when the original cannot fit."""
    original = decomposition.query
    if len(original) <= budget_chars:
        return f"<request>\n{original}\n</request>"
    head = trim(original, max(0, budget_chars - len(decomposition.working_query) - 200))
    return (
        "<request note=\"condensed; every requirement below still applies\">\n"
        f"{decomposition.working_query}\n</request>\n"
        f"<request_original_start>\n{head}\n</request_original_start>"
    )


def evidence_block(evidence: Sequence[Evidence], budget_chars: int) -> str:
    """Evidence best-first, each tagged, until the budget is spent."""
    if not evidence:
        return "<evidence>\nnone retrieved\n</evidence>"
    lines = []
    spent = 0
    for item in sorted(evidence, key=lambda e: -e.score):
        source = f" source=\"{item.origin}\"" if item.origin else ""
        text = trim(item.text.strip(), max(200, budget_chars // max(1, len(evidence))))
        entry = f"[{item.ref}]{source} {text}"
        if lines and spent + len(entry) > budget_chars:
            break
        lines.append(entry)
        spent += len(entry)
    return "<evidence>\n" + "\n\n".join(lines) + "\n</evidence>"


def dependency_block(results: Sequence[Tuple[str, str]]) -> str:
    if not results:
        return ""
    body = "\n\n".join(f"Result of {job_id}:\n{text}" for job_id, text in results)
    return f"<accepted_results>\n{body}\n</accepted_results>"


def memory_block(memory: MemoryContext, budget_chars: int) -> str:
    if not memory.summary and not memory.turns:
        return ""
    parts = []
    if memory.summary:
        parts.append("Summary so far: " + trim(memory.summary, budget_chars // 2))
    remaining = budget_chars - sum(len(p) for p in parts)
    turns = []
    for role, text in reversed(memory.turns):
        line = f"{role.capitalize()}: {text}"
        if turns and len(line) > remaining:
            break
        turns.append(trim(line, max(200, remaining)))
        remaining -= len(line)
    parts.extend(reversed(turns))
    return "<conversation_so_far>\n" + "\n".join(parts) + "\n</conversation_so_far>"


def join_blocks(*blocks: str) -> str:
    return "\n\n".join(block for block in blocks if block)
