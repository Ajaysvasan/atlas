"""The final answer's prompt: a stable prefix to cache, then this turn's request, results and evidence."""

from typing import Dict, List, NamedTuple, Sequence, Tuple

from config import Config
from prompt_registry import load_section
from thinking_layer.accumulator import trim
from thinking_layer.models import Evidence, JobKind, ThinkingResult, ThinkingStatus

SECTION = "generation_layer"
CHARS_PER_TOKEN = 4


class GenerationPrompts(NamedTuple):
    answer: str


def load_generation_prompts(path=None) -> GenerationPrompts:
    return GenerationPrompts(**load_section(SECTION, GenerationPrompts._fields, path))


class GenerationBudget(NamedTuple):
    """Characters per section; sections are cut, never dropped, so the order of priority holds."""

    request_chars: int = 12_000
    results_chars: int = 16_000
    evidence_chars: int = 10_000
    history_chars: int = 8_000
    summary_chars: int = 2_000
    history_exchanges: int = Config.GENERATION_HISTORY_TURNS

    @staticmethod
    def for_tokens(tokens: int = Config.GENERATION_CONTEXT_TOKENS) -> "GenerationBudget":
        chars = tokens * CHARS_PER_TOKEN
        return GenerationBudget(
            request_chars=chars // 4,
            results_chars=chars // 3,
            evidence_chars=chars // 5,
            history_chars=chars // 6,
            summary_chars=chars // 24,
        )


class KnowledgeStatus(NamedTuple):
    memory: str | None = None
    global_retrieval: str | None = None
    acquisition: str | None = None


class GenerationContext(NamedTuple):
    prefix: List[Dict[str, str]]
    turn: Dict[str, str]
    evidence_used: Tuple[str, ...]
    history_exchanges: int


def history_window(exchanges: int, window: int) -> int:
    """Index of the first exchange to replay.

    It moves in steps of half the window rather than one exchange at a time,
    so the replayed history, and with it the saved prefix, stays the same for
    several turns in a row instead of changing on every one.
    """
    if window < 1 or exchanges <= window:
        return 0
    step = max(1, window // 2)
    return ((exchanges - window) // step + 1) * step


def _pairs(turns: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """User/assistant exchanges in order; an unanswered user turn is not replayed."""
    pairs, pending = [], None
    for role, text in turns:
        if role == "user":
            pending = text
        elif role == "assistant" and pending is not None:
            pairs.append((pending, text))
            pending = None
    return pairs


def assemble(
    prompts: GenerationPrompts,
    thinking: ThinkingResult,
    evidence: Sequence[Evidence],
    *,
    summary: str | None = None,
    history: Sequence[Tuple[str, str]] = (),
    history_offset: int = 0,
    knowledge: KnowledgeStatus = KnowledgeStatus(),
    budget: GenerationBudget = GenerationBudget(),
) -> GenerationContext:
    """Deterministic: the same inputs always produce the same messages."""
    system = prompts.answer
    if summary:
        system += "\n\nWhat this conversation has covered so far:\n" + trim(summary, budget.summary_chars)
    prefix = [{"role": "system", "content": system}]

    # `history_offset` is how many exchanges precede `history` in the whole
    # conversation, so the window is placed by absolute position and does not
    # slide with every new turn.
    exchanges = _pairs(history)
    start = history_window(history_offset + len(exchanges), budget.history_exchanges)
    exchanges = exchanges[max(0, start - history_offset):]
    share = budget.history_chars // max(1, 2 * len(exchanges))
    for user, assistant in exchanges:
        prefix.append({"role": "user", "content": trim(user, share)})
        prefix.append({"role": "assistant", "content": trim(assistant, share)})

    sections = [_request(thinking, budget.request_chars)]
    sections.append(_results(thinking, budget.results_chars))
    evidence_text, used = _evidence(thinking, evidence, budget.evidence_chars)
    sections.append(evidence_text)
    sections.append(_knowledge(knowledge))
    if thinking.limitations:
        sections.append(
            "<known_limitations>\n"
            + "\n".join(f"- {item}" for item in thinking.limitations[:8])
            + "\n</known_limitations>"
        )
    sections.append("Write the final answer to the request now.")
    turn = {"role": "user", "content": "\n\n".join(s for s in sections if s)}
    return GenerationContext(prefix, turn, used, len(exchanges))


def _request(thinking: ThinkingResult, limit: int) -> str:
    original = thinking.original_query
    if len(original) <= limit:
        return f"<request>\n{original}\n</request>"
    return (
        "<request note=\"long request; condensed form first, then the original as far as it fits\">\n"
        f"{thinking.effective_query}\n</request>\n"
        f"<request_original>\n{trim(original, max(0, limit - len(thinking.effective_query)))}\n</request_original>"
    )


def _results(thinking: ThinkingResult, limit: int) -> str:
    if not thinking.accepted:
        return (
            "<accepted_results>\nNo reasoning result was accepted. Answer from the evidence and "
            "general knowledge, and say what could not be established.\n</accepted_results>"
        )
    final = thinking.final_output
    if final is not None:
        body = trim(final, limit)
    else:
        parts = [o for o in thinking.accepted if o.kind is not JobKind.STITCH]
        share = limit // max(1, len(parts))
        body = "\n\n".join(f"{o.description}:\n{trim(o.output or '', share)}" for o in parts)
    note = " status=\"partial\"" if thinking.status is not ThinkingStatus.COMPLETED else ""
    return f"<accepted_results{note}>\n{body}\n</accepted_results>"


def _evidence(thinking: ThinkingResult, evidence: Sequence[Evidence], limit: int) -> Tuple[str, Tuple[str, ...]]:
    """Evidence the accepted results cite first, then the rest by score, within the budget."""
    if not evidence:
        return "<evidence>\nnone retrieved\n</evidence>", ()
    cited = {item.ref for item in thinking.evidence}
    ordered = sorted(evidence, key=lambda e: (e.ref not in cited, -e.score, e.ref))
    lines, used, spent = [], [], 0
    per_item = max(300, limit // max(1, len(ordered)))
    for item in ordered:
        source = f" source=\"{item.origin}\"" if item.origin else ""
        line = f"[{item.ref}]{source} {trim(item.text.strip(), per_item)}"
        if lines and spent + len(line) > limit:
            break
        lines.append(line)
        used.append(item.ref)
        spent += len(line)
    return "<evidence>\n" + "\n\n".join(lines) + "\n</evidence>", tuple(used)


def _knowledge(knowledge: KnowledgeStatus) -> str:
    parts = [f"{name}: {value}" for name, value in (
        ("memory", knowledge.memory),
        ("knowledge base", knowledge.global_retrieval),
        ("acquisition", knowledge.acquisition),
    ) if value]
    if not parts:
        return ""
    return "<knowledge_status>\n" + "\n".join(parts) + "\n</knowledge_status>"
