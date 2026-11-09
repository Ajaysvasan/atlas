"""Plain-text rendering of status reports and conversation lists."""

from datetime import datetime
from typing import List, Sequence

from observability.health import ComponentHealth, Health
from observability.telemetry import RequestRecord

_STAGE_LABELS = {
    "decomposition": "Query decomposition",
    "memory": "Memory resolution",
    "retrieval": "Retrieval",
    "ksv": "KSV",
    "acquisition": "Knowledge acquisition",
    "thinking": "Thinking (total)",
    "thinking.planning": "  planning",
    "thinking.execution": "  job execution + job critique",
    "thinking.plan_critique": "  plan critique",
    "generation": "Generation",
    "persistence": "Persistence",
}


def duration(ms: float | None) -> str:
    if ms is None:
        return "-"
    return f"{ms:,.0f} ms" if ms < 10_000 else f"{ms / 1000:,.1f} s"


def timestamp(value: str) -> str:
    try:
        return datetime.fromisoformat(value).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    except ValueError:
        return value


def health_report(results: Sequence[ComponentHealth]) -> str:
    width = max(len(r.name) for r in results)
    lines = ["Atlas status:", ""]
    for r in results:
        lines.append(f"  {r.name:<{width}}  {r.status.value:<15} {r.detail}")
    counts = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    summary = ", ".join(f"{counts[s]} {s.value}" for s in Health if s in counts)
    lines += ["", f"  {summary}"]
    return "\n".join(lines)


def time_report(record: RequestRecord | None) -> str:
    if record is None:
        return "No query has run in this session yet."
    lines = [f"Last query: {record.request_id}  ({record.status}"
             + (f", {record.error_category}" if record.error_category else "") + ")", ""]
    for name, ms in record.stages.items():
        lines.append(f"  {_STAGE_LABELS.get(name, name):<34} {duration(ms):>10}")
    cache = record.cache
    if cache:
        lines.append(f"  {'  KV restore (within generation)':<34} {duration(cache.get('restore_ms')):>10}")
        lines.append(f"  {'  KV save (within generation)':<34} {duration(cache.get('save_ms')):>10}")
    lines.append(f"  {'Total (end to end)':<34} {duration(record.total_ms):>10}")
    detail = record.retrieval.get("thinking_detail_ms")
    if detail:
        lines += ["", "  Summed across jobs (they may overlap, so not part of the total):"]
        for name, ms in detail.items():
            lines.append(f"    {name.replace('_sum', '').replace('_', ' ')}: {duration(ms)}")
    if cache:
        lines += ["", f"  KV cache: {cache.get('outcome')}; {cache.get('evaluated_prompt_tokens')} of "
                      f"{cache.get('prompt_tokens')} prompt tokens evaluated"
                      + (f"; save failed: {cache['save_error']}" if cache.get("save_error") else "")]
    return "\n".join(lines)


def retries_report(record: RequestRecord | None) -> str:
    if record is None:
        return "No query has run in this session yet."
    if record.path is None:
        return f"Last query: {record.request_id} ended before reasoning ({record.status})."
    lines = [
        f"Last query: {record.request_id}  (path: {record.path}, {record.jobs} job(s))",
        "",
        "Planner/Critique retries:",
        f"  Job-level retries:     {record.job_retries}",
        f"  Plan-level revisions:  {record.plan_revisions}",
        f"  Total reasoning retries: {record.reasoning_retries}",
    ]
    if record.retried_jobs:
        lines.append(f"  Retried jobs: {', '.join(record.retried_jobs)}")
    if record.retry_reasons:
        lines.append("  Reasons:")
        lines += [f"    - {reason[:160]}" for reason in record.retry_reasons]
    lines += [
        f"  Retry budget exhausted: {'yes' if record.retries_exhausted else 'no'}",
        "",
        f"Model/runtime retries (not reasoning): {record.runtime_retries}",
        f"Thinking status: {record.thinking_status}",
        f"Final status: {record.status.upper()}",
    ]
    return "\n".join(lines)


def conversation_list(records, active_id: str | None) -> str:
    if not records:
        return "You have no conversations yet. Type a query to start one."
    lines: List[str] = ["Available conversations:", ""]
    for number, record in enumerate(records, start=1):
        marker = "  (active)" if record.conversation_id == active_id else ""
        lines += [
            f"{number}. {record.conversation_name}{marker}",
            f"   ID: {record.conversation_id}",
            f"   Created: {timestamp(record.created_at)}",
            "",
        ]
    return "\n".join(lines).rstrip()
