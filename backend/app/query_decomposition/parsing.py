"""The JSON each prompt must reply with, and turning a reply into something typed."""

import json
from typing import Dict, List, Sequence, Tuple

from query_decomposition.models import Analysis, Mode, TopicPath
from query_decomposition.query_decomposition_exceptions import UnusableReply

COMPLEXITY = "complexity"
ANALYSIS = "analysis"
EXPANSION = "expansion"

MAX_TOPICS = 3
MAX_SUB_TOPICS = 5

COMPLEXITY_SCHEMA = {
    "type": "object",
    "properties": {"mode": {"type": "string", "enum": [mode.value for mode in Mode]}},
    "required": ["mode"],
}

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "topics": {
            "type": "array",
            "maxItems": MAX_TOPICS,
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "sub_topics": {
                        "type": "array",
                        "maxItems": MAX_SUB_TOPICS,
                        "items": {"type": "string"},
                    },
                },
                "required": ["topic", "sub_topics"],
            },
        },
        "sufficient": {"type": "boolean"},
    },
    "required": ["topics", "sufficient"],
}


def expansion_schema(max_sub_queries: int) -> Dict:
    return {
        "type": "object",
        "properties": {
            "sub_queries": {
                "type": "array",
                "maxItems": max_sub_queries,
                "items": {"type": "string"},
            }
        },
        "required": ["sub_queries"],
    }


def _object(stage: str, reply: str) -> Dict:
    text = reply.strip() if isinstance(reply, str) else ""
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise UnusableReply(stage, f"not JSON ({error.msg})") from error
    if not isinstance(value, dict):
        raise UnusableReply(stage, "not a JSON object")
    return value


def _texts(values, limit: int) -> Tuple[str, ...]:
    """The non-empty strings in `values`, stripped, each once, at most `limit`."""
    seen = set()
    kept: List[str] = []
    for value in values if isinstance(values, list) else []:
        if not isinstance(value, str) or not value.strip():
            continue
        text = " ".join(value.split())
        if text.casefold() in seen:
            continue
        seen.add(text.casefold())
        kept.append(text)
    return tuple(kept[:limit])


def read_mode(reply: str) -> Mode:
    value = _object(COMPLEXITY, reply).get("mode")
    try:
        return Mode(value.strip().lower() if isinstance(value, str) else value)
    except ValueError as error:
        raise UnusableReply(COMPLEXITY, f"unknown mode {value!r}") from error


def read_analysis(reply: str, existing_topics: Sequence[str]) -> Analysis:
    """Topics and sufficiency; a topic matching an existing one takes its exact name."""
    value = _object(ANALYSIS, reply)
    sufficient = value.get("sufficient")
    if not isinstance(sufficient, bool):
        raise UnusableReply(ANALYSIS, f"'sufficient' is {sufficient!r}, not a boolean")
    entries = value.get("topics")
    if not isinstance(entries, list):
        raise UnusableReply(ANALYSIS, "'topics' is not a list")

    known = {name.casefold(): name for name in existing_topics}
    seen = set()
    topics: List[TopicPath] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        names = _texts([entry.get("topic")], 1)
        if not names:
            continue
        name = known.get(names[0].casefold(), names[0])
        if name.casefold() in seen:
            continue
        seen.add(name.casefold())
        topics.append(TopicPath(name, _texts(entry.get("sub_topics"), MAX_SUB_TOPICS)))
    return Analysis(tuple(topics[:MAX_TOPICS]), sufficient)


def read_sub_queries(reply: str, limit: int) -> Tuple[str, ...]:
    value = _object(EXPANSION, reply).get("sub_queries")
    if not isinstance(value, list):
        raise UnusableReply(EXPANSION, "'sub_queries' is not a list")
    return _texts(value, limit)
