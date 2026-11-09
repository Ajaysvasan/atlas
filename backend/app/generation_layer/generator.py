"""The user-facing answer, written by the main model from the assembled context."""

import time
from typing import NamedTuple, Tuple

from config import Config, get_logger
from generation_layer.context import GenerationContext
from inference.inference_exceptions import InferenceError
from inference.sessions import CacheReport, KVSessionManager

logger = get_logger(__name__)


class GenerationFailed(InferenceError):
    """The model produced no usable answer."""


class GenerationResult(NamedTuple):
    text: str
    truncated: bool
    completion_tokens: int
    evidence_used: Tuple[str, ...]
    cache: CacheReport
    milliseconds: float


class Generator:
    """Writes the answer and nothing else: no retrieval, no planning, no memory writes."""

    def __init__(self, sessions: KVSessionManager, max_tokens: int = Config.GENERATION_MAX_TOKENS) -> None:
        self.sessions = sessions
        self.max_tokens = max_tokens

    def generate(self, conversation_id: str, user_id: str, context: GenerationContext) -> GenerationResult:
        started = time.monotonic()
        output = self.sessions.generate(
            conversation_id, user_id, context.prefix, context.turn,
            max_tokens=self.max_tokens, temperature=0.0,
        )
        if not output.text.strip():
            raise GenerationFailed("the model returned an empty answer")
        truncated = output.finish_reason == "length"
        if truncated:
            logger.warning("The answer for %s hit the %d-token limit", conversation_id, self.max_tokens)
        return GenerationResult(
            output.text, truncated, output.completion_tokens, context.evidence_used, output.cache,
            (time.monotonic() - started) * 1000.0,
        )
