"""Main generation for a conversation: restore its saved prefix if it still matches, answer, save the prefix again."""

import time
from typing import Dict, List, NamedTuple, Sequence

import numpy as np

from config import get_logger
from inference.checkpoints import HIT, INVALID, CheckpointStore
from inference.inference_exceptions import InferenceError, PromptTooLarge
from inference.model_port import MAIN
from inference.shared_model import SharedModel

logger = get_logger(__name__)

CHECKPOINT = "kv_checkpoint"
MIN_REUSE_TOKENS = 32


class CacheReport(NamedTuple):
    conversation_id: str
    outcome: str
    reason: str | None
    prompt_tokens: int
    evaluated_prompt_tokens: int
    resident_prefix_tokens: int
    restored_prefix_tokens: int
    restore_ms: float
    save_ms: float
    saved_tokens: int
    saved_bytes: int
    save_error: str | None

    @property
    def reused_tokens(self) -> int:
        return max(0, self.prompt_tokens - self.evaluated_prompt_tokens)


class SessionGeneration(NamedTuple):
    text: str
    finish_reason: str | None
    completion_tokens: int
    cache: CacheReport


class FormattedPrompt(NamedTuple):
    tokens: List[int]
    stop: List[str]
    stopping_criteria: object


def common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    count = 0
    for x, y in zip(a, b):
        if int(x) != int(y):
            break
        count += 1
    return count


def _formatter(llm, add_generation_prompt: bool):
    from llama_cpp import llama_chat_format

    template = llm.metadata.get("tokenizer.chat_template")
    if not template:
        raise InferenceError("the model carries no chat template")
    eos_id = llm.token_eos()
    bos_id = llm.token_bos()
    model = getattr(llm, "_model", None)
    text = (lambda token: model.token_get_text(token) if model is not None and token >= 0 else "")
    return llama_chat_format.Jinja2ChatFormatter(
        template=template,
        eos_token=text(eos_id),
        bos_token=text(bos_id),
        add_generation_prompt=add_generation_prompt,
        stop_token_ids=[eos_id],
    )


def format_messages(llm, messages: List[Dict[str, str]], add_generation_prompt: bool) -> FormattedPrompt:
    """Tokens exactly as create_chat_completion would build them from the model's own template."""
    result = _formatter(llm, add_generation_prompt)(messages=messages)
    tokens = llm.tokenize(result.prompt.encode("utf-8"), add_bos=not result.added_special, special=True)
    stop = [] if result.stop is None else ([result.stop] if isinstance(result.stop, str) else list(result.stop))
    return FormattedPrompt(list(tokens), stop, result.stopping_criteria)


class KVSessionManager:
    """Owns which conversation's state is in the model. CLI and pipeline never see llama state."""

    def __init__(
        self,
        model: SharedModel,
        store: CheckpointStore,
        min_reuse_tokens: int = MIN_REUSE_TOKENS,
        formatter=format_messages,
    ) -> None:
        self.model = model
        self.store = store
        self.min_reuse_tokens = min_reuse_tokens
        self.format = formatter
        self.active_conversation: str | None = None
        self.restores = 0
        self.misses = 0
        self.invalidations = 0
        self.save_failures = 0

    def generate(
        self,
        conversation_id: str,
        user_id: str,
        prefix: List[Dict[str, str]],
        turn: Dict[str, str],
        *,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> SessionGeneration:
        """Answer `turn` after `prefix`; afterwards the state saved is `prefix` alone, with no evidence in it."""
        with self.model.exclusive() as llm:
            full = self.format(llm, prefix + [turn], True)
            saved_prefix = self.format(llm, prefix, False)
            if len(full.tokens) + max_tokens > self.model.spec.n_ctx:
                raise PromptTooLarge(len(full.tokens), self.model.spec.n_ctx - max_tokens)

            resident = common_prefix(llm.input_ids[: llm.n_tokens], full.tokens)
            outcome, reason, restored, restore_ms = self.__restore(
                llm, conversation_id, user_id, full.tokens, resident
            )
            self.active_conversation = conversation_id
            output = self.model.completion(
                full.tokens,
                purpose=MAIN,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=full.stop,
                stopping_criteria=full.stopping_criteria,
            )
            evaluated = self.model.last_evaluated_prompt_tokens
            choice = output["choices"][0]
            save_ms, saved_tokens, saved_bytes, save_error = self.__save(
                llm, conversation_id, user_id, saved_prefix.tokens
            )
        report = CacheReport(
            conversation_id, outcome, reason, len(full.tokens),
            len(full.tokens) if evaluated is None else evaluated,
            resident, restored, restore_ms, save_ms, saved_tokens, saved_bytes, save_error,
        )
        logger.info(
            "Main generation for %s: cache %s, %d of %d prompt tokens evaluated",
            conversation_id, outcome, report.evaluated_prompt_tokens, report.prompt_tokens,
        )
        usage = output.get("usage") or {}
        return SessionGeneration(
            (choice.get("text") or "").strip(),
            choice.get("finish_reason"),
            int(usage.get("completion_tokens") or 0),
            report,
        )

    def __restore(self, llm, conversation_id, user_id, tokens, resident):
        loaded = self.store.load(
            conversation_id, user_id, MAIN, self.model.fingerprint, self.model.spec.n_ctx
        )
        if loaded.status == INVALID:
            self.invalidations += 1
            return "invalid", loaded.reason, 0, loaded.load_ms
        if loaded.status != HIT:
            self.misses += 1
            return ("resident" if resident >= self.min_reuse_tokens else "miss"), loaded.reason, 0, loaded.load_ms
        checkpoint = loaded.checkpoint
        usable = common_prefix(checkpoint.tokens, tokens)
        if usable < self.min_reuse_tokens:
            self.store.invalidate(checkpoint.row, f"prefix diverges after {usable} token(s)")
            self.invalidations += 1
            return "stale", f"prefix diverges after {usable} token(s)", 0, loaded.load_ms
        if usable <= resident:
            return "resident", "the model already holds as long a prefix", 0, loaded.load_ms
        started = time.monotonic()
        try:
            from llama_cpp import LlamaState

            input_ids = np.zeros(self.model.spec.n_ctx, dtype=np.intc)
            input_ids[: checkpoint.tokens.size] = checkpoint.tokens
            # Sampling happens inside llama.cpp; restored logits are never read
            # (load_state forces a re-evaluation), so one broadcast row stands
            # in for the n_tokens x n_vocab array save_state would carry.
            llm.load_state(LlamaState(
                input_ids=input_ids,
                scores=np.zeros((1, llm.n_vocab()), dtype=np.single),
                n_tokens=int(checkpoint.tokens.size),
                llama_state=checkpoint.state,
                llama_state_size=len(checkpoint.state),
                seed=checkpoint.seed,
            ))
        except Exception as error:
            # A state llama.cpp refuses leaves the context undefined: clear it
            # and let the prompt be evaluated from nothing.
            llm.reset()
            self.store.invalidate(checkpoint.row, f"llama.cpp refused it: {error}")
            self.invalidations += 1
            return "invalid", str(error), 0, loaded.load_ms + (time.monotonic() - started) * 1000.0
        self.restores += 1
        return "restored", None, usable, loaded.load_ms + (time.monotonic() - started) * 1000.0

    def __save(self, llm, conversation_id, user_id, prefix_tokens):
        if len(prefix_tokens) < self.min_reuse_tokens:
            return 0.0, 0, 0, None
        started = time.monotonic()
        try:
            # Truncates the context back to the prefix: the shared part stays,
            # one token is re-decoded, and the evidence and answer drop out.
            self.model.completion(prefix_tokens, purpose=CHECKPOINT, max_tokens=1, temperature=0.0)
            state = llm.save_state()
            tokens = np.asarray(state.input_ids[: state.n_tokens], dtype=np.intc)
            if tokens.size != len(prefix_tokens) or common_prefix(tokens, prefix_tokens) != len(prefix_tokens):
                raise InferenceError("the model's state does not hold exactly the prefix")
            row = self.store.save(
                conversation_id, user_id, MAIN, self.model.fingerprint, self.model.spec.n_ctx,
                tokens, bytes(state.llama_state[: state.llama_state_size]), state.seed,
            )
            return (time.monotonic() - started) * 1000.0, row.n_tokens, row.size_bytes, None
        except Exception as error:
            self.save_failures += 1
            logger.warning("Could not checkpoint conversation %s: %s", conversation_id, error)
            return (time.monotonic() - started) * 1000.0, 0, 0, str(error)

    def checkpoint_status(self, conversation_id: str, user_id: str) -> str:
        row = self.store.row_for(conversation_id, user_id, MAIN)
        if row is None:
            return "none"
        if row.model_fingerprint != self.model.fingerprint or row.n_ctx != self.model.spec.n_ctx:
            return "incompatible"
        return f"{row.n_tokens} tokens, {row.size_bytes / 1e6:.1f} MB"

    def forget(self, conversation_id: str, user_id: str | None = None) -> int:
        return self.store.delete_conversation(conversation_id, user_id)
