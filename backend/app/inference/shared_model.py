"""The one resident llama.cpp model, and the lock every use of it goes through."""

import hashlib
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterator, List, NamedTuple

from config import Config, get_logger
from inference.inference_exceptions import (
    InferenceError,
    ModelFileMissing,
    ModelUnavailable,
    PromptTooLarge,
)
from inference.model_port import DRAFT

logger = get_logger(__name__)

# Bumped whenever the checkpoint file layout or its validation rules change;
# every older checkpoint then fails validation and is recomputed.
CHECKPOINT_FORMAT = 1

# Role markers the chat template adds, which tokenize() does not count.
TEMPLATE_TOKENS = 64

_FINGERPRINT_SAMPLE = 4 * 1024 * 1024


class ModelSpec(NamedTuple):
    path: Path
    n_ctx: int = Config.SHARED_MODEL_CONTEXT_WINDOW
    n_threads: int | None = Config.MODEL_THREADS
    seed: int = 7

    @staticmethod
    def draft() -> "ModelSpec":
        return ModelSpec(Path(Config.DRAFT_MODEL_PATH) / Config.DRAFT_MODEL_FILE)


def file_fingerprint(path: Path) -> str:
    """Identity of a model file: size plus its first and last 4 MiB, not the whole 2 GB."""
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode())
    with open(path, "rb") as handle:
        digest.update(handle.read(_FINGERPRINT_SAMPLE))
        if size > _FINGERPRINT_SAMPLE:
            handle.seek(max(_FINGERPRINT_SAMPLE, size - _FINGERPRINT_SAMPLE))
            digest.update(handle.read(_FINGERPRINT_SAMPLE))
    return digest.hexdigest()


class InferenceStats(NamedTuple):
    calls: Dict[str, int]
    prompt_tokens: int
    evaluated_prompt_tokens: int
    completion_tokens: int
    failures: int
    busy_ms: float

    @property
    def reused_prompt_tokens(self) -> int:
        return max(0, self.prompt_tokens - self.evaluated_prompt_tokens)


class _EvalMeter:
    """Counts the prompt tokens llama.cpp actually decodes, which is how prefix reuse is measured."""

    def __init__(self, llm) -> None:
        self.__eval = llm.eval
        self.first_batch: int | None = None
        llm.eval = self

    def __call__(self, tokens):
        if self.first_batch is None:
            self.first_batch = len(tokens)
        return self.__eval(tokens)

    def start(self) -> None:
        self.first_batch = None


def _default_loader(spec: ModelSpec):
    from llama_cpp import Llama  # not a dependency; installed separately

    options = dict(model_path=str(spec.path), n_ctx=spec.n_ctx, seed=spec.seed, verbose=False)
    if spec.n_threads:
        options["n_threads"] = spec.n_threads
    return Llama(**options)


class SharedModel:
    """Loaded once, on first use; every completion holds the lock, so calls never overlap."""

    def __init__(self, spec: ModelSpec, loader: Callable[[ModelSpec], object] = _default_loader) -> None:
        self.spec = spec
        self.__loader = loader
        self.__lock = threading.RLock()
        self.__llm = None
        self.__meter: _EvalMeter | None = None
        self.__fingerprint: str | None = None
        self.load_error: str | None = None
        self.load_ms: float | None = None
        self.__calls: Dict[str, int] = {}
        self.__prompt_tokens = 0
        self.__evaluated = 0
        self.__completion_tokens = 0
        self.__failures = 0
        self.__busy_ms = 0.0
        self.last_evaluated_prompt_tokens: int | None = None

    @property
    def loaded(self) -> bool:
        return self.__llm is not None

    @property
    def fingerprint(self) -> str:
        """Model file plus the settings a saved state depends on."""
        if self.__fingerprint is None:
            try:
                model = file_fingerprint(self.spec.path)
            except OSError as error:
                raise ModelUnavailable(f"cannot read {self.spec.path}: {error}") from error
            payload = f"{model}|n_ctx={self.spec.n_ctx}|format={CHECKPOINT_FORMAT}|{_runtime_version()}"
            self.__fingerprint = hashlib.sha256(payload.encode()).hexdigest()
        return self.__fingerprint

    def __load(self):
        if self.__llm is not None:
            return self.__llm
        if not self.spec.path.exists():
            self.load_error = f"model file not found at {self.spec.path}"
            raise ModelFileMissing(
                f"Model not found at {self.spec.path}.\n"
                "Run:  uv run python download_models/download_draft_model.py"
            )
        started = time.monotonic()
        try:
            llm = self.__loader(self.spec)
        except Exception as error:
            self.load_error = str(error)
            raise ModelUnavailable(f"could not load {self.spec.path}: {error}") from error
        self.load_ms = (time.monotonic() - started) * 1000.0
        self.load_error = None
        self.__meter = _EvalMeter(llm) if hasattr(llm, "eval") else None
        self.__llm = llm
        logger.info("Loaded %s (n_ctx=%d) in %.0f ms", self.spec.path.name, self.spec.n_ctx,
                    self.load_ms)
        return llm

    def ensure_loaded(self) -> None:
        with self.__lock:
            self.__load()

    @contextmanager
    def exclusive(self) -> Iterator[object]:
        """The llama instance, held: nothing else can run or change its state until release."""
        with self.__lock:
            yield self.__load()

    def count_tokens(self, text: str) -> int:
        with self.exclusive() as llm:
            return len(llm.tokenize(text.encode("utf-8")))

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
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        options = {}
        if schema is not None:
            options["response_format"] = {"type": "json_object", "schema": schema}
        with self.exclusive() as llm:
            prompt_tokens = len(llm.tokenize(f"{system}\n{user}".encode("utf-8"))) + TEMPLATE_TOKENS
            if prompt_tokens + max_tokens > self.spec.n_ctx:
                raise PromptTooLarge(prompt_tokens, self.spec.n_ctx - max_tokens)
            output = self.chat_completion(
                messages=messages, max_tokens=max_tokens, temperature=temperature,
                purpose=purpose, **options,
            )
        return (output["choices"][0]["message"]["content"] or "").strip()

    def chat_completion(self, *, purpose: str = DRAFT, **request) -> Dict:
        """create_chat_completion under the lock, metered."""
        return self.__metered(purpose, lambda llm: llm.create_chat_completion(**request))

    def completion(self, prompt: List[int] | str, *, purpose: str = DRAFT, **request) -> Dict:
        """create_completion under the lock, metered."""
        return self.__metered(purpose, lambda llm: llm.create_completion(prompt=prompt, **request))

    def __metered(self, purpose: str, call) -> Dict:
        with self.exclusive() as llm:
            if self.__meter is not None:
                self.__meter.start()
            started = time.monotonic()
            try:
                output = call(llm)
            except InferenceError:
                self.__failures += 1
                raise
            except Exception as error:
                self.__failures += 1
                raise InferenceError(f"{purpose} inference failed: {error}") from error
            finally:
                self.__busy_ms += (time.monotonic() - started) * 1000.0
            usage = output.get("usage") or {}
            prompt = int(usage.get("prompt_tokens") or 0)
            evaluated = self.__meter.first_batch if self.__meter and self.__meter.first_batch is not None else prompt
            self.__calls[purpose] = self.__calls.get(purpose, 0) + 1
            self.__prompt_tokens += prompt
            self.__evaluated += min(prompt, evaluated)
            self.__completion_tokens += int(usage.get("completion_tokens") or 0)
            self.last_evaluated_prompt_tokens = evaluated
            return output

    def stats(self) -> InferenceStats:
        with self.__lock:
            return InferenceStats(
                dict(self.__calls),
                self.__prompt_tokens,
                self.__evaluated,
                self.__completion_tokens,
                self.__failures,
                self.__busy_ms,
            )

    def chat_adapter(self, purpose: str) -> "ChatAdapter":
        return ChatAdapter(self, purpose)

    def close(self) -> None:
        with self.__lock:
            llm, self.__llm = self.__llm, None
            if llm is not None:
                close = getattr(llm, "close", None)
                if close is not None:
                    close()
                logger.info("Unloaded %s", self.spec.path.name)


class ChatAdapter:
    """Looks like a Llama to code that calls create_chat_completion, but runs on the shared one."""

    shared = True

    def __init__(self, model: SharedModel, purpose: str) -> None:
        self.model = model
        self.purpose = purpose

    def create_chat_completion(self, **request) -> Dict:
        return self.model.chat_completion(purpose=self.purpose, **request)


def _runtime_version() -> str:
    try:
        import llama_cpp
    except ImportError:
        return "llama_cpp=absent"
    version = getattr(llama_cpp, "__version__", None)
    return f"llama_cpp={version if isinstance(version, str) else 'unknown'}"
