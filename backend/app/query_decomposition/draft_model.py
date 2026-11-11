"""The draft model for one query's prompts: a lease on the shared instance, or a private load from a given file."""

import gc
from pathlib import Path
from typing import Dict

from config import Config, get_logger
from query_decomposition.query_decomposition_exceptions import QueryTooLarge

logger = get_logger(__name__)

# Role markers the chat template adds, which tokenize() does not count.
_TEMPLATE_TOKENS = 64


class DraftModel:
    """Usable inside a `with` block only.

    With no `model_file` it leases the process's shared draft model, which
    stays resident after the block. With a `model_file` it loads that file
    privately and releases it on leaving the block, whatever happened inside.
    """

    def __init__(
        self,
        context_window: int = Config.QUERY_DECOMPOSITION_CONTEXT_WINDOW,
        model_file: str | Path | None = None,
    ) -> None:
        self.context_window = context_window
        self.__shared = model_file is None
        self.model_file = Path(
            Path(Config.DRAFT_MODEL_PATH) / Config.DRAFT_MODEL_FILE
            if model_file is None
            else model_file
        )
        self.__model = None
        self.__lease = None

    def __enter__(self) -> "DraftModel":
        if self.__shared:
            from inference.runtime import shared_runtime

            self.__lease = shared_runtime().draft
            self.__lease.ensure_loaded()
            return self

        from llama_cpp import Llama  # not a dependency; installed separately

        if not self.model_file.exists():
            raise FileNotFoundError(
                f"Draft model not found at {self.model_file}.\n"
                "Run:  python download_models/download_draft_model.py"
            )
        from inference.devices import gpu_layers

        logger.info("Loading draft model from %s", self.model_file)
        options = dict(model_path=str(self.model_file), n_ctx=self.context_window, verbose=False)
        layers = gpu_layers()
        try:
            self.__model = Llama(**options, **({"n_gpu_layers": layers} if layers else {}))
        except Exception as error:
            if not layers:
                raise
            logger.warning("Could not load the draft model on the GPU (%s); loading it on the CPU", error)
            self.__model = Llama(**options)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.__lease = None
        model, self.__model = self.__model, None
        if model is None:
            return
        logger.debug("Unloading draft model")
        close = getattr(model, "close", None)
        if close is not None:
            close()
        del model
        gc.collect()

    def infer(
        self,
        system: str,
        user: str,
        schema: Dict | None = None,
        max_tokens: int = 512,
    ) -> str:
        """One chat completion; with a schema the reply is constrained to JSON matching it."""
        if self.__lease is not None:
            from inference.inference_exceptions import PromptTooLarge

            try:
                return self.__lease.infer(system, user, schema=schema, max_tokens=max_tokens)
            except PromptTooLarge as error:
                raise QueryTooLarge(error.prompt_tokens, error.limit) from error
        if self.__model is None:
            raise RuntimeError("The draft model is used inside its `with` block only")
        prompt_tokens = (
            len(self.__model.tokenize(f"{system}\n{user}".encode("utf-8")))
            + _TEMPLATE_TOKENS
        )
        if prompt_tokens + max_tokens > self.context_window:
            raise QueryTooLarge(prompt_tokens, self.context_window - max_tokens)
        options = {}
        if schema is not None:
            options["response_format"] = {"type": "json_object", "schema": schema}
        output = self.__model.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
            **options,
        )
        return (output["choices"][0]["message"]["content"] or "").strip()
