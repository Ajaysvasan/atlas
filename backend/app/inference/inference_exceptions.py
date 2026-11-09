"""Errors raised by the shared inference runtime."""


class InferenceError(RuntimeError):
    """Base for every inference failure; the thinking layer retries these as model failures."""


class ModelUnavailable(InferenceError):
    """The model file is missing or could not be loaded."""


class ModelFileMissing(ModelUnavailable, FileNotFoundError):
    """The model file is not on disk; a FileNotFoundError too, as callers caught before."""


class PromptTooLarge(InferenceError):
    """A prompt plus its reply would not fit the context window. Retrying cannot help."""

    def __init__(self, prompt_tokens: int, limit: int) -> None:
        super().__init__(f"prompt of {prompt_tokens} tokens exceeds the {limit} available")
        self.prompt_tokens = prompt_tokens
        self.limit = limit


class CheckpointError(InferenceError):
    """A checkpoint could not be written or read; the caller recomputes instead."""
