"""The process's models: one draft instance, and the main role served by it unless configured apart."""

import threading
from pathlib import Path

from config import Config, get_logger
from inference.shared_model import ModelSpec, SharedModel

logger = get_logger(__name__)


class InferenceRuntime:
    def __init__(self, draft: SharedModel, main: SharedModel | None = None) -> None:
        self.draft = draft
        self.main = draft if main is None else main

    @property
    def separate_main(self) -> bool:
        return self.main is not self.draft

    def close(self) -> None:
        self.draft.close()
        if self.separate_main:
            self.main.close()


_runtime: InferenceRuntime | None = None
_runtime_lock = threading.Lock()


def build_runtime() -> InferenceRuntime:
    draft = SharedModel(ModelSpec.draft())
    main = None
    if Config.MAIN_MODEL_PATH:
        main = SharedModel(ModelSpec(Path(Config.MAIN_MODEL_PATH)))
    return InferenceRuntime(draft, main)


def shared_runtime() -> InferenceRuntime:
    """The process-wide runtime; nothing is loaded until a model is first used."""
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = build_runtime()
        return _runtime


def install_runtime(runtime: InferenceRuntime | None) -> None:
    """Replace the process-wide runtime (tests, or an entry point that built its own)."""
    global _runtime
    with _runtime_lock:
        _runtime = runtime


class ReasoningModel:
    """Worker and stitch jobs go to the main model; planning, critique and naming stay on the draft."""

    def __init__(self, runtime: InferenceRuntime) -> None:
        self.runtime = runtime

    def infer(self, system, user, **options) -> str:
        from inference.model_port import WORKER

        model = self.runtime.main if options.get("purpose") == WORKER else self.runtime.draft
        return model.infer(system, user, **options)
