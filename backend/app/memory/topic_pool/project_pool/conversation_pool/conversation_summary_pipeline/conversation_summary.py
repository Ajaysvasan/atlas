from __future__ import annotations

import gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from config import Config
from memory.topic_pool.project_pool.conversation_pool.conversation_data_management.conversationVectorMetaManager import (
    ConversationVectorMetaDataRepository,
)
from memory.topic_pool.project_pool.conversation_pool.full_conversation_bucket import (
    FullConversation,
)

_CHARS_PER_TOKEN = 4
_SYSTEM_OVERHEAD_TOKENS = 200
_OUTPUT_RESERVE_TOKENS = 512
_WINDOW_OVERLAP_CHUNKS = 50  # look-back overlap added to every context window


class ConversationSummary:
    def __init__(
        self,
        full_conversation_dir: str | Path,
        project_id: str,
        project_name: str,
        main_model_context_window_length: int,
        draft_model_context_window_length: int,
    ) -> None:
        self.full_conversation = FullConversation(
            full_conversation_dir=full_conversation_dir,
            project_id=project_id,
            project_name=project_name,
        )
        self.main_model_context_window_length = main_model_context_window_length
        self.summary_repo = ConversationVectorMetaDataRepository(
            full_conversation_dir, project_id
        )
        self.draft_model_context_window_length = draft_model_context_window_length

    def __get_current_conversation(self, chunk_sequence_number: int) -> str:
        """
        Returns the current conversation window as a single string.
        Extends the window back by _WINDOW_OVERLAP_CHUNKS so no context
        is lost at window boundaries, and clamps the start to 0 so
        negative indices never reach the DB (fixes Bug 4.29).
        """
        start = max(
            0,
            chunk_sequence_number
            - self.main_model_context_window_length
            - _WINDOW_OVERLAP_CHUNKS,
        )
        rows = self.full_conversation.get_context(start, chunk_sequence_number)
        return " ".join(row[0] for row in rows)

    def __get_latest_summary(self) -> str | None:
        return self.summary_repo.get_latest_summary()

    def __load_model(self):
        """Loads the GGUF draft model from Config.DRAFT_MODEL_PATH."""
        from llama_cpp import Llama  # lazy — not needed on every import

        model_file = Path(Config.DRAFT_MODEL_PATH) / Config.DRAFT_MODEL_FILE
        if not model_file.exists():
            raise FileNotFoundError(
                f"Draft model not found at {model_file}.\n"
                "Run:  python download_models/download_draft_model.py"
            )
        return Llama(
            model_path=str(model_file),
            n_ctx=Config.DRAFT_MODEL_CONTEXT_WINDOW,
            verbose=False,
        )

    def __unload_model(self, model) -> None:
        """Removes the model from RAM, frees VRAM KV cache if CUDA is present."""
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def __build_prompt(
        self, latest_summary: str | None, conversation_chunk: str
    ) -> tuple[str, str]:
        """Returns (system_prompt, user_content) ready for chat completion."""
        system = (
            "You are a concise summariser. "
            "Given the previous cumulative summary (if any) and a new block of "
            "conversation, produce a single updated cumulative summary. "
            "Preserve key facts and decisions. Be concise."
        )
        if latest_summary:
            user_content = (
                f"Previous summary:\n{latest_summary}\n\n"
                f"New conversation:\n{conversation_chunk}\n\n"
                "Produce an updated cumulative summary."
            )
        else:
            user_content = (
                f"Conversation:\n{conversation_chunk}\n\n"
                "Produce a concise summary of this conversation."
            )
        return system, user_content

    def __estimate_tokens(self, text: str) -> int:
        return len(text) // _CHARS_PER_TOKEN

    def __split_into_batches(
        self, text: str, max_chars: int, overlap_chars: int
    ) -> List[str]:
        """
        Splits text into batches each at most max_chars long.
        Adjacent batches share overlap_chars of text so no context
        is dropped at batch boundaries.
        """
        if len(text) <= max_chars:
            return [text]
        batches: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            batches.append(text[start:end])
            if end == len(text):
                break
            start = end - overlap_chars
        return batches

    def __run_inference(self, model, system: str, user_content: str) -> str:
        output = model.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            max_tokens=512,
            temperature=0.1,
        )
        return output["choices"][0]["message"]["content"].strip()

    def __generate_summary(
        self, latest_summary: str | None, current_conversation: str
    ) -> str:
        """
        Full LLM pipeline:
        - Thread A loads the model while Thread B computes conversation batches.
        - Batches are processed sequentially; each batch's output becomes the
          running summary fed into the next batch (rolling summarisation).
        - If the combined token budget fits in one pass, only one inference call
          is made.
        - Model is unloaded and VRAM freed unconditionally before returning.
        """
        available_tokens = (
            Config.DRAFT_MODEL_CONTEXT_WINDOW
            - _SYSTEM_OVERHEAD_TOKENS
            - _OUTPUT_RESERVE_TOKENS
        )
        summary_tokens = self.__estimate_tokens(latest_summary or "")
        max_conv_chars = max(1, (available_tokens - summary_tokens) * _CHARS_PER_TOKEN)
        overlap_chars = _WINDOW_OVERLAP_CHUNKS * _CHARS_PER_TOKEN

        # Thread A: load model | Thread B: split conversation into batches
        with ThreadPoolExecutor(max_workers=2) as executor:
            model_future = executor.submit(self.__load_model)
            batches_future = executor.submit(
                self.__split_into_batches,
                current_conversation,
                max_conv_chars,
                overlap_chars,
            )
            model = model_future.result()
            batches = batches_future.result()

        try:
            running_summary: str | None = latest_summary
            for batch in batches:
                system, user_content = self.__build_prompt(running_summary, batch)
                running_summary = self.__run_inference(model, system, user_content)
        finally:
            self.__unload_model(model)

        return running_summary or ""

    def get_current_conversation(self, chunk_sequence_number: int) -> str:
        return self.__get_current_conversation(chunk_sequence_number)

    def get_current_summary(self) -> str | None:
        return self.__get_latest_summary()

    def make_summary(self, chunk_sequence_number: int) -> str:
        """
        Fetches the latest cumulative summary and the current conversation
        window, then runs the draft-model LLM pipeline and returns the new
        cumulative summary string.
        """
        latest_summary = self.get_current_summary()
        current_conversation = self.get_current_conversation(chunk_sequence_number)
        return self.__generate_summary(latest_summary, current_conversation)
