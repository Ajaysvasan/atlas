from __future__ import annotations

import gc
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import numpy as np

from config import Config
from memory.topic_pool.project_pool.conversation_pool.conversation_data_management.conversationVectorMetaManager import (
    ConversationVectorMetaDataRepository,
)
from memory.topic_pool.project_pool.conversation_pool.full_conversation_bucket import (
    FullConversation,
)
from memory.topic_pool.project_pool.conversation_pool.fullconversation_repository.fullconversation_repository import (
    utc_now,
)
from memory.topic_pool.project_pool.conversation_pool.snapshot import SnapShot

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
        self.conversation_dir = full_conversation_dir
        self.project_id = project_id
        self.project_name = project_name
        self.main_model_context_window_length = main_model_context_window_length
        self.summary_repo = ConversationVectorMetaDataRepository(
            full_conversation_dir, project_id
        )
        self.draft_model_context_window_length = draft_model_context_window_length

        # Same directory as the summary repo, so the snapshot it writes is the
        # one get_latest_summary() reads back on the next round. The repository
        # is handed over rather than rebuilt, so both sides share one connection.
        self.snap_shot = SnapShot(
            conversation_dir=full_conversation_dir,
            project_id=project_id,
            project_name=project_name,
            meta_repo=self.summary_repo,
        )
        # SentenceTransformer weights are ~100MB; only pay for them if a
        # snapshot is actually taken.
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            from data_layer.ingestion.embedding.EmbeddingManager import (
                EmbeddingManager,
            )

            self._embedder = EmbeddingManager()
        return self._embedder

    def __window_start(self, chunk_sequence_number: int) -> int:
        """Where the summarisation window begins.

        The look-back alone is not enough. It is anchored to the newest turn, so
        whenever more unsummarised turns have piled up than the window spans —
        a bulk import, a large snapshot_every_n_turns, a period where
        snapshotting failed — every turn older than the window was skipped,
        while the watermark still advanced to the newest one and declared them
        summarised. They could never be recovered.

        Starting no later than the first unsummarised turn closes that gap; the
        window may then be larger than the look-back, which is what the batching
        in __generate_summary is for.
        """
        look_back = max(
            0,
            chunk_sequence_number
            - self.main_model_context_window_length
            - _WINDOW_OVERLAP_CHUNKS,
        )
        watermark = self.summary_repo.get_highest_summarised_sequence() or 0
        return max(0, min(look_back, watermark + 1))

    def __get_current_conversation(self, chunk_sequence_number: int) -> str:
        """
        Returns the current conversation window as a single string.
        Extends the window back by _WINDOW_OVERLAP_CHUNKS so no context
        is lost at window boundaries, and clamps the start to 0 so
        negative indices never reach the DB (fixes Bug 4.29).
        """
        start = self.__window_start(chunk_sequence_number)
        rows = self.full_conversation.get_context(start, chunk_sequence_number)
        return " ".join(row[0] for row in rows)

    def __get_covered_chunks(self, chunk_sequence_number: int):
        """The chunk rows this snapshot will claim to cover."""
        start = self.__window_start(chunk_sequence_number)
        return self.full_conversation.get_context_rows(start, chunk_sequence_number)

    def __cumulative_vector_id(self, summary: str, time_of_snapshot: str) -> int:
        """Snapshot-unique id for the cumulative vector.

        The embedder derives ids from content, which is right for chunks — the
        same text should map to the same vector — but wrong here:
        cumulative_vector_id is a PRIMARY KEY and two snapshots can legitimately
        produce identical summary text (a conversation whose gist stops
        changing, at temperature 0.1). That collided with IntegrityError and
        lost the snapshot. Binding the timestamp makes the id unique per
        snapshot while staying deterministic.
        """
        payload = f"{self.project_id}\x00{time_of_snapshot}\x00{summary}"
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        return (
            int.from_bytes(digest[:8], byteorder="little", signed=False)
            & Config.VECTOR_ID_MASK
        )

    def __persist_snapshot(self, summary: str, covered_rows) -> None:
        """Embed the summary and its covered chunks, then store the snapshot.

        Two levels are written: one cumulative vector for the summary as a whole
        (what SnapShot.search() ranks on) and one vector per covered chunk
        (reachable afterwards through summary_snapshot_map).
        """
        time_of_snapshot = utc_now()
        cumulative = self.embedder.embed_text(summary)
        chunk_embeddings = self.embedder.embed_texts(
            [row[1] for row in covered_rows],
            [row[0] for row in covered_rows],
        )

        self.snap_shot.add(
            time_of_snapshot=time_of_snapshot,
            len_of_the_summary=len(summary),
            summary_vector_ids=[e.vector_id for e in chunk_embeddings],
            summary_vectors=np.array(
                [e.vector for e in chunk_embeddings], dtype=np.float32
            ),
            chunk_ids=[row[0] for row in covered_rows],
            chunks=[tuple(row) for row in covered_rows],
            summary=summary,
            cumulative_summary_vector_id=self.__cumulative_vector_id(
                summary, time_of_snapshot
            ),
            cumulative_summary_vector=cumulative.vector,
        )

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
            # The injected window, not the global: loading with a different
            # n_ctx than the token budget below assumed would let a prompt sized
            # against the budget overflow the model's actual context.
            n_ctx=self.draft_model_context_window_length,
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
        max_chars = max(1, max_chars)
        if len(text) <= max_chars:
            return [text]

        # The cursor advances by (max_chars - overlap_chars) each pass. If the
        # overlap were allowed to reach or exceed the batch size that step would
        # be zero or negative, and since a batch is appended every iteration the
        # loop would never terminate — it consumes memory until the process is
        # killed rather than raising, so nothing upstream could catch it.
        # Clamping keeps progress strictly positive.
        overlap_chars = max(0, min(overlap_chars, max_chars - 1))

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
            self.draft_model_context_window_length
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

        # Everything after the model exists belongs inside the guard. Resolving
        # the batches future out here previously meant that if it raised, the
        # finally below was never reached and a multi-gigabyte model stayed
        # resident with its VRAM unreleased.
        try:
            batches = batches_future.result()
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

    def take_snapshot(self, chunk_sequence_number: int) -> str | None:
        """Summarise the window up to chunk_sequence_number and persist it.

        This is the round trip make_summary() alone never completed: the summary
        is embedded, stored as a snapshot, and becomes the "previous cumulative
        summary" the next call reads back. Returns the summary, or None when the
        window is empty and there is nothing to summarise.
        """
        covered_rows = self.__get_covered_chunks(chunk_sequence_number)
        if not covered_rows:
            return None

        latest_summary = self.get_current_summary()
        conversation = " ".join(row[1] for row in covered_rows)
        summary = self.__generate_summary(latest_summary, conversation)
        if not summary:
            return None

        self.__persist_snapshot(summary, covered_rows)
        return summary

    def close(self) -> None:
        """Release the metadata connection shared with the SnapShot."""
        self.summary_repo.close()
