from __future__ import annotations

import gc
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import numpy as np

from config import Config, get_logger, log_timing
from memory.topic_pool.project_pool.conversation_pool.conversation_data_management.conversationVectorMetaManager import (
    ConversationVectorMetaDataRepository,
)
from memory.topic_pool.project_pool.conversation_pool.full_conversation_bucket import (
    FullConversation,
)
from memory.topic_pool.project_pool.conversation_pool.fullconversation_repository.fullconversation_repository import (
    Turn,
    utc_now,
)
from memory.topic_pool.project_pool.conversation_pool.snapshot import SnapShot

logger = get_logger(__name__)

_CHARS_PER_TOKEN = 4
_SYSTEM_OVERHEAD_TOKENS = 200
_OUTPUT_RESERVE_TOKENS = 512
_WINDOW_OVERLAP_CHUNKS = 50  # look-back overlap added to every context window


def _speaker(turn: Turn) -> str:
    return f"{turn.role.capitalize()}: "


def render_transcript(turns: List[Turn]) -> str:
    """The conversation as the draft model reads it: one labelled line per turn."""
    return "\n".join(_speaker(turn) + turn.text for turn in turns)


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

        self.snap_shot = SnapShot(
            conversation_dir=full_conversation_dir,
            project_id=project_id,
            project_name=project_name,
            meta_repo=self.summary_repo,
        )
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            from data_layer.ingestion.embedding.EmbeddingManager import (
                EmbeddingManager,
            )

            logger.debug("Loading the embedder for project %s", self.project_id)
            self._embedder = EmbeddingManager()
        return self._embedder

    def __window_start(self, chunk_sequence_number: int) -> int:
        """Where the summarisation window begins."""
        look_back = max(
            0,
            chunk_sequence_number
            - self.main_model_context_window_length
            - _WINDOW_OVERLAP_CHUNKS,
        )
        watermark = self.summary_repo.get_highest_summarised_sequence() or 0
        return max(0, min(look_back, watermark + 1))

    def __window_turns(self, chunk_sequence_number: int) -> List[Turn]:
        """The turns in the current conversation window."""
        start = self.__window_start(chunk_sequence_number)
        return self.full_conversation.get_turns(start, chunk_sequence_number)

    def __get_current_conversation(self, chunk_sequence_number: int) -> str:
        """The current window as a speaker-labelled transcript."""
        return render_transcript(self.__window_turns(chunk_sequence_number))

    def __cumulative_vector_id(self, summary: str, time_of_snapshot: str) -> int:
        """Snapshot-unique id for the cumulative vector."""
        payload = f"{self.project_id}\x00{time_of_snapshot}\x00{summary}"
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        return (
            int.from_bytes(digest[:8], byteorder="little", signed=False)
            & Config.VECTOR_ID_MASK
        )

    def __persist_snapshot(self, summary: str, covered_rows) -> None:
        """Embed the summary and its covered chunks, then store the snapshot."""
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
        logger.info("Loading draft model from %s", model_file)
        return Llama(
            model_path=str(model_file),
            n_ctx=self.draft_model_context_window_length,
            verbose=False,
        )

    def __unload_model(self, model) -> None:
        """Removes the model from RAM, frees VRAM KV cache if CUDA is present."""
        logger.debug("Unloading draft model")
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
            "Each line of the conversation begins with its speaker; keep track "
            "of who said what, and attribute requests, decisions and answers "
            "to the speaker they came from. "
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
        """Splits text into batches each at most max_chars long."""
        max_chars = max(1, max_chars)
        if len(text) <= max_chars:
            return [text]

        # An overlap at or above the batch size makes the step zero or
        # negative, and the loop never terminates.
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

    def __labelled_pieces(
        self, turn: Turn, max_chars: int, overlap_chars: int
    ) -> List[str]:
        """One turn as transcript lines, each at most max_chars, each labelled."""
        label = _speaker(turn)
        if len(label) + len(turn.text) <= max_chars:
            return [label + turn.text]
        budget = max_chars - len(label)
        if budget < 1:
            return self.__split_into_batches(label + turn.text, max_chars, overlap_chars)
        return [
            label + piece
            for piece in self.__split_into_batches(turn.text, budget, overlap_chars)
        ]

    @staticmethod
    def __trailing_lines(lines: List[str], limit: int) -> List[str]:
        """The longest run of whole lines from the end whose transcript fits limit."""
        tail: List[str] = []
        size = -1
        for line in reversed(lines):
            size += len(line) + 1
            if size > limit:
                break
            tail.append(line)
        tail.reverse()
        return tail

    def __batch_turns(
        self, turns: List[Turn], max_chars: int, overlap_chars: int
    ) -> List[str]:
        """Pack speaker-labelled turns into transcripts of at most max_chars."""
        max_chars = max(1, max_chars)
        lines: List[str] = []
        for turn in turns:
            lines.extend(self.__labelled_pieces(turn, max_chars, overlap_chars))
        if not lines:
            return [""]

        batches: List[str] = []
        current: List[str] = []
        size = 0
        for line in lines:
            if current and size + 1 + len(line) > max_chars:
                batches.append("\n".join(current))
                # Carry only as much as still leaves room for this line, so the
                # overlap can never push a batch past max_chars.
                room = min(overlap_chars, max_chars - len(line) - 1)
                current = self.__trailing_lines(current, room)
                size = len("\n".join(current))
            size += len(line) + (1 if current else 0)
            current.append(line)
        batches.append("\n".join(current))
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
        self, latest_summary: str | None, turns: List[Turn]
    ) -> str:
        """Full LLM pipeline:"""
        available_tokens = (
            self.draft_model_context_window_length
            - _SYSTEM_OVERHEAD_TOKENS
            - _OUTPUT_RESERVE_TOKENS
        )
        summary_tokens = self.__estimate_tokens(latest_summary or "")
        max_conv_chars = max(1, (available_tokens - summary_tokens) * _CHARS_PER_TOKEN)
        overlap_chars = _WINDOW_OVERLAP_CHUNKS * _CHARS_PER_TOKEN

        with ThreadPoolExecutor(max_workers=2) as executor:
            model_future = executor.submit(self.__load_model)
            batches_future = executor.submit(
                self.__batch_turns,
                turns,
                max_conv_chars,
                overlap_chars,
            )
            model = model_future.result()

        # Inside the guard: if resolving the batches raises out here, the
        # finally never runs and the model stays resident.
        try:
            batches = batches_future.result()
            running_summary: str | None = latest_summary
            for position, batch in enumerate(batches, start=1):
                system, user_content = self.__build_prompt(running_summary, batch)
                with log_timing(
                    logger,
                    f"summarising batch {position}/{len(batches)}",
                    level=logging.DEBUG,
                    chars=len(batch),
                ):
                    running_summary = self.__run_inference(model, system, user_content)
        finally:
            self.__unload_model(model)

        return running_summary or ""

    def get_current_conversation(self, chunk_sequence_number: int) -> str:
        return self.__get_current_conversation(chunk_sequence_number)

    def get_current_summary(self) -> str | None:
        return self.__get_latest_summary()

    def make_summary(self, chunk_sequence_number: int) -> str:
        """Fetches the latest cumulative summary and the current conversation"""
        latest_summary = self.get_current_summary()
        turns = self.__window_turns(chunk_sequence_number)
        return self.__generate_summary(latest_summary, turns)

    def take_snapshot(self, chunk_sequence_number: int) -> str | None:
        """Summarise the window up to chunk_sequence_number and persist it."""
        # One start for both reads: it depends on the watermark, which a
        # snapshot landing in between would move.
        start = self.__window_start(chunk_sequence_number)
        covered_rows = self.full_conversation.get_context_rows(
            start, chunk_sequence_number
        )
        if not covered_rows:
            logger.debug(
                "Nothing to summarise up to sequence %d", chunk_sequence_number
            )
            return None

        latest_summary = self.get_current_summary()
        turns = self.full_conversation.get_turns(start, chunk_sequence_number)
        summary = self.__generate_summary(latest_summary, turns)
        if not summary:
            logger.warning(
                "Draft model returned an empty summary for project %s; "
                "no snapshot taken",
                self.project_id,
            )
            return None

        self.__persist_snapshot(summary, covered_rows)
        logger.info(
            "Snapshot taken for project %s: %d chunk(s), %d-character summary",
            self.project_id,
            len(covered_rows),
            len(summary),
        )
        return summary

    def close(self) -> None:
        """Release every connection this object opened."""
        self.snap_shot.close()
        self.summary_repo.close()
