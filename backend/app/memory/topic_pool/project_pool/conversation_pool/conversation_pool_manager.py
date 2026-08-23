"""
One conversation's worth of memory, assembled into a single object.

FullConversation (storage), ConversationSummary (summarisation) and SnapShot
(snapshot history and search) each work in isolation but have to be wired
together consistently: the same directory, the same project identity, and one
shared SnapShot so its cursors survive between calls. Building them ad hoc is
how the summariser ended up reading a different database than the snapshot
writer. This manager is the only place that wiring lives.
"""

from pathlib import Path
from typing import Iterable, List, Tuple

from config import Config

from .conversation_summary_pipeline.conversation_summary import ConversationSummary
from .full_conversation_bucket import FullConversation


class ConversationPoolManager:
    def __init__(
        self,
        conversation_dir: str | Path,
        project_id: str,
        project_name: str,
        main_model_context_window_length: int = Config.MAIN_MODEL_CONTEXT_WINDOW_TURNS,
        draft_model_context_window_length: int = Config.DRAFT_MODEL_CONTEXT_WINDOW,
        snapshot_every_n_turns: int = Config.SNAPSHOT_EVERY_N_TURNS,
    ) -> None:
        if snapshot_every_n_turns < 1:
            raise ValueError("snapshot_every_n_turns must be at least 1")

        self.conversation_dir = Path(conversation_dir)
        self.project_id = project_id
        self.project_name = project_name
        self.snapshot_every_n_turns = snapshot_every_n_turns

        self.full_conversation = FullConversation(
            full_conversation_dir=self.conversation_dir,
            project_id=project_id,
            project_name=project_name,
        )
        self.summariser = ConversationSummary(
            full_conversation_dir=self.conversation_dir,
            project_id=project_id,
            project_name=project_name,
            main_model_context_window_length=main_model_context_window_length,
            draft_model_context_window_length=draft_model_context_window_length,
        )
        # Deliberately the summariser's own instances, not new ones. A second
        # SnapShot would carry its own cursors and the two would drift apart.
        self.snap_shot = self.summariser.snap_shot
        self.meta_repo = self.summariser.summary_repo

        self.__restore_cursors()

    def __restore_cursors(self) -> None:
        """Point the snapshot cursors at history already on disk.

        Cursors are in-memory only, so a manager constructed against an existing
        project starts at -1/-1 and search() would scan an empty range even with
        snapshots stored.
        """
        if self.meta_repo.get_cumulative_vector_meta_data_ids():
            self.snap_shot.sync_cursors()

    def add_turn(self, role: str, text: str) -> int:
        """Append one turn. Returns its sequence number."""
        return self.full_conversation.append_turn(role, text)

    def add_turns(self, turns: Iterable[Tuple[str, str]]) -> List[int]:
        return self.full_conversation.append_turns(turns)

    def record_turn(self, role: str, text: str) -> Tuple[int, str | None]:
        """Append a turn and snapshot if enough have accumulated.

        The one call a caller needs per conversation turn. Returns
        (sequence_number, summary) where summary is None if no snapshot was due.
        """
        sequence = self.add_turn(role, text)
        return sequence, self.maybe_snapshot()

    def latest_sequence(self) -> int:
        """Sequence number of the most recent turn; 0 when empty."""
        return self.full_conversation.next_sequence_number() - 1

    def size(self) -> int:
        return self.full_conversation.size()

    def history(self):
        return self.full_conversation.get_full_conversation()

    def recent(self, n: int):
        return self.full_conversation.get_last_n_chunks(n)

    def context(self, start: int, end: int):
        return self.full_conversation.get_context(start, end)

    def since(self, sequence: int):
        return self.full_conversation.get_conversation_since(sequence)

    def current_summary(self) -> str | None:
        return self.summariser.get_current_summary()

    def summarised_upto(self) -> int:
        """Highest sequence number covered by a snapshot; 0 when none."""
        return self.meta_repo.get_highest_summarised_sequence() or 0

    def turns_since_last_snapshot(self) -> int:
        return max(0, self.latest_sequence() - self.summarised_upto())

    def should_snapshot(self) -> bool:
        return self.turns_since_last_snapshot() >= self.snapshot_every_n_turns

    def snapshot_now(self) -> str | None:
        """Snapshot the current window regardless of the trigger threshold.

        Returns the new summary, or None when there is nothing to summarise.
        """
        latest = self.latest_sequence()
        if latest <= 0:
            return None
        return self.summariser.take_snapshot(latest)

    def maybe_snapshot(self) -> str | None:
        """Snapshot only if enough unsummarised turns have accumulated."""
        if not self.should_snapshot():
            return None
        return self.snapshot_now()

    def search(self, query: str):
        """Find the snapshot whose summary best matches a query string."""
        query_vector = self.summariser.embedder.embed_text(query).vector
        return self.search_vector(query_vector)

    def search_vector(self, query_vector):
        return self.snap_shot.search(query_vector)

    def snapshot_ids(self):
        return self.meta_repo.get_cumulative_vector_meta_data_ids()

    def snapshot_detail(self, cumulative_vector_id: int):
        return self.meta_repo.get_cumulative_vector_meta_data(cumulative_vector_id)

    def advance(self) -> None:
        self.snap_shot.advance()

    def prev(self) -> None:
        self.snap_shot.prev()

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Release the SQLite connection this manager holds open.

        One manager per live conversation, so without an explicit release the
        connections are only reclaimed whenever the garbage collector happens to
        run __del__ — and they scale with concurrent conversations until the
        process runs out of file descriptors.
        """
        self.summariser.close()

    def __enter__(self) -> "ConversationPoolManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
