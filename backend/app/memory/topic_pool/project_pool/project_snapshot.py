"""Builds a project's snapshot from the conversation snapshots beneath it."""

from pathlib import Path
from typing import Callable, List, Sequence, Tuple

from numpy import ndarray

from config import get_logger
from memory.snapshot import PROJECT, SnapShot
from storage.timestamps import utc_now
from memory.topic_pool.project_pool.conversation_pool.conversation_data_management.conversationVectorMetaManager import (
    ConversationVectorMetaDataRepository,
)
from memory.topic_pool.project_pool.project_data_repo.project_snapshot_repo import (
    ProjectSnapshotRepository,
)

logger = get_logger(__name__)

# Split system/user to match the draft model's chat interface, the same shape
# ConversationSummary.__build_prompt returns.
PROJECT_SNAPSHOT_SYSTEM = (
    "You are maintaining a running description of a software project. "
    "Write what the project is about and what has been done so far. "
    "Describe the project, not the conversations it was discussed in: no "
    "speaker labels, no mention of questions asked or answered."
)


def render_prompt(previous: str | None, new_summaries: Sequence[str]) -> str:
    """The user half of the prompt for one incremental project snapshot."""
    parts = []
    if previous:
        parts.append(
            "This is the description so far. Keep what still holds and revise "
            f"what has moved on:\n{previous}"
        )
    else:
        parts.append("There is no description yet; write the first one.")
    joined = "\n\n".join(f"- {summary}" for summary in new_summaries)
    parts.append(f"These conversation summaries are new since then:\n{joined}")
    return "\n\n".join(parts)


class ProjectSnapshot:
    """One project's rolling description, rebuilt as its conversations advance.

    Incremental by construction: each snapshot summarises the previous one
    together with the conversation summaries written since, so the cost of a
    snapshot tracks what is new rather than the size of the project. The
    watermark is `seq` on `cumulative_vector_meta_data`, which is allocated
    monotonically on write — `created_at` is caller-supplied and cannot order
    anything reliably.
    """

    def __init__(
        self,
        project_id: str,
        project_name: str,
        meta_repo: ConversationVectorMetaDataRepository,
        embed: Callable[[str], ndarray],
        conversation_dir: str | Path | None = None,
        project_db_path: str | Path | None = None,
        snapshot_repo: ProjectSnapshotRepository | None = None,
    ) -> None:
        self.project_id = project_id
        self.project_name = project_name
        self.meta_repo = meta_repo
        self.embed = embed
        self.snap_shot = SnapShot(
            conversation_dir=conversation_dir or meta_repo.conversation_dir,
            project_id=project_id,
            project_name=project_name,
            scope=PROJECT,
            snapshot_repo=snapshot_repo,
            project_db_path=project_db_path,
        )

    @property
    def snapshot_repo(self) -> ProjectSnapshotRepository:
        return self.snap_shot.snapshot_repo

    def pending(self) -> List[Tuple[int, int, str]]:
        """The conversation summaries this project has not folded in yet."""
        return self.meta_repo.get_project_snapshots_since(
            self.snapshot_repo.last_seq_included()
        )

    def take(self, summarise: Callable[[str, str], str]) -> int | None:
        """Summarise what is new into a fresh project snapshot.

        `summarise(system, user)` is supplied per call rather than held, because
        the draft model is expensive to load and its lifetime belongs to the
        caller — which lets this run inside a window where a model is already
        resident instead of loading a second one.

        Returns the new snapshot's id, or None when nothing has changed since
        the last one — the common case, since this runs on every conversation
        snapshot.
        """
        new = self.pending()
        if not new:
            logger.debug(
                "No new conversation summaries for project %s; "
                "leaving its snapshot alone",
                self.project_id,
            )
            return None

        previous = self.snapshot_repo.latest()
        prompt = render_prompt(
            previous.summary if previous is not None else None,
            [summary for _, _, summary in new],
        )
        summary = summarise(PROJECT_SNAPSHOT_SYSTEM, prompt)
        if not summary or not summary.strip():
            # A draft model that returns nothing must not advance the watermark:
            # doing so would drop those conversation summaries from every future
            # snapshot, since "new since" would step past them.
            logger.warning(
                "The summariser returned nothing for project %s; "
                "the watermark stays at %d",
                self.project_id, self.snapshot_repo.last_seq_included(),
            )
            return None

        highest_seq = max(seq for seq, _, _ in new)
        snapshot_id = self.snap_shot.add_project_snapshot(
            summary=summary,
            last_seq_included=highest_seq,
            summary_vector=self.embed(summary),
            created_at=utc_now(),
        )
        logger.info(
            "Project %s snapshot %s folded in %d new conversation summary(ies) "
            "up to seq %d",
            self.project_id, snapshot_id, len(new), highest_seq,
        )
        return snapshot_id

    def close(self) -> None:
        self.snap_shot.close()
