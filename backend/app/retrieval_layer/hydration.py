"""Turning search results back into text."""

import sqlite3
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from config import Config, get_logger
from retrieval_layer.models import Passage, ScoredId

logger = get_logger(__name__)

# One statement covering whichever chunk tables exist. A chunk lives in
# `Chunks` when its document had sections and in `RecursiveChunks` when it did
# not, and a corpus of only one kind has only one of the tables — naming both
# unconditionally fails the whole query with "no such table".
_CHUNK_TABLES = (("Chunks", "contextId"), ("RecursiveChunks", "documentId"))

_LEG = """
    select v.vectorId, v.chunkId, c.chunk, c.startoffset, c.endoffset, c.{parent}
    from vector_meta_data v
    join {table} c on c.chunkId = v.chunkId
    where v.vectorId in ({placeholders})
"""


class Hydration:
    """Reads the text for a page of search results in a single query.

    The one stage where an obvious implementation is quadratically wrong: a
    search returns k ids at once, and fetching them one at a time is the N+1
    that makes retrieval slow long before the index does.
    """

    def __init__(self, chunk_store_path: str | Path = Config.DB_PATH) -> None:
        self.chunk_store_path = Path(chunk_store_path)

    def __rows(self, vector_ids: Sequence[int]) -> Dict[int, tuple]:
        if not vector_ids:
            return {}
        if not self.chunk_store_path.exists():
            logger.warning("No chunk store at %s", self.chunk_store_path)
            return {}
        placeholders = ",".join("?" * len(vector_ids))
        connection = sqlite3.connect(
            f"file:{self.chunk_store_path}?mode=ro", uri=True
        )
        try:
            present = {
                row[0] for row in connection.execute(
                    "select name from sqlite_master where type='table'"
                )
            }
            legs = [
                _LEG.format(table=table, parent=parent, placeholders=placeholders)
                for table, parent in _CHUNK_TABLES
                if table in present
            ]
            if not legs:
                logger.warning("No chunk tables in %s", self.chunk_store_path)
                return {}
            statement = " union all ".join(legs)
            params = tuple(vector_ids) * len(legs)
            return {row[0]: row for row in connection.execute(statement, params)}
        except sqlite3.OperationalError as error:
            logger.warning("Could not hydrate: %s", error)
            return {}
        finally:
            connection.close()

    def hydrate(self, scored: Sequence[ScoredId]) -> Tuple[List[Passage], int]:
        """Passages in the order given, and how many ids had no text.

        An id can fail to resolve: DiskANN pads a short result set with labels
        that were never allocated, and an index can outlive the chunks it was
        built from. Those are dropped rather than raised — a stale entry is not
        a reason to fail a search that found other things.
        """
        if not scored:
            return [], 0
        rows = self.__rows([item.vector_id for item in scored])

        passages: List[Passage] = []
        for item in scored:
            row = rows.get(item.vector_id)
            if row is None:
                continue
            _, chunk_id, text, start, end, parent = row
            passages.append(Passage(
                vector_id=item.vector_id,
                chunk_id=chunk_id,
                text=text,
                score=item.score,
                source=item.source,
                document_id=parent,
                start_offset=start,
                end_offset=end,
            ))
        dropped = len(scored) - len(passages)
        if dropped:
            logger.debug("%d search result(s) had no chunk behind them", dropped)
        return passages, dropped
