"""Lexical search over the chunk text, using SQLite's own FTS5 and bm25."""

import re
import sqlite3
from pathlib import Path
from typing import List

from config import Config, get_logger
from retrieval_layer.models import KEYWORD, QueryPlan, ScoredId, as_ranked
from storage.sqlite_setup import connect, enable_wal

logger = get_logger(__name__)

FTS_PATH = Config.DATA_DIR / Path("retrieval/keyword_index.sql")

_WORD = re.compile(r"[A-Za-z0-9_]+")


def match_expression(text: str) -> str:
    """Turn free text into something FTS5 will accept.

    A query typed by a person contains quotes, hyphens and words like AND and
    NEAR, all of which mean something to FTS5 and raise a syntax error when
    they are not meant. Pulling out the word characters and quoting each one
    leaves a query that always parses and matches on any term.
    """
    terms = _WORD.findall(text)
    return " OR ".join(f'"{term}"' for term in terms)


class KeywordSearch:
    """The sparse half of retrieval.

    Its index lives in a file of its own rather than inside the chunk store,
    so rebuilding it cannot disturb the ingested corpus.
    """

    def __init__(
        self,
        mapping,
        chunk_store_path: str | Path = Config.DB_PATH,
        fts_path: str | Path = FTS_PATH,
    ) -> None:
        self.mapping = mapping
        self.chunk_store_path = Path(chunk_store_path)
        self.fts_path = Path(fts_path)
        self.fts_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = connect(self.fts_path, check_same_thread=False)
        self.journal_mode = enable_wal(self.connection, self.fts_path)
        self.__init_db()

    def __init_db(self) -> None:
        # porter, not the default tokenizer: without stemming a query for
        # "write ahead logging" misses a chunk that says "writers" and
        # "journals", which is most of them.
        self.connection.executescript("""
            create virtual table if not exists chunk_fts
                using fts5(chunkId unindexed, body,
                           tokenize='porter unicode61');
            create table if not exists fts_synced(chunkId text primary key);
        """)
        self.connection.commit()

    def sync(self) -> int:
        """Bring the index up to date with the chunk store. Returns rows added.

        Incremental by design: a full rebuild would re-tokenise the whole corpus
        on every ingest, and the corpus only ever grows.
        """
        if not self.chunk_store_path.exists():
            logger.warning("No chunk store at %s to index", self.chunk_store_path)
            return 0
        source = sqlite3.connect(f"file:{self.chunk_store_path}?mode=ro", uri=True)
        try:
            rows = []
            for table in ("Chunks", "RecursiveChunks"):
                try:
                    rows.extend(
                        source.execute(f"select chunkId, chunk from {table}")
                    )
                except sqlite3.OperationalError:
                    continue
        finally:
            source.close()

        known = {
            row[0] for row in self.connection.execute("select chunkId from fts_synced")
        }
        fresh = [(cid, text) for cid, text in rows if cid not in known]
        if not fresh:
            return 0
        self.connection.executemany(
            "insert into chunk_fts(chunkId, body) values (?, ?)", fresh
        )
        self.connection.executemany(
            "insert or ignore into fts_synced(chunkId) values (?)",
            [(cid,) for cid, _ in fresh],
        )
        self.connection.commit()
        logger.info("Indexed %d new chunk(s) for keyword search", len(fresh))
        return len(fresh)

    def search(self, plan: QueryPlan, k: int) -> List[ScoredId]:
        """The k best lexical matches, best first."""
        if k < 1:
            return []
        expression = match_expression(plan.terms)
        if not expression:
            logger.debug("Query had no searchable terms")
            return []
        try:
            rows = self.connection.execute(
                "select chunkId, bm25(chunk_fts) as score from chunk_fts "
                "where chunk_fts match ? order by score limit ?",
                (expression, k),
            ).fetchall()
        except sqlite3.OperationalError as error:
            logger.warning("Keyword search failed: %s", error)
            return []

        # bm25 returns a negative number, more negative being a better match,
        # so this ranks the same way a distance does.
        labels = self.mapping.vector_ids_for([row[0] for row in rows])
        pairs = [
            (labels[chunk_id], score)
            for chunk_id, score in rows
            if chunk_id in labels
        ]
        ranked = as_ranked(pairs, KEYWORD, descending=False)
        logger.debug("Keyword search returned %d candidate(s)", len(ranked))
        return ranked

    def close(self) -> None:
        conn = getattr(self, "connection", None)
        if conn is not None:
            conn.close()
            self.connection = None
