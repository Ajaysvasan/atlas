"""Lexical search over a project's conversation turns, using SQLite's FTS5 and bm25."""

import sqlite3
from pathlib import Path
from typing import Dict, List, Sequence

from config import get_logger
from data_layer.ingestion.embedding.vector_ids import vector_id_for
from memory.local_retrieval.turn_scope import TurnScope
from memory.memory_database import MemoryDatabase, Schema
from memory.topic_pool.project_pool.conversation_pool.fullconversation_repository.fullconversation_repository import (
    SCHEMA as TURN_SCHEMA,
    ScopedTurn,
    conversation_ids,
    turns_after_sequence,
)
from retrieval_layer.keyword_search import match_expression
from retrieval_layer.models import KEYWORD, QueryPlan, ScoredId, as_ranked

logger = get_logger(__name__)


def _create_tables(cursor: sqlite3.Cursor) -> None:
    # turn_key is the turn's vector id, so a keyword hit and a vector hit on
    # the same turn meet in fusion under one key.
    cursor.execute("""
        create table if not exists turn_index(
            turn_key integer primary key,
            chunk_id text not null unique references summary_chunks(chunk_id),
            project_id text not null references project_table(project_id),
            conversation_id text not null,
            sequence_number integer not null,
            foreign key (conversation_id, sequence_number)
                references full_conversation(conversation_id, sequence_number)
        )
    """)
    cursor.execute(
        "create index if not exists idx_turn_index_position "
        "on turn_index(project_id, conversation_id, sequence_number)"
    )
    cursor.execute("""
        create virtual table if not exists turn_fts
            using fts5(body, content='', tokenize='porter unicode61')
    """)


SCHEMA = Schema("turn_search_index", _create_tables, requires=(TURN_SCHEMA,))


class TurnKeywordSearch:
    """The sparse half of local retrieval, and the registry of every indexed turn's key."""

    def __init__(self, database: MemoryDatabase | str | Path | None = None) -> None:
        self.database = MemoryDatabase.of(database)
        self.database.ensure(SCHEMA)

    def __unindexed(self, scope: TurnScope) -> List[ScopedTurn]:
        # Sequence numbers are allocated in order under BEGIN IMMEDIATE, so a
        # conversation's highest indexed one is everything it has seen.
        query = (
            "select conversation_id, max(sequence_number) from turn_index "
            "where project_id = ?"
        )
        params = (scope.project_id,)
        if scope.conversation_only:
            query += " and conversation_id = ?"
            params += (scope.conversation_id,)
        with self.database.reading() as cursor:
            watermarks = dict(cursor.execute(query + " group by conversation_id", params))
        conversations = (
            [scope.conversation_id]
            if scope.conversation_only
            else conversation_ids(scope.project_id, self.database)
        )
        fresh: List[ScopedTurn] = []
        for conversation_id in conversations:
            fresh.extend(turns_after_sequence(
                scope.project_id, conversation_id,
                watermarks.get(conversation_id, 0), self.database,
            ))
        return fresh

    def sync(self, scope: TurnScope) -> int:
        """Index the turns in scope written since the last sync. Returns how many."""
        fresh = self.__unindexed(scope)
        if not fresh:
            return 0
        added = 0
        with self.database.writing() as cursor:
            for turn in fresh:
                key = vector_id_for(turn.chunk_id)
                cursor.execute(
                    "insert or ignore into turn_index"
                    "(turn_key, chunk_id, project_id, conversation_id, sequence_number) "
                    "values (?, ?, ?, ?, ?)",
                    (key, turn.chunk_id, scope.project_id, turn.conversation_id,
                     turn.sequence_number),
                )
                # turn_fts takes a rowid twice without complaint, so a turn a
                # concurrent sync has just indexed must not reach it again.
                if cursor.rowcount == 1:
                    cursor.execute(
                        "insert into turn_fts(rowid, body) values (?, ?)", (key, turn.text)
                    )
                    added += 1
        logger.debug("Indexed %d turn(s) for keyword search", added)
        return added

    def size(self, scope: TurnScope) -> int:
        """How many turns in scope are indexed."""
        query = "select count(*) from turn_index where project_id = ?"
        params = (scope.project_id,)
        if scope.conversation_only:
            query += " and conversation_id = ?"
            params += (scope.conversation_id,)
        with self.database.reading() as cursor:
            return cursor.execute(query, params).fetchone()[0]

    def chunk_ids_for(self, keys: Sequence[int]) -> Dict[int, str]:
        """The chunk id behind each indexed key; keys never indexed are absent."""
        if not keys:
            return {}
        placeholders = ",".join("?" * len(keys))
        with self.database.reading() as cursor:
            rows = cursor.execute(
                f"select turn_key, chunk_id from turn_index where turn_key in ({placeholders})",
                tuple(int(key) for key in keys),
            ).fetchall()
        return dict(rows)

    def search(self, plan: QueryPlan, k: int, scope: TurnScope) -> List[ScoredId]:
        """The k best lexical matches in scope, best first."""
        if k < 1:
            return []
        expression = match_expression(plan.terms)
        if not expression:
            logger.debug("Query had no searchable terms")
            return []
        clauses = ["i.project_id = ?"]
        params: list = [scope.project_id]
        if scope.conversation_only:
            clauses.append("i.conversation_id = ?")
            params.append(scope.conversation_id)
        if scope.before_sequence is not None:
            clauses.append("not (i.conversation_id = ? and i.sequence_number >= ?)")
            params += [scope.conversation_id, scope.before_sequence]
        try:
            with self.database.reading() as cursor:
                rows = cursor.execute(
                    "select i.turn_key, bm25(turn_fts) as score from turn_fts "
                    "join turn_index as i on i.turn_key = turn_fts.rowid "
                    f"where turn_fts match ? and {' and '.join(clauses)} "
                    "order by score limit ?",
                    (expression, *params, k),
                ).fetchall()
        except sqlite3.OperationalError as error:
            logger.warning("Keyword search over turns failed: %s", error)
            return []
        # bm25 is negative and more negative is better, so it ranks like a distance.
        ranked = as_ranked(rows, KEYWORD, descending=False)
        logger.debug("Keyword search returned %d turn(s)", len(ranked))
        return ranked
