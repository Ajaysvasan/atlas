"""Every query the memory layer runs on a hot path uses an index.

A query plan is the only honest check here: the SQL does not change when an
index is missing, it just gets slower as the table grows, and no functional test
would ever notice. Each case below scanned the table before its index existed —
`full_conversation(chunk_id)` was 33x slower, and the watermark join 399x,
because SQLite built a transient index on every call.
"""

import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("psycopg", MagicMock())
sys.modules.setdefault("dotenv", MagicMock())

from memory.topic_pool.project_pool.conversation_pool.conversation_data_management.conversationVectorMetaManager import (  # noqa: E402
    ConversationVectorMetaDataRepository,
)
from memory.topic_pool.project_pool.conversation_pool.fullconversation_repository.fullconversation_repository import (  # noqa: E402
    FullConversationRepository,
)
from memory.topic_pool.project_pool.project_data_repo.project_meta_data import (  # noqa: E402
    ProjectMetaData,
)
from memory.topic_pool.topic_pool_repo.topic_pool_meta_handler import (  # noqa: E402
    TopicPoolMetaHandler,
)


ROWS = 400


@pytest.fixture
def stores(tmp_path):
    """One of every database the memory layer creates, with rows in them.

    The rows are the point. SQLite plans from row estimates, so an empty table
    hides the very choices these tests exist to pin — with nothing in it the
    planner will not bother building the AUTOMATIC index that a populated one
    provokes.
    """
    topic = TopicPoolMetaHandler(tmp_path / "topic.sql")
    project = ProjectMetaData(
        "p", "t", db_path=tmp_path / "project.sql", vector_repository=MagicMock()
    )
    FullConversationRepository(tmp_path / "conv", "p", "n")
    meta = ConversationVectorMetaDataRepository(tmp_path / "conv", "p")

    paths = {
        "topic": tmp_path / "topic.sql",
        "project": tmp_path / "project.sql",
        "conversation": tmp_path / "conv" / "p_conversation.db",
    }
    with sqlite3.connect(paths["topic"]) as conn:
        conn.executemany(
            "insert into topics_mapping_table values (?,?,'2026','t')",
            [(f"id{i}", f"topic {i}") for i in range(ROWS)],
        )
    with sqlite3.connect(paths["project"]) as conn:
        conn.executemany(
            "insert into project_table values (?,?,?,'2026','2026','s',null)",
            [(f"p{i}", f"name{i}", f"t{i % 20}") for i in range(ROWS)],
        )
        conn.executemany(
            "insert into project_mapping_table values (?,?,?,'2026')",
            [(f"p{i}", f"t{i % 20}", i) for i in range(ROWS)],
        )
    with sqlite3.connect(paths["conversation"]) as conn:
        conn.executemany(
            "insert into summary_chunks values (?,?,'2026','turn')",
            [(f"chunk{i}", f"text {i}") for i in range(ROWS)],
        )
        conn.executemany(
            "insert into full_conversation values ('p',?,?,'user','2026')",
            [(i, f"chunk{i}") for i in range(ROWS)],
        )
        conn.executemany(
            "insert into summary_vector_meta_data values (?,?,'p')",
            [(i, f"chunk{i}") for i in range(ROWS)],
        )

    yield paths
    topic.close()
    project.close()
    meta.close()


def plan(db: Path, sql: str, params=()) -> str:
    with sqlite3.connect(db) as conn:
        return " / ".join(row[-1] for row in conn.execute("explain query plan " + sql, params))


HOT_QUERIES = [
    ("topic", "a topic by name",
     "select topic_id from topics_mapping_table where topic_name = ? and is_active = 't'", ("x",)),
    ("project", "the projects in a topic",
     "select project_id, project_name, project_summary from project_table where topic_id = ?", ("t",)),
    ("project", "the vector ids in a topic",
     "select project_id, project_summary_vector_id from project_mapping_table where topic_id = ?", ("t",)),
    ("conversation", "a turn by chunk id",
     "select sequence_number from full_conversation where chunk_id = ?", ("c",)),
    ("conversation", "the summarised watermark",
     "select max(f.sequence_number) from summary_vector_meta_data m"
     " join full_conversation f on f.chunk_id = m.chunk_id where m.project_id = ?", ("p",)),
]


@pytest.mark.parametrize("store,label,sql,params", HOT_QUERIES,
                         ids=[case[1] for case in HOT_QUERIES])
def test_the_query_uses_an_index(stores, store, label, sql, params):
    result = plan(stores[store], sql, params)
    assert "SCAN" not in result, f"{label}: {result}"


def test_the_watermark_join_does_not_build_its_own_index(stores):
    """SQLite writes an AUTOMATIC index when it needs one that does not exist —
    per call, every call. It is a correct answer arrived at the expensive way."""
    result = plan(
        stores["conversation"],
        "select max(f.sequence_number) from summary_vector_meta_data m"
        " join full_conversation f on f.chunk_id = m.chunk_id where m.project_id = ?",
        ("p",),
    )
    assert "AUTOMATIC" not in result, result


def test_primary_key_lookups_need_no_extra_index(stores):
    """Not every column wants one. These are already served by the implicit
    index behind their primary key, and a second would only cost writes."""
    for store, sql, params in [
        ("project", "select project_name from project_table where project_id = ?", ("p",)),
        ("project", "select project_description from project_description_table"
                    " where project_id = ? and project_description_id = ?", ("p", "d")),
        ("conversation", "select chunk_id from full_conversation where sequence_number = ?", (1,)),
        ("conversation", "select chunk from summary_chunks where chunk_id = ?", ("c",)),
    ]:
        assert "SEARCH" in plan(stores[store], sql, params)


def test_indexes_are_created_on_an_existing_database(tmp_path):
    """They go in with CREATE INDEX IF NOT EXISTS at open, so a database written
    before they existed picks them up rather than needing a migration."""
    db = tmp_path / "project.sql"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            "create table project_table(project_id text primary key, project_name text,"
            " topic_id text not null, created_at date not null, updated_at date not null,"
            " project_summary text not null, user_id text);"
        )
    meta = ProjectMetaData("p", "t", db_path=db, vector_repository=MagicMock())
    with sqlite3.connect(db) as conn:
        names = {r[0] for r in conn.execute(
            "select name from sqlite_master where type='index' and name not like 'sqlite_%'")}
    meta.close()
    assert "idx_project_topic" in names
