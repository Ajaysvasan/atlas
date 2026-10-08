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

from memory.memory_database import MemoryDatabase

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
    """Every memory table, in the one database, with rows in them.

    The rows are the point. SQLite plans from row estimates, so an empty table
    hides the very choices these tests exist to pin — with nothing in it the
    planner will not bother building the AUTOMATIC index that a populated one
    provokes. They go in through the database with foreign keys on, so every
    child row has its parent, as it would in use.
    """
    path = tmp_path / "memory.db"
    database = MemoryDatabase(path)
    TopicPoolMetaHandler(database)
    ProjectMetaData("p", "t", database, vector_repository=MagicMock())
    FullConversationRepository("p", "n", "c1", database)
    ConversationVectorMetaDataRepository("p", "conv_abc", database)

    with database.writing() as cursor:
        cursor.executemany(
            "insert into topics_mapping_table values (?,?,'2026','t')",
            [("t", "t")] + [(f"t{j}", f"topic t{j}") for j in range(20)]
            + [(f"id{i}", f"topic {i}") for i in range(ROWS)],
        )
        cursor.executemany(
            "insert into project_table values (?,?,?,'2026','2026','s',null)",
            [("p", "name", "t")] + [(f"p{i}", f"name{i}", f"t{i % 20}") for i in range(ROWS)],
        )
        cursor.executemany(
            "insert into project_mapping_table values (?,?,?,'2026')",
            [(f"p{i}", f"t{i % 20}", i) for i in range(ROWS)],
        )
        cursor.executemany(
            "insert into summary_chunks values (?,'c1',?,'2026','turn')",
            [(f"chunk{i}", f"text {i}") for i in range(ROWS)],
        )
        cursor.executemany(
            "insert into full_conversation values ('p','c1',?,?,'user','2026')",
            [(i, f"chunk{i}") for i in range(ROWS)],
        )
        cursor.executemany(
            "insert into summary_vector_meta_data values (?,?,'p')",
            [(i, f"chunk{i}") for i in range(ROWS)],
        )

    yield {"topic": path, "project": path, "conversation": path}
    database.close()


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
     " join full_conversation f on f.chunk_id = m.chunk_id"
     " where m.project_id = ? and f.conversation_id = ?", ("p", "c1")),
    ("conversation", "a turn by its chunk id",
     "select sequence_number from full_conversation where chunk_id = ? and conversation_id = ?",
     ("c", "c1")),
]


@pytest.mark.parametrize("store,label,sql,params", HOT_QUERIES,
                         ids=[case[1] for case in HOT_QUERIES])
def test_the_query_uses_an_index(stores, store, label, sql, params):
    result = plan(stores[store], sql, params)
    assert "SCAN" not in result, f"{label}: {result}"


def test_the_watermark_join_does_not_build_its_own_index(stores):
    """SQLite writes an AUTOMATIC index when it needs one that does not exist —
    per call, every call. It is a correct answer arrived at the expensive way."""
    result = plan(stores["conversation"], _WATERMARK, ("p", "c1"))
    assert "AUTOMATIC" not in result, result


def test_primary_key_lookups_need_no_extra_index(stores):
    """Not every column wants one. These are already served by the implicit
    index behind their primary key, and a second would only cost writes."""
    for store, sql, params in [
        ("project", "select project_name from project_table where project_id = ?", ("p",)),
        ("project", "select project_description from project_description_table"
                    " where project_id = ? and project_description_id = ?", ("p", "d")),
        # The primary key is (conversation_id, sequence_number) since the
        # conversation_id change, so a scoped lookup is the one it serves --
        # and the only shape the repository now issues.
        ("conversation", "select chunk_id from full_conversation"
                         " where conversation_id = ? and sequence_number = ?", ("c1", 1)),
        ("conversation", "select chunk from summary_chunks where chunk_id = ?", ("c",)),
    ]:
        assert "SEARCH" in plan(stores[store], sql, params)


def test_indexes_are_created_on_an_existing_database(tmp_path):
    """They go in with CREATE INDEX IF NOT EXISTS at open, so a database written
    before one existed picks it up rather than needing a migration."""
    path = tmp_path / "memory.db"
    first = MemoryDatabase(path)
    ProjectMetaData("p", "t", first, vector_repository=MagicMock())
    with first.writing() as cursor:
        cursor.execute("drop index idx_project_topic")
    first.close()

    reopened = MemoryDatabase(path)
    ProjectMetaData("p", "t", reopened, vector_repository=MagicMock())
    with reopened.reading() as cursor:
        names = {r[0] for r in cursor.execute(
            "select name from sqlite_master where type='index' and name not like 'sqlite_%'")}
    reopened.close()
    assert "idx_project_topic" in names


# ---------------------------------------------------------------------------
# Bug 4.68: the shape of idx_full_conversation_chunk, re-measured
# ---------------------------------------------------------------------------

_WATERMARK = (
    "select max(f.sequence_number) from summary_vector_meta_data m"
    " join full_conversation f on f.chunk_id = m.chunk_id"
    " where m.project_id = ? and f.conversation_id = ?"
)


def test_the_watermark_walks_one_conversation_and_probes_by_chunk(stores):
    """The watermark is per conversation: project-wide, one conversation read
    another's progress as its own. Scoped to the conversation, the planner
    drives from its primary key and probes summary_vector_meta_data through
    idx_summary_vector_chunk. Measured at 5 conversations x 8000 turns: 5.3 ms
    without that index, 3-5 us with it.
    """
    result = plan(stores["conversation"], _WATERMARK, ("p", "c1"))

    assert "conversation_id=?" in result, result
    assert "idx_summary_vector_chunk" in result, result
    assert "SCAN" not in result, result


def test_the_conversation_scoped_readers_are_served_by_the_primary_key(stores):
    """Why no second index was added for them.

    The primary key is (conversation_id, sequence_number), so every reader that
    filters a conversation and walks its sequence numbers is already covered.
    Measured: no index variant on full_conversation moved any of them.
    """
    result = plan(
        stores["conversation"],
        "select f.sequence_number, f.role from full_conversation f"
        " where f.conversation_id = ? order by f.sequence_number",
        ("c1",),
    )

    assert "SCAN" not in result, result


def test_the_snapshot_chain_is_read_through_its_primary_key(tmp_path):
    """project_snapshot_mapping's key leads with project_id, so a separate
    index on project_id duplicated it: same plan, 12.7 us against 15.9 us for
    latest(), and one more index to maintain on every write."""
    from memory.topic_pool.project_pool.project_data_repo.project_snapshot_repo import (
        ProjectSnapshotRepository,
    )

    database = MemoryDatabase(tmp_path / "memory.db")
    ProjectSnapshotRepository("p", database)
    with database.reading() as cursor:
        names = {r[0] for r in cursor.execute(
            "select name from sqlite_master where type='index' "
            "and tbl_name='project_snapshot_mapping'")}
        result = " / ".join(r[-1] for r in cursor.execute(
            "explain query plan select project_snapshot_id from project_snapshot_mapping "
            "where project_id = ?", ("p",)))
    database.close()
    assert "idx_project_snapshot_mapping_project" not in names
    assert "sqlite_autoindex_project_snapshot_mapping_1" in result, result
