"""ConversationTable against a real PostgreSQL: column types, the CHECK, and the timestamps psycopg hands back."""

import uuid
from datetime import datetime

import pytest

pytestmark = pytest.mark.live


@pytest.fixture
def table():
    from query_decomposition.conversation_table import ConversationTable

    table = ConversationTable()
    conversation_id = f"__live_test__{uuid.uuid4().hex}"
    try:
        yield table, conversation_id
    finally:
        with table.conn.cursor() as cursor:
            cursor.execute(
                "delete from conversation_table where conversation_id = %s",
                (conversation_id,),
            )
        table.conn.commit()
        table.close()


def test_a_query_round_trips_awaiting_its_response(table):
    table, conversation_id = table
    recorded = table.record_query(conversation_id, "user", "why vectors?")
    stored = table.get(recorded.exchange_id)

    assert stored == recorded
    assert stored.query == "why vectors?"
    assert isinstance(stored.query_created_at, datetime)
    assert stored.query_created_at.tzinfo is not None
    assert (stored.system_response, stored.system_response_time) == (None, None)


def test_the_response_and_its_time_are_written_together(table):
    from query_decomposition.query_decomposition_exceptions import (
        ExchangeNotFound,
        ResponseAlreadyRecorded,
    )

    table, conversation_id = table
    recorded = table.record_query(conversation_id, "user", "why?")
    answered = table.record_response(recorded.exchange_id, "because")

    assert answered.system_response == "because"
    assert answered.system_response_time >= answered.query_created_at
    with pytest.raises(ResponseAlreadyRecorded):
        table.record_response(recorded.exchange_id, "again")
    with pytest.raises(ExchangeNotFound):
        table.record_response(uuid.uuid4().hex, "because")


def test_a_response_without_its_time_is_refused_by_the_table(table):
    import psycopg

    table, conversation_id = table
    recorded = table.record_query(conversation_id, "user", "why?")
    with pytest.raises(psycopg.errors.CheckViolation):
        with table.conn.cursor() as cursor:
            cursor.execute(
                "update conversation_table set system_response = 'x' "
                "where exchange_id = %s",
                (recorded.exchange_id,),
            )
    table.conn.rollback()


def test_history_is_one_users_conversation_oldest_first(table):
    table, conversation_id = table
    first = table.record_query(conversation_id, "user", "one")
    second = table.record_query(conversation_id, "user", "two")
    table.record_query(conversation_id, "someone else", "three")

    assert [row.exchange_id for row in table.history(conversation_id, "user")] == [
        first.exchange_id,
        second.exchange_id,
    ]
