"""ListConversations and ListExchanges over a real loopback server: ownership, order, paging, failures, no gate."""

import threading

import grpc
import pytest

from grpc_server.generated import atlas_service_pb2 as pb
from grpc_server.service import InvalidCursor, decode_cursor, encode_cursor

from .conftest import ALICE, AUTH, BOB, trailer


def conversations(stub, user=ALICE, limit=0, cursor="", metadata=AUTH):
    return stub.ListConversations(pb.ListConversationsRequest(request_id="req-list", user_id=user, limit=limit,
                                                              cursor=cursor), metadata=metadata, timeout=5)


def exchanges(stub, conversation_id, user=ALICE, limit=0, cursor=""):
    return stub.ListExchanges(pb.ListExchangesRequest(request_id="req-ex", user_id=user,
                                                      conversation_id=conversation_id, limit=limit, cursor=cursor),
                              metadata=AUTH, timeout=5)


def start(registry, user, name, second):
    return registry.create(user, name, created_at=f"2026-01-01T00:00:{second:02d}.000000+00:00")


def every_page(fetch):
    seen, cursor = [], ""
    while True:
        page = fetch(cursor)
        seen.append(page)
        cursor = page.next_cursor
        if not cursor:
            return seen


class TestCursor:
    def test_round_trip(self):
        assert decode_cursor(encode_cursor("2026-11-09T10:00:00.000001+00:00", "abc")) == (
            "2026-11-09T10:00:00.000001+00:00", "abc")

    @pytest.mark.parametrize("cursor", ["!!!", "bm90IGpzb24", encode_cursor("yesterday", "abc"),
                                        encode_cursor("2026-11-09T10:00:00+00:00", ""), "A" * 600])
    def test_garbage_is_refused(self, cursor):
        with pytest.raises(InvalidCursor):
            decode_cursor(cursor)


class TestListConversations:
    def test_a_new_user_has_none(self, stub):
        response = conversations(stub)
        assert response.request_id == "req-list"
        assert list(response.conversations) == [] and response.next_cursor == ""

    def test_most_recently_active_first_with_counts(self, stub, conversation_registry, conversation_rows):
        old = start(conversation_registry, ALICE, "Raft", 1)
        new = start(conversation_registry, ALICE, "Paxos", 2)
        exchange = conversation_rows.record_query(old.conversation_id, ALICE, "What is a term?")
        conversation_rows.record_response(exchange.exchange_id, "An epoch.")
        conversation_rows.record_query(old.conversation_id, ALICE, "And a log?")

        rows = conversations(stub).conversations
        assert [r.conversation_id for r in rows] == [old.conversation_id, new.conversation_id]
        assert (rows[0].conversation_name, rows[0].exchange_count) == ("Raft", 2)
        assert rows[0].last_activity_at > rows[0].created_at == "2026-01-01T00:00:01.000000+00:00"
        assert (rows[1].exchange_count, rows[1].last_activity_at) == (0, rows[1].created_at)

    def test_only_the_callers_conversations(self, stub, conversation_registry):
        start(conversation_registry, ALICE, "Mine", 1)
        start(conversation_registry, BOB, "Theirs", 2)
        assert [r.conversation_name for r in conversations(stub, user=ALICE).conversations] == ["Mine"]
        assert [r.conversation_name for r in conversations(stub, user=BOB).conversations] == ["Theirs"]

    def test_pages_cover_everything_once(self, stub, conversation_registry):
        ids = [start(conversation_registry, ALICE, f"Topic {n}", n).conversation_id for n in range(5)]
        pages = every_page(lambda cursor: conversations(stub, limit=2, cursor=cursor))
        assert [len(p.conversations) for p in pages] == [2, 2, 1]
        assert [r.conversation_id for p in pages for r in p.conversations] == ids[::-1]

    def test_a_full_last_page_has_no_cursor(self, stub, conversation_registry):
        for n in range(2):
            start(conversation_registry, ALICE, f"Topic {n}", n)
        assert conversations(stub, limit=2).next_cursor == ""

    @pytest.mark.parametrize("limit", [-1, 201])
    def test_a_page_size_out_of_range_is_invalid(self, stub, limit):
        with pytest.raises(grpc.RpcError) as caught:
            conversations(stub, limit=limit)
        assert caught.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert trailer(caught.value) == "INVALID_REQUEST"

    def test_an_unreadable_cursor_is_invalid(self, stub):
        with pytest.raises(grpc.RpcError) as caught:
            conversations(stub, cursor="not-a-cursor")
        assert caught.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert trailer(caught.value) == "INVALID_CURSOR"

    def test_a_malformed_user_is_invalid(self, stub):
        with pytest.raises(grpc.RpcError) as caught:
            conversations(stub, user="../etc")
        assert caught.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    def test_the_service_token_is_required(self, stub):
        with pytest.raises(grpc.RpcError) as caught:
            conversations(stub, metadata=())
        assert caught.value.code() == grpc.StatusCode.UNAUTHENTICATED

    def test_a_database_failure_is_unavailable_and_says_nothing_inside(self, stub, conversation_registry,
                                                                        monkeypatch):
        failure = type("OperationalError", (Exception,), {"__module__": "psycopg.errors"})

        def broken(*args, **kwargs):
            raise failure("connection to /var/run/postgresql failed")

        monkeypatch.setattr(conversation_registry, "summaries", broken)
        with pytest.raises(grpc.RpcError) as caught:
            conversations(stub)
        assert caught.value.code() == grpc.StatusCode.UNAVAILABLE
        assert trailer(caught.value) == "DATABASE_UNAVAILABLE"
        assert "postgresql" not in caught.value.details()

    def test_it_answers_while_a_query_holds_the_pipeline(self, stub, fakes, conversation_registry):
        pipeline = fakes[0]
        pipeline.release.clear()
        running = threading.Thread(target=lambda: stub.Query(
            pb.QueryRequest(request_id="req-q", user_id=ALICE, query="Explain MVCC."), metadata=AUTH, timeout=10))
        running.start()
        try:
            assert pipeline.started.wait(5)
            # The gate admits one operation and lets one wait; a listing is neither.
            rows = conversations(stub).conversations
            assert [r.conversation_name for r in rows] == ["PostgreSQL MVCC"]
        finally:
            pipeline.release.set()
            running.join(10)


class TestListExchanges:
    def test_the_owner_reads_them_oldest_first(self, stub, conversation_registry, conversation_rows):
        conversation = start(conversation_registry, ALICE, "Raft", 1)
        first = conversation_rows.record_query(conversation.conversation_id, ALICE, "What is a term?")
        conversation_rows.record_response(first.exchange_id, "An epoch.")
        conversation_rows.record_query(conversation.conversation_id, ALICE, "Still running?")

        response = exchanges(stub, conversation.conversation_id)
        assert (response.conversation_id, response.conversation_name) == (conversation.conversation_id, "Raft")
        answered, pending = response.exchanges
        assert (answered.query, answered.answer) == ("What is a term?", "An epoch.")
        assert answered.HasField("answered_at") and answered.answered_at >= answered.asked_at
        assert pending.query == "Still running?"
        assert not pending.HasField("answer") and not pending.HasField("answered_at")
        assert response.next_cursor == ""

    def test_another_users_conversation_does_not_exist_for_them(self, stub, conversation_registry,
                                                               conversation_rows):
        conversation = start(conversation_registry, ALICE, "Raft", 1)
        conversation_rows.record_query(conversation.conversation_id, ALICE, "secret")
        with pytest.raises(grpc.RpcError) as caught:
            exchanges(stub, conversation.conversation_id, user=BOB)
        assert caught.value.code() == grpc.StatusCode.NOT_FOUND
        assert trailer(caught.value) == "CONVERSATION_NOT_FOUND"

    @pytest.mark.parametrize("conversation_id", ["", "no-such-conversation", "../../etc/passwd"])
    def test_unknown_or_malformed_ids_are_not_found(self, stub, conversation_id):
        with pytest.raises(grpc.RpcError) as caught:
            exchanges(stub, conversation_id)
        assert caught.value.code() == grpc.StatusCode.NOT_FOUND

    def test_pages_cover_everything_once(self, stub, conversation_registry, conversation_rows):
        conversation = start(conversation_registry, ALICE, "Raft", 1)
        asked = [conversation_rows.record_query(conversation.conversation_id, ALICE, f"q{n}").exchange_id
                 for n in range(5)]
        pages = every_page(lambda cursor: exchanges(stub, conversation.conversation_id, limit=2, cursor=cursor))
        assert [e.exchange_id for p in pages for e in p.exchanges] == asked

    def test_an_unreadable_cursor_is_invalid(self, stub, conversation_registry):
        conversation = start(conversation_registry, ALICE, "Raft", 1)
        with pytest.raises(grpc.RpcError) as caught:
            exchanges(stub, conversation.conversation_id, cursor="%%%")
        assert trailer(caught.value) == "INVALID_CURSOR"
