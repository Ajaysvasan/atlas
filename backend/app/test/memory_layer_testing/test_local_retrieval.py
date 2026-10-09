"""The local retrieval layer: a project's own conversation turns, searched with
the global layer's stages over the memory layer's vectors and chunks.

Everything runs against a real memory database. Turns are summarised through
SnapShot.add, the same write ConversationSummary makes, so the vector ids,
chunk rows and snapshot metadata are the ones production writes. Only pgvector
is stood in for, by the root conftest's in-memory store, and the models by
stubs.
"""

import hashlib
import threading

import numpy as np
import pytest

from data_layer.ingestion.embedding.vector_ids import vector_id_for
from memory.local_retrieval import turn_keyword_search, turn_vector_search
from memory.local_retrieval.local_retriever import LocalRetriever
from memory.local_retrieval.turn_keyword_search import TurnKeywordSearch
from memory.local_retrieval.turn_scope import CONVERSATION, PROJECT, TurnScope
from memory.local_retrieval.turn_vector_search import TurnVectorSearch
from memory.memory_database import MemoryDatabase
from memory.snapshot import SnapShot
from memory.topic_pool.project_pool.conversation_pool.conversation_data_management.conversationVectorMetaManager import (
    turn_vector_ids,
)
from memory.topic_pool.project_pool.conversation_pool.full_conversation_bucket import (
    FullConversation,
)
from memory.topic_pool.project_pool.conversation_pool.fullconversation_repository.fullconversation_repository import (
    conversation_ids,
    turns_after_sequence,
    turns_for_chunks,
)
from retrieval_layer.models import QueryPlan, RetrievalRequest
from retrieval_layer.query_preparation import QueryPreparation
from retrieval_layer.reranking import Reranker
from retrieval_layer.retrieval_exceptions import EmptyQuery, InvalidRetrievalSetting
from retrieval_layer.settings import RetrievalSettings
from storage.timestamps import utc_now

DB = "memory.db"
P1, P2 = "proj_local", "proj_elsewhere"
C1, C2, C3 = "conv_one", "conv_two", "conv_three"
DIMENSIONS = 128

# Shares no word with any turn below, so only the vector search can find a turn for it.
ASK = "zebra"


def unit(seed):
    v = np.random.default_rng(seed).standard_normal(DIMENSIONS).astype(np.float32)
    return v / np.linalg.norm(v)


QUERY = unit(1)


def toward(cosine, seed):
    """A unit vector at an exact cosine to the query."""
    other = unit(seed)
    other = other - QUERY * float(other @ QUERY)
    other = other / np.linalg.norm(other)
    v = QUERY * cosine + other * np.sqrt(1 - cosine**2)
    return (v / np.linalg.norm(v)).astype(np.float32)


def encode(texts):
    return np.array([
        QUERY if text == ASK
        else unit(int(hashlib.md5(text.encode()).hexdigest()[:8], 16))
        for text in texts
    ], dtype=np.float32)


def never_encode(texts):
    raise AssertionError("the embedder was loaded for a scope with nothing in it")


@pytest.fixture
def database(tmp_path, seed_projects):
    seed_projects(tmp_path / DB, "topic", P1, P2)
    return MemoryDatabase.of(tmp_path / DB)


@pytest.fixture
def vectors(chunk_vectors):
    """pgvector, as the root conftest stands it in."""
    return chunk_vectors


def say(database, *turns, project=P1, conversation=C1):
    return FullConversation(project, project, conversation, database).append_turns(turns)


def summarise(database, vectors, start, end, vector_of, project=P1, conversation=C1,
              summary_vector=QUERY):
    """Snapshot turns start..end as ConversationSummary does, embedding each with vector_of."""
    bucket = FullConversation(project, project, conversation, database)
    rows = bucket.get_context_rows(start, end)
    turns = bucket.get_turns(start, end)
    snapshot = SnapShot(project, project, conversation, database=database)
    snapshot._vector_manager = vectors
    snapshot.add(
        time_of_snapshot=utc_now(),
        len_of_the_summary=7,
        summary_vector_ids=[vector_id_for(row[0]) for row in rows],
        summary_vectors=np.array([vector_of(turn) for turn in turns], dtype=np.float32),
        chunk_ids=[row[0] for row in rows],
        chunks=[tuple(row) for row in rows],
        summary="summary",
        cumulative_summary_vector_id=vector_id_for(f"{project}/{conversation}/{end}"),
        cumulative_summary_vector=summary_vector,
    )


def by_sequence(vectors_by_sequence, default=0.0):
    """vector_of for summarise(): the turn at sequence n sits at the cosine given for n."""
    return lambda turn: toward(vectors_by_sequence.get(turn.sequence_number, default),
                               turn.sequence_number + 100)


def retriever(database, vectors, conversation=C1, project=P1, encoder=encode, **settings):
    settings.setdefault("rerank", False)
    return LocalRetriever(
        project, project, conversation, RetrievalSettings(**settings),
        database=database,
        preparation=QueryPreparation(encode=encoder),
        vector_search=TurnVectorSearch(project, project, database, vector_store=vectors),
    )


def texts(result):
    return [passage.text for passage in result.passages]


def plan(text=ASK):
    return QueryPlan(text=text, vector=QUERY, terms=text)


# --------------------------------------------------------------------------- #
# The reads the memory layer owns
# --------------------------------------------------------------------------- #

class TestTheOwnersReads:
    def test_turns_for_chunks_names_the_conversation_and_the_speaker(self, database):
        say(database, ("user", "hello"), ("assistant", "hi there"))
        chunk_id = FullConversation(P1, P1, C1, database).get_turns(2, 2)[0].chunk_id
        (turn,) = turns_for_chunks([chunk_id, "no-such-chunk"], database)
        assert (turn.conversation_id, turn.sequence_number, turn.role, turn.text) == (
            C1, 2, "assistant", "hi there")

    def test_turns_after_sequence_reads_only_the_named_project(self, database):
        say(database, ("user", "one"), ("user", "two"))
        assert [t.text for t in turns_after_sequence(P1, C1, 1, database)] == ["two"]
        assert turns_after_sequence(P2, C1, 0, database) == []

    def test_conversation_ids_lists_each_conversation_once(self, database):
        say(database, ("user", "a"), ("user", "b"))
        say(database, ("user", "c"), conversation=C2)
        say(database, ("user", "d"), project=P2, conversation=C3)
        assert sorted(conversation_ids(P1, database)) == [C1, C2]

    def test_turn_vector_ids_lists_only_summarised_turns_in_scope(self, database, vectors):
        say(database, ("user", "a"), ("user", "b"), ("user", "c"))
        say(database, ("user", "d"), conversation=C2)
        summarise(database, vectors, 1, 2, by_sequence({}))
        summarise(database, vectors, 1, 1, by_sequence({}), conversation=C2)
        assert {(r.conversation_id, r.sequence_number)
                for r in turn_vector_ids(P1, C1, database)} == {(C1, 1), (C1, 2)}
        assert len(turn_vector_ids(P1, None, database)) == 3
        assert turn_vector_ids(P2, None, database) == []


# --------------------------------------------------------------------------- #
# What a retrieval returns
# --------------------------------------------------------------------------- #

class TestWhatComesBack:
    def test_a_summarised_turn_is_found_by_its_vector(self, database, vectors):
        say(database, ("user", "apples"), ("assistant", "oranges"), ("user", "pears"))
        summarise(database, vectors, 1, 3, by_sequence({2: 0.95}, default=0.1))
        result = retriever(database, vectors).retrieve(RetrievalRequest(ASK, top_k=1))
        assert texts(result) == ["oranges"]

    def test_a_turn_no_snapshot_has_covered_is_found_by_its_words(self, database, vectors):
        say(database, ("user", "we picked postgres"), ("assistant", "noted"))
        result = retriever(database, vectors).retrieve("postgres")
        assert texts(result) == ["we picked postgres"]

    def test_every_passage_says_who_spoke(self, database, vectors):
        say(database, ("user", "postgres question"), ("assistant", "postgres answer"),
            ("system", "postgres rule"))
        result = retriever(database, vectors).retrieve("postgres")
        assert {(p.text, p.role) for p in result.passages} == {
            ("postgres question", "user"), ("postgres answer", "assistant"),
            ("postgres rule", "system")}

    def test_a_passage_carries_its_conversation_and_position(self, database, vectors):
        say(database, ("user", "first"), ("user", "about postgres"))
        (passage,) = retriever(database, vectors).retrieve("postgres").passages
        assert (passage.document_id, passage.start_offset, passage.end_offset) == (C1, 2, 2)

    def test_a_passage_is_keyed_by_its_vector_id(self, database, vectors):
        say(database, ("user", "about postgres"))
        (passage,) = retriever(database, vectors).retrieve("postgres").passages
        assert passage.vector_id == vector_id_for(passage.chunk_id)

    def test_turns_come_back_in_conversation_order(self, database, vectors):
        say(database, ("user", "postgres"), ("assistant", "other"),
            ("user", "postgres postgres postgres"), ("assistant", "postgres again"))
        result = retriever(database, vectors).retrieve("postgres")
        assert [p.start_offset for p in result.passages] == [1, 3, 4]

    def test_one_turn_found_both_ways_appears_once(self, database, vectors):
        say(database, ("user", "postgres"), ("user", "other"))
        summarise(database, vectors, 1, 2, by_sequence({1: 0.9}))
        result = retriever(database, vectors).retrieve(RetrievalRequest("postgres"))
        assert texts(result) == ["postgres", "other"]
        assert result.dropped_unresolvable == 0

    def test_the_speaker_survives_reranking(self, database, vectors):
        say(database, ("user", "postgres one"), ("assistant", "postgres two"))
        local = retriever(database, vectors, rerank=True)
        local.reranker = Reranker(scorer=lambda pairs: [float(len(b)) for _, b in pairs])
        assert {p.role for p in local.retrieve("postgres").passages} == {"user", "assistant"}


class TestScope:
    def test_another_conversation_is_not_searched_by_default(self, database, vectors):
        say(database, ("user", "postgres here"))
        say(database, ("user", "postgres there"), conversation=C2)
        assert texts(retriever(database, vectors).retrieve("postgres")) == ["postgres here"]

    def test_the_project_scope_searches_every_conversation(self, database, vectors):
        say(database, ("user", "postgres here"))
        say(database, ("user", "postgres there"), conversation=C2)
        say(database, ("user", "zebra-free text"), conversation=C3)
        summarise(database, vectors, 1, 1, by_sequence({1: 0.9}), conversation=C3)
        result = retriever(database, vectors).retrieve(
            RetrievalRequest("postgres"), scope=PROJECT)
        assert set(texts(result)) == {"postgres here", "postgres there", "zebra-free text"}

    def test_another_project_is_never_searched(self, database, vectors):
        say(database, ("user", "postgres here"))
        say(database, ("user", "postgres elsewhere"), project=P2, conversation=C2)
        summarise(database, vectors, 1, 1, by_sequence({1: 0.99}),
                  project=P2, conversation=C2)
        result = retriever(database, vectors).retrieve("postgres", scope=PROJECT)
        assert texts(result) == ["postgres here"]

    def test_turns_indexed_for_the_project_stay_out_of_one_conversation(self, database, vectors):
        """The index is shared, so a project-wide search has already put C2 in it."""
        say(database, ("user", "postgres here"))
        say(database, ("user", "postgres there"), conversation=C2)
        local = retriever(database, vectors)
        local.retrieve("postgres", scope=PROJECT)
        assert texts(local.retrieve("postgres")) == ["postgres here"]

    def test_turns_indexed_for_another_project_stay_out_of_this_one(self, database, vectors):
        say(database, ("user", "postgres here"))
        say(database, ("user", "postgres elsewhere"), project=P2, conversation=C2)
        retriever(database, vectors, project=P2, conversation=C2).retrieve("postgres")
        result = retriever(database, vectors).retrieve("postgres", scope=PROJECT)
        assert texts(result) == ["postgres here"]

    def test_a_conversation_of_another_project_finds_nothing_here(self, database, vectors):
        say(database, ("user", "postgres elsewhere"), project=P2, conversation=C2)
        assert retriever(database, vectors, conversation=C2).retrieve("postgres").passages == []

    def test_a_snapshot_summary_is_never_returned_as_a_turn(self, database, vectors):
        say(database, ("user", "a"), ("user", "b"))
        # The cumulative summary's vector is the query itself, the nearest
        # thing in the store, and it shares the project's slot in pgvector.
        summarise(database, vectors, 1, 2, by_sequence({}, default=0.1), summary_vector=QUERY)
        search = TurnVectorSearch(P1, P1, database, vector_store=vectors)
        found = {s.vector_id for s in search.search(plan(), 10, TurnScope.of(P1, C1))}
        assert found == {r.vector_id for r in turn_vector_ids(P1, C1, database)}

    def test_an_unknown_scope_is_refused(self, database, vectors):
        with pytest.raises(InvalidRetrievalSetting):
            retriever(database, vectors).retrieve("postgres", scope="topic")


class TestBeforeSequence:
    def test_turns_the_prompt_already_carries_are_left_out(self, database, vectors):
        say(database, *[("user", f"postgres {n}") for n in range(1, 5)])
        summarise(database, vectors, 1, 4, by_sequence({1: 0.5, 2: 0.6, 3: 0.9, 4: 0.95}))
        local = retriever(database, vectors)
        dense = local.vector_search.search(plan(), 10, TurnScope.of(P1, C1, before_sequence=3))
        assert {s.vector_id for s in dense} == {
            r.vector_id for r in turn_vector_ids(P1, C1, database) if r.sequence_number < 3}
        result = local.retrieve("postgres", before_sequence=3)
        assert [p.start_offset for p in result.passages] == [1, 2]

    def test_the_cut_applies_only_to_the_current_conversation(self, database, vectors):
        say(database, ("user", "postgres 1"), ("user", "postgres 2"))
        say(database, ("user", "postgres a"), ("user", "postgres b"), conversation=C2)
        result = retriever(database, vectors).retrieve(
            "postgres", scope=PROJECT, before_sequence=2)
        assert set(texts(result)) == {"postgres 1", "postgres a", "postgres b"}

    @pytest.mark.parametrize("value", [0, -1, True, "3", 2.0])
    def test_an_unusable_cut_is_refused(self, database, vectors, value):
        with pytest.raises(InvalidRetrievalSetting):
            retriever(database, vectors).retrieve("postgres", before_sequence=value)


# --------------------------------------------------------------------------- #
# The keyword index
# --------------------------------------------------------------------------- #

class TestTheKeywordIndex:
    def test_it_indexes_only_what_is_new(self, database):
        index = TurnKeywordSearch(database)
        scope = TurnScope.of(P1, C1)
        say(database, ("user", "one"), ("user", "two"))
        assert index.sync(scope) == 2
        assert index.sync(scope) == 0
        say(database, ("user", "three"))
        assert index.sync(scope) == 1
        assert index.size(scope) == 3

    def test_a_sync_reads_only_past_the_last_indexed_turn(self, database, monkeypatch):
        read_after = []

        def spy(project_id, conversation_id, sequence_number, database):
            read_after.append(sequence_number)
            return turns_after_sequence(project_id, conversation_id, sequence_number, database)

        monkeypatch.setattr(turn_keyword_search, "turns_after_sequence", spy)
        index = TurnKeywordSearch(database)
        say(database, ("user", "one"), ("user", "two"))
        index.sync(TurnScope.of(P1, C1))
        index.sync(TurnScope.of(P1, C1))
        assert read_after == [0, 2]

    def test_the_project_scope_indexes_every_conversation(self, database):
        say(database, ("user", "one"))
        say(database, ("user", "two"), conversation=C2)
        say(database, ("user", "three"), project=P2, conversation=C3)
        index = TurnKeywordSearch(database)
        assert index.sync(TurnScope.of(P1, C1, PROJECT)) == 2
        assert index.size(TurnScope.of(P1, C1)) == 1

    def test_a_turn_is_registered_under_its_vector_id(self, database):
        say(database, ("user", "one"))
        index = TurnKeywordSearch(database)
        index.sync(TurnScope.of(P1, C1))
        chunk_id = FullConversation(P1, P1, C1, database).get_turns(1, 1)[0].chunk_id
        assert index.chunk_ids_for([vector_id_for(chunk_id), 12345]) == {
            vector_id_for(chunk_id): chunk_id}

    def test_two_indexers_racing_index_each_turn_once(self, database, monkeypatch):
        """turn_fts takes a rowid twice; only turn_index can refuse the second."""
        say(database, ("user", "postgres one"), ("user", "postgres two"))
        both_read = threading.Barrier(2, timeout=5)

        def read_then_wait(*args):
            turns = turns_after_sequence(*args)
            both_read.wait()
            return turns

        monkeypatch.setattr(turn_keyword_search, "turns_after_sequence", read_then_wait)
        first, second = TurnKeywordSearch(database), TurnKeywordSearch(database)
        scope = TurnScope.of(P1, C1)
        added = []
        threads = [threading.Thread(target=lambda i=i: added.append(i.sync(scope)))
                   for i in (first, second)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert sorted(added) == [0, 2]
        found = first.search(plan("postgres"), 10, scope)
        assert len(found) == len({s.vector_id for s in found}) == 2

    def test_a_query_with_nothing_to_match_finds_nothing(self, database):
        say(database, ("user", "one"))
        index = TurnKeywordSearch(database)
        index.sync(TurnScope.of(P1, C1))
        assert index.search(plan("?!"), 10, TurnScope.of(P1, C1)) == []


# --------------------------------------------------------------------------- #
# The vector search
# --------------------------------------------------------------------------- #

class CountingStore:
    def __init__(self, inner):
        self.inner, self.asked = inner, []

    def vectors_for(self, vector_ids):
        self.asked.append(list(vector_ids))
        return self.inner.vectors_for(vector_ids)

    def close(self):
        pass


class TestTheVectorSearch:
    def test_each_vector_is_fetched_once(self, database, vectors):
        say(database, ("user", "a"), ("user", "b"))
        summarise(database, vectors, 1, 2, by_sequence({}))
        store = CountingStore(vectors)
        search = TurnVectorSearch(P1, P1, database, vector_store=store)
        search.search(plan(), 5, TurnScope.of(P1, C1))
        say(database, ("user", "c"))
        summarise(database, vectors, 3, 3, by_sequence({}))
        search.search(plan(), 5, TurnScope.of(P1, C1))
        search.search(plan(), 5, TurnScope.of(P1, C1))
        assert [len(ids) for ids in store.asked] == [2, 1]

    def test_nearest_comes_first(self, database, vectors):
        say(database, *[("user", str(n)) for n in range(1, 6)])
        summarise(database, vectors, 1, 5, by_sequence({1: 0.2, 2: 0.9, 3: 0.5, 4: 0.7, 5: 0.1}))
        search = TurnVectorSearch(P1, P1, database, vector_store=vectors)
        ranked = search.search(plan(), 3, TurnScope.of(P1, C1))
        sequence_of = {r.vector_id: r.sequence_number for r in turn_vector_ids(P1, C1, database)}
        assert [sequence_of[s.vector_id] for s in ranked] == [2, 4, 3]
        assert [s.rank for s in ranked] == [1, 2, 3]

    def test_a_vector_missing_from_the_store_is_left_out(self, database, vectors):
        say(database, ("user", "a"), ("user", "b"))
        summarise(database, vectors, 1, 2, by_sequence({}))
        lost = turn_vector_ids(P1, C1, database)[0].vector_id
        del vectors.rows[lost]
        search = TurnVectorSearch(P1, P1, database, vector_store=vectors)
        found = search.search(plan(), 5, TurnScope.of(P1, C1))
        assert len(found) == 1 and found[0].vector_id != lost

    def test_with_postgresql_down_keyword_search_carries_on(self, database, vectors, caplog):
        say(database, ("user", "postgres"), ("user", "other"))
        summarise(database, vectors, 1, 2, by_sequence({2: 0.9}))
        vectors.down = True
        with caplog.at_level("WARNING"):
            result = retriever(database, vectors).retrieve("postgres")
        assert texts(result) == ["postgres"]
        assert "keyword search carries on" in caplog.text

    def test_a_failed_connection_is_replaced_on_the_next_search(self, database, vectors, monkeypatch):
        say(database, ("user", "a"))
        summarise(database, vectors, 1, 1, by_sequence({1: 0.9}))
        opened = []

        class Store:
            def __init__(self, *_):
                self.broken, self.closed = not opened, False
                opened.append(self)

            def vectors_for(self, vector_ids):
                if self.broken:
                    raise ConnectionError("server closed the connection")
                return vectors.vectors_for(vector_ids)

            def close(self):
                self.closed = True

        monkeypatch.setattr(turn_vector_search, "ConversationVectorManager", Store)
        search = TurnVectorSearch(P1, P1, database)
        assert search.search(plan(), 5, TurnScope.of(P1, C1)) == []
        assert len(search.search(plan(), 5, TurnScope.of(P1, C1))) == 1
        assert [store.closed for store in opened] == [True, False]


# --------------------------------------------------------------------------- #
# The retriever around them
# --------------------------------------------------------------------------- #

class TestCaching:
    def test_a_repeat_question_is_served_from_the_cache(self, database, vectors):
        say(database, ("user", "postgres"))
        local = retriever(database, vectors)
        local.retrieve("postgres")
        assert local.retrieve("postgres").cached

    def test_a_new_turn_means_a_fresh_answer(self, database, vectors):
        say(database, ("user", "postgres one"))
        local = retriever(database, vectors)
        local.retrieve("postgres")
        say(database, ("assistant", "postgres two"))
        result = local.retrieve("postgres")
        assert not result.cached
        assert texts(result) == ["postgres one", "postgres two"]

    def test_a_new_snapshot_means_a_fresh_answer(self, database, vectors):
        say(database, ("user", "apples"), ("user", "pears"))
        local = retriever(database, vectors)
        ask = RetrievalRequest(ASK, top_k=1)
        assert local.retrieve(ask).passages == []
        summarise(database, vectors, 1, 2, by_sequence({2: 0.9}))
        result = local.retrieve(ask)
        assert not result.cached
        assert texts(result) == ["pears"]

    def test_a_turn_in_another_conversation_freshens_a_project_answer(self, database, vectors):
        say(database, ("user", "postgres one"))
        local = retriever(database, vectors)
        local.retrieve("postgres", scope=PROJECT)
        say(database, ("user", "postgres two"), conversation=C2)
        result = local.retrieve("postgres", scope=PROJECT)
        assert not result.cached and len(result.passages) == 2

    @pytest.mark.parametrize("first, second", [
        ({"scope": CONVERSATION}, {"scope": PROJECT}),
        ({}, {"before_sequence": 2}),
    ])
    def test_different_scopes_do_not_share_answers(self, database, vectors, first, second):
        say(database, ("user", "postgres one"), ("user", "postgres two"))
        local = retriever(database, vectors)
        local.retrieve("postgres", **first)
        assert not local.retrieve("postgres", **second).cached


class TestWhatItAccepts:
    def test_a_scope_with_no_turns_loads_no_model(self, database, vectors):
        result = retriever(database, vectors, encoder=never_encode).retrieve("postgres")
        assert result.passages == [] and not result.cached

    @pytest.mark.parametrize("query", ["", "   "])
    def test_nothing_to_search_for(self, database, vectors, query):
        with pytest.raises(EmptyQuery):
            retriever(database, vectors).retrieve(query)

    def test_asking_for_no_results(self, database, vectors):
        with pytest.raises(InvalidRetrievalSetting):
            retriever(database, vectors).retrieve(RetrievalRequest("postgres", top_k=0))

    def test_every_stage_is_timed(self, database, vectors):
        say(database, ("user", "postgres"))
        result = retriever(database, vectors).retrieve("postgres")
        assert [t.stage for t in result.timings] == [
            "index", "cache", "prepare", "vector search", "keyword search",
            "fusion", "hydration", "diversity", "assembly"]


class TestWhatItOwns:
    def test_it_closes_the_vector_search_it_opened(self, database, monkeypatch):
        closed = []
        monkeypatch.setattr(turn_vector_search.TurnVectorSearch, "close",
                            lambda self: closed.append(self))
        with LocalRetriever(P1, P1, C1, database=database) as local:
            pass
        assert closed == [local.vector_search]

    def test_it_does_not_close_what_it_was_handed(self, database, vectors):
        handed = TurnVectorSearch(P1, P1, database, vector_store=vectors)
        handed.close = lambda: pytest.fail("closed a search it was handed")
        LocalRetriever(P1, P1, C1, database=database, vector_search=handed).close()
