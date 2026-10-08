"""The sparse half: FTS5 and bm25 over the chunk text."""

import sqlite3

import pytest

from retrieval_layer.keyword_search import KeywordSearch, match_expression
from retrieval_layer.models import KEYWORD, QueryPlan


class FakeMapping:
    def __init__(self, pairs):
        self.pairs = dict(pairs)

    def vector_ids_for(self, chunk_ids):
        return {c: self.pairs[c] for c in chunk_ids if c in self.pairs}


def chunk_store(tmp_path, rows, table="Chunks", parent="contextId"):
    path = tmp_path / "chunks.db"
    conn = sqlite3.connect(path)
    conn.execute(
        f"create table {table}(chunkId text primary key, {parent} text, "
        f"chunk text, startoffset int, endoffset int)"
    )
    conn.executemany(
        f"insert into {table} values (?, null, ?, 0, 0)", rows
    )
    conn.commit()
    conn.close()
    return str(path)


def search_over(tmp_path, rows, mapping=None):
    store = chunk_store(tmp_path, rows)
    mapping = mapping or FakeMapping({cid: i + 1 for i, (cid, _) in enumerate(rows)})
    ks = KeywordSearch(mapping, store, tmp_path / "kw.sql")
    ks.sync()
    return ks


def plan(text):
    return QueryPlan(text, None, text)


class TestTheMatchExpression:
    def test_words_become_quoted_terms(self):
        assert match_expression("write ahead") == '"write" OR "ahead"'

    @pytest.mark.parametrize("text", [
        'cache-control "immutable"', "AND OR NOT NEAR", "a (b) c*", "x^y:z",
        "what's this?", "100% sure", 'he said "hi"',
    ])
    def test_punctuation_and_operators_cannot_break_the_query(self, text, tmp_path):
        """FTS5 reads AND, NEAR, quotes and * as syntax and raises on bad ones."""
        ks = search_over(tmp_path, [("c1", "some ordinary body text")])
        ks.search(plan(text), 3)
        ks.close()

    def test_text_with_no_words_yields_no_expression(self):
        assert match_expression("!!! ??? ...") == ""

    def test_a_query_with_no_words_searches_nothing(self, tmp_path):
        ks = search_over(tmp_path, [("c1", "body")])
        assert ks.search(plan("!!!"), 3) == []
        ks.close()


class TestStemming:
    def test_a_query_word_matches_an_inflected_one(self, tmp_path):
        """Without porter, "journal" misses a chunk that says "journals"."""
        ks = search_over(tmp_path, [("c1", "the writer appends to journals")])
        assert len(ks.search(plan("journal"), 3)) == 1
        ks.close()

    def test_an_unrelated_word_still_does_not_match(self, tmp_path):
        ks = search_over(tmp_path, [("c1", "the writer appends to journals")])
        assert ks.search(plan("photosynthesis"), 3) == []
        ks.close()


class TestBeyondAscii:
    """The tokenizer is unicode61; a query split on ASCII before it gets there
    searches "naïve" as "na" OR "ve" and finds nothing."""

    def test_an_accented_word_matches_itself(self, tmp_path):
        ks = search_over(tmp_path, [("c1", "a naïve bayes classifier"),
                                    ("c2", "unrelated text about vectors")])
        assert [h.vector_id for h in ks.search(plan("naïve"), 3)] == [1]
        ks.close()

    def test_an_unaccented_query_finds_the_accented_word(self, tmp_path):
        """unicode61 folds diacritics, so this works once nothing upstream
        has already broken the word apart."""
        ks = search_over(tmp_path, [("c1", "the café opened at nine")])
        assert len(ks.search(plan("cafe"), 3)) == 1
        ks.close()

    def test_a_non_latin_script(self, tmp_path):
        ks = search_over(tmp_path, [("c1", "индекс базы данных"),
                                    ("c2", "an index of a database")])
        assert [h.vector_id for h in ks.search(plan("индекс"), 3)] == [1]
        ks.close()

    def test_the_expression_keeps_the_word_whole(self):
        assert match_expression("naïve café") == '"naïve" OR "café"'


class TestRanking:
    def test_the_better_match_ranks_first(self, tmp_path):
        ks = search_over(tmp_path, [
            ("c1", "indexing"),
            ("c2", "write ahead logging write ahead logging journal"),
        ])
        hits = ks.search(plan("write ahead logging"), 2)
        assert hits[0].vector_id == 2
        ks.close()

    def test_results_are_marked_as_keyword(self, tmp_path):
        ks = search_over(tmp_path, [("c1", "write ahead")])
        assert ks.search(plan("write"), 1)[0].source == KEYWORD
        ks.close()

    def test_k_bounds_the_results(self, tmp_path):
        ks = search_over(tmp_path, [(f"c{i}", "write ahead") for i in range(5)])
        assert len(ks.search(plan("write"), 2)) == 2
        ks.close()


class TestTranslatingToVectorIds:
    def test_chunk_ids_become_vector_ids(self, tmp_path):
        """Fusion compares candidates from both searchers, so both speak labels."""
        ks = search_over(tmp_path, [("c1", "write ahead")],
                         mapping=FakeMapping({"c1": 42}))
        assert ks.search(plan("write"), 1)[0].vector_id == 42
        ks.close()

    def test_a_chunk_with_no_vector_is_dropped(self, tmp_path):
        """Indexed for keywords but never embedded: it has no label to fuse on."""
        ks = search_over(tmp_path, [("c1", "write ahead"), ("c2", "write ahead")],
                         mapping=FakeMapping({"c1": 1}))
        assert [h.vector_id for h in ks.search(plan("write"), 5)] == [1]
        ks.close()


class TestSync:
    def test_it_indexes_the_chunk_store(self, tmp_path):
        ks = search_over(tmp_path, [("c1", "body one"), ("c2", "body two")])
        assert ks.connection.execute(
            "select count(*) from chunk_fts").fetchone()[0] == 2
        ks.close()

    def test_a_second_sync_adds_nothing(self, tmp_path):
        """Re-tokenising the whole corpus on every ingest would be the cost."""
        ks = search_over(tmp_path, [("c1", "body")])
        assert ks.sync() == 0
        ks.close()

    def test_new_chunks_are_picked_up(self, tmp_path):
        store = chunk_store(tmp_path, [("c1", "body")])
        ks = KeywordSearch(FakeMapping({"c1": 1, "c2": 2}), store, tmp_path / "kw.sql")
        ks.sync()
        conn = sqlite3.connect(store)
        conn.execute("insert into Chunks values ('c2', null, 'more body', 0, 0)")
        conn.commit()
        conn.close()
        assert ks.sync() == 1
        ks.close()

    def test_a_missing_chunk_store_is_survived(self, tmp_path):
        ks = KeywordSearch(FakeMapping({}), tmp_path / "absent.db", tmp_path / "kw.sql")
        assert ks.sync() == 0
        ks.close()

    def test_its_index_is_a_file_of_its_own(self, tmp_path):
        """Rebuilding it must not be able to disturb the ingested corpus."""
        store = chunk_store(tmp_path, [("c1", "body")])
        ks = KeywordSearch(FakeMapping({"c1": 1}), store, tmp_path / "kw.sql")
        assert ks.fts_path != tmp_path / "chunks.db"
        assert (tmp_path / "kw.sql").exists()
        ks.close()
