"""Turning search results back into text, in one query rather than k."""

import sqlite3

import pytest

from data_layer.ingestion.embedding.vector_ids import vector_id_for
from data_layer.vector_db_manager.repository.vectorMetaDataRepository import CREATE
from retrieval_layer.hydration import Hydration
from retrieval_layer.models import ScoredId, VECTOR


def store_with(tmp_path, hierarchical=(), recursive=(), mapped=()):
    """A chunk store holding whichever chunk kinds the test needs."""
    path = tmp_path / "chunks.db"
    conn = sqlite3.connect(path)
    conn.execute(CREATE)
    if hierarchical:
        conn.execute("create table Chunks(chunkId text primary key, contextId text, "
                     "chunk text, startoffset int, endoffset int)")
        conn.executemany("insert into Chunks values (?, 'ctx', ?, 0, 10)", hierarchical)
    if recursive:
        conn.execute("create table RecursiveChunks(chunkId text primary key, "
                     "documentId text, chunk text, startoffset int, endoffset int)")
        conn.executemany("insert into RecursiveChunks values (?, 'doc', ?, 0, 10)",
                         recursive)
    for label, chunk_id in mapped:
        conn.execute("insert into vector_meta_data values (?, ?, ?, 'm', 128)",
                     (label, vector_id_for(chunk_id), chunk_id))
    conn.commit()
    conn.close()
    return str(path)


def scored(*ids):
    return [ScoredId(i, 1.0 / (n + 1), VECTOR, n + 1) for n, i in enumerate(ids)]


class TestItReturnsTheText:
    def test_a_hierarchical_chunk_hydrates(self, tmp_path):
        path = store_with(tmp_path, hierarchical=[("c1", "the text")],
                          mapped=[(1, "c1")])
        passages, dropped = Hydration(path).hydrate(scored(1))
        assert passages[0].text == "the text"
        assert dropped == 0

    def test_a_recursive_chunk_hydrates(self, tmp_path):
        path = store_with(tmp_path, recursive=[("r1", "flat text")],
                          mapped=[(1, "r1")])
        passages, _ = Hydration(path).hydrate(scored(1))
        assert passages[0].text == "flat text"

    def test_both_kinds_in_one_call(self, tmp_path):
        path = store_with(tmp_path, hierarchical=[("c1", "sectioned")],
                          recursive=[("r1", "flat")], mapped=[(1, "c1"), (2, "r1")])
        passages, _ = Hydration(path).hydrate(scored(1, 2))
        assert [p.text for p in passages] == ["sectioned", "flat"]

    def test_the_offsets_and_parent_come_too(self, tmp_path):
        path = store_with(tmp_path, hierarchical=[("c1", "the text")],
                          mapped=[(1, "c1")])
        passage = Hydration(path).hydrate(scored(1))[0][0]
        assert (passage.start_offset, passage.end_offset) == (0, 10)
        assert passage.document_id == "ctx"

    def test_the_score_and_source_survive(self, tmp_path):
        path = store_with(tmp_path, hierarchical=[("c1", "t")], mapped=[(1, "c1")])
        passage = Hydration(path).hydrate(scored(1))[0][0]
        assert passage.source == VECTOR
        assert passage.score == 1.0


class TestOrder:
    def test_the_ranked_order_is_kept(self, tmp_path):
        """SQL returns rows in its own order; the ranking is what matters."""
        path = store_with(
            tmp_path,
            hierarchical=[("c1", "first"), ("c2", "second"), ("c3", "third")],
            mapped=[(1, "c1"), (2, "c2"), (3, "c3")],
        )
        passages, _ = Hydration(path).hydrate(scored(3, 1, 2))
        assert [p.text for p in passages] == ["third", "first", "second"]


class TestWhatCannotBeResolved:
    def test_an_unmapped_label_is_dropped_and_counted(self, tmp_path):
        """DiskANN pads short result sets with labels never allocated."""
        path = store_with(tmp_path, hierarchical=[("c1", "t")], mapped=[(1, "c1")])
        passages, dropped = Hydration(path).hydrate(scored(1, 999))
        assert len(passages) == 1
        assert dropped == 1

    def test_a_mapping_whose_chunk_is_gone_is_dropped(self, tmp_path):
        """An index can outlive the chunks it was built from."""
        path = store_with(tmp_path, hierarchical=[("c1", "t")],
                          mapped=[(1, "c1"), (2, "deleted_chunk")])
        passages, dropped = Hydration(path).hydrate(scored(1, 2))
        assert len(passages) == 1 and dropped == 1

    def test_everything_unresolvable_is_not_an_error(self, tmp_path):
        path = store_with(tmp_path, hierarchical=[("c1", "t")], mapped=[(1, "c1")])
        passages, dropped = Hydration(path).hydrate(scored(97, 98, 99))
        assert passages == [] and dropped == 3


class TestAnIncompleteStore:
    def test_a_corpus_with_only_one_chunk_kind_still_hydrates(self, tmp_path):
        """Only the hierarchical table exists when every document had sections.

        Naming both tables unconditionally fails the whole query with
        "no such table", which cost every result rather than none.
        """
        path = store_with(tmp_path, hierarchical=[("c1", "t")], mapped=[(1, "c1")])
        with sqlite3.connect(path) as conn:
            names = {r[0] for r in conn.execute(
                "select name from sqlite_master where type='table'")}
        assert "RecursiveChunks" not in names
        assert Hydration(path).hydrate(scored(1))[0][0].text == "t"

    def test_a_missing_store_returns_nothing_rather_than_raising(self, tmp_path):
        passages, dropped = Hydration(tmp_path / "absent.db").hydrate(scored(1))
        assert passages == [] and dropped == 1

    def test_a_store_with_no_chunk_tables_is_survived(self, tmp_path):
        path = store_with(tmp_path, mapped=[(1, "c1")])
        assert Hydration(path).hydrate(scored(1)) == ([], 1)

    def test_nothing_in_nothing_out(self, tmp_path):
        path = store_with(tmp_path, hierarchical=[("c1", "t")], mapped=[(1, "c1")])
        assert Hydration(path).hydrate([]) == ([], 0)


class TestItIsOneQuery:
    def test_k_results_cost_one_round_trip(self, tmp_path, monkeypatch):
        """The N+1 that makes retrieval slow long before the index does."""
        path = store_with(
            tmp_path,
            hierarchical=[(f"c{i}", f"text {i}") for i in range(20)],
            mapped=[(i + 1, f"c{i}") for i in range(20)],
        )
        opened = []
        real_connect = sqlite3.connect

        def counting_connect(*args, **kwargs):
            opened.append(args[0])
            return real_connect(*args, **kwargs)

        monkeypatch.setattr(sqlite3, "connect", counting_connect)
        passages, _ = Hydration(path).hydrate(scored(*range(1, 21)))

        assert len(passages) == 20
        assert len(opened) == 1
