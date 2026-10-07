"""Bugs 5.2, 5.3 and the id-space bugs they hid.

A DiskANN label is all a search returns, and the id the project derives for a
chunk is a one-way hash. Without this table a hit cannot become text, which is
the one thing retrieval needs.
"""

import sqlite3

import numpy as np
import pytest

from data_layer.ingestion.Chunker.DB_Manager import Manager
from data_layer.vector_db_manager.repository.vectorMetaDataRepository import (
    VectorMetaDataRepository,
)
from config import Config


@pytest.fixture
def chunk_store(tmp_path):
    """A real chunk store, which is where the mapping table has to live."""
    path = str(tmp_path / "chunks.db")
    Manager(path, True)
    Manager(path, False)
    return path


@pytest.fixture
def repo(chunk_store):
    r = VectorMetaDataRepository(chunk_store)
    yield r
    r.close()


class TestItLivesWithTheChunks:
    def test_the_table_is_created_in_the_chunk_store(self, repo, chunk_store):
        with sqlite3.connect(chunk_store) as conn:
            tables = {r[0] for r in conn.execute(
                "select name from sqlite_master where type='table'"
            )}
        assert {"vector_meta_data", "Chunks", "RecursiveChunks"} <= tables

    def test_a_hit_reaches_its_text_in_one_join(self, repo, chunk_store):
        """The reason it is co-located rather than in a database of its own."""
        with sqlite3.connect(chunk_store) as conn:
            conn.execute(
                "insert into Chunks values ('c1', null, 'the text', 0, 8)"
            )
            conn.commit()
        label = repo.allocate("c1")

        with sqlite3.connect(chunk_store) as conn:
            row = conn.execute(
                "select c.chunk from vector_meta_data v "
                "join Chunks c on c.chunkId = v.chunkId where v.vectorId = ?",
                (label,),
            ).fetchone()
        assert row[0] == "the text"

    def test_an_insert_succeeds(self, repo):
        """Bug 5.2: the old foreign key named a table in another database."""
        repo.insert(1, "chunk_a", "all-MiniLM-L6-v2", 128)
        assert repo.count() == 1

    def test_a_recursive_chunk_maps_too(self, repo, chunk_store):
        """A chunk id lives in one of two tables, which is why there is no FK."""
        with sqlite3.connect(chunk_store) as conn:
            conn.execute(
                "insert into RecursiveChunks values ('r1', null, 'flat text', 0, 9)"
            )
            conn.commit()
        label = repo.allocate("r1")
        assert repo.chunk_ids_for([label]) == {label: "r1"}


class TestTheLabelsFitDiskann:
    def test_an_allocated_label_fits_uint32(self, repo):
        """diskannpy indexes uint32; the project's derived ids are 63-bit."""
        label = repo.allocate("c1")
        assert 0 < label <= np.iinfo(np.uint32).max

    def test_the_whole_index_fits(self, repo):
        """Sequential labels cannot collide, which hashing into 32 bits would."""
        assert Config.MAX_VECTORS < np.iinfo(np.uint32).max

    def test_labels_are_unique(self, repo):
        labels = repo.allocate_many([f"c{i}" for i in range(50)])
        assert len(set(labels)) == 50

    def test_labels_are_not_reused_after_a_delete(self, repo, chunk_store):
        first = repo.allocate("c1")
        with sqlite3.connect(chunk_store) as conn:
            conn.execute("delete from vector_meta_data where vectorId = ?", (first,))
            conn.commit()
        assert repo.allocate("c2") > first

    def test_allocate_many_returns_them_in_order(self, repo):
        labels = repo.allocate_many(["a", "b", "c"])
        assert [repo.chunk_ids_for([l])[l] for l in labels] == ["a", "b", "c"]

    def test_allocating_nothing_is_not_an_error(self, repo):
        assert repo.allocate_many([]) == []


class TestResolving:
    def test_a_batch_resolves_in_one_query(self, repo):
        labels = repo.allocate_many([f"c{i}" for i in range(5)])
        assert len(repo.chunk_ids_for(labels)) == 5

    def test_an_unmapped_label_is_absent_rather_than_raising(self, repo):
        """A stale index can return a label whose chunk has since gone.

        DiskANN also pads a short result set, so a search for k neighbours over
        fewer vectors returns labels that were never allocated.
        """
        label = repo.allocate("c1")
        assert repo.chunk_ids_for([label, 999_999]) == {label: "c1"}

    def test_resolving_nothing_is_not_an_error(self, repo):
        assert repo.chunk_ids_for([]) == {}

    def test_the_reverse_direction_works(self, repo):
        label = repo.allocate("c1")
        assert repo.vector_ids_for(["c1"]) == {"c1": label}

    def test_an_unknown_column_is_refused(self, repo):
        from data_layer.datalayer_exceptions.datalayer_exceptions import (
            InvalidColumnNameException,
        )
        repo.allocate("c1")
        with pytest.raises(InvalidColumnNameException):
            repo.get_meta_data(1, "chunkId; drop table Chunks")

    def test_an_unknown_label_raises(self, repo):
        from data_layer.datalayer_exceptions.datalayer_exceptions import (
            InvalidVectorID,
        )
        with pytest.raises(InvalidVectorID):
            repo.get_meta_data(424242, "chunkId")


class TestTheStore:
    def test_wal_is_in_force(self, repo):
        assert repo.journal_mode == "wal"

    def test_close_is_safe_to_call_twice(self, chunk_store):
        repo = VectorMetaDataRepository(chunk_store)
        repo.close()
        repo.close()

    def test_mismatched_batch_sizes_are_refused(self, repo):
        from data_layer.datalayer_exceptions.datalayer_exceptions import (
            InvalidBatchSize,
        )
        with pytest.raises(InvalidBatchSize):
            repo.batch_insert([1, 2], ["only_one"], "model", 128)
