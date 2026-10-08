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


class TestTheVectorIsKept:
    """Bug 5.20: diskannpy cannot load an index it saved, so the vectors kept
    here are what the index is rebuilt from after a restart."""

    def vecs(self, n, seed=0):
        return list(np.random.default_rng(seed).standard_normal((n, 128)).astype(np.float32))

    def test_a_batch_comes_back_exactly(self, repo):
        stored = self.vecs(5)
        labels = repo.allocate_many([f"c{i}" for i in range(5)], stored)
        (got_labels, got_vectors), = list(repo.vectors())
        assert got_labels.tolist() == labels
        np.testing.assert_array_equal(got_vectors, np.stack(stored))

    def test_a_single_allocation_keeps_its_vector(self, repo):
        vector = self.vecs(1)[0]
        label = repo.allocate("c1", vector)
        (got_labels, got_vectors), = list(repo.vectors())
        assert got_labels.tolist() == [label]
        np.testing.assert_array_equal(got_vectors[0], vector)

    def test_a_vector_is_stored_as_float32(self, repo):
        repo.allocate("c1", np.ones(128, dtype=np.float64))
        (_, got), = list(repo.vectors())
        assert got.dtype == np.float32

    def test_one_vector_per_chunk_is_required(self, repo):
        from data_layer.datalayer_exceptions.datalayer_exceptions import InvalidBatchSize

        with pytest.raises(InvalidBatchSize):
            repo.allocate_many(["a", "b"], self.vecs(1))
        assert repo.count() == 0

    def test_a_vector_of_the_wrong_width_is_refused_before_anything_is_written(self, repo):
        from data_layer.datalayer_exceptions.datalayer_exceptions import (
            InvalidVectorDimension,
        )

        with pytest.raises(InvalidVectorDimension):
            repo.allocate_many(["a", "b"], [np.ones(128), np.ones(64)])
        assert repo.count() == 0

    def test_they_come_back_a_page_at_a_time_in_label_order(self, repo):
        repo.allocate_many([f"c{i}" for i in range(10)], self.vecs(10))
        pages = list(repo.vectors(batch_size=3))
        assert [len(labels) for labels, _ in pages] == [3, 3, 3, 1]
        everything = np.concatenate([labels for labels, _ in pages]).tolist()
        assert everything == sorted(everything) and len(everything) == 10

    def test_only_labels_after_the_one_given(self, repo):
        labels = repo.allocate_many([f"c{i}" for i in range(6)], self.vecs(6))
        later = np.concatenate([l for l, _ in repo.vectors(after=labels[3])]).tolist()
        assert later == labels[4:]

    def test_nothing_after_the_last_label(self, repo):
        labels = repo.allocate_many(["a"], self.vecs(1))
        assert list(repo.vectors(after=labels[-1])) == []

    def test_a_label_without_a_vector_is_skipped_and_counted(self, repo):
        repo.allocate("no vector")
        repo.allocate("has one", self.vecs(1)[0])
        labels = np.concatenate([l for l, _ in repo.vectors()]).tolist()
        assert len(labels) == 1
        assert repo.missing_vectors() == 1

    def test_another_models_vectors_are_never_mixed_in(self, repo):
        """A different model is a different space; its neighbours are noise."""
        repo.allocate("old", self.vecs(1)[0], embeddingModelUsed="some/other-model")
        repo.allocate("new", self.vecs(1, seed=1)[0])
        labels = np.concatenate([l for l, _ in repo.vectors()]).tolist()
        assert len(labels) == 1
        assert repo.missing_vectors() == 1


class TestMigratingAnOldTable:
    """Stores written before vectors were kept have the table without the
    column; they are upgraded in place and their rows kept."""

    def old_store(self, tmp_path, rows=3):
        path = str(tmp_path / "old.db")
        with sqlite3.connect(path) as conn:
            conn.execute("""create table vector_meta_data(
                vectorId integer primary key autoincrement,
                chunkId text not null, embeddingModelUsed text not null,
                dimensions integer not null)""")
            conn.executemany(
                "insert into vector_meta_data(chunkId, embeddingModelUsed, dimensions) "
                "values (?, ?, ?)",
                [(f"c{i}", Config.EMBEDDING_MODEL, 128) for i in range(rows)])
        return path

    def test_the_column_is_added_and_the_rows_kept(self, tmp_path):
        path = self.old_store(tmp_path)
        repo = VectorMetaDataRepository(path)
        columns = {r[1] for r in repo.connection.execute(
            "pragma table_info(vector_meta_data)")}
        assert "vector" in columns
        assert repo.count() == 3
        repo.close()

    def test_old_rows_are_reported_as_unsearchable(self, tmp_path):
        repo = VectorMetaDataRepository(self.old_store(tmp_path))
        assert repo.missing_vectors() == 3
        repo.close()

    def test_new_rows_keep_their_vectors_after_the_upgrade(self, tmp_path):
        repo = VectorMetaDataRepository(self.old_store(tmp_path))
        label = repo.allocate("new", np.ones(128, dtype=np.float32))
        assert np.concatenate([l for l, _ in repo.vectors()]).tolist() == [label]
        repo.close()

    def test_opening_an_upgraded_store_again_changes_nothing(self, tmp_path):
        path = self.old_store(tmp_path)
        VectorMetaDataRepository(path).close()
        repo = VectorMetaDataRepository(path)
        assert repo.count() == 3
        repo.close()


class TestTheSuiteNeverTouchesTheRealChunkStore:
    """bugs.md 7.5: the pipeline's end-to-end test wrote a label into the
    developer's own chunk store on every run."""

    def test_the_configured_path_is_a_test_file(self, tmp_path_factory):
        real = str(Config.ABS_PATH / "data" / "hierarchical_db")
        assert Config.DB_PATH != real
        assert str(tmp_path_factory.getbasetemp()) in Config.DB_PATH

    def test_a_chunker_follows_the_path_configured_when_it_is_built(self, tmp_path, monkeypatch):
        from data_layer.ingestion.Chunker.chunker import Chunker

        monkeypatch.setattr(Config, "DB_PATH", str(tmp_path / "late"))
        assert Chunker().db_path == str(tmp_path / "late")

    @pytest.mark.parametrize("build", ["hydration", "vector_search"])
    def test_retrieval_follows_the_path_configured_when_it_is_built(self, tmp_path, monkeypatch, build):
        from retrieval_layer.hydration import Hydration
        from retrieval_layer.vector_search import VectorSearch

        monkeypatch.setattr(Config, "DB_PATH", str(tmp_path / "late"))
        made = Hydration() if build == "hydration" else VectorSearch(index=object())
        assert str(made.chunk_store_path) == str(tmp_path / "late")


class TestOneLabelPerChunk:
    """Bug 5.21: allocation inserted unconditionally, so a real store held one
    chunk under 45 labels after repeated ingests."""

    def vec(self, seed=0):
        return np.random.default_rng(seed).standard_normal(128).astype(np.float32)

    def test_allocating_a_chunk_again_returns_its_label(self, repo):
        first = repo.allocate("c1", self.vec())
        assert repo.allocate("c1", self.vec()) == first
        assert repo.count() == 1

    def test_a_batch_reuses_known_labels_and_keeps_its_order(self, repo):
        known = repo.allocate_many(["a", "b"], [self.vec(1), self.vec(2)])
        mixed = repo.allocate_many(["x", "a", "y", "b"], [self.vec(i) for i in range(4)])
        assert mixed[1] == known[0] and mixed[3] == known[1]
        assert len({mixed[0], mixed[2]} - set(known)) == 2
        assert repo.count() == 4

    def test_the_same_chunk_twice_in_one_batch_gets_one_label(self, repo):
        labels = repo.allocate_many(["a", "a"], [self.vec(), self.vec()])
        assert labels[0] == labels[1]
        assert repo.count() == 1

    def test_another_model_gets_its_own_label(self, repo):
        mine = repo.allocate("c1", self.vec())
        theirs = repo.allocate("c1", self.vec(), embeddingModelUsed="some/other-model")
        assert mine != theirs

    def test_a_label_without_a_vector_gains_one(self, repo):
        """A label written before vectors were kept, re-ingested: same label,
        and now searchable."""
        label = repo.allocate("c1")
        assert repo.allocate("c1", self.vec(3)) == label
        (labels, vectors), = list(repo.vectors())
        assert labels.tolist() == [label]
        np.testing.assert_array_equal(vectors[0], self.vec(3))

    def test_a_stored_vector_is_not_replaced(self, repo):
        repo.allocate("c1", self.vec(1))
        repo.allocate("c1", self.vec(2))
        (_, vectors), = list(repo.vectors())
        np.testing.assert_array_equal(vectors[0], self.vec(1))

    def test_the_database_refuses_a_second_label(self, repo):
        repo.allocate("c1", self.vec())
        with pytest.raises(sqlite3.IntegrityError):
            with repo._writing() as cursor:
                cursor.execute(
                    "insert into vector_meta_data(chunkId, embeddingModelUsed, dimensions) "
                    "values ('c1', ?, 128)", (Config.EMBEDDING_MODEL,))


class TestCollapsingAStoreWrittenBeforeTheConstraint:
    def old_store(self, tmp_path):
        path = str(tmp_path / "old.db")
        vector = np.ones(128, dtype=np.float32).tobytes()
        with sqlite3.connect(path) as conn:
            conn.execute("""create table vector_meta_data(
                vectorId integer primary key autoincrement, chunkId text not null,
                embeddingModelUsed text not null, dimensions integer not null, vector blob)""")
            conn.execute("create index idx_vector_meta_chunk on vector_meta_data(chunkId)")
            rows = [("c1", None), ("c1", vector), ("c1", vector), ("c2", None), ("c2", None)]
            conn.executemany(
                "insert into vector_meta_data(chunkId, embeddingModelUsed, dimensions, vector) "
                "values (?, ?, 128, ?)", [(c, Config.EMBEDDING_MODEL, v) for c, v in rows])
        return path

    def test_each_chunk_keeps_one_label(self, tmp_path):
        repo = VectorMetaDataRepository(self.old_store(tmp_path))
        assert repo.count() == 2
        repo.close()

    def test_the_label_kept_is_the_one_with_a_vector(self, tmp_path):
        """Labels 2 and 3 have vectors; the lowest of them survives."""
        repo = VectorMetaDataRepository(self.old_store(tmp_path))
        assert repo.vector_ids_for(["c1"]) == {"c1": 2}
        assert repo.vector_ids_for(["c2"]) == {"c2": 4}
        repo.close()

    def test_the_removal_is_reported(self, tmp_path, caplog):
        with caplog.at_level("WARNING"):
            VectorMetaDataRepository(self.old_store(tmp_path)).close()
        assert any("3 duplicate label" in r.getMessage() for r in caplog.records)

    def test_reopening_removes_nothing_more(self, tmp_path):
        path = self.old_store(tmp_path)
        VectorMetaDataRepository(path).close()
        repo = VectorMetaDataRepository(path)
        assert repo.count() == 2
        repo.close()

