"""`vector_meta_data`: a DiskANN label back to its vector id and chunk.

Bugs 5.2, 5.3 and the id-space bugs they hid, and 5.22: the vector id is the
one every store keys on, so it is passed in and required — never generated
here. The table generates only the DiskANN label, which is a different number.
"""

import sqlite3

import numpy as np
import pytest

from config import Config
from data_layer.datalayer_exceptions.datalayer_exceptions import (
    EmbeddingModelMismatch,
    InvalidBatchSize,
    InvalidColumnNameException,
    InvalidVectorID,
    MalformedVectorId,
    MissingVectorId,
    VectorIdConflict,
)
from data_layer.ingestion.Chunker.DB_Manager import Manager
from data_layer.ingestion.embedding.vector_ids import vector_id_for
from data_layer.vector_db_manager.repository.vectorMetaDataRepository import (
    VectorMetaDataRepository,
)


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


def add(repo, *chunk_ids):
    """Labels for chunks, under the vector ids the embedder would give them."""
    return repo.batch_insert([vector_id_for(c) for c in chunk_ids], list(chunk_ids))


class TestTheVectorIdIsRequired:
    def test_a_missing_vector_id_is_refused(self, repo):
        with pytest.raises(MissingVectorId, match="c1"):
            repo.insert(None, "c1")
        assert repo.count() == 0

    def test_one_missing_id_refuses_the_whole_batch_before_anything_is_written(self, repo):
        with pytest.raises(MissingVectorId):
            repo.batch_insert([vector_id_for("a"), None], ["a", "b"])
        assert repo.count() == 0

    @pytest.mark.parametrize("bad", [-1, 1 << 63, 1.5, "12", True])
    def test_a_vector_id_that_is_not_one_is_refused(self, repo, bad):
        with pytest.raises(MalformedVectorId):
            repo.insert(bad, "c1")
        assert repo.count() == 0

    def test_the_largest_vector_id_is_accepted(self, repo):
        repo.insert(Config.VECTOR_ID_MASK, "c1")
        assert repo.get_meta_data(Config.VECTOR_ID_MASK, "chunkId") == "c1"

    def test_a_numpy_integer_is_accepted(self, repo):
        repo.insert(np.int64(42), "c1")
        assert repo.get_meta_data(42, "chunkId") == "c1"

    @pytest.mark.parametrize("columns, values", [
        ("chunkId, embeddingModelUsed, dimensions", "'c1', 'm', 128"),
        ("vectorId, chunkId, embeddingModelUsed, dimensions", "null, 'c1', 'm', 128"),
        ("vectorId, chunkId, embeddingModelUsed, dimensions", "'abc', 'c1', 'm', 128"),
        ("vectorId, chunkId, embeddingModelUsed, dimensions", "-5, 'c1', 'm', 128"),
    ])
    def test_the_database_refuses_it_too(self, repo, columns, values):
        """A write that bypasses this class still cannot store a label without one."""
        with pytest.raises(sqlite3.IntegrityError):
            with repo._writing() as cursor:
                cursor.execute(f"insert into vector_meta_data({columns}) values ({values})")

    def test_the_vector_id_is_stored_as_given(self, repo):
        label = repo.insert(vector_id_for("c1"), "c1")
        assert repo.get_meta_data(vector_id_for("c1"), "label") == label
        assert repo.get_meta_data(vector_id_for("c1"), "chunkId") == "c1"


class TestTheLabelIsAllocated:
    def test_an_allocated_label_fits_uint32(self, repo):
        """diskannpy indexes uint32; the vector ids are 63-bit."""
        label, = add(repo, "c1")
        assert 0 < label <= np.iinfo(np.uint32).max

    def test_the_whole_index_fits(self):
        """Sequential labels cannot collide, which hashing into 32 bits would."""
        assert Config.MAX_VECTORS < np.iinfo(np.uint32).max

    def test_labels_are_unique(self, repo):
        labels = add(repo, *[f"c{i}" for i in range(50)])
        assert len(set(labels)) == 50

    def test_labels_are_not_reused_after_a_delete(self, repo, chunk_store):
        first, = add(repo, "c1")
        with sqlite3.connect(chunk_store) as conn:
            conn.execute("delete from vector_meta_data where label = ?", (first,))
        assert add(repo, "c2")[0] > first

    def test_a_label_past_uint32_is_refused_rather_than_wrapped(self, repo):
        with repo._writing() as cursor:
            cursor.execute(
                "insert into sqlite_sequence(name, seq) values ('vector_meta_data', 4294967295)"
            )
        with pytest.raises(sqlite3.IntegrityError):
            add(repo, "c1")

    def test_a_batch_comes_back_in_order(self, repo):
        labels = add(repo, "a", "b", "c")
        assert [repo.chunk_ids_for([l])[l] for l in labels] == ["a", "b", "c"]

    def test_allocating_nothing_is_not_an_error(self, repo):
        assert repo.batch_insert([], []) == []

    def test_mismatched_batch_sizes_are_refused(self, repo):
        with pytest.raises(InvalidBatchSize):
            repo.batch_insert([1, 2], ["only_one"])


class TestOneLabelPerChunk:
    """Bug 5.21: allocation inserted unconditionally, so a real store held one
    chunk under 45 labels after repeated ingests."""

    def test_the_same_chunk_again_gets_its_label_back(self, repo):
        first, = add(repo, "c1")
        assert add(repo, "c1") == [first]
        assert repo.count() == 1

    def test_a_batch_reuses_known_labels_and_keeps_its_order(self, repo):
        known = add(repo, "a", "b")
        mixed = add(repo, "x", "a", "y", "b")
        assert mixed[1] == known[0] and mixed[3] == known[1]
        assert len({mixed[0], mixed[2]} - set(known)) == 2
        assert repo.count() == 4

    def test_the_same_chunk_twice_in_one_batch_gets_one_label(self, repo):
        labels = add(repo, "a", "a")
        assert labels[0] == labels[1]
        assert repo.count() == 1

    def test_a_chunk_under_a_different_vector_id_is_refused(self, repo):
        add(repo, "c1")
        with pytest.raises(VectorIdConflict, match="c1"):
            repo.insert(vector_id_for("c1") + 1, "c1")
        assert repo.count() == 1

    def test_a_vector_id_already_mapped_to_another_chunk_is_refused(self, repo):
        add(repo, "c1")
        with pytest.raises(VectorIdConflict, match="c1"):
            repo.insert(vector_id_for("c1"), "c2")

    def test_another_model_is_refused_rather_than_mixed_in(self, repo):
        """A vector id belongs to the chunk, so a chunk holds one model's vector."""
        add(repo, "c1")
        with pytest.raises(EmbeddingModelMismatch):
            repo.insert(vector_id_for("c1"), "c1", embeddingModelUsed="some/other-model")

    def test_a_refused_row_rolls_back_the_batch(self, repo):
        add(repo, "c1")
        with pytest.raises(VectorIdConflict):
            repo.batch_insert([vector_id_for("new"), vector_id_for("c1") + 1], ["new", "c1"])
        assert repo.labels_for(["new"]) == {}


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
            conn.execute("insert into Chunks values ('c1', null, 'the text', 0, 8)")
        label, = add(repo, "c1")
        with sqlite3.connect(chunk_store) as conn:
            row = conn.execute(
                "select c.chunk from vector_meta_data v "
                "join Chunks c on c.chunkId = v.chunkId where v.label = ?",
                (label,),
            ).fetchone()
        assert row[0] == "the text"

    def test_a_recursive_chunk_maps_too(self, repo, chunk_store):
        """A chunk id lives in one of two tables, which is why there is no FK."""
        with sqlite3.connect(chunk_store) as conn:
            conn.execute("insert into RecursiveChunks values ('r1', null, 'flat text', 0, 9)")
        label, = add(repo, "r1")
        assert repo.chunk_ids_for([label]) == {label: "r1"}

    def test_there_is_no_vector_column(self, repo):
        """The vectors live in PostgreSQL; this table only short-circuits to the chunk."""
        columns = {r[1] for r in repo.connection.execute("pragma table_info(vector_meta_data)")}
        assert columns == {"label", "vectorId", "chunkId", "embeddingModelUsed", "dimensions"}


class TestResolving:
    def test_a_batch_resolves_in_one_query(self, repo):
        labels = add(repo, *[f"c{i}" for i in range(5)])
        assert len(repo.chunk_ids_for(labels)) == 5

    def test_an_unmapped_label_is_absent_rather_than_raising(self, repo):
        """A stale index can return a label whose chunk has since gone."""
        label, = add(repo, "c1")
        assert repo.chunk_ids_for([label, 999_999]) == {label: "c1"}

    def test_resolving_nothing_is_not_an_error(self, repo):
        assert repo.chunk_ids_for([]) == {}
        assert repo.labels_for([]) == {}

    def test_the_reverse_direction_works(self, repo):
        label, = add(repo, "c1")
        assert repo.labels_for(["c1"]) == {"c1": label}

    def test_an_unknown_column_is_refused(self, repo):
        add(repo, "c1")
        with pytest.raises(InvalidColumnNameException):
            repo.get_meta_data(vector_id_for("c1"), "chunkId; drop table Chunks")

    def test_an_unknown_vector_id_raises(self, repo):
        with pytest.raises(InvalidVectorID):
            repo.get_meta_data(424242, "chunkId")


class TestPaging:
    def test_labels_come_back_a_page_at_a_time_in_order(self, repo):
        labels = add(repo, *[f"c{i}" for i in range(10)])
        pages = list(repo.labels(batch_size=3))
        assert [len(page_labels) for page_labels, _ in pages] == [3, 3, 3, 1]
        assert np.concatenate([p for p, _ in pages]).tolist() == labels

    def test_each_label_comes_with_its_vector_id(self, repo):
        add(repo, "a", "b")
        (labels, vector_ids), = list(repo.labels())
        assert vector_ids.tolist() == [vector_id_for("a"), vector_id_for("b")]
        assert labels.dtype == np.uint32 and vector_ids.dtype == np.int64

    def test_only_labels_after_the_one_given(self, repo):
        labels = add(repo, *[f"c{i}" for i in range(6)])
        later = np.concatenate([l for l, _ in repo.labels(after=labels[3])]).tolist()
        assert later == labels[4:]
        assert repo.pending(labels[3]) == 2

    def test_nothing_after_the_last_label(self, repo):
        labels = add(repo, "a")
        assert list(repo.labels(after=labels[-1])) == []
        assert repo.pending(labels[-1]) == 0

    def test_another_model_is_never_paged_in(self, tmp_path):
        """A store written by another model holds vectors in a different space."""
        repo = VectorMetaDataRepository(str(tmp_path / "other"))
        repo.insert(vector_id_for("old"), "old", embeddingModelUsed="some/other-model")
        repo.insert(vector_id_for("new"), "new")
        labels = np.concatenate([l for l, _ in repo.labels()]).tolist()
        assert labels == [repo.labels_for(["new"])["new"]]
        assert repo.pending() == 1
        repo.close()


class TestTheStore:
    def test_wal_is_in_force(self, repo):
        assert repo.journal_mode == "wal"

    def test_close_is_safe_to_call_twice(self, chunk_store):
        repo = VectorMetaDataRepository(chunk_store)
        repo.close()
        repo.close()


class TestMigratingAnOldTable:
    """Bug 5.22: tables from before the vector id was required kept the label
    in `vectorId`, and some kept the vector itself."""

    def old_store(self, tmp_path, rows):
        path = str(tmp_path / "old.db")
        with sqlite3.connect(path) as conn:
            conn.execute("""create table vector_meta_data(
                vectorId integer primary key autoincrement, chunkId text not null,
                embeddingModelUsed text not null, dimensions integer not null, vector blob)""")
            conn.executemany(
                "insert into vector_meta_data(vectorId, chunkId, embeddingModelUsed, "
                "dimensions, vector) values (?, ?, ?, 128, ?)", rows)
        return path

    def test_labels_are_kept_and_vector_ids_derived(self, tmp_path):
        repo = VectorMetaDataRepository(self.old_store(
            tmp_path, [(3, "c1", Config.EMBEDDING_MODEL, b"x"), (7, "c2", Config.EMBEDDING_MODEL, None)]))
        assert repo.labels_for(["c1", "c2"]) == {"c1": 3, "c2": 7}
        assert repo.get_meta_data(vector_id_for("c2"), "label") == 7
        repo.close()

    def test_the_vector_column_is_gone(self, tmp_path):
        repo = VectorMetaDataRepository(self.old_store(tmp_path, [(1, "c1", Config.EMBEDDING_MODEL, b"x")]))
        columns = {r[1] for r in repo.connection.execute("pragma table_info(vector_meta_data)")}
        assert "vector" not in columns and "label" in columns
        repo.close()

    def test_duplicates_keep_the_configured_models_oldest_label(self, tmp_path):
        repo = VectorMetaDataRepository(self.old_store(tmp_path, [
            (2, "c1", "some/other-model", None),
            (4, "c1", Config.EMBEDDING_MODEL, None),
            (5, "c1", Config.EMBEDDING_MODEL, None),
        ]))
        assert repo.count() == 1
        assert repo.labels_for(["c1"]) == {"c1": 4}
        repo.close()

    def test_a_label_that_does_not_fit_uint32_is_dropped(self, tmp_path):
        """The first schema stored 63-bit derived ids there, never labels."""
        repo = VectorMetaDataRepository(self.old_store(
            tmp_path, [(vector_id_for("c1"), "c1", Config.EMBEDDING_MODEL, None)]))
        assert repo.count() == 0
        repo.close()

    def test_labels_the_old_table_handed_out_are_not_handed_out_again(self, tmp_path):
        repo = VectorMetaDataRepository(self.old_store(
            tmp_path, [(3, "c1", Config.EMBEDDING_MODEL, None), (9, "c1", "x", None)]))
        assert add(repo, "new") == [10]
        repo.close()

    def test_the_rebuild_is_reported(self, tmp_path, caplog):
        with caplog.at_level("WARNING"):
            VectorMetaDataRepository(self.old_store(
                tmp_path, [(1, "c1", Config.EMBEDDING_MODEL, None)])).close()
        assert any("kept 1 row" in r.getMessage() for r in caplog.records)

    def test_opening_it_again_changes_nothing(self, tmp_path, caplog):
        path = self.old_store(tmp_path, [(1, "c1", Config.EMBEDDING_MODEL, None)])
        VectorMetaDataRepository(path).close()
        caplog.clear()
        with caplog.at_level("WARNING"):
            repo = VectorMetaDataRepository(path)
        assert repo.count() == 1 and not caplog.records
        repo.close()

    def test_an_empty_old_table_becomes_an_empty_new_one(self, tmp_path):
        repo = VectorMetaDataRepository(self.old_store(tmp_path, []))
        assert repo.count() == 0
        assert add(repo, "c1") == [1]
        repo.close()

    def test_the_first_schema_with_its_nullable_key_is_migrated_too(self, tmp_path):
        path = str(tmp_path / "first.db")
        with sqlite3.connect(path) as conn:
            conn.execute("create table vector_meta_data(vectorId int primary key, chunkId text, "
                         "embeddingModelUsed text, dimensions int)")
            conn.execute("insert into vector_meta_data values (null, 'c1', 'm', 128)")
        repo = VectorMetaDataRepository(path)
        assert repo.count() == 0
        repo.close()


class TestTheSuiteNeverTouchesTheRealStores:
    """bugs.md 7.5: the pipeline's end-to-end test wrote a label into the
    developer's own chunk store on every run."""

    def test_the_configured_paths_are_test_files(self, tmp_path_factory):
        base = str(tmp_path_factory.getbasetemp())
        assert Config.DB_PATH != str(Config.ABS_PATH / "data" / "hierarchical_db")
        assert base in Config.DB_PATH
        assert base in Config.INDEX_PATH

    def test_no_test_undoes_every_patch(self):
        """`monkeypatch.undo()` also reverts conftest's redirects, and the rest
        of that test then reaches the developer's real PostgreSQL. Scope a
        temporary patch with `pytest.MonkeyPatch.context()` instead."""
        from pathlib import Path

        tests = Path(__file__).resolve().parents[1]
        offenders = [str(p.relative_to(tests)) for p in tests.rglob("*.py")
                     if p != Path(__file__).resolve()
                     and "monkeypatch.undo(" in p.read_text(encoding="utf-8")]
        assert offenders == [], offenders

    def test_the_chunk_vector_store_is_not_postgresql(self, chunk_vectors):
        from data_layer.vector_db_manager import stored_vectors

        assert stored_vectors.chunk_vector_store() is chunk_vectors

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
