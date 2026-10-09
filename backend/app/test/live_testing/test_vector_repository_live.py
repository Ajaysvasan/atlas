"""VectorRepository against a real PostgreSQL, where the mocks cannot reach.

Bugs 5.13 and 5.14 both passed the mocked suite: a MagicMock cursor adapts any
object handed to it and returns whatever it is told to, so neither the adapter
registration order nor the type that comes back is observable there.
"""

import numpy as np
import pytest

from config import Config

pytestmark = pytest.mark.live


def a_vector(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random(Config.EMBEDDING_DIMENSIONS).astype(
        np.float32
    )


class TestTheWritePath:
    """Bug 5.13: the cursor bound its adapters before the vector types existed."""

    def test_a_numpy_array_can_be_inserted(self, repository):
        repository.insert(1, a_vector())

    def test_a_numpy_array_can_be_batch_inserted(self, repository):
        repository.batch_insert([2, 3], [a_vector(2), a_vector(3)])

    def test_a_numpy_array_can_be_updated(self, repository):
        repository.insert(4, a_vector(4))
        repository.update(4, a_vector(40))

        assert np.array_equal(repository.search(4), a_vector(40))


class TestTheReadPath:
    """Bug 5.14: pgvector returns its own Vector, which numpy cannot coerce."""

    def test_a_read_returns_a_numpy_array(self, repository):
        repository.insert(5, a_vector(5))
        returned = repository.search(5)

        assert isinstance(returned, np.ndarray)
        assert returned.dtype == np.float32
        assert returned.shape == (Config.EMBEDDING_DIMENSIONS,)

    def test_a_batch_read_returns_numpy_arrays(self, repository):
        repository.batch_insert([6, 7], [a_vector(6), a_vector(7)])
        returned = repository.batch_search([6, 7])

        assert isinstance(returned, np.ndarray)
        assert returned.shape == (2, Config.EMBEDDING_DIMENSIONS)


class TestTheRoundTrip:
    def test_what_comes_back_is_what_went_in(self, repository):
        written = a_vector(8)
        repository.insert(8, written)

        assert np.array_equal(repository.search(8), written)

    def test_a_batch_comes_back_in_the_order_asked_for(self, repository):
        first, second = a_vector(9), a_vector(10)
        repository.batch_insert([9, 10], [first, second])

        returned = repository.batch_search([10, 9])

        assert np.array_equal(returned[0], second)
        assert np.array_equal(returned[1], first)

    def test_a_deleted_vector_is_gone(self, repository):
        from data_layer.datalayer_exceptions.datalayer_exceptions import (
            VectorNotFoundEror,
        )

        repository.insert(11, a_vector(11))
        repository.delete(11)

        with pytest.raises(VectorNotFoundEror):
            repository.search(11)


class TestTheBulkRead:
    """What the index is built from: every stored vector among a page of ids,
    in one query. Vector ids use all 63 bits, which psycopg sends as numeric
    unless the query casts them."""

    def test_ids_past_32_bits_come_back_exactly(self, repository):
        from data_layer.ingestion.embedding.vector_ids import vector_id_for

        ids = [vector_id_for(f"chunk-{i}") for i in range(3)]
        assert max(ids) > 2**32
        written = [a_vector(20 + i) for i in range(3)]
        repository.batch_insert(ids, written)

        found = repository.vectors_for(ids)

        assert set(found) == set(ids)
        for vector_id, vector in zip(ids, written):
            assert found[vector_id].dtype == np.float32
            assert np.array_equal(found[vector_id], vector)

    def test_ids_not_stored_are_absent(self, repository):
        repository.insert(30, a_vector(30))
        assert set(repository.vectors_for([30, 31])) == {30}

    def test_another_projects_vectors_are_not_returned(self, repository):
        from data_layer.vector_db_manager.repository.vectorRepository import (
            VectorRepository,
        )

        other = VectorRepository(repository.project_id + "_other")
        try:
            other.insert(40, a_vector(40))
            assert repository.vectors_for([40]) == {}
        finally:
            other.curr.execute("delete from vectors where project_id = %s", (other.project_id,))
            other.conn.commit()
            other.close()

    def test_nothing_asked_for_is_not_a_query(self, repository):
        assert repository.vectors_for([]) == {}

    def test_stored_vectors_pair_labels_with_what_postgresql_holds(self, repository, tmp_path):
        from data_layer.ingestion.embedding.vector_ids import vector_id_for
        from data_layer.vector_db_manager.repository.vectorMetaDataRepository import (
            VectorMetaDataRepository,
        )
        from data_layer.vector_db_manager.stored_vectors import StoredVectors

        mapping = VectorMetaDataRepository(str(tmp_path / "chunks"))
        chunk_ids = [f"chunk-{i}" for i in range(4)]
        ids = [vector_id_for(c) for c in chunk_ids]
        written = np.stack([a_vector(50 + i) for i in range(4)])
        repository.batch_insert(ids[:3], written[:3])
        labels = mapping.batch_insert(ids, chunk_ids)
        source = StoredVectors(mapping, repository)

        page, = list(source.pages())
        mapping.close()

        assert page.labels.tolist() == labels[:3]
        assert np.array_equal(page.vectors, written[:3])
        assert page.through == labels[3] and source.missing == 1
