"""Document chunk vectors: labels from the chunk store, vectors from pgvector."""

from typing import Iterator, NamedTuple

import numpy as np
from numpy import float32, ndarray, uint32

from config import Config, get_logger
from data_layer.vector_db_manager.repository.vectorMetaDataRepository import (
    PAGE,
    VectorMetaDataRepository,
)

logger = get_logger(__name__)


def chunk_vector_store():
    """The pgvector store document chunk vectors live in."""
    from data_layer.vector_db_manager.repository.vectorRepository import (
        VectorRepository,
    )

    return VectorRepository(Config.GLOBAL_VECTOR_SCOPE)


class Page(NamedTuple):
    labels: ndarray
    vectors: ndarray
    through: int


class StoredVectors:
    """What an index is built from: each label with its vector, in label order."""

    def __init__(self, mapping: VectorMetaDataRepository, store) -> None:
        self.mapping = mapping
        self.store = store
        self.missing = 0

    def pending(self, after: int = 0) -> int:
        return self.mapping.pending(after)

    def pages(self, after: int = 0, batch_size: int = PAGE) -> Iterator[Page]:
        """Labels above `after` with their vectors; `through` is the last label looked at.

        A label whose vector is not stored is skipped and counted in `missing`,
        and still moves `through` on, so it is not looked for again.
        """
        for labels, vector_ids in self.mapping.labels(after, batch_size):
            found = self.store.vectors_for(vector_ids.tolist())
            keep = [i for i, vector_id in enumerate(vector_ids.tolist()) if vector_id in found]
            self.missing += len(labels) - len(keep)
            through = int(labels[-1])
            if not keep:
                yield Page(np.empty(0, uint32), np.empty((0, Config.EMBEDDING_DIMENSIONS), float32), through)
                continue
            vectors = np.stack([found[int(vector_ids[i])] for i in keep]).astype(float32, copy=False)
            yield Page(labels[keep], vectors, through)
