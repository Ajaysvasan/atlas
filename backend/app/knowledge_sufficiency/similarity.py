"""Cosine scoring, shared by every caller that ranks vectors against a query."""

import numpy as np
from numpy import float32, ndarray
from numpy.typing import NDArray


def cosine_scores(query: ndarray, matrix: NDArray[float32]) -> NDArray[float32]:
    """Cosine of the query against each row. A zero row scores -1."""
    if matrix.size == 0:
        return np.empty(0, dtype=float32)
    scale = np.linalg.norm(matrix, axis=1) * float(np.linalg.norm(query))
    # Guarded twice over: a zero row has no direction, and dividing by its norm
    # would make every score nan, which then propagates through max().
    scores = matrix @ query / np.where(scale == 0, 1.0, scale)
    return np.where(scale == 0, -1.0, scores).astype(float32)
