"""FAISS o su plan B, y ausencia de fuga en el esquema confirmado."""

import numpy as np

from plena.faiss_index import FlatL2Index, l2_matches_weighted
from plena.distance import ColumnMap


def test_faiss_or_numpy_neighbors_match():
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [5.0, 5.0]])
    database = np.vstack([center + rng.normal(0, 0.05, size=(8, 2)) for center in centers])
    queries = centers + 0.01
    index = FlatL2Index(database)
    found = index.search(queries, 3)
    reference = index.search_numpy(queries, 3, dtype=np.float32 if index.backend == "faiss" else np.float64)
    for row in range(len(queries)):
        assert set(found.indices[row]) == set(reference.indices[row])


def test_weighted_l2_identity():
    column_map = ColumnMap(["a", "b0", "b1"], ["a", "b", "b"], ["a", "b"])
    matrix = np.array([[0.2, 1.0, 0.0], [0.8, 0.0, 1.0], [0.1, 1.0, 0.0]])
    weights = np.array([0.4, 0.6])
    assert l2_matches_weighted(matrix, weights, column_map)
