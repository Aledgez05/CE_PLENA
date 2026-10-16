"""Índice de vecinos sobre vectores ponderados, sin normalizar a norma unitaria."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from plena.distance import ColumnMap, transform_weighted, weighted_squared_distances


def weighted_vectors(matrix: np.ndarray, weights: np.ndarray, column_map: ColumnMap) -> np.ndarray:
    """Aplica ``sqrt(w)`` por bloque. No normaliza las filas."""
    return transform_weighted(matrix, weights, column_map)


def l2_matches_weighted(
    matrix: np.ndarray,
    weights: np.ndarray,
    column_map: ColumnMap,
    atol: float = 1e-8,
) -> bool:
    """Comprueba que L2 al cuadrado sobre los vectores ponderados reproduce la distancia."""
    weighted = weighted_vectors(matrix, weights, column_map)
    left = np.sum((weighted[:, None, :] - weighted[None, :, :]) ** 2, axis=2)
    right = weighted_squared_distances(matrix, matrix, weights, column_map)
    return bool(np.allclose(left, right, atol=atol))


@dataclass
class SearchResult:
    distances: np.ndarray
    indices: np.ndarray
    backend: str


class FlatL2Index:
    """``IndexFlatL2`` de FAISS o la misma búsqueda exacta en NumPy.

    FAISS devuelve la distancia al cuadrado. El plan B de NumPy también.
    """

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors64 = np.ascontiguousarray(np.asarray(vectors, dtype=np.float64))
        self.backend = "numpy"
        self._index = None
        self.faiss_error = ""
        try:
            import faiss

            self._index = faiss.IndexFlatL2(self.vectors64.shape[1])
            self._index.add(np.ascontiguousarray(self.vectors64, dtype=np.float32))
            self.backend = "faiss"
        except Exception as exc:  # ImportError o incompatibilidad de la rueda
            self.faiss_error = str(exc)
            self._index = None
            self.backend = "numpy"

    def search(self, queries: np.ndarray, k: int) -> SearchResult:
        queries64 = np.ascontiguousarray(np.asarray(queries, dtype=np.float64))
        k = min(k, len(self.vectors64))
        if self.backend == "faiss":
            distances, indices = self._index.search(
                np.ascontiguousarray(queries64, dtype=np.float32), k
            )
            return SearchResult(distances.astype(np.float64), indices.astype(int), "faiss")
        return self.search_numpy(queries64, k)

    def search_numpy(self, queries: np.ndarray, k: int, dtype: type = np.float64) -> SearchResult:
        db = np.asarray(self.vectors64, dtype=dtype)
        queries = np.asarray(queries, dtype=dtype)
        k = min(k, len(db))
        squared_norm_db = np.sum(db**2, axis=1)
        squared_norm_q = np.sum(queries**2, axis=1)
        cross = queries @ db.T
        distances = squared_norm_q[:, None] + squared_norm_db[None, :] - 2.0 * cross
        distances = np.maximum(distances, 0.0)
        indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        row = np.arange(len(queries))[:, None]
        order = np.argsort(distances[row, indices], axis=1, kind="mergesort")
        indices = indices[row, order]
        chosen = distances[row, indices]
        return SearchResult(np.asarray(chosen, dtype=np.float64), indices.astype(int), "numpy")


def neighbors_agree(reference: np.ndarray, candidate: np.ndarray, distances: np.ndarray, atol: float = 1e-5) -> bool:
    """Coincidencia de vecinos. Si hay empate de distancia, basta el conjunto."""
    if reference.shape != candidate.shape:
        return False
    for row in range(len(reference)):
        if np.array_equal(reference[row], candidate[row]):
            continue
        ref_dist = distances[row]
        if np.max(ref_dist) - np.min(ref_dist) > atol:
            # Sin empate global: se comparan conjuntos solo si las distancias coinciden.
            if set(reference[row]) != set(candidate[row]):
                return False
        elif set(reference[row]) != set(candidate[row]):
            return False
    return True


def benchmark_latency(
    index: FlatL2Index,
    queries: np.ndarray,
    k: int,
    repeats: int = 5,
) -> dict[str, float]:
    """Media y percentil 95 por consulta, FAISS contra fuerza bruta."""
    queries = np.asarray(queries, dtype=np.float64)
    if len(queries) == 0:
        raise ValueError("No hay consultas para el benchmark.")

    def _time(searcher) -> list[float]:
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            searcher(queries, k)
            elapsed = time.perf_counter() - start
            samples.append(elapsed / len(queries))
        return samples

    faiss_times = _time(index.search) if index.backend == "faiss" else []
    numpy_times = _time(index.search_numpy)
    memory_bytes = int(index.vectors64.nbytes)
    report = {
        "backend": index.backend,
        "memory_bytes": memory_bytes,
        "numpy_mean_s": float(np.mean(numpy_times)),
        "numpy_p95_s": float(np.percentile(numpy_times, 95)),
        "n_queries": int(len(queries)),
        "k": int(k),
    }
    if faiss_times:
        report["faiss_mean_s"] = float(np.mean(faiss_times))
        report["faiss_p95_s"] = float(np.percentile(faiss_times, 95))
    else:
        report["faiss_mean_s"] = float("nan")
        report["faiss_p95_s"] = float("nan")
        report["faiss_error"] = index.faiss_error
    return report
