"""Baselines: pesos iguales, búsqueda aleatoria y referencia por permutación."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score

from plena.chromosome import equal_weight_vector
from plena.fitness import FitnessBreakdown, FitnessEvaluator


@dataclass
class RandomSearchResult:
    best_genes: np.ndarray
    best: FitnessBreakdown
    n_evaluations: int
    seed: int


def equal_weights(n_features: int) -> np.ndarray:
    return equal_weight_vector(n_features)


def random_search(
    evaluator: FitnessEvaluator,
    n_genes: int,
    n_evaluations: int,
    rng: np.random.Generator,
    seed: int,
) -> RandomSearchResult:
    """Misma decodificación y la misma función de aptitud que el algoritmo genético."""
    if n_evaluations < 1:
        raise ValueError("La búsqueda aleatoria necesita al menos una evaluación.")
    best_genes = None
    best_detail = None
    for _ in range(n_evaluations):
        genes = rng.uniform(-1.0, 1.0, size=n_genes)
        detail = evaluator(genes)
        if best_detail is None or detail.score > best_detail.score:
            best_detail = detail
            best_genes = genes.copy()
    assert best_genes is not None and best_detail is not None
    return RandomSearchResult(best_genes, best_detail, n_evaluations, seed)


def permutation_reference(
    matrix: np.ndarray,
    y_true: np.ndarray,
    labels: np.ndarray,
    *,
    n_permutations: int,
    seed: int,
    k: int,
) -> dict[str, float]:
    """Referencia por azar.

    ARI: etiqueta permutada contra los clusters obtenidos.
    Silhouette: particiones aleatorias con el mismo k, en el mismo espacio.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    labels = np.asarray(labels)
    ari_values = []
    silhouette_values = []
    for _ in range(n_permutations):
        permuted = rng.permutation(y_true)
        ari_values.append(float(adjusted_rand_score(permuted, labels)))
        random_labels = rng.integers(0, k, size=len(matrix))
        if len(np.unique(random_labels)) < 2:
            continue
        silhouette_values.append(float(silhouette_score(matrix, random_labels, metric="euclidean")))
    return {
        "null_ari_mean": float(np.mean(ari_values)) if ari_values else float("nan"),
        "null_ari_std": float(np.std(ari_values)) if ari_values else float("nan"),
        "null_silhouette_mean": float(np.mean(silhouette_values)) if silhouette_values else float("nan"),
        "null_silhouette_std": float(np.std(silhouette_values)) if silhouette_values else float("nan"),
        "n_permutations": int(n_permutations),
    }
