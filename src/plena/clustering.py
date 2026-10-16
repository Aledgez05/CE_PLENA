"""K-Means, asignación por centroide y KNN sobre vecinos de train."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


@dataclass
class ClusterModel:
    k: int
    centroids: np.ndarray
    labels: np.ndarray
    random_state: int
    n_init: int
    inertia: float


def validate_k(k: int, n_samples: int) -> None:
    if k < 2 or k >= n_samples:
        raise ValueError(f"k={k} no es válido para {n_samples} muestras. Se rechazan k=1 y k=n.")


def fit_kmeans(
    matrix: np.ndarray,
    k: int,
    random_state: int,
    n_init: int = 10,
) -> ClusterModel:
    matrix = np.asarray(matrix, dtype=float)
    validate_k(k, len(matrix))
    model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    model.fit(matrix)
    labels = predict_centroids(model.cluster_centers_, matrix)
    return ClusterModel(
        k=k,
        centroids=np.asarray(model.cluster_centers_, dtype=float),
        labels=labels,
        random_state=random_state,
        n_init=n_init,
        inertia=float(model.inertia_),
    )


def predict_centroids(centroids: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Asigna cada fila al centroide más cercano en distancia euclidiana al cuadrado."""
    centroids = np.asarray(centroids, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    diff = matrix[:, None, :] - centroids[None, :, :]
    distances = np.sum(diff**2, axis=2)
    return np.argmin(distances, axis=1).astype(int)


def predict_knn(
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    query_matrix: np.ndarray,
    n_neighbors: int,
) -> np.ndarray:
    """Asigna el cluster por mayoría de los vecinos de train.

    Los empates se rompen con el identificador de cluster más bajo.
    """
    train_matrix = np.asarray(train_matrix, dtype=float)
    query_matrix = np.asarray(query_matrix, dtype=float)
    train_labels = np.asarray(train_labels)
    n_neighbors = min(n_neighbors, len(train_matrix))
    if n_neighbors < 1:
        raise ValueError("k de vecinos debe ser al menos 1.")
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    neighbors.fit(train_matrix)
    indices = neighbors.kneighbors(query_matrix, return_distance=False)
    predicted = []
    for row in indices:
        votes = train_labels[row]
        values, counts = np.unique(votes, return_counts=True)
        best = counts.max()
        winners = np.sort(values[counts == best])
        predicted.append(int(winners[0]))
    return np.asarray(predicted, dtype=int)


def kmedoids_alternate(
    distance: np.ndarray,
    k: int,
    random_state: int,
    n_init: int = 5,
    max_iter: int = 30,
) -> np.ndarray:
    """K-medoids por alternancia de asignación y medoide.

    Es el plan B si la librería ``kmedoids`` no está disponible. No usa FAISS.
    """
    distance = np.asarray(distance, dtype=float)
    n = distance.shape[0]
    validate_k(k, n)
    rng = np.random.default_rng(random_state)
    best_labels = None
    best_cost = np.inf
    for init in range(n_init):
        medoids = list(rng.choice(n, size=k, replace=False))
        labels = np.argmin(distance[:, medoids], axis=1)
        for _ in range(max_iter):
            updated = []
            for cluster in range(k):
                members = np.flatnonzero(labels == cluster)
                if len(members) == 0:
                    updated.append(medoids[cluster])
                    continue
                costs = distance[np.ix_(members, members)].sum(axis=1)
                updated.append(int(members[int(np.argmin(costs))]))
            # Completa medooides repetidos para conservar k.
            unique = list(dict.fromkeys(updated))
            if len(unique) < k:
                missing = [i for i in range(n) if i not in unique]
                unique.extend(missing[: k - len(unique)])
            medoids = unique[:k]
            new_labels = np.argmin(distance[:, medoids], axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
        cost = float(distance[np.arange(n), np.array(medoids)[labels]].sum())
        if cost < best_cost:
            best_cost = cost
            best_labels = labels.copy()
        _ = init
    return best_labels
