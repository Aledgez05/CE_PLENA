"""Penalización de particiones que enumeran patrones idénticos.

Una solución degenerada pone cada fila distinta de la representación ponderada
en su propio cluster. La distancia interna es cero y el Silhouette euclidiano
llega a 1. Esta medida no usa la etiqueta.
"""

from __future__ import annotations

import numpy as np


def n_unique_rows(matrix: np.ndarray, decimals: int = 6) -> int:
    """Cuenta filas distintas tras redondear.

    El redondeo absorbe polvo de punto flotante. No fusiona patrones cuya
    diferencia es del orden de una variable con peso real: un salto de una
    binaria estandarizada, multiplicado por ``sqrt(w)``, queda muy por encima
    de ``1e-6``.
    """
    rounded = np.round(np.asarray(matrix, dtype=float), decimals)
    if len(rounded) == 0:
        return 0
    return int(np.unique(rounded, axis=0).shape[0])


def collapse_report(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    tau: float = 0.15,
    decimals: int = 6,
) -> dict[str, float]:
    """Qué tan cerca está la partición de separar cada patrón en un cluster.

    Para cada punto, ``a`` es la distancia a su centroide y ``b`` la distancia
    de ese centroide al centroide más cercano. El término

        ``exp(-a / (tau * b))``

    vale 1 si el cluster tiene radio cero y cae en cuanto el radio es una
    fracción apreciable del hueco entre clusters. ``tau = 0.15`` hace que un
    radio del 15 % del hueco deje el término en ``exp(-1) ≈ 0.37``, y un radio
    comparable al hueco lo deje cerca de cero. Un ruido de ``1e-8`` no cambia
    el resultado: seguiría haciendo falta dispersión real dentro del cluster.

    ``enumerated_fraction`` es la fracción de filas que caen en un cluster de
    un solo patrón, contado con el mismo redondeo que ``n_unique``.
    """
    matrix = np.asarray(matrix, dtype=float)
    labels = np.asarray(labels)
    n_rows = len(matrix)
    empty = {
        "n_unique": 0.0,
        "enumerated_fraction": 0.0,
        "near_collapse": 0.0,
        "mean_intra": 0.0,
        "min_gap": 0.0,
    }
    if n_rows == 0:
        return empty
    if tau <= 0:
        raise ValueError("tau de la penalización de colapso debe ser positivo.")
    n_unique = n_unique_rows(matrix, decimals)
    classes = np.unique(labels)
    centroids = np.vstack([matrix[labels == lab].mean(axis=0) for lab in classes])
    if len(classes) == 1:
        gap = np.array([np.inf])
    else:
        delta = centroids[:, None, :] - centroids[None, :, :]
        distances = np.sqrt(np.sum(delta**2, axis=2))
        np.fill_diagonal(distances, np.inf)
        gap = distances.min(axis=1)
    near_total = 0.0
    intra_total = 0.0
    enumerated = 0
    finite_gaps = [float(value) for value in gap if np.isfinite(value)]
    for index, lab in enumerate(classes):
        block = matrix[labels == lab]
        intra = np.sqrt(np.sum((block - centroids[index]) ** 2, axis=1))
        scale = tau * max(float(gap[index]), 1e-12)
        near_total += float(np.exp(-intra / scale).sum())
        intra_total += float(intra.sum())
        if n_unique_rows(block, decimals) <= 1:
            enumerated += int(len(block))
    return {
        "n_unique": float(n_unique),
        "enumerated_fraction": float(enumerated / n_rows),
        "near_collapse": float(near_total / n_rows),
        "mean_intra": float(intra_total / n_rows),
        "min_gap": float(min(finite_gaps)) if finite_gaps else 0.0,
    }
