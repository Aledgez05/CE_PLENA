"""Distancia euclidiana ponderada por bloques y distancia de Gower."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ColumnMap:
    """Relación entre columnas transformadas y la variable original."""

    transformed_names: list[str]
    variable_of: list[str]
    variable_names: list[str]

    def __post_init__(self) -> None:
        if len(self.transformed_names) != len(self.variable_of):
            raise ValueError("El mapeo de columnas está incompleto.")
        missing = set(self.variable_of) - set(self.variable_names)
        if missing:
            raise ValueError(f"Hay columnas sin variable original: {sorted(missing)}")

    @property
    def block_sizes(self) -> dict[str, int]:
        sizes = {name: 0 for name in self.variable_names}
        for variable in self.variable_of:
            sizes[variable] += 1
        return sizes

    def index_of(self) -> dict[str, int]:
        return {name: pos for pos, name in enumerate(self.variable_names)}


def transform_weighted(
    matrix: np.ndarray,
    weights: np.ndarray,
    column_map: ColumnMap,
) -> np.ndarray:
    """Escala cada bloque de la variable j por ``sqrt(w_j / n_cols_j)``.

    Con esa escala, la distancia euclidiana al cuadrado equivale a
    ``Σ w_j`` por la distancia al cuadrado del bloque, promediada si el
    bloque tiene varias columnas (one-hot).
    """
    matrix = np.asarray(matrix, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("La matriz debe ser bidimensional.")
    if len(weights) != len(column_map.variable_names):
        raise ValueError("Los pesos no coinciden con las variables originales.")
    if matrix.shape[1] != len(column_map.variable_of):
        raise ValueError("La matriz no coincide con el mapeo de columnas.")
    if not np.isfinite(matrix).all() or not np.isfinite(weights).all():
        raise ValueError("Hay valores no finitos en la matriz o en los pesos.")
    index = column_map.index_of()
    sizes = column_map.block_sizes
    scale = np.empty(matrix.shape[1], dtype=float)
    for col, variable in enumerate(column_map.variable_of):
        weight = float(weights[index[variable]])
        scale[col] = np.sqrt(max(weight, 0.0) / sizes[variable])
    return matrix * scale


def weighted_squared_distances(
    left: np.ndarray,
    right: np.ndarray,
    weights: np.ndarray,
    column_map: ColumnMap,
) -> np.ndarray:
    """Matriz de distancias euclidianas al cuadrado ya ponderadas."""
    left_w = transform_weighted(left, weights, column_map)
    right_w = transform_weighted(right, weights, column_map)
    diff = left_w[:, None, :] - right_w[None, :, :]
    return np.sum(diff**2, axis=2)


def gower_component_matrices(
    frame: pd.DataFrame,
    types: dict[str, str],
) -> dict[str, np.ndarray]:
    """Matrices de disimilitud por variable para la distancia de Gower.

    Continuas y discretas usan el rango observado. Nominales y binarias usan
    coincidencia. Esta ruta no se indexa con FAISS.
    """
    components: dict[str, np.ndarray] = {}
    for column, ftype in types.items():
        series = frame[column]
        values = series.to_numpy()
        if ftype in {"continuous", "discrete", "ordinal"}:
            numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            span = np.nanmax(numeric) - np.nanmin(numeric)
            if not np.isfinite(span) or span <= 0:
                span = 1.0
            distance = np.abs(numeric[:, None] - numeric[None, :]) / span
            distance = np.nan_to_num(distance, nan=1.0)
        elif ftype in {"binary", "nominal"}:
            distance = (values[:, None] != values[None, :]).astype(float)
        else:
            raise ValueError(f"Gower no cubre el tipo {ftype} de {column}.")
        components[column] = distance
    return components


def gower_distance(components: dict[str, np.ndarray], weights: np.ndarray, names: list[str]) -> np.ndarray:
    """``D(w) = Σ w_j D_j / Σ w_j``."""
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(names):
        raise ValueError("Los pesos no coinciden con las variables de Gower.")
    total = None
    for weight, name in zip(weights, names):
        term = weight * components[name]
        total = term if total is None else total + term
    denom = float(weights.sum())
    if denom <= 0:
        raise ValueError("La suma de pesos de Gower es cero.")
    return total / denom
