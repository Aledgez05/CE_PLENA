"""Decodificación del cromosoma real en [-1, 1]."""

from __future__ import annotations

import numpy as np


def n_eff(weights: np.ndarray) -> float:
    """Número efectivo de variables: ``1 / Σ w_j²`` sobre el vector completo."""
    weights = np.asarray(weights, dtype=float)
    denom = float(np.sum(weights**2))
    if denom <= 0:
        return 0.0
    return float(1.0 / denom)


def project_upper_simplex(raw: np.ndarray, w_max: float) -> np.ndarray:
    """Proyecta pesos no negativos al símplex con cota superior ``w_max``.

    El vector de entrada puede contener ceros (variables inactivas). Esos
    ceros se conservan. Hace falta que ``n_activos * w_max >= 1``.
    """
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 1:
        raise ValueError("Los pesos deben ser un vector.")
    active = np.flatnonzero(raw > 0)
    out = np.zeros(raw.shape, dtype=float)
    if len(active) == 0:
        raise ValueError("No hay pesos positivos que proyectar.")
    if len(active) * w_max < 1 - 1e-9:
        raise ValueError("El tope w_max no permite que los activos sumen 1.")
    values = raw[active].astype(float)
    values = values / values.sum()
    free = np.ones(len(values), dtype=bool)
    fixed = np.zeros(len(values), dtype=float)
    for _ in range(len(values)):
        over = free & (values > w_max + 1e-10)
        if not np.any(over):
            break
        fixed[over] = w_max
        free[over] = False
        remain = 1.0 - float(fixed.sum())
        if not np.any(free):
            break
        portion = np.maximum(values[free], 0.0)
        if portion.sum() <= 0:
            values[free] = remain / free.sum()
        else:
            values[free] = portion / portion.sum() * remain
    projected = np.where(free, values, fixed)
    projected = np.clip(projected, 0.0, w_max)
    gap = 1.0 - float(projected.sum())
    if gap > 1e-8:
        room = w_max - projected
        if room.sum() <= 0:
            raise ValueError("No queda margen para ajustar la suma de pesos.")
        projected = projected + room / room.sum() * gap
    elif gap < -1e-8:
        projected = projected / projected.sum()
    projected = np.clip(projected, 0.0, w_max)
    if abs(float(projected.sum()) - 1.0) > 1e-6 or float(projected.max()) > w_max + 1e-6:
        raise ValueError("La proyección no cumplió suma 1 y tope w_max.")
    out[active] = projected
    return out


def decode_uncapped(
    genes: np.ndarray,
    tau: float = 0.0,
    w_max: float = 0.3,
    min_active: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convierte un cromosoma en máscara, pesos recortados y pesos antes del tope.

    ``g_j <= tau`` excluye la variable. ``g_j > tau`` propone ``w_j = g_j``.
    Si hay menos de ``min_active`` activas, se activan las de mayor gen.
    Si el tope ``w_max`` impide sumar 1, se activan más variables, también por
    gen decreciente. Los pesos activos se normalizan, se recortan a ``w_max``
    y se renormalizan.
    """
    genes = np.asarray(genes, dtype=float)
    if genes.ndim != 1:
        raise ValueError("El cromosoma debe ser un vector.")
    if len(genes) == 0:
        raise ValueError("El cromosoma está vacío.")
    repaired = np.nan_to_num(genes, nan=-1.0, posinf=1.0, neginf=-1.0)
    repaired = np.clip(repaired, -1.0, 1.0)
    dimension = len(repaired)
    if min_active > dimension:
        raise ValueError("min_active es mayor que la longitud del cromosoma.")
    if w_max * dimension < 1 - 1e-9:
        raise ValueError("w_max no es compatible con la longitud del cromosoma.")
    n_needed = max(min_active, int(np.ceil(1.0 / w_max - 1e-12)))
    n_needed = min(n_needed, dimension)
    active = repaired > tau
    if int(active.sum()) < n_needed:
        order = np.argsort(-repaired, kind="mergesort")
        active = np.zeros(dimension, dtype=bool)
        active[order[:n_needed]] = True
    raw = np.zeros(dimension, dtype=float)
    raw[active] = np.maximum(repaired[active], 1e-6)
    before = np.zeros(dimension, dtype=float)
    if float(raw.sum()) > 0:
        before = raw / raw.sum()
    weights = project_upper_simplex(raw, w_max)
    mask = weights > 1e-12
    return mask, weights, before


def decode(
    genes: np.ndarray,
    tau: float = 0.0,
    w_max: float = 0.3,
    min_active: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Máscara y pesos ya recortados a ``w_max``."""
    mask, weights, _before = decode_uncapped(genes, tau=tau, w_max=w_max, min_active=min_active)
    return mask, weights


def equal_weight_vector(dimension: int) -> np.ndarray:
    """Pesos iguales sin pasar por el tope del algoritmo genético."""
    if dimension < 1:
        raise ValueError("Se necesita al menos una variable.")
    return np.ones(dimension, dtype=float) / dimension


def genes_all_active(dimension: int, value: float = 0.5) -> np.ndarray:
    """Cromosoma que, con tau 0, deja todas las variables por encima del umbral."""
    return np.full(dimension, value, dtype=float)
