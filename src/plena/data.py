"""Carga de tablas y particiones train / validación / test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from plena.names import normalize_column_name


def load_table(path: str | Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Lee un CSV y normaliza los nombres de columna.

    Devuelve el marco y el mapeo ``nombre_original → nombre_normalizado``.
    """
    frame = pd.read_csv(path, encoding="utf-8-sig")
    mapping: dict[str, str] = {}
    used: dict[str, str] = {}
    renamed = []
    for original in frame.columns:
        normalized = normalize_column_name(original)
        if not normalized:
            raise ValueError(f"La columna {original!r} quedó vacía al normalizar.")
        if normalized in used:
            raise ValueError(
                f"Colisión de nombres: {original!r} y {used[normalized]!r} pasan a {normalized!r}."
            )
        used[normalized] = str(original)
        mapping[str(original)] = normalized
        renamed.append(normalized)
    frame = frame.copy()
    frame.columns = renamed
    return frame, mapping


def drop_missing_target(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, int]:
    """Quita filas sin etiqueta. La etiqueta no se imputa."""
    if target not in frame.columns:
        raise ValueError(f"No existe la columna etiqueta {target}.")
    kept = frame.dropna(subset=[target]).copy()
    removed = len(frame) - len(kept)
    kept = kept.reset_index(drop=False).rename(columns={"index": "source_row"})
    return kept, removed


def _mode_label(values: np.ndarray) -> object:
    _unique, counts = np.unique(values, return_counts=True)
    return _unique[int(np.argmax(counts))]


def split_indices(
    y: np.ndarray,
    *,
    train: float,
    val: float,
    test: float,
    seed: int,
    groups: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Partición reproducible.

    Si hay grupos, se usa StratifiedGroupKFold con 5 pliegues: el 0 es prueba,
    el 1 es validación y el resto es train. Ningún grupo cae en dos partes.
    La etiqueta solo equilibra las partes; no entra al agrupamiento.
    """
    y = np.asarray(y)
    n = len(y)
    if n == 0:
        raise ValueError("No hay filas para particionar.")
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError("Las proporciones deben sumar 1.")
    if groups is None:
        return _split_rows(np.arange(n), y, train, val, test, seed)
    groups = np.asarray(groups)
    if len(groups) != n:
        raise ValueError("Los grupos no coinciden con las filas.")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_of_row = np.full(n, -1, dtype=int)
    for fold, (_train_idx, test_idx) in enumerate(splitter.split(np.zeros((n, 1)), y, groups)):
        fold_of_row[test_idx] = fold
    if np.any(fold_of_row < 0):
        raise ValueError("StratifiedGroupKFold dejó filas sin pliegue.")
    assigned = {
        "test": np.sort(np.flatnonzero(fold_of_row == 0)),
        "val": np.sort(np.flatnonzero(fold_of_row == 1)),
        "train": np.sort(np.flatnonzero(fold_of_row >= 2)),
    }
    _assert_disjoint(assigned)
    seen_groups: dict[object, str] = {}
    for name, idx in assigned.items():
        for group_id in np.unique(groups[idx]):
            if group_id in seen_groups:
                raise ValueError(f"El grupo {group_id} cayó en más de una partición.")
            seen_groups[group_id] = name
    return assigned


def _split_rows(
    indices: np.ndarray,
    y: np.ndarray,
    train: float,
    val: float,
    test: float,
    seed: int,
    stratify: bool = True,
) -> dict[str, np.ndarray]:
    indices = np.asarray(indices)
    y = np.asarray(y)
    strat = y if stratify else None
    try:
        train_val, test_idx = train_test_split(
            indices,
            test_size=test,
            random_state=seed,
            stratify=strat,
        )
    except ValueError:
        if not stratify:
            raise
        return _split_rows(indices, y, train, val, test, seed, stratify=False)
    relative_val = val / (train + val)
    strat_tv = y[train_val] if stratify else None
    try:
        train_idx, val_idx = train_test_split(
            train_val,
            test_size=relative_val,
            random_state=seed,
            stratify=strat_tv,
        )
    except ValueError:
        if not stratify:
            raise
        return _split_rows(indices, y, train, val, test, seed, stratify=False)
    parts = {
        "train": np.sort(np.asarray(train_idx)),
        "val": np.sort(np.asarray(val_idx)),
        "test": np.sort(np.asarray(test_idx)),
    }
    _assert_disjoint(parts)
    return parts


def _assert_disjoint(parts: dict[str, np.ndarray]) -> None:
    seen: set[int] = set()
    for name, values in parts.items():
        current = set(int(v) for v in values)
        overlap = seen & current
        if overlap:
            raise ValueError(f"La partición {name} se solapa con otra.")
        seen |= current


def prevalence(y: np.ndarray, positive=1) -> float:
    y = np.asarray(y)
    if len(y) == 0:
        return float("nan")
    return float(np.mean(y == positive))


def save_splits(path: str | Path, splits: dict[str, np.ndarray], metadata: dict) -> None:
    payload = {
        "metadata": metadata,
        "indices": {name: [int(v) for v in values] for name, values in splits.items()},
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def feature_frame(frame: pd.DataFrame, feature_names: list[str], banned: set[str]) -> pd.DataFrame:
    """Selecciona las columnas de entrada y rechaza identificadores o la etiqueta."""
    leaked = [name for name in feature_names if name in banned]
    if leaked:
        raise ValueError("Columnas prohibidas en la matriz de características: " + ", ".join(leaked))
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ValueError("Faltan características: " + ", ".join(missing))
    return frame.loc[:, feature_names].copy()


def banned_columns(target: str, id_columns: list[str], exclusions: list[str]) -> set[str]:
    return set(id_columns) | set(exclusions) | {target}
