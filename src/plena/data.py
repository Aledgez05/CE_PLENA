"""Carga, partición 60/20/20 y escalado ajustado solo en train."""

import hashlib

import numpy as np
import pandas as pd

from plena.settings import CSV_SHA256, DATA, SEED_SPLIT


def load():
    digest = hashlib.sha256(DATA.read_bytes()).hexdigest()
    if digest != CSV_SHA256:
        raise SystemExit(f"El CSV no es el del cuaderno: {digest}")
    frame = pd.read_csv(DATA)
    y = frame["depressed"].to_numpy(np.int64)
    names = [c for c in frame.columns if c not in ("depressed", "Survey_id", "Ville_id")]
    return frame[names].to_numpy(np.float64), y, names


def split(y):
    """Misma partición para todas las semillas. Estratificada por la etiqueta."""
    rng = np.random.default_rng(SEED_SPLIT)
    parts = []
    for label in np.unique(y):
        ids = rng.permutation(np.flatnonzero(y == label))
        n_train, n_val = int(0.60 * len(ids)), int(0.20 * len(ids))
        parts.append((ids[:n_train], ids[n_train : n_train + n_val], ids[n_train + n_val :]))
    train, val, test = (np.concatenate(group) for group in zip(*parts))
    return tuple(rng.permutation(idx) for idx in (train, val, test))


def scale(x_train, names):
    """Mediana, indicador de faltante y estandarizado."""
    median = np.nanmedian(x_train, axis=0)
    missing = np.flatnonzero(np.isnan(x_train).any(axis=0))
    used = list(names) + [f"{names[j]}__faltante" for j in missing]

    def transform(x):
        flags = np.isnan(x[:, missing]).astype(float)
        filled = np.where(np.isnan(x), median, x)
        return np.column_stack([filled, flags])

    train = transform(x_train)
    mean, std = train.mean(0), train.std(0)
    std = np.where(std < 1e-8, 1.0, std)

    def apply(x):
        return np.ascontiguousarray((transform(x) - mean) / std, np.float32)

    return apply, used
