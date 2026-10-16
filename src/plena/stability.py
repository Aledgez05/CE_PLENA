"""Estabilidad de pesos, ablaciones, perfiles y subgrupos."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from plena.chromosome import n_eff
from plena.evaluation import stability_ari


def jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    left = np.asarray(mask_a, dtype=bool)
    right = np.asarray(mask_b, dtype=bool)
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(left, right).sum() / union)


def weight_stability(weight_rows: list[np.ndarray], masks: list[np.ndarray]) -> dict[str, float]:
    """Spearman medio entre vectores de peso y Jaccard medio entre máscaras."""
    spearman_values = []
    jaccard_values = []
    for i in range(len(weight_rows)):
        for j in range(i + 1, len(weight_rows)):
            coef = spearmanr(weight_rows[i], weight_rows[j]).statistic
            spearman_values.append(float(coef) if np.isfinite(coef) else float("nan"))
            jaccard_values.append(jaccard(masks[i], masks[j]))
    return {
        "spearman_mean": float(np.nanmean(spearman_values)) if spearman_values else float("nan"),
        "jaccard_mean": float(np.nanmean(jaccard_values)) if jaccard_values else float("nan"),
        "n_pairs": len(spearman_values),
    }


def assignment_stability(label_rows: list[np.ndarray]) -> float:
    values = []
    for i in range(len(label_rows)):
        for j in range(i + 1, len(label_rows)):
            values.append(stability_ari(label_rows[i], label_rows[j]))
    return float(np.mean(values)) if values else float("nan")


def ablate_weights(weights: np.ndarray, names: list[str], drop: list[str]) -> np.ndarray:
    """Pone a cero las variables indicadas y renormaliza el resto."""
    updated = np.array(weights, dtype=float, copy=True)
    index = {name: pos for pos, name in enumerate(names)}
    for name in drop:
        if name in index:
            updated[index[name]] = 0.0
    total = float(updated.sum())
    if total <= 0:
        raise ValueError("La ablación dejó todos los pesos en cero.")
    return updated / total


def cluster_profiles(
    frame: pd.DataFrame,
    labels: np.ndarray,
    feature_types: dict[str, str],
    y: np.ndarray | None = None,
) -> pd.DataFrame:
    """Medianas o proporciones por cluster. No se infiere causalidad."""
    labels = np.asarray(labels)
    rows = []
    for cluster in sorted(np.unique(labels)):
        subset = frame.loc[labels == cluster]
        row: dict[str, object] = {
            "cluster": f"cluster_{int(cluster)}",
            "size": int(len(subset)),
        }
        if y is not None:
            row["label_prevalence"] = float(np.mean(np.asarray(y)[labels == cluster] == 1))
        for name, ftype in feature_types.items():
            if name not in subset.columns:
                continue
            if ftype in {"continuous", "discrete", "ordinal"}:
                row[f"median_{name}"] = float(pd.to_numeric(subset[name], errors="coerce").median())
            elif ftype == "binary":
                numeric = pd.to_numeric(subset[name], errors="coerce")
                row[f"rate_{name}"] = float(numeric.mean())
            elif ftype == "nominal":
                mode = subset[name].mode(dropna=True)
                row[f"mode_{name}"] = "" if mode.empty else str(mode.iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def subgroup_prevalence(
    frame: pd.DataFrame,
    y: np.ndarray,
    columns: list[str],
    max_levels: int = 12,
) -> dict[str, list[dict]]:
    """Tamaño y prevalencia de la etiqueta por subgrupo. Solo describe."""
    y = np.asarray(y)
    report = {}
    for column in columns:
        if column not in frame.columns:
            continue
        series = frame[column]
        entries = []
        for value, positions in series.groupby(series).groups.items():
            idx = np.asarray(list(positions))
            if len(entries) >= max_levels:
                break
            entries.append(
                {
                    "value": str(value),
                    "n": int(len(idx)),
                    "label_prevalence": float(np.mean(y[idx] == 1)),
                }
            )
        report[column] = entries
    return report


def effective_variables(weights: np.ndarray) -> float:
    return n_eff(weights)
