"""Métricas internas, externas y de clasificación por mapeo de clusters."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    normalized_mutual_info_score,
    rand_score,
    roc_auc_score,
    silhouette_score,
)
from scipy.stats import chi2_contingency


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def internal_metrics(matrix: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Métricas internas. No usan la etiqueta ``depressed``."""
    labels = np.asarray(labels)
    matrix = np.asarray(matrix, dtype=float)
    n_clusters = int(len(np.unique(labels)))
    result = {
        "silhouette": float("nan"),
        "davies_bouldin": float("nan"),
        "calinski_harabasz": float("nan"),
        "n_clusters_observed": n_clusters,
    }
    if n_clusters < 2 or n_clusters >= len(labels):
        return result
    result["silhouette"] = _safe_float(silhouette_score(matrix, labels, metric="euclidean"))
    result["davies_bouldin"] = _safe_float(davies_bouldin_score(matrix, labels))
    result["calinski_harabasz"] = _safe_float(calinski_harabasz_score(matrix, labels))
    return result


def cluster_prevalence(y_true: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Proporción de la clase positiva por ``cluster_i``. No interpreta la clase."""
    y_true = np.asarray(y_true)
    labels = np.asarray(labels)
    positive = _positive_value(y_true)
    output = {}
    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        output[f"cluster_{int(cluster)}"] = float(np.mean(y_true[mask] == positive)) if mask.any() else float("nan")
    return output


def _positive_value(y_true: np.ndarray):
    values = pd.unique(y_true)
    if set(map(_plain, values)) <= {0, 1, 0.0, 1.0}:
        return 1
    # Sin semántica confirmada: la segunda clase ordenada se usa solo como marcador binario.
    ordered = sorted(values, key=lambda item: str(item))
    return ordered[-1]


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def is_binary(y_true: np.ndarray) -> bool:
    return len(pd.unique(y_true)) == 2


def external_metrics(y_true: np.ndarray, labels: np.ndarray) -> dict:
    """RI, ARI, AMI, NMI, prevalencia, chi-cuadrado y AUC de pertenencia."""
    y_true = np.asarray(y_true)
    labels = np.asarray(labels)
    result = {
        "ri": _safe_float(rand_score(y_true, labels)),
        "ari": _safe_float(adjusted_rand_score(y_true, labels)),
        "ami": _safe_float(adjusted_mutual_info_score(y_true, labels)),
        "nmi": _safe_float(normalized_mutual_info_score(y_true, labels)),
        "prevalence_by_cluster": cluster_prevalence(y_true, labels),
        "chi2": float("nan"),
        "chi2_p": float("nan"),
        "auc_membership": float("nan"),
    }
    table = pd.crosstab(labels, y_true)
    if table.shape[0] > 1 and table.shape[1] > 1:
        chi2, p_value, _dof, _expected = chi2_contingency(table)
        result["chi2"] = float(chi2)
        result["chi2_p"] = float(p_value)
    if is_binary(y_true):
        result["auc_membership"] = _auc_from_rates(y_true, labels, y_true, labels)
    return result


def _auc_from_rates(y_train, labels_train, y_eval, labels_eval) -> float:
    positive = _positive_value(y_train)
    rates = {}
    global_rate = float(np.mean(np.asarray(y_train) == positive))
    for cluster in np.unique(labels_train):
        mask = labels_train == cluster
        rates[int(cluster)] = float(np.mean(y_train[mask] == positive)) if mask.any() else global_rate
    scores = np.array([rates.get(int(cluster), global_rate) for cluster in labels_eval], dtype=float)
    y_binary = (np.asarray(y_eval) == positive).astype(int)
    if len(np.unique(y_binary)) < 2 or len(np.unique(scores)) < 2:
        return float("nan")
    return float(roc_auc_score(y_binary, scores))


def majority_mapping(y_train: np.ndarray, labels_train: np.ndarray) -> dict[int, object]:
    """Etiqueta mayoritaria de train para cada cluster. Los empates usan el valor menor en texto."""
    mapping = {}
    y_train = np.asarray(y_train)
    labels_train = np.asarray(labels_train)
    for cluster in np.unique(labels_train):
        subset = y_train[labels_train == cluster]
        values, counts = np.unique(subset, return_counts=True)
        best = counts.max()
        winners = values[counts == best]
        winners = sorted(winners, key=lambda item: str(item))
        mapping[int(cluster)] = winners[0]
    return mapping


def classification_from_clusters(
    y_train: np.ndarray,
    labels_train: np.ndarray,
    y_eval: np.ndarray,
    labels_eval: np.ndarray,
) -> dict[str, float]:
    """Balanced accuracy, F1 y ROC-AUC a partir del mapeo aprendido en train."""
    mapping = majority_mapping(y_train, labels_train)
    fallback = _positive_value(y_train)
    # El fallback solo cubre un cluster que no apareció en train.
    predicted = np.array([mapping.get(int(cluster), fallback) for cluster in labels_eval])
    result = {
        "balanced_accuracy": float("nan"),
        "f1_macro": float("nan"),
        "roc_auc": float("nan"),
    }
    if len(np.unique(y_eval)) < 2:
        return result
    result["balanced_accuracy"] = _safe_float(balanced_accuracy_score(y_eval, predicted))
    result["f1_macro"] = _safe_float(f1_score(y_eval, predicted, average="macro"))
    if is_binary(y_eval):
        result["roc_auc"] = _auc_from_rates(y_train, labels_train, y_eval, labels_eval)
    return result


def stability_ari(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    return float(adjusted_rand_score(labels_a, labels_b))
