"""Regresión logística y árbol, como referencia supervisada aparte del agrupamiento."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier


def _scores(y_true, predicted, probability) -> dict[str, float]:
    y_true = np.asarray(y_true)
    result = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "f1_macro": float(f1_score(y_true, predicted, average="macro")),
        "roc_auc": float("nan"),
    }
    if probability is not None and len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, probability))
    return result


def supervised_reference(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Modelos supervisados sobre la misma matriz. No se comparan con Silhouette.

    La etiqueta puede ser un indicador de encuesta, no un diagnóstico.
    """
    y_train = np.asarray(y_train).astype(int)
    y_eval = np.asarray(y_eval).astype(int)
    logistic = LogisticRegression(max_iter=500, random_state=seed)
    logistic.fit(x_train, y_train)
    tree = DecisionTreeClassifier(max_depth=4, random_state=seed)
    tree.fit(x_train, y_train)
    return {
        "logistic_regression": _scores(
            y_eval, logistic.predict(x_eval), logistic.predict_proba(x_eval)[:, 1]
        ),
        "decision_tree": _scores(y_eval, tree.predict(x_eval), tree.predict_proba(x_eval)[:, 1]),
    }
