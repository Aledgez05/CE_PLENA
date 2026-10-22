"""Tablas y las cuatro figuras del protocolo."""

import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    rand_score,
    recall_score,
    roc_auc_score,
)

from plena.baselines import majority_prediction, predict_logistic, predict_mlp
from plena.cluster import assign, safe_silhouette, weigh
from plena.settings import COLORS, METHODS, OUT, SEEDS


def _counts(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    return tn, fp, fn, tp


def classify(y, pred, score):
    tn, fp, fn, tp = _counts(y, pred)
    return {
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, score),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def predict(model, x_train, y_train, x_test):
    if model["kind"] == "cluster":
        clusters = assign(model["model"], x_test, model["weights"])
        return model["mapping"][clusters], model["probs"][clusters], clusters
    if model["kind"] == "logistic":
        score = predict_logistic(x_test, model["beta"])
        return (score >= model["threshold"]).astype(int), score, None
    if model["kind"] == "mlp":
        score = predict_mlp(model["model"], x_test)
        return (score >= model["threshold"]).astype(int), score, None
    pred, score = majority_prediction(y_train, len(x_test))
    return pred, score, None


def evaluate(models, x_train, y_train, x_test, y_test):
    rows, predictions = [], {}
    fields = ["adjusted_rand", "balanced_accuracy", "f1", "roc_auc", "rand_index", "silhouette_own", "silhouette_common"]
    for model in models:
        pred, score, clusters = predict(model, x_train, y_train, x_test)
        partition = clusters if clusters is not None else pred
        row = {
            "method": model["method"],
            "seed": model["seed"],
            "rand_index": rand_score(y_test, partition),
            "adjusted_rand": adjusted_rand_score(y_test, partition),
            "silhouette_common": safe_silhouette(x_test, partition),
            "silhouette_own": safe_silhouette(weigh(x_test, model["weights"]), partition) if model["kind"] == "cluster" else np.nan,
            **classify(y_test, pred, score),
        }
        rows.append(row)
        predictions[(model["method"], model["seed"])] = pred
    summary = {}
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        summary[method] = {}
        for field in fields:
            values = np.array([row[field] for row in subset], float)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                summary[method][field] = (np.nan, np.nan)
            elif len(values) == 1:
                summary[method][field] = (float(values[0]), 0.0)
            else:
                summary[method][field] = (float(values.mean()), float(values.std(ddof=1)))
    return rows, summary, predictions


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_figures(summary, histories, models, names, y_test, predictions):
    folder = OUT / "figures"
    folder.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    panels = zip(
        axes,
        ["adjusted_rand", "balanced_accuracy", "f1"],
        ["Rand ajustado", "Balanced accuracy", "F1 de la clase deprimida"],
    )
    for ax, field, title in panels:
        means = [summary[m][field][0] for m in METHODS]
        errors = [summary[m][field][1] for m in METHODS]
        ax.barh(np.arange(len(METHODS)), means, xerr=errors, color=COLORS, capsize=3)
        ax.set_yticks(np.arange(len(METHODS)), METHODS if ax is axes[0] else [""] * len(METHODS))
        ax.set_title(title)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Prueba. La barra de error es la desviación entre las tres semillas")
    fig.tight_layout()
    fig.savefig(folder / "comparacion_prueba.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for (method, seed), history in histories.items():
        axes[0 if "Silhouette" in method else 1].plot(history[:, 0], -history[:, 1], label=f"semilla {seed}")
    for ax, title in zip(axes, ["Mejor Silhouette de validación", "Mejor Rand de validación"]):
        ax.set_title(title)
        ax.set_xlabel("Generación")
        ax.set_ylabel("Métrica (mayor es mejor)")
        ax.legend()
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(folder / "convergencia_ga.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    seed_colors = {17: "#4477aa", 42: "#cc6677", 91: "#228833"}
    for ax, method, cutoff in zip(axes, ["FAISS + GA Silhouette", "FAISS + GA Rand"], (0.05, 0.10)):
        chosen = [m for m in models if m["method"] == method]
        shares = np.vstack([m["weights"] ** 2 / np.sum(m["weights"] ** 2) for m in chosen])
        keep = np.flatnonzero(shares.max(0) >= cutoff)
        order = keep[np.argsort(shares[:, keep].max(0))]
        y = np.arange(len(order))
        for offset, model, row in zip((-0.25, 0.0, 0.25), chosen, shares):
            ax.barh(y + offset, row[order], height=0.25, color=seed_colors[model["seed"]], label=f"semilla {model['seed']}")
        ax.set_yticks(y, np.array(names)[order])
        ax.set_xlim(0, 1)
        ax.set_xlabel("w² / suma(w²)")
        ax.set_title(method)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Peso relativo por semilla. Cada color es una corrida")
    fig.tight_layout()
    fig.savefig(folder / "pesos.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, method in zip(axes.ravel(), METHODS):
        pred = predictions[(method, SEEDS[0])]
        matrix = np.array([[np.sum((y_test == i) & (pred == j)) for j in (0, 1)] for i in (0, 1)])
        ax.imshow(matrix, cmap="Blues")
        for i, j in np.ndindex(2, 2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Pred. 0", "Pred. 1"])
        ax.set_yticks([0, 1], ["Real 0", "Real 1"])
        ax.set_title(method, fontsize=10)
    fig.suptitle(f"Matrices de confusión en prueba, semilla {SEEDS[0]}")
    fig.tight_layout()
    fig.savefig(folder / "matrices_confusion.png", dpi=140)
    plt.close(fig)


def write_tables(rows, summary, models, names):
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "metricas_prueba.csv", rows)
    write_csv(OUT / "matrices_confusion.csv", [
        {"method": r["method"], "seed": r["seed"], "tn": r["tn"], "fp": r["fp"], "fn": r["fn"], "tp": r["tp"]} for r in rows
    ])
    write_csv(OUT / "resumen.csv", [
        {"method": method, **{f"{field}_{stat}": summary[method][field][i] for field in summary[method] for i, stat in enumerate(("mean", "std"))}}
        for method in METHODS
    ])
    weight_rows = []
    for model in models:
        if model["kind"] != "cluster":
            continue
        share = model["weights"] ** 2 / np.sum(model["weights"] ** 2)
        weight_rows.extend(
            {"method": model["method"], "seed": model["seed"], "feature": name, "weight": float(value), "distance_share": float(part)}
            for name, value, part in zip(names, model["weights"], share)
        )
    write_csv(OUT / "pesos.csv", weight_rows)
