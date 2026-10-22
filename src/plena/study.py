"""Entrena sin mirar la prueba y escribe las tablas y las figuras."""

import numpy as np
from sklearn.metrics import rand_score

from plena.baselines import best_threshold, fit_logistic, fit_mlp, predict_mlp
from plena.cluster import assign, cluster_to_class, kmeans, run_ga, safe_silhouette, weigh
from plena.data import load, scale, split
from plena.report import evaluate, save_figures, write_tables
from plena.settings import METHODS, OUT, SEEDS


def train(x_train, y_train, x_val, y_val):
    models, histories = [], {}
    logistic = fit_logistic(x_train, y_train, x_val, y_val)
    print(f"Logística: L2={logistic['l2']}  umbral={logistic['threshold']:.4f}  BA validación={logistic['val_ba']:.4f}")
    for seed in SEEDS:
        print(f"Semilla {seed}")
        for name, objective in (
            ("FAISS base", None),
            ("FAISS + GA Silhouette", "silhouette"),
            ("FAISS + GA Rand", "rand"),
        ):
            if objective is None:
                weights = np.ones(x_train.shape[1])
            else:
                weights, history = run_ga(x_train, x_val, y_val, objective, seed)
                histories[(name, seed)] = history
            model = kmeans(x_train, weights, seed)
            mapping, probs = cluster_to_class(assign(model, x_train, weights), y_train)
            val_c = assign(model, x_val, weights)
            print(
                f"  {name}: Rand val {rand_score(y_val, val_c):.4f} | "
                f"Silhouette val {safe_silhouette(weigh(x_val, weights), val_c):.4f}"
            )
            models.append({
                "method": name, "seed": seed, "kind": "cluster",
                "weights": weights, "model": model, "mapping": mapping, "probs": probs,
            })
        net = fit_mlp(x_train, y_train, x_val, y_val, seed)
        threshold, balanced = best_threshold(y_val, predict_mlp(net, x_val))
        print(f"  Red neuronal: BA validación {balanced:.4f}  umbral {threshold:.4f}")
        models.append({"method": "Red neuronal ligera", "seed": seed, "kind": "mlp", "model": net, "threshold": threshold})
        models.append({"method": "Regresión logística", "seed": seed, "kind": "logistic", **logistic})
        models.append({"method": "Clase mayoritaria", "seed": seed, "kind": "majority"})
    return models, histories


def main():
    x, y, names = load()
    print(f"Filas: {len(y)} | Positivos: {int(y.sum())} ({y.mean():.1%})")
    train_idx, val_idx, test_idx = split(y)
    apply, used = scale(x[train_idx], names)
    x_train, x_val, x_test = apply(x[train_idx]), apply(x[val_idx]), apply(x[test_idx])
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    for label, part in ("train", y_train), ("validación", y_val), ("prueba", y_test):
        print(f"{label:12} {len(part):4} filas  clases {np.bincount(part).tolist()}")

    models, histories = train(x_train, y_train, x_val, y_val)
    rows, summary, predictions = evaluate(models, x_train, y_train, x_test, y_test)
    write_tables(rows, summary, models, used)
    save_figures(summary, histories, models, used, y_test, predictions)

    print("\nPrueba, media ± desviación")
    for method in METHODS:
        ari, ba, f1, auc = (summary[method][k] for k in ("adjusted_rand", "balanced_accuracy", "f1", "roc_auc"))
        print(f"{method:28} ARI {ari[0]:7.3f}  BA {ba[0]:7.3f}  F1 {f1[0]:7.3f}  AUC {auc[0]:7.3f}")
    print(f"Archivos en {OUT}")
