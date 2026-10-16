"""Figuras y texto de la comparación. No declara una mejora sin la prueba pareada."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def format_mean_std(values: list[float]) -> str:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return "n/d"
    if len(array) == 1:
        return f"{array[0]:.3f}"
    return f"{array.mean():.3f} ± {array.std(ddof=1):.3f}"


def plot_convergence(histories: list[list[dict]], path: Path, title: str) -> None:
    if not histories:
        return
    length = max(len(history) for history in histories)
    best = np.full((len(histories), length), np.nan)
    for row, history in enumerate(histories):
        for item in history:
            best[row, int(item["generation"])] = item["best"]
    mean = np.nanmean(best, axis=0)
    std = np.nanstd(best, axis=0)
    generations = np.arange(length)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(generations, mean, label="mejor aptitud")
    ax.fill_between(generations, mean - std, mean + std, alpha=0.25, label="± desviación")
    ax.set_xlabel("Generación")
    ax.set_ylabel("Aptitud")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_pareto(frame: pd.DataFrame, path: Path) -> None:
    subset = frame[(frame["method"] == "ga") & (frame["split"] == "val")]
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for variant, group in subset.groupby("variant"):
        ax.scatter(group["n_active"], group["silhouette"], label=str(variant))
    ax.set_xlabel("Variables activas")
    ax.set_ylabel("Silhouette en validación")
    ax.set_title("Silhouette frente a variables activas")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_weights(names: list[str], weights: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, weights)
    ax.set_ylabel("Peso")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_comparison_markdown(
    path: Path,
    summary: pd.DataFrame,
    tests: list[dict],
    selection_note: str,
) -> None:
    lines = [
        "# Comparación",
        "",
        selection_note,
        "",
        "Un ARI cercano a 0 es un resultado legítimo: no se presenta como fallo ni como éxito.",
        "Los clusters se nombran cluster_0, cluster_1, … y no son niveles clínicos.",
        "",
        "## Media en validación",
        "",
    ]
    try:
        table = summary.to_markdown(index=False)
    except (ImportError, ModuleNotFoundError, ValueError):
        table = "```\n" + summary.to_csv(index=False) + "```"
    lines.extend(
        [
            table,
            "",
            "## Pruebas pareadas de Wilcoxon",
            "",
            "Se compara el algoritmo genético contra cada baseline en las semillas emparejadas.",
        "«Mejora», «empata» o «empeora» se decide solo si el contraste está definido.",
        "El umbral es p < 0.05 en Wilcoxon bilateral. Con 5 parejas, el p mínimo",
        "cuando todas las diferencias tienen el mismo signo es 0.0625: una ventaja",
        "descriptiva puede quedar como empate. La diferencia media se reporta igual.",
        "",
        ]
    )
    for item in tests:
            p_text = f"{item['p']:.4f}" if item["p"] == item["p"] else "n/d"
            lines.append(
                f"- {item['variant']}, k={item['k']}, {item['metric']}, GA vs {item['baseline']}: "
                f"{item['outcome']} (diferencia media {item['mean_diff']:.4f}, p={p_text})."
            )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
