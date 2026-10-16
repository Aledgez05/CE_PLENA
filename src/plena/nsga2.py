"""NSGA-II opcional. Tres objetivos: Silhouette, ARI de train y variables activas."""

from __future__ import annotations

import numpy as np

from plena.fitness import FitnessEvaluator


def run_nsga2(
    evaluator_v1: FitnessEvaluator,
    evaluator_v2: FitnessEvaluator,
    *,
    population: int,
    generations: int,
    seed: int,
) -> dict:
    """Frente de Pareto. Maximizar S y ARI equivale a minimizar sus negativos.

    No se ejecuta en la ruta crítica. Si ``pymoo`` no importa, se documenta y se detiene esta extensión.
    """
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem
        from pymoo.optimize import minimize
    except Exception as exc:
        return {"available": False, "error": str(exc), "front": []}

    dimension = evaluator_v1.column_map and len(evaluator_v1.column_map.variable_names)

    class WeightProblem(Problem):
        def __init__(self) -> None:
            super().__init__(n_var=dimension, n_obj=3, xl=-1.0, xu=1.0)

        def _evaluate(self, genomes, out, *args, **kwargs):
            objectives = []
            for genes in genomes:
                unsupervised = evaluator_v1(genes)
                supervised = evaluator_v2(genes)
                silhouette = unsupervised.silhouette if unsupervised.valid else -1.0
                ari = supervised.ari_train if supervised.ari_train is not None else -1.0
                n_active = unsupervised.n_active if unsupervised.valid else dimension
                objectives.append([-silhouette, -ari, n_active])
            out["F"] = np.asarray(objectives, dtype=float)

    algorithm = NSGA2(pop_size=population)
    result = minimize(
        WeightProblem(),
        algorithm,
        ("n_gen", generations),
        seed=seed,
        verbose=False,
    )
    front = []
    if result.X is not None:
        genomes = np.atleast_2d(result.X)
        values = np.atleast_2d(result.F)
        for genes, value in zip(genomes, values):
            front.append(
                {
                    "genes": genes.tolist(),
                    "silhouette": float(-value[0]),
                    "ari_train": float(-value[1]),
                    "n_active": float(value[2]),
                }
            )
    return {"available": True, "error": "", "front": front}


def select_from_front(front: list[dict]) -> dict | None:
    """Criterio declarado antes de mirar test: máximo Silhouette y, en empate, menos variables."""
    if not front:
        return None
    return max(front, key=lambda item: (item["silhouette"], -item["n_active"]))
