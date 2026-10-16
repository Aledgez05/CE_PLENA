"""CMA-ES opcional, con el mismo presupuesto de evaluaciones que el algoritmo genético."""

from __future__ import annotations

import numpy as np

from plena.fitness import FitnessEvaluator


def run_cma(
    evaluator: FitnessEvaluator,
    n_genes: int,
    n_evaluations: int,
    seed: int,
    sigma: float = 0.3,
) -> dict:
    try:
        import cma
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    start = np.zeros(n_genes)

    def objective(genes: np.ndarray) -> float:
        detail = evaluator(np.asarray(genes, dtype=float))
        return -float(detail.score)

    options = {
        "bounds": [-1.0, 1.0],
        "maxfevals": int(n_evaluations),
        "seed": int(seed),
        "verbose": -9,
    }
    result = cma.fmin(objective, start, sigma, options)
    genes = np.asarray(result[0], dtype=float)
    detail = evaluator(genes)
    return {
        "available": True,
        "error": "",
        "genes": genes.tolist(),
        "score": detail.score,
        "n_active": detail.n_active,
        "n_eff": detail.n_eff,
    }
