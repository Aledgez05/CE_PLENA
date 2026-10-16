"""Corridas con checkpoint, presupuesto y avance en consola."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score, silhouette_score

from plena.baselines import random_search, random_search_seed
from plena.config import ExperimentConfig
from plena.experiments import DatasetBundle, _row, score_weights
from plena.fitness import FitnessEvaluator
from plena.genetic_algorithm import run_ga


def checkpoint_path(output_dir: Path, split_id: str, method: str, variant: str, k: int, seed: int) -> Path:
    return output_dir / "checkpoints" / f"{split_id}__{method}__{variant}__k{k}__s{seed}.json"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    temporary.replace(path)


def _json_default(value):
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"No se puede serializar {type(value)}")


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def slice_offsets(start: int, end: int | None, count: int) -> list[int]:
    stop = count if end is None else end
    return [offset for offset in range(start, stop) if 0 <= offset < count]


def plan_jobs(
    experiment: ExperimentConfig,
    *,
    k_values: list[int],
    variants: tuple[str, ...],
    seed_start: int,
    seed_end: int | None,
    random_for_k: dict[int, int],
    split_id: str,
    methods: tuple[str, ...] = ("ga", "random_search"),
) -> list[dict]:
    jobs = []
    ga_offsets = slice_offsets(seed_start, seed_end, experiment.n_seeds_ga)
    for k in k_values:
        random_offsets = slice_offsets(seed_start, seed_end, random_for_k.get(k, 0))
        for variant in variants:
            if "ga" in methods:
                for offset in ga_offsets:
                    jobs.append(_job("ga", variant, k, offset, experiment.seed, split_id))
            if "random_search" in methods:
                for offset in random_offsets:
                    jobs.append(_job("random_search", variant, k, offset, experiment.seed, split_id))
    return jobs


def _job(method: str, variant: str, k: int, offset: int, base_seed: int, split_id: str) -> dict:
    return {
        "method": method,
        "variant": variant,
        "k": int(k),
        "offset": int(offset),
        "seed": int(base_seed + offset),
        "split_id": split_id,
    }


def execute_job(bundle: DatasetBundle, experiment: ExperimentConfig, job: dict) -> dict:
    """Una corrida. El proceso hijo limita BLAS a un hilo."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    try:
        from threadpoolctl import threadpool_limits

        limiter = threadpool_limits(limits=1)
        limiter.__enter__()
    except Exception:
        limiter = None
    try:
        evaluator = FitnessEvaluator(
            bundle.x_fit,
            bundle.y_fit,
            bundle.column_map,
            experiment,
            k=job["k"],
            variant=job["variant"],
        )
        started = time.perf_counter()
        if job["method"] == "ga":
            found = run_ga(evaluator, len(bundle.feature_names), experiment.ga, job["seed"])
            best = found.best
            genes = found.best_genes
            n_evaluations = found.n_evaluations
            history = found.history
        else:
            rng = np.random.default_rng(
                random_search_seed(experiment.seed, job["offset"], job["k"], job["variant"])
            )
            found = random_search(
                evaluator,
                len(bundle.feature_names),
                experiment.ga.population * experiment.ga.generations,
                rng,
                job["seed"],
            )
            best = found.best
            genes = found.best_genes
            n_evaluations = found.n_evaluations
            history = None
        elapsed = time.perf_counter() - started
        if not best.valid or best.weights is None:
            return {"job": job, "valid": False, "opt_seconds": elapsed}
        metrics = score_weights(bundle, best.weights, job["k"], experiment, "val")
        record = _row(job["method"], job["variant"], job["k"], job["seed"], "val", metrics, elapsed)
        before = None if best.weights_before_cap is None else best.weights_before_cap.tolist()
        stored = {
            "method": job["method"],
            "variant": job["variant"],
            "k": job["k"],
            "seed": job["seed"],
            "split_id": job["split_id"],
            "weights": best.weights.tolist(),
            "weights_before_cap": before,
            "mask": None if best.mask is None else best.mask.astype(int).tolist(),
            "genes": np.asarray(genes).tolist(),
            "n_evaluations": int(n_evaluations),
            "val_silhouette": metrics["silhouette"],
            "val_ari": metrics["ari"],
            "n_active": metrics["n_active"],
            "n_eff": metrics["n_eff"],
        }
        payload = {"job": job, "valid": True, "record": record, "stored": stored}
        if history is not None:
            payload["history"] = {"variant": job["variant"], "k": job["k"], "seed": job["seed"], "history": history}
        return payload
    finally:
        if limiter is not None:
            limiter.__exit__(None, None, None)


def run_jobs(
    bundle: DatasetBundle,
    experiment: ExperimentConfig,
    output_dir: Path,
    jobs: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Ejecuta las corridas que no tengan checkpoint y devuelve todas."""
    pending = []
    for job in jobs:
        path = checkpoint_path(
            output_dir, job["split_id"], job["method"], job["variant"], job["k"], job["seed"]
        )
        if load_json(path) is None:
            pending.append(job)
    total = len(jobs)
    done = total - len(pending)
    print(f"Corridas listas: {done} de {total}. Pendientes: {len(pending)}.", flush=True)
    if pending:
        started = time.perf_counter()
        if experiment.n_jobs <= 1:
            for index, job in enumerate(pending, start=1):
                _store_result(output_dir, execute_job(bundle, experiment, job))
                _progress(done + index, total, started, index)
        else:
            generator = Parallel(n_jobs=experiment.n_jobs, return_as="generator", prefer="processes")(
                delayed(execute_job)(bundle, experiment, job) for job in pending
            )
            for completed, result in enumerate(generator, start=1):
                _store_result(output_dir, result)
                _progress(done + completed, total, started, completed)
    return _collect(_reload_pending_results(output_dir, jobs))


def _reload_pending_results(output_dir: Path, jobs: list[dict]) -> list[dict]:
    loaded = []
    for job in jobs:
        path = checkpoint_path(
            output_dir, job["split_id"], job["method"], job["variant"], job["k"], job["seed"]
        )
        item = load_json(path)
        if item is not None:
            loaded.append(item)
    return loaded


def _store_result(output_dir: Path, result: dict) -> None:
    job = result["job"]
    path = checkpoint_path(output_dir, job["split_id"], job["method"], job["variant"], job["k"], job["seed"])
    write_json(path, result)


def _progress(done: int, total: int, started: float, completed_now: int) -> None:
    elapsed = time.perf_counter() - started
    remaining = elapsed / max(completed_now, 1) * max(total - done, 0)
    print(
        f"Corrida {done} de {total}. Tiempo restante estimado: {remaining / 60:.1f} min.",
        flush=True,
    )


def _collect(payloads: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    records = []
    stored = []
    histories = []
    for item in payloads:
        if not item.get("valid"):
            continue
        records.append(item["record"])
        stored.append(item["stored"])
        if "history" in item:
            histories.append(item["history"])
    return records, stored, histories


def probe_fitness(bundle: DatasetBundle, experiment: ExperimentConfig, n_eval: int = 100) -> float:
    """Tiempo medio de una evaluación con un solo hilo de BLAS."""
    evaluator = FitnessEvaluator(
        bundle.x_fit,
        bundle.y_fit,
        bundle.column_map,
        experiment,
        k=experiment.force_k,
        variant="v1",
    )
    rng = np.random.default_rng(experiment.seed)
    genes = [rng.uniform(-1.0, 1.0, size=len(bundle.feature_names)) for _ in range(n_eval)]
    try:
        from threadpoolctl import threadpool_limits

        context = threadpool_limits(limits=1)
    except Exception:
        context = None
    if context is not None:
        context.__enter__()
    started = time.perf_counter()
    for gene in genes:
        evaluator(gene)
    elapsed = time.perf_counter() - started
    if context is not None:
        context.__exit__(None, None, None)
    return elapsed / max(n_eval, 1)


def budget_table(
    experiment: ExperimentConfig,
    one_eval: float,
    cuts: list[str],
    estimated_hours: float,
    one_ga_minutes: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cpu_count": os.cpu_count() or 1,
                "n_jobs": experiment.n_jobs,
                "one_fitness_seconds": one_eval,
                "population": experiment.ga.population,
                "generations": experiment.ga.generations,
                "n_seeds_ga": experiment.n_seeds_ga,
                "n_random_runs": experiment.n_random_runs,
                "n_random_runs_other": experiment.n_random_runs_other,
                "one_ga_minutes": one_ga_minutes,
                "estimated_hours": estimated_hours,
                "cuts": ";".join(cuts) if cuts else "ninguno",
                "max_hours": experiment.max_hours,
            }
        ]
    )


def choose_budget(experiment: ExperimentConfig, one_eval: float) -> tuple[list[str], float, float]:
    """Recorta solo en el orden pedido si la estimación pasa de max_hours."""
    cuts: list[str] = []
    original_other = experiment.n_random_runs_other

    def wall(pop: int, gen: int, other: int | None) -> tuple[float, float]:
        hours, minutes = _wall_hours(experiment, one_eval, pop, gen, other)
        return hours, minutes

    other = original_other
    pop = experiment.ga.population
    gen = experiment.ga.generations
    hours, minutes = wall(pop, gen, other)
    if hours > experiment.max_hours and other is None:
        other = min(10, experiment.n_random_runs)
        cuts.append("busquedas_aleatorias_10_fuera_de_k3_y_k_elegido")
        hours, minutes = wall(pop, gen, other)
    if hours > experiment.max_hours and gen > 30:
        gen = 30
        cuts.append("generaciones_30")
        hours, minutes = wall(pop, gen, other)
    if hours > experiment.max_hours and pop > 30:
        pop = 30
        cuts.append("poblacion_30")
        hours, minutes = wall(pop, gen, other)
    experiment.ga.population = pop
    experiment.ga.generations = gen
    experiment.n_random_runs_other = other
    return cuts, hours, minutes


def _wall_hours(
    experiment: ExperimentConfig,
    one_eval: float,
    pop: int,
    gen: int,
    other: int | None,
) -> tuple[float, float]:
    n_k = len(experiment.k_values)
    n_ga = experiment.n_seeds_ga * n_k * 2
    if other is None:
        n_random = experiment.n_random_runs * n_k * 2
    else:
        high = 2 if n_k > 1 else 1
        low = max(0, n_k - high)
        n_random = (high * experiment.n_random_runs + low * other) * 2
    n_robust = experiment.robustness_partitions * experiment.robustness_seeds * 2 * 2
    evaluations = (n_ga + n_random + n_robust) * pop * gen
    jobs = max(experiment.n_jobs, 1)
    seconds = evaluations * one_eval / jobs * 1.15
    return seconds / 3600, (one_eval * pop * gen) / 60


def random_counts(experiment: ExperimentConfig, k_values: list[int], chosen_k: int | None) -> dict[int, int]:
    counts = {}
    other = experiment.n_random_runs_other
    for k in k_values:
        if other is None or chosen_k is None or k in {experiment.force_k, chosen_k}:
            counts[int(k)] = experiment.n_random_runs
        else:
            counts[int(k)] = other
    return counts


def group_bootstrap(
    y: np.ndarray,
    labels: np.ndarray,
    matrix: np.ndarray,
    groups: np.ndarray | None,
    n_boot: int,
    seed: int,
) -> dict:
    """Intervalos por remuestreo de grupos. Si no hay grupos, remuestrea filas."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    labels = np.asarray(labels)
    matrix = np.asarray(matrix)
    silhouettes = []
    aris = []
    if groups is None:
        units = None
    else:
        units = np.unique(groups)
    for _ in range(n_boot):
        if units is None:
            chosen = rng.integers(0, len(y), size=len(y))
        else:
            picked = rng.choice(units, size=len(units), replace=True)
            chosen = np.concatenate([np.flatnonzero(groups == unit) for unit in picked])
        if len(chosen) < 3 or len(np.unique(labels[chosen])) < 2:
            continue
        silhouettes.append(float(silhouette_score(matrix[chosen], labels[chosen], metric="euclidean")))
        aris.append(float(adjusted_rand_score(y[chosen], labels[chosen])))

    def summarize(values: list[float]) -> dict:
        if len(values) < 20:
            return {"n": len(values), "low": None, "high": None, "mean": None}
        return {
            "n": len(values),
            "low": float(np.percentile(values, 2.5)),
            "high": float(np.percentile(values, 97.5)),
            "mean": float(np.mean(values)),
        }

    return {
        "unit": "grupo" if groups is not None else "fila",
        "silhouette": summarize(silhouettes),
        "ari": summarize(aris),
    }
