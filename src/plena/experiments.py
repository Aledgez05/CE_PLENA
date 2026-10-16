"""Orquesta baselines, algoritmo genético, validación y una sola lectura de test."""

from __future__ import annotations

import json
import platform
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from plena import __version__
from plena.baselines import equal_weights, permutation_reference
from plena.chromosome import n_eff
from plena.clustering import fit_kmeans, kmedoids_alternate, predict_centroids, predict_knn
from plena.config import ExperimentConfig, SchemaConfig, assert_weights_feasible, validate_schema
from plena.data import (
    banned_columns,
    drop_missing_target,
    feature_frame,
    load_table,
    prevalence,
    save_splits,
    split_indices,
)
from plena.distance import gower_component_matrices, gower_distance
from plena.evaluation import (
    classification_from_clusters,
    external_metrics,
    internal_metrics,
)
from plena.faiss_index import (
    FlatL2Index,
    benchmark_latency,
    classify_neighbor_rows,
    l2_matches_weighted,
    weighted_vectors,
)
from plena.fitness import FitnessEvaluator
from plena.genetic_algorithm import run_ga
from plena.preprocessing import build_preprocessor, fit_route_a
from plena.reporting import (
    format_mean_std,
    plot_convergence,
    plot_pareto,
    plot_weights,
    write_comparison_markdown,
)
from plena.seed import set_seed
from plena.stability import (
    ablate_weights,
    assignment_stability,
    cluster_profiles,
    subgroup_prevalence,
    weight_stability,
)
from plena.supervised import supervised_reference

SELECTION_NOTE = (
    "El conjunto de prueba no se usó para elegir k, pesos, lambda, alpha ni ningún "
    "hiperparámetro. La configuración se congeló con métricas de validación y solo "
    "después se calcularon las métricas de prueba, una vez."
)


@dataclass
class DatasetBundle:
    feature_names: list[str]
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    x_fit: np.ndarray
    y_fit: np.ndarray
    column_map: object
    transformer: object
    raw_train: pd.DataFrame
    raw_val: pd.DataFrame
    raw_test: pd.DataFrame
    types: dict[str, str]
    splits: dict[str, np.ndarray]
    source_rows_train: np.ndarray
    groups_train: np.ndarray | None = None
    groups_val: np.ndarray | None = None
    groups_test: np.ndarray | None = None
    generative_train: np.ndarray | None = None
    generative_val: np.ndarray | None = None
    generative_test: np.ndarray | None = None


def prepare_dataset(schema: SchemaConfig, experiment: ExperimentConfig) -> DatasetBundle:
    frame, _mapping = load_table(schema.dataset_path)
    frame, removed = drop_missing_target(frame, schema.target)
    validate_schema(schema, columns=list(frame.columns), allow_unconfirmed=False)
    assert_weights_feasible(len(schema.feature_names), experiment)
    y_all = frame[schema.target].to_numpy()
    groups = frame[schema.group_column].to_numpy() if schema.group_column else None
    splits = split_indices(
        y_all,
        train=experiment.split.train,
        val=experiment.split.val,
        test=experiment.split.test,
        seed=experiment.seed,
        groups=groups,
    )
    banned = banned_columns(
        schema.target,
        schema.id_columns,
        [item.column for item in schema.exclusions],
    )
    features = feature_frame(frame, schema.feature_names, banned)
    transformer, step_map = build_preprocessor(schema, experiment.scaler)
    raw_train = features.iloc[splits["train"]].reset_index(drop=True)
    raw_val = features.iloc[splits["val"]].reset_index(drop=True)
    raw_test = features.iloc[splits["test"]].reset_index(drop=True)
    column_map, x_train, x_val, x_test = fit_route_a(
        transformer,
        step_map,
        schema.feature_names,
        raw_train,
        raw_val,
        raw_test,
    )
    y_train = y_all[splits["train"]]
    y_val = y_all[splits["val"]]
    y_test = y_all[splits["test"]]
    positives = {
        "train": int(np.sum(np.asarray(y_train).astype(float) == 1)),
        "val": int(np.sum(np.asarray(y_val).astype(float) == 1)),
        "test": int(np.sum(np.asarray(y_test).astype(float) == 1)),
    }
    if experiment.positive_minimum and any(count < experiment.positive_minimum for count in positives.values()):
        raise RuntimeError(
            "Alguna partición tiene muy pocos positivos "
            f"({positives}; mínimo {experiment.positive_minimum}). "
            "No se continúa el estudio."
        )
    groups_train = groups_val = groups_test = None
    if groups is not None:
        groups_train = groups[splits["train"]]
        groups_val = groups[splits["val"]]
        groups_test = groups[splits["test"]]
    generative = frame["generative_cluster"].to_numpy() if "generative_cluster" in frame.columns else None
    x_fit, y_fit = _fitness_sample(
        np.asarray(x_train, dtype=float),
        y_train,
        experiment.fitness_sample_size,
        experiment.seed,
    )
    _ = removed
    return DatasetBundle(
        feature_names=list(schema.feature_names),
        x_train=np.asarray(x_train, dtype=float),
        x_val=np.asarray(x_val, dtype=float),
        x_test=np.asarray(x_test, dtype=float),
        y_train=np.asarray(y_train),
        y_val=np.asarray(y_val),
        y_test=np.asarray(y_test),
        x_fit=x_fit,
        y_fit=y_fit,
        column_map=column_map,
        transformer=transformer,
        raw_train=raw_train,
        raw_val=raw_val,
        raw_test=raw_test,
        types={name: spec.type for name, spec in schema.features.items()},
        splits=splits,
        source_rows_train=frame.iloc[splits["train"]]["source_row"].to_numpy(),
        groups_train=None if groups_train is None else np.asarray(groups_train),
        groups_val=None if groups_val is None else np.asarray(groups_val),
        groups_test=None if groups_test is None else np.asarray(groups_test),
        generative_train=None if generative is None else np.asarray(generative)[splits["train"]],
        generative_val=None if generative is None else np.asarray(generative)[splits["val"]],
        generative_test=None if generative is None else np.asarray(generative)[splits["test"]],
    )


def _fitness_sample(matrix: np.ndarray, y: np.ndarray, size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    take = min(size, len(matrix))
    chosen = np.sort(rng.choice(len(matrix), size=take, replace=False))
    return matrix[chosen], y[chosen]


def score_weights(
    bundle: DatasetBundle,
    weights: np.ndarray,
    k: int,
    experiment: ExperimentConfig,
    split: str,
) -> dict:
    """Ajusta K-Means en train completo y puntúa la partición pedida."""
    from plena.distance import transform_weighted

    train_w = transform_weighted(bundle.x_train, weights, bundle.column_map)
    model = fit_kmeans(
        train_w,
        k,
        random_state=experiment.kmeans_random_state,
        n_init=experiment.kmeans_n_init,
    )
    if split == "train":
        matrix = train_w
        labels = model.labels
        y_eval = bundle.y_train
    else:
        source = bundle.x_val if split == "val" else bundle.x_test
        y_eval = bundle.y_val if split == "val" else bundle.y_test
        matrix = transform_weighted(source, weights, bundle.column_map)
        labels = predict_centroids(model.centroids, matrix)
    internal = internal_metrics(matrix, labels)
    external = external_metrics(y_eval, labels)
    classification = classification_from_clusters(bundle.y_train, model.labels, y_eval, labels)
    start = time.perf_counter()
    predict_centroids(model.centroids, matrix[: min(32, len(matrix))])
    infer_seconds = (time.perf_counter() - start) / max(min(32, len(matrix)), 1)
    return {
        "silhouette": internal["silhouette"],
        "davies_bouldin": internal["davies_bouldin"],
        "calinski_harabasz": internal["calinski_harabasz"],
        "ri": external["ri"],
        "ari": external["ari"],
        "ami": external["ami"],
        "nmi": external["nmi"],
        "auc_membership": external["auc_membership"],
        "balanced_accuracy": classification["balanced_accuracy"],
        "f1_macro": classification["f1_macro"],
        "roc_auc": classification["roc_auc"],
        "prevalence_by_cluster": external["prevalence_by_cluster"],
        "n_active": int(np.sum(weights > 1e-12)),
        "n_eff": n_eff(weights),
        "infer_seconds": float(infer_seconds),
        "labels_eval": labels,
        "labels_train": model.labels,
        "centroids": model.centroids,
    }


def _row(method: str, variant: str, k: int, seed: int, split: str, metrics: dict, seconds: float) -> dict:
    return {
        "method": method,
        "variant": variant,
        "representation": "route_a_euclidean",
        "k": k,
        "seed": seed,
        "split": split,
        "silhouette": metrics["silhouette"],
        "ari": metrics["ari"],
        "ri": metrics["ri"],
        "ami": metrics["ami"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "f1_macro": metrics["f1_macro"],
        "n_active": metrics["n_active"],
        "n_eff": metrics["n_eff"],
        "opt_seconds": seconds,
        "infer_seconds": metrics["infer_seconds"],
        "test_used_for_selection": False,
    }


def run_study(
    schema: SchemaConfig,
    experiment: ExperimentConfig,
    output_dir: Path,
    *,
    k_values: list[int] | None = None,
    variants: tuple[str, ...] = ("v1", "v2"),
    seed_start: int = 0,
    seed_end: int | None = None,
    measure_cost: bool = False,
    finalize: bool | None = None,
) -> dict:
    """Corre la comparación. Test entra solo al cerrar el estudio."""
    from plena.study_loop import (
        budget_table,
        choose_budget,
        plan_jobs,
        probe_fitness,
        random_counts,
        run_jobs,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    started_clock = time.perf_counter()
    criterion = output_dir / "selection_criterion.txt"
    criterion.write_text(experiment.selection_criterion + "\n" + SELECTION_NOTE + "\n", encoding="utf-8")
    bundle = prepare_dataset(schema, experiment)
    positives = {
        name: int(np.sum(np.asarray(getattr(bundle, f"y_{name}")).astype(float) == 1))
        for name in ("train", "val", "test")
    }
    print(
        "Positivos por partición: "
        + ", ".join(f"{name}={count}" for name, count in positives.items()),
        flush=True,
    )
    save_splits(
        output_dir / "splits.json",
        bundle.splits,
        {
            "seed": experiment.seed,
            "train": experiment.split.train,
            "val": experiment.split.val,
            "test": experiment.split.test,
            "prevalence_train": prevalence(bundle.y_train),
            "prevalence_val": prevalence(bundle.y_val),
            "prevalence_test": prevalence(bundle.y_test),
            "positives": positives,
            "n_train": int(len(bundle.y_train)),
            "n_val": int(len(bundle.y_val)),
            "n_test": int(len(bundle.y_test)),
            "note": "Índices sobre las filas que conservan etiqueta. Test no entra en la búsqueda.",
        },
    )
    if measure_cost:
        one_eval = probe_fitness(bundle, experiment, n_eval=100)
        cuts, hours, minutes = choose_budget(experiment, one_eval)
        budget_table(experiment, one_eval, cuts, hours, minutes).to_csv(
            output_dir / "cost_estimate.csv", index=False
        )
        print(
            f"Estimación: {hours:.2f} h con {experiment.n_jobs} procesos. "
            f"Una corrida de GA: {minutes:.2f} min. Recortes: {cuts or 'ninguno'}.",
            flush=True,
        )
        if hours > experiment.max_hours or minutes > 10:
            raise RuntimeError(
                "El presupuesto sigue por encima del límite después de los recortes permitidos. "
                f"Estimación {hours:.2f} h, una corrida {minutes:.2f} min. Recortes: {cuts or 'ninguno'}."
            )
    else:
        _write_cost(bundle, experiment, output_dir / "cost_estimate.csv")
    selected_k = list(experiment.k_values if k_values is None else k_values)
    counts = random_counts(experiment, selected_k, None)
    ga_jobs = plan_jobs(
        experiment,
        k_values=selected_k,
        variants=variants,
        seed_start=seed_start,
        seed_end=seed_end,
        random_for_k=counts,
        split_id="main",
        methods=("ga",),
    )
    records, stored, histories = run_jobs(bundle, experiment, output_dir, ga_jobs)
    chosen_preview = _select(stored, experiment)
    chosen_k = None
    if "v1" in chosen_preview:
        chosen_k = int(chosen_preview["v1"]["k"])
    elif chosen_preview:
        chosen_k = int(next(iter(chosen_preview.values()))["k"])
    full_slice = (
        k_values is None
        and variants == ("v1", "v2")
        and seed_start == 0
        and (seed_end is None or seed_end >= experiment.n_seeds_ga)
    )
    if full_slice and chosen_k is not None:
        counts = random_counts(experiment, selected_k, chosen_k)
    random_jobs = plan_jobs(
        experiment,
        k_values=selected_k,
        variants=variants,
        seed_start=seed_start,
        seed_end=seed_end,
        random_for_k=counts,
        split_id="main",
        methods=("random_search",),
    )
    extra_records, extra_stored, _extra_hist = run_jobs(bundle, experiment, output_dir, random_jobs)
    records.extend(extra_records)
    stored.extend(extra_stored)
    for k in selected_k:
        for variant in variants:
            weights = equal_weights(len(bundle.feature_names))
            metrics = score_weights(bundle, weights, k, experiment, "val")
            records.append(_row("equal_weights", variant, k, experiment.seed, "val", metrics, 0.0))
    frame = pd.DataFrame(records)
    do_finalize = full_slice if finalize is None else finalize
    if not do_finalize:
        if not frame.empty:
            frame.to_csv(output_dir / "comparison.partial.csv", index=False)
        return {"frame": frame, "chosen": {}, "tests": [], "manifest": {"partial": True}}
    chosen = _select(stored, experiment)
    chosen["reported_k3"] = _select(
        [item for item in stored if item["k"] == experiment.force_k],
        experiment,
    )
    test_rows, test_details = _evaluate_test(bundle, experiment, {k: v for k, v in chosen.items() if k in {"v1", "v2"}}, stored)
    frame = pd.concat([frame, pd.DataFrame(test_rows)], ignore_index=True)
    tests = _wilcoxon(frame)
    _write_tables(output_dir, frame, tests, histories, chosen, bundle, schema, experiment, test_details)
    _bootstrap_test(bundle, experiment, chosen, test_details, output_dir)
    extras = _robustness_and_inference(bundle, schema, experiment, chosen, stored, output_dir)
    _representation_comparisons(schema, experiment, bundle, chosen, output_dir)
    if measure_cost and experiment.robustness_partitions:
        _robustness_partitions(schema, experiment, chosen, output_dir)
    if measure_cost:
        remaining = experiment.max_hours * 3600 - (time.perf_counter() - started_clock)
        _maybe_extensions(bundle, experiment, chosen, output_dir, remaining)
    _write_environment(output_dir, schema, experiment)
    manifest = {
        "test_used_for_selection": False,
        "selection_note": SELECTION_NOTE,
        "selection_criterion": experiment.selection_criterion,
        "chosen": chosen,
        "library_notes": extras.get("library_notes", []),
        "positives": positives,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"frame": frame, "chosen": chosen, "tests": tests, "manifest": manifest}


def _select(stored: list[dict], experiment: ExperimentConfig) -> dict[str, dict]:
    chosen = {}
    dimension = None
    for variant in ("v1", "v2"):
        candidates = [item for item in stored if item["method"] == "ga" and item["variant"] == variant]
        if not candidates:
            continue
        dimension = len(candidates[0]["weights"])

        def key(item: dict, variant: str = variant, dimension: int = dimension) -> tuple:
            if variant == "v1":
                primary = item["val_silhouette"]
            else:
                primary = (
                    experiment.v2_alpha * item["val_silhouette"]
                    + (1.0 - experiment.v2_alpha) * item["val_ari"]
                    - experiment.ga.lambda_active * (item["n_active"] / dimension)
                )
            return (primary, item["n_eff"], -item["n_active"], -item["seed"])

        best = max(candidates, key=key)
        chosen[variant] = best
    return chosen


def _evaluate_test(bundle, experiment, chosen, stored) -> tuple[list[dict], dict]:
    """Única fase que lee el conjunto de prueba."""
    rows = []
    details = {}
    for variant, selected in chosen.items():
        weights = np.asarray(selected["weights"], dtype=float)
        metrics = score_weights(bundle, weights, selected["k"], experiment, "test")
        rows.append(_row("ga", variant, selected["k"], selected["seed"], "test", metrics, 0.0))
        details[variant] = metrics
        equal = equal_weights(len(bundle.feature_names))
        equal_metrics = score_weights(bundle, equal, selected["k"], experiment, "test")
        rows.append(_row("equal_weights", variant, selected["k"], experiment.seed, "test", equal_metrics, 0.0))
        random_candidates = [
            item
            for item in stored
            if item["method"] == "random_search" and item["variant"] == variant and item["k"] == selected["k"]
        ]
        if random_candidates:
            if variant == "v1":
                best_random = max(random_candidates, key=lambda item: item["val_silhouette"])
            else:
                best_random = max(random_candidates, key=lambda item: item["val_ari"])
            random_metrics = score_weights(
                bundle, np.asarray(best_random["weights"]), selected["k"], experiment, "test"
            )
            rows.append(
                _row("random_search", variant, selected["k"], best_random["seed"], "test", random_metrics, 0.0)
            )
    return rows, details


def _bootstrap_test(bundle, experiment, chosen, test_details, output_dir: Path) -> None:
    from plena.distance import transform_weighted
    from plena.study_loop import group_bootstrap

    report = {}
    for variant in ("v1", "v2"):
        if variant not in chosen or variant not in test_details:
            continue
        weights = np.asarray(chosen[variant]["weights"], dtype=float)
        matrix = transform_weighted(bundle.x_test, weights, bundle.column_map)
        report[variant] = group_bootstrap(
            bundle.y_test,
            test_details[variant]["labels_eval"],
            matrix,
            bundle.groups_test,
            experiment.bootstrap_replicates,
            experiment.seed,
        )
    (output_dir / "bootstrap_test.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


def _representation_comparisons(schema, experiment, bundle, chosen, output_dir: Path) -> None:
    """Comparaciones de codificación, sin una segunda búsqueda genética."""
    from dataclasses import replace

    from plena.config import FeatureSpec
    from plena.preprocessing import build_preprocessor, fit_route_a

    rows = []
    targets = []
    if "v1" in chosen:
        targets.append(int(chosen["v1"]["k"]))
    targets.append(experiment.force_k)
    k_values = sorted(set(targets))
    if "education_level" in schema.features:
        edu = schema.features["education_level"]
        nominal = replace(edu, type="nominal", order=None, transform=None)
        features = dict(schema.features)
        features["education_level"] = nominal
        rows.extend(
            _score_representation(
                replace(schema, features=features),
                experiment,
                bundle,
                k_values,
                "education_nominal",
                list(features),
            )
        )
    if schema.group_column and (schema.dataset_path.is_file()):
        frame, _mapping = load_table(schema.dataset_path)
        frame, _removed = drop_missing_target(frame, schema.target)
        features = dict(schema.features)
        features[schema.group_column] = FeatureSpec(
            name=schema.group_column,
            type="nominal",
            status="confirmed",
            justification="Ablación: la aldea entra como nominal.",
        )
        rows.extend(
            _score_representation(
                replace(schema, features=features),
                experiment,
                bundle,
                k_values,
                "ville_nominal",
                list(features),
                source=frame,
            )
        )
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "representation_ablation.csv", index=False)
    _ = build_preprocessor, fit_route_a


def _score_representation(schema, experiment, bundle, k_values, name, columns, source=None) -> list[dict]:
    from dataclasses import replace

    from plena.preprocessing import build_preprocessor, fit_route_a

    if source is None:
        raw_train, raw_val, raw_test = bundle.raw_train, bundle.raw_val, bundle.raw_test
        raw_train = raw_train[columns]
        raw_val = raw_val[columns]
        raw_test = raw_test[columns]
    else:
        raw_train = source.iloc[bundle.splits["train"]][columns].reset_index(drop=True)
        raw_val = source.iloc[bundle.splits["val"]][columns].reset_index(drop=True)
        raw_test = source.iloc[bundle.splits["test"]][columns].reset_index(drop=True)
    transformer, step_map = build_preprocessor(schema, experiment.scaler)
    column_map, x_train, x_val, x_test = fit_route_a(
        transformer, step_map, columns, raw_train, raw_val, raw_test
    )
    alt = replace(
        bundle,
        feature_names=list(columns),
        x_train=np.asarray(x_train, dtype=float),
        x_val=np.asarray(x_val, dtype=float),
        x_test=np.asarray(x_test, dtype=float),
        column_map=column_map,
        raw_train=raw_train,
        raw_val=raw_val,
        raw_test=raw_test,
    )
    rows = []
    weights = equal_weights(len(columns))
    note = (
        "En la partición por aldea, los niveles de ville_id de validación no aparecen en train."
        if name == "ville_nominal"
        else "Comparación de codificación con pesos iguales, sin una segunda búsqueda."
    )
    for k in k_values:
        for split in ("train", "val"):
            metrics = score_weights(alt, weights, k, experiment, split)
            rows.append(
                {
                    "representation": name,
                    "k": k,
                    "split": split,
                    "silhouette": metrics["silhouette"],
                    "ari": metrics["ari"],
                    "n_features": len(columns),
                    "note": note,
                }
            )
    return rows


def _robustness_partitions(schema, experiment, chosen, output_dir: Path) -> None:
    """Otras particiones, solo en el k elegido y en k=3, sin reoptimizar la malla."""
    from dataclasses import replace

    from plena.study_loop import plan_jobs, run_jobs

    if "v1" not in chosen:
        return
    k_values = sorted({experiment.force_k, int(chosen["v1"]["k"])})
    collected = []
    for index in range(experiment.robustness_partitions):
        alt = replace(
            experiment,
            seed=experiment.seed + 100 + index,
            robustness_partitions=0,
            n_seeds_ga=experiment.robustness_seeds,
        )
        alt_bundle = prepare_dataset(schema, alt)
        print(
            f"Robustez, partición {index + 1} de {experiment.robustness_partitions}, semilla {alt.seed}.",
            flush=True,
        )
        jobs = plan_jobs(
            alt,
            k_values=k_values,
            variants=("v1", "v2"),
            seed_start=0,
            seed_end=experiment.robustness_seeds,
            random_for_k={k: 0 for k in k_values},
            split_id=f"robust_{alt.seed}",
            methods=("ga",),
        )
        records, _stored, _histories = run_jobs(alt_bundle, alt, output_dir, jobs)
        for row in records:
            row = dict(row)
            row["partition_seed"] = alt.seed
            collected.append(row)
    if collected:
        pd.DataFrame(collected).to_csv(output_dir / "robustness.csv", index=False)


def _maybe_extensions(bundle, experiment, chosen, output_dir: Path, remaining_seconds: float) -> None:
    """CMA-ES primero. NSGA-II solo si todavía queda tiempo."""
    from plena.cma_es import run_cma
    from plena.fitness import FitnessEvaluator
    from plena.nsga2 import run_nsga2

    if "v1" not in chosen:
        return
    if remaining_seconds < 600:
        (output_dir / "extensions_skipped.txt").write_text(
            "No quedó tiempo para CMA-ES ni NSGA-II.\n", encoding="utf-8"
        )
        return
    budget = experiment.ga.population * experiment.ga.generations
    k = int(chosen["v1"]["k"])
    reports = {}
    for variant in ("v1", "v2"):
        evaluator = FitnessEvaluator(
            bundle.x_fit, bundle.y_fit, bundle.column_map, experiment, k=k, variant=variant
        )
        reports[variant] = run_cma(evaluator, len(bundle.feature_names), budget, experiment.seed)
    (output_dir / "cma_es.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    if remaining_seconds < 2400:
        (output_dir / "extensions_skipped.txt").write_text(
            "CMA-ES quedó registrado. No quedó tiempo para NSGA-II.\n", encoding="utf-8"
        )
        return
    evaluator_v1 = FitnessEvaluator(
        bundle.x_fit, bundle.y_fit, bundle.column_map, experiment, k=experiment.force_k, variant="v1"
    )
    evaluator_v2 = FitnessEvaluator(
        bundle.x_fit, bundle.y_fit, bundle.column_map, experiment, k=experiment.force_k, variant="v2"
    )
    front = run_nsga2(
        evaluator_v1,
        evaluator_v2,
        population=min(experiment.ga.population, 20),
        generations=min(experiment.ga.generations, 15),
        seed=experiment.seed,
    )
    (output_dir / "nsga2.json").write_text(json.dumps(front, indent=2), encoding="utf-8")


def _wilcoxon(frame: pd.DataFrame) -> list[dict]:
    tests = []
    validation = frame[frame["split"] == "val"]
    for variant in sorted(validation["variant"].unique()):
        for k in sorted(validation["k"].unique()):
            ga = validation[(validation["method"] == "ga") & (validation["variant"] == variant) & (validation["k"] == k)]
            if ga.empty:
                continue
            for baseline in ("equal_weights", "random_search"):
                base = validation[
                    (validation["method"] == baseline) & (validation["variant"] == variant) & (validation["k"] == k)
                ]
                for metric in ("silhouette", "ari"):
                    paired = _paired(ga, base, metric)
                    if paired is None:
                        continue
                    left, right = paired
                    tests.append(
                        {
                            "variant": variant,
                            "k": int(k),
                            "metric": metric,
                            "baseline": baseline,
                            **_outcome(left, right),
                        }
                    )
    return tests


def _paired(ga: pd.DataFrame, baseline: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray] | None:
    left = []
    right = []
    baseline_by_seed = {int(row.seed): float(getattr(row, metric)) for row in baseline.itertuples()}
    constant = None
    if len(baseline_by_seed) == 1:
        constant = next(iter(baseline_by_seed.values()))
    for row in ga.itertuples():
        if constant is not None and int(row.seed) not in baseline_by_seed:
            other = constant
        elif int(row.seed) in baseline_by_seed:
            other = baseline_by_seed[int(row.seed)]
        else:
            continue
        left.append(float(getattr(row, metric)))
        right.append(other)
    if len(left) < 2:
        return None
    return np.asarray(left), np.asarray(right)


def _outcome(left: np.ndarray, right: np.ndarray) -> dict:
    diff = left - right
    mean_diff = float(np.mean(diff))
    if np.allclose(diff, 0.0):
        p_value = 1.0
    else:
        try:
            p_value = float(wilcoxon(diff, alternative="two-sided").pvalue)
        except ValueError:
            p_value = float("nan")
    if np.isfinite(p_value) and p_value < 0.05 and mean_diff > 0:
        outcome = "mejora"
    elif np.isfinite(p_value) and p_value < 0.05 and mean_diff < 0:
        outcome = "empeora"
    else:
        outcome = "empata"
    return {"outcome": outcome, "mean_diff": mean_diff, "p": p_value}


def _write_cost(bundle: DatasetBundle, experiment: ExperimentConfig, path: Path) -> None:
    evaluator = FitnessEvaluator(
        bundle.x_fit,
        bundle.y_fit,
        bundle.column_map,
        experiment,
        k=experiment.force_k,
        variant="v1",
    )
    genes = np.full(len(bundle.feature_names), 0.5)
    start = time.perf_counter()
    evaluator(genes)
    one = time.perf_counter() - start
    budget = experiment.ga.population * experiment.ga.generations
    total = one * budget * experiment.n_seeds_ga * len(experiment.k_values) * 2
    table = pd.DataFrame(
        [
            {
                "one_fitness_seconds": one,
                "fitness_sample_size": len(bundle.x_fit),
                "configured_evaluations_per_run": budget,
                "estimated_ga_seconds": total,
                "budget_seconds": experiment.budget_seconds,
                "exceeds_budget": bool(experiment.budget_seconds and total > experiment.budget_seconds),
            }
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def _write_tables(output_dir, frame, tests, histories, chosen, bundle, schema, experiment, test_details) -> None:
    public = frame.drop(columns=[col for col in frame.columns if col.startswith("_")], errors="ignore")
    public.to_csv(output_dir / "comparison.csv", index=False)
    baseline = public[public["method"] == "equal_weights"]
    baseline.to_csv(output_dir / "baseline.csv", index=False)
    summary = (
        public[public["split"] == "val"]
        .groupby(["method", "variant", "k"], as_index=False)
        .agg(
            silhouette=("silhouette", lambda s: format_mean_std(s.tolist())),
            ari=("ari", lambda s: format_mean_std(s.tolist())),
            n_active=("n_active", "mean"),
            n_eff=("n_eff", "mean"),
            opt_seconds=("opt_seconds", "mean"),
        )
    )
    summary.to_csv(output_dir / "comparison_summary.csv", index=False)
    write_comparison_markdown(output_dir / "comparison.md", summary, tests, SELECTION_NOTE)
    (output_dir / "ga_history.json").write_text(json.dumps(histories), encoding="utf-8")
    (output_dir / "chosen.json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    figures = output_dir / "figures"
    plot_convergence(
        [item["history"] for item in histories if item["variant"] == "v1" and item["k"] == experiment.force_k],
        figures / "convergence_v1.png",
        f"Convergencia V1, k={experiment.force_k}",
    )
    plot_pareto(public, figures / "pareto_silhouette.png")
    if "v1" in chosen:
        plot_weights(
            bundle.feature_names,
            np.asarray(chosen["v1"]["weights"]),
            figures / "weights_v1.png",
            "Pesos del individuo V1 elegido en validación",
        )
    _ = schema, test_details


def _robustness_and_inference(bundle, schema, experiment, chosen, stored, output_dir: Path) -> dict:
    notes = []
    if "v1" not in chosen:
        return {"library_notes": ["No hubo un individuo V1 válido."]}
    selected = chosen["v1"]
    weights = np.asarray(selected["weights"], dtype=float)
    same_k = [
        item
        for item in stored
        if item["method"] == "ga" and item["variant"] == "v1" and item["k"] == selected["k"]
    ]
    masks = [np.asarray(item["weights"]) > 1e-12 for item in same_k]
    weight_rows = [np.asarray(item["weights"]) for item in same_k]
    stability = weight_stability(weight_rows, masks)
    label_rows = []
    for item in same_k:
        scored = score_weights(bundle, np.asarray(item["weights"]), selected["k"], experiment, "val")
        label_rows.append(scored["labels_eval"])
    stability["assignment_ari"] = assignment_stability(label_rows)
    (output_dir / "stability.json").write_text(json.dumps(stability, indent=2), encoding="utf-8")
    ablation_rows = []
    for group, columns in schema.ablation_groups.items():
        columns = [col for col in columns if col in bundle.feature_names]
        if not columns:
            continue
        ablated = ablate_weights(weights, bundle.feature_names, columns)
        metrics = score_weights(bundle, ablated, selected["k"], experiment, "val")
        ablation_rows.append({"group_removed": group, "silhouette": metrics["silhouette"], "ari": metrics["ari"]})
    if ablation_rows:
        pd.DataFrame(ablation_rows).to_csv(output_dir / "ablation.csv", index=False)
    train_metrics = score_weights(bundle, weights, selected["k"], experiment, "train")
    profiles = cluster_profiles(
        bundle.raw_train.reset_index(drop=True),
        train_metrics["labels_train"],
        bundle.types,
        bundle.y_train,
    )
    profiles.to_csv(output_dir / "cluster_profiles.csv", index=False)
    if schema.sensitive_columns:
        subgroups = subgroup_prevalence(bundle.raw_train, bundle.y_train, schema.sensitive_columns)
        (output_dir / "subgroups.json").write_text(json.dumps(subgroups, indent=2), encoding="utf-8")
    _route_b(bundle, experiment, output_dir, notes)
    _faiss(bundle, weights, train_metrics, selected, experiment, output_dir, notes)
    supervised = supervised_reference(
        weighted_vectors(bundle.x_train, weights, bundle.column_map),
        bundle.y_train.astype(int),
        weighted_vectors(bundle.x_val, weights, bundle.column_map),
        bundle.y_val.astype(int),
        experiment.seed,
    )
    (output_dir / "supervised_reference.json").write_text(
        json.dumps(
            {
                "disclaimer": "Estas métricas son supervisadas y no se comparan con Silhouette. La etiqueta puede ser un indicador de encuesta, no un diagnóstico.",
                "models": supervised,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    reference = permutation_reference(
        weighted_vectors(bundle.x_fit, weights, bundle.column_map),
        bundle.y_fit,
        train_metrics["labels_train"][: len(bundle.x_fit)]
        if False
        else fit_kmeans(
            weighted_vectors(bundle.x_fit, weights, bundle.column_map),
            selected["k"],
            experiment.kmeans_random_state,
            experiment.kmeans_n_init,
        ).labels,
        n_permutations=experiment.n_permutations,
        seed=experiment.seed,
        k=selected["k"],
    )
    (output_dir / "permutation_reference.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")
    return {"library_notes": notes}


def _route_b(bundle, experiment, output_dir: Path, notes: list[str]) -> None:
    sample = min(180, len(bundle.raw_train))
    frame = bundle.raw_train.iloc[:sample]
    try:
        components = gower_component_matrices(frame, bundle.types)
        weights = equal_weights(len(bundle.feature_names))
        distance = gower_distance(components, weights, bundle.feature_names)
        try:
            from kmedoids import fasterpam

            labels = np.asarray(fasterpam(distance, experiment.force_k, random_state=experiment.seed).labels)
            backend = "kmedoids.fasterpam"
        except Exception as exc:
            labels = kmedoids_alternate(distance, experiment.force_k, experiment.seed)
            backend = f"alternating-k-medoids ({exc})"
            notes.append(f"Ruta B usa el plan B de K-Medoids: {exc}")
        pd.DataFrame(
            [{"representation": "gower", "k": experiment.force_k, "backend": backend, "n": sample}]
        ).to_csv(output_dir / "route_b.csv", index=False)
        notes.append("La distancia de Gower no se acelera con FAISS. Su Silhouette no se compara con el de la ruta A.")
        _ = labels
    except Exception as exc:
        notes.append(f"Ruta B no se completó: {exc}")


def _faiss(bundle, weights, train_metrics, selected, experiment, output_dir: Path, notes: list[str]) -> None:
    vectors = weighted_vectors(bundle.x_train, weights, bundle.column_map)
    if not l2_matches_weighted(bundle.x_train[:40], weights, bundle.column_map):
        notes.append("La comprobación L2 de los vectores ponderados no coincidió.")
    index = FlatL2Index(vectors)
    if index.backend != "faiss":
        notes.append(f"FAISS no está disponible y se usa NumPy. Detalle: {index.faiss_error}")
    queries = weighted_vectors(bundle.x_val, weights, bundle.column_map)
    k_neighbors = min(experiment.knn_k, len(vectors))
    found = index.search(queries, k_neighbors)
    brute = index.search_numpy(queries, k_neighbors, dtype=np.float32 if index.backend == "faiss" else np.float64)
    kinds = classify_neighbor_rows(brute.indices, found.indices, brute.distances, found.distances)
    agree = kinds["exact"] + kinds["same_set"] + kinds["boundary_tie"]
    knn_labels = predict_knn(vectors, train_metrics["labels_train"], queries, k_neighbors)
    centroid_labels = predict_centroids(train_metrics["centroids"], queries)
    from sklearn.metrics import adjusted_rand_score

    agreement = float(adjusted_rand_score(centroid_labels, knn_labels))
    latency = benchmark_latency(index, queries[: min(50, len(queries))], k_neighbors, repeats=3)
    report = {
        "backend": index.backend,
        "neighbor_row_agreement": agree / max(len(queries), 1),
        "neighbor_exact": kinds["exact"],
        "neighbor_same_set": kinds["same_set"],
        "neighbor_boundary_tie": kinds["boundary_tie"],
        "neighbor_mismatch": kinds["mismatch"],
        "neighbor_note": (
            "La fuerza bruta de referencia está en float32, igual que FAISS. "
            "Un desacuerdo cuya distancia cae junto a la k-ésima se cuenta como empate."
        ),
        "centroid_vs_knn_ari": agreement,
        "latency": latency,
        "note": "Si el conjunto es pequeño, FAISS puede no ser más rápido que la fuerza bruta.",
    }
    (output_dir / "faiss_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    artifacts = output_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "transformer": bundle.transformer,
            "weights": weights,
            "feature_names": bundle.feature_names,
            "k": selected["k"],
            "centroids": train_metrics["centroids"],
            "column_map": bundle.column_map,
        },
        artifacts / "model.joblib",
    )
    np.savez(artifacts / "weighted_train.npz", vectors=vectors, source_row=bundle.source_rows_train)
    if index.backend == "faiss":
        import faiss

        faiss.write_index(index._index, str(artifacts / "index.faiss"))
    (artifacts / "id_map.json").write_text(
        json.dumps({"source_row": [int(v) for v in bundle.source_rows_train]}),
        encoding="utf-8",
    )


def _write_environment(output_dir: Path, schema: SchemaConfig, experiment: ExperimentConfig) -> None:
    import sys

    lines = [
        f"python={sys.version}",
        f"platform={platform.platform()}",
        f"plena={__version__}",
    ]
    try:
        from importlib.metadata import version

        for package in ("numpy", "pandas", "scikit-learn", "scipy", "faiss-cpu", "PyYAML", "matplotlib"):
            try:
                lines.append(f"{package}={version(package)}")
            except Exception:
                lines.append(f"{package}=missing")
    except Exception as exc:
        lines.append(str(exc))
    (output_dir / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if schema.source_path:
        shutil.copy(schema.source_path, output_dir / "schema.used.yaml")
    if experiment.source_path:
        shutil.copy(experiment.source_path, output_dir / "experiment.used.yaml")


def informative_weight_fraction(weights: np.ndarray, names: list[str], informative: list[str]) -> float:
    index = {name: pos for pos, name in enumerate(names)}
    total = float(np.sum(weights))
    if total <= 0:
        return 0.0
    return float(sum(weights[index[name]] for name in informative) / total)


def planted_signal_report(
    schema: SchemaConfig,
    experiment: ExperimentConfig,
    k: int | None = None,
) -> dict:
    """Corre V1 y mide cuánto peso cae en las variables informativas."""
    set_seed(experiment.seed)
    bundle = prepare_dataset(schema, experiment)
    k = experiment.force_k if k is None else k
    from sklearn.metrics import adjusted_rand_score

    fractions = []
    before_caps = []
    generative_aris = []
    for offset in range(experiment.n_seeds_ga):
        seed = experiment.seed + offset
        evaluator = FitnessEvaluator(
            bundle.x_fit, bundle.y_fit, bundle.column_map, experiment, k=k, variant="v1"
        )
        result = run_ga(evaluator, len(bundle.feature_names), experiment.ga, seed)
        if not result.best.valid or result.best.weights is None:
            fractions.append(0.0)
            before_caps.append(None)
            generative_aris.append(float("nan"))
            continue
        fractions.append(
            informative_weight_fraction(result.best.weights, bundle.feature_names, schema.informative_variables)
        )
        before = result.best.weights_before_cap
        before_caps.append(None if before is None else before.tolist())
        if bundle.generative_val is None:
            generative_aris.append(float("nan"))
        else:
            metrics = score_weights(bundle, result.best.weights, k, experiment, "val")
            generative_aris.append(float(adjusted_rand_score(bundle.generative_val, metrics["labels_eval"])))
    chance = len(schema.informative_variables) / max(len(bundle.feature_names), 1)
    fractions_array = np.asarray(fractions, dtype=float)
    above = int(np.sum(fractions_array > chance))
    return {
        "fractions": fractions,
        "chance": chance,
        "n_features": len(bundle.feature_names),
        "seeds_above_chance": above,
        "mean_fraction": float(np.mean(fractions_array)),
        "ari_generative_val": generative_aris,
        "weights_before_cap": before_caps,
        "passed": bool(above > len(fractions) / 2 and float(np.mean(fractions_array)) > chance),
    }


def repeat_histories_match(schema: SchemaConfig, experiment: ExperimentConfig) -> bool:
    """Dos corridas con la misma semilla comparten historial; otra semilla no."""
    bundle = prepare_dataset(schema, experiment)
    k = experiment.force_k

    def once(seed: int):
        evaluator = FitnessEvaluator(
            bundle.x_fit, bundle.y_fit, bundle.column_map, experiment, k=k, variant="v1"
        )
        return run_ga(evaluator, len(bundle.feature_names), experiment.ga, seed).history

    first = once(experiment.seed)
    second = once(experiment.seed)
    third = once(experiment.seed + 1)
    return first == second and first != third
