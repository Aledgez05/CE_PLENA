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
from plena.baselines import equal_weights, permutation_reference, random_search
from plena.chromosome import decode, n_eff
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
from plena.faiss_index import FlatL2Index, benchmark_latency, l2_matches_weighted, weighted_vectors
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
    types: dict[str, str]
    splits: dict[str, np.ndarray]
    source_rows_train: np.ndarray


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
        types={name: spec.type for name, spec in schema.features.items()},
        splits=splits,
        source_rows_train=frame.iloc[splits["train"]]["source_row"].to_numpy(),
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


def run_study(schema: SchemaConfig, experiment: ExperimentConfig, output_dir: Path) -> dict:
    """Corre la comparación. Test entra solo en ``_evaluate_test``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = prepare_dataset(schema, experiment)
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
            "note": "Índices sobre las filas que conservan etiqueta. Test no entra en la búsqueda.",
        },
    )
    _write_cost(bundle, experiment, output_dir / "cost_estimate.csv")
    records = []
    histories = []
    stored = []
    configured_budget = experiment.ga.population * experiment.ga.generations
    for k in experiment.k_values:
        for variant in ("v1", "v2"):
            weights = equal_weights(len(bundle.feature_names))
            metrics = score_weights(bundle, weights, k, experiment, "val")
            records.append(_row("equal_weights", variant, k, experiment.seed, "val", metrics, 0.0))
            for offset in range(experiment.n_seeds_ga):
                seed = experiment.seed + offset
                evaluator = FitnessEvaluator(
                    bundle.x_fit, bundle.y_fit, bundle.column_map, experiment, k=k, variant=variant
                )
                started = time.perf_counter()
                ga_result = run_ga(evaluator, len(bundle.feature_names), experiment.ga, seed)
                elapsed = time.perf_counter() - started
                if not ga_result.best.valid or ga_result.best.weights is None:
                    continue
                metrics = score_weights(bundle, ga_result.best.weights, k, experiment, "val")
                records.append(_row("ga", variant, k, seed, "val", metrics, elapsed))
                histories.append({"variant": variant, "k": k, "seed": seed, "history": ga_result.history})
                stored.append(
                    {
                        "method": "ga",
                        "variant": variant,
                        "k": k,
                        "seed": seed,
                        "weights": ga_result.best.weights.tolist(),
                        "mask": ga_result.best.mask.astype(int).tolist(),
                        "genes": ga_result.best_genes.tolist(),
                        "n_evaluations": ga_result.n_evaluations,
                        "val_silhouette": metrics["silhouette"],
                        "val_ari": metrics["ari"],
                        "n_active": metrics["n_active"],
                        "n_eff": metrics["n_eff"],
                    }
                )
            for offset in range(experiment.n_random_runs):
                seed = experiment.seed + offset
                evaluator = FitnessEvaluator(
                    bundle.x_fit, bundle.y_fit, bundle.column_map, experiment, k=k, variant=variant
                )
                rng = np.random.default_rng(seed + 1000 + k)
                started = time.perf_counter()
                searched = random_search(
                    evaluator,
                    len(bundle.feature_names),
                    configured_budget,
                    rng,
                    seed,
                )
                elapsed = time.perf_counter() - started
                if not searched.best.valid or searched.best.weights is None:
                    continue
                metrics = score_weights(bundle, searched.best.weights, k, experiment, "val")
                records.append(_row("random_search", variant, k, seed, "val", metrics, elapsed))
                stored.append(
                    {
                        "method": "random_search",
                        "variant": variant,
                        "k": k,
                        "seed": seed,
                        "weights": searched.best.weights.tolist(),
                        "n_evaluations": searched.n_evaluations,
                        "val_silhouette": metrics["silhouette"],
                        "val_ari": metrics["ari"],
                        "n_active": metrics["n_active"],
                        "n_eff": metrics["n_eff"],
                    }
                )
    frame = pd.DataFrame(records)
    chosen = _select(stored, experiment)
    test_rows, test_details = _evaluate_test(bundle, experiment, chosen, stored)
    frame = pd.concat([frame, pd.DataFrame(test_rows)], ignore_index=True)
    tests = _wilcoxon(frame)
    _write_tables(output_dir, frame, tests, histories, chosen, bundle, schema, experiment, test_details)
    extras = _robustness_and_inference(bundle, schema, experiment, chosen, stored, output_dir)
    _write_environment(output_dir, schema, experiment)
    manifest = {
        "test_used_for_selection": False,
        "selection_note": SELECTION_NOTE,
        "chosen": chosen,
        "library_notes": extras.get("library_notes", []),
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
    agree = 0
    for row in range(len(queries)):
        if set(found.indices[row]) != set(brute.indices[row]):
            continue
        agree += 1
    knn_labels = predict_knn(vectors, train_metrics["labels_train"], queries, k_neighbors)
    centroid_labels = predict_centroids(train_metrics["centroids"], queries)
    from sklearn.metrics import adjusted_rand_score

    agreement = float(adjusted_rand_score(centroid_labels, knn_labels))
    latency = benchmark_latency(index, queries[: min(50, len(queries))], k_neighbors, repeats=3)
    report = {
        "backend": index.backend,
        "neighbor_row_agreement": agree / max(len(queries), 1),
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
    fractions = []
    for offset in range(experiment.n_seeds_ga):
        seed = experiment.seed + offset
        evaluator = FitnessEvaluator(
            bundle.x_fit, bundle.y_fit, bundle.column_map, experiment, k=k, variant="v1"
        )
        result = run_ga(evaluator, len(bundle.feature_names), experiment.ga, seed)
        if not result.best.valid or result.best.weights is None:
            fractions.append(0.0)
            continue
        fractions.append(
            informative_weight_fraction(result.best.weights, bundle.feature_names, schema.informative_variables)
        )
    majority = float(np.mean(np.asarray(fractions) >= 0.70))
    return {"fractions": fractions, "majority_at_least_70": majority, "passed": bool(majority > 0.5)}


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
