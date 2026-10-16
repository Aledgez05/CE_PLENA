"""Agrupamiento, métricas, aptitud y operadores."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from plena.chromosome import decode
from plena.clustering import fit_kmeans, predict_centroids, validate_k
from plena.config import ExperimentConfig, GAConfig
from plena.distance import ColumnMap
from plena.evaluation import classification_from_clusters, external_metrics, internal_metrics
from plena.fitness import FitnessEvaluator
from plena.genetic_algorithm import blend_crossover, gaussian_mutation, next_generation, run_ga
import pytest


def test_k_bounds():
    with pytest.raises(ValueError):
        validate_k(1, 20)
    with pytest.raises(ValueError):
        validate_k(20, 20)


def test_predict_train_matches_fit():
    matrix, _ = make_blobs(n_samples=80, centers=3, random_state=0, cluster_std=0.4)
    model = fit_kmeans(matrix, 3, random_state=0, n_init=5)
    again = predict_centroids(model.centroids, matrix)
    assert np.array_equal(model.labels, again)


def test_internal_metrics_do_not_need_labels():
    matrix, labels = make_blobs(n_samples=60, centers=3, random_state=1, cluster_std=0.3)
    metrics = internal_metrics(matrix, labels)
    assert metrics["silhouette"] > 0.5
    assert np.isfinite(metrics["davies_bouldin"])


def test_external_metrics_known_partition():
    y_true = np.array([0, 0, 1, 1])
    labels = np.array([0, 0, 1, 1])
    metrics = external_metrics(y_true, labels)
    assert abs(metrics["ari"] - 1) < 1e-8
    assert abs(metrics["ri"] - 1) < 1e-8
    classification = classification_from_clusters(y_true, labels, y_true, labels)
    assert abs(classification["balanced_accuracy"] - 1) < 1e-8


def _column_map(dimension: int) -> ColumnMap:
    names = [f"v{i}" for i in range(dimension)]
    return ColumnMap(names, names, names)


def _experiment() -> ExperimentConfig:
    experiment = ExperimentConfig()
    experiment.ga = GAConfig(population=8, generations=3, elitism=2, min_active=3, w_max=0.4)
    experiment.kmeans_n_init = 2
    experiment.penalties.neff_min = 2
    return experiment


def test_children_stay_in_bounds_and_elites_are_intact():
    rng = np.random.default_rng(0)
    population = rng.uniform(-1, 1, size=(8, 5))
    scores = rng.normal(size=8)
    config = _experiment().ga
    nxt = next_generation(population, scores, config, rng)
    assert np.all(nxt >= -1) and np.all(nxt <= 1)
    order = np.argsort(-scores, kind="mergesort")
    assert np.array_equal(nxt[0], population[order[0]])
    assert np.array_equal(nxt[1], population[order[1]])
    child_a, child_b = blend_crossover(population[0], population[1], np.random.default_rng(1))
    mutated = gaussian_mutation(child_a, np.random.default_rng(2), sigma=0.5, rate=1)
    assert np.all(mutated >= -1) and np.all(mutated <= 1)
    assert np.all(child_b >= -1)


def test_same_chromosome_same_fitness_and_penalties():
    matrix, labels = make_blobs(n_samples=70, centers=3, n_features=4, random_state=2, cluster_std=0.5)
    y = (labels == 0).astype(int)
    column_map = _column_map(matrix.shape[1])
    experiment = _experiment()
    evaluator = FitnessEvaluator(matrix, y, column_map, experiment, k=3, variant="v1")
    genes = np.array([0.8, -0.5, 0.4, 0.2])
    first = evaluator(genes)
    second = evaluator(genes)
    assert first.valid
    assert first.score == second.score
    assert evaluator.cache_hits == 1
    detail = evaluator(np.array([-1.0, -1.0, -1.0, -1.0]))
    assert np.isfinite(detail.score)
    experiment.penalties.neff_min = 100
    penalized = FitnessEvaluator(matrix, y, column_map, experiment, k=3, variant="v1")
    broad = penalized(np.array([0.2, 0.2, 0.2, 0.2]))
    assert broad.penalties["neff"] > 0


def test_invalid_solution_does_not_raise():
    matrix = np.random.default_rng(0).normal(size=(12, 4))
    experiment = _experiment()
    evaluator = FitnessEvaluator(
        matrix,
        np.array([0, 1] * 6),
        _column_map(4),
        experiment,
        k=3,
        variant="v2",
    )
    detail = evaluator(np.array([np.nan, np.inf, -np.inf, 0.2]))
    assert np.isfinite(detail.score)


def test_ga_is_reproducible_on_blobs():
    matrix, labels = make_blobs(n_samples=50, centers=3, n_features=4, random_state=0, cluster_std=0.4)
    y = labels
    column_map = _column_map(matrix.shape[1])
    experiment = _experiment()
    experiment.ga.population = 6
    experiment.ga.generations = 2
    experiment.ga.stagnation_patience = 5

    def once(seed: int):
        evaluator = FitnessEvaluator(matrix, y, column_map, experiment, k=3, variant="v1")
        return run_ga(evaluator, matrix.shape[1], experiment.ga, seed)

    assert once(1).history == once(1).history
    assert once(1).history != once(2).history


def test_small_cluster_penalty_is_finite():
    # Dos puntos muy separados y el resto encimado: puede salir un cluster chico.
    matrix = np.vstack([np.zeros((20, 2)), np.ones((2, 2)) * 50])
    y = np.array([0] * 20 + [1] * 2)
    experiment = _experiment()
    experiment.penalties.min_cluster_frac = 0.2
    evaluator = FitnessEvaluator(matrix, y, _column_map(2), experiment, k=2, variant="v1")
    detail = evaluator(np.array([0.9, 0.9]))
    assert np.isfinite(detail.score)
    if detail.valid:
        assert "small_cluster" in detail.penalties
