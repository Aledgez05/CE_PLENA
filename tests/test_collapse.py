"""La penalización de colapso castiga el catálogo de celdas y no usa la etiqueta."""

import numpy as np

from plena.collapse import collapse_report, n_unique_rows
from plena.collapse_probe import MARGIN, decide, is_eligible
from plena.config import ExperimentConfig, GAConfig
from plena.distance import ColumnMap
from plena.fitness import FitnessEvaluator


def _corners(copies: int = 12) -> np.ndarray:
    corners = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [4.0, 4.0]])
    return np.repeat(corners, copies, axis=0)


def _labels_isolated(copies: int = 12) -> np.ndarray:
    return np.repeat(np.arange(4), copies)


def _labels_merged(copies: int = 12) -> np.ndarray:
    return np.array([0] * (2 * copies) + [1] * (2 * copies))


def test_unique_rows_ignore_float_dust():
    matrix = _corners()
    dusty = matrix + 1e-8
    assert n_unique_rows(matrix) == 4
    assert n_unique_rows(dusty) == 4


def test_exact_enumeration_is_collapsed_and_dust_does_not_save_it():
    clean = collapse_report(_corners(), _labels_isolated())
    dusty = collapse_report(_corners() + 1e-8, _labels_isolated())
    assert clean["n_unique"] == 4
    assert clean["enumerated_fraction"] == 1
    assert clean["near_collapse"] > 0.95
    assert dusty["near_collapse"] > 0.95


def test_merging_patterns_drops_the_penalty():
    merged = collapse_report(_corners(), _labels_merged())
    isolated = collapse_report(_corners(), _labels_isolated())
    assert merged["near_collapse"] < 0.1
    assert isolated["near_collapse"] - merged["near_collapse"] > 0.8
    enumerated_score = 1.0 - isolated["near_collapse"]
    merged_score = 0.55 - merged["near_collapse"]
    assert merged_score > enumerated_score


def test_v1_penalty_uses_geometry_and_ignores_the_label():
    matrix = _corners()
    names = ["x", "y"]
    column_map = ColumnMap(names, names, names)
    experiment = ExperimentConfig()
    experiment.ga = GAConfig(population=6, generations=2, elitism=1, min_active=2, w_max=0.6)
    experiment.kmeans_n_init = 5
    experiment.kmeans_random_state = 0
    experiment.penalties.neff_min = 1
    experiment.penalties.collapse_coef = 1.0
    y = np.zeros(len(matrix), dtype=int)
    y[: len(y) // 2] = 1
    evaluator = FitnessEvaluator(matrix, y, column_map, experiment, k=4, variant="v1")
    genes = np.array([0.8, 0.8])
    found = evaluator(genes)
    shuffled = FitnessEvaluator(
        matrix, np.random.default_rng(0).permutation(y), column_map, experiment, k=4, variant="v1"
    )(genes)
    assert found.valid
    assert found.ari_train is None
    assert found.penalties["near_collapse"] > 0.95
    assert abs(found.score - shuffled.score) < 1e-12
    plain = ExperimentConfig()
    plain.ga = experiment.ga
    plain.kmeans_n_init = experiment.kmeans_n_init
    plain.kmeans_random_state = experiment.kmeans_random_state
    plain.penalties.neff_min = 1
    plain.penalties.collapse_coef = 0.0
    unpenalized = FitnessEvaluator(matrix, y, column_map, plain, k=4, variant="v1")(genes)
    assert unpenalized.score > found.score
    assert abs((unpenalized.score - found.score) - found.penalties["collapse"]) < 1e-9


def test_decision_rejects_a_lucky_seed_and_does_not_claim_a_general_effect():
    rows = []
    for seed in range(5):
        rows.append(
            {
                "k": 3,
                "seed": seed,
                "ari": 0.04 if seed == 0 else -0.02,
                "n_unique": 40,
                "near_collapse": 0.1,
                "silhouette": 0.4,
                "n_eff": 4,
                "n_active": 6,
            }
        )
    outcome = decide(rows, {3: 0.0})
    assert outcome["status"] == "rechazado"
    assert outcome["statistical_claim"] is False
    assert MARGIN == 0.02


def test_decision_accepts_only_a_descriptive_margin():
    rows = []
    for seed in range(5):
        rows.append(
            {
                "k": 2,
                "seed": seed,
                "ari": 0.05,
                "n_unique": 30,
                "near_collapse": 0.2,
                "silhouette": 0.45,
                "n_eff": 5.0 if seed else 4.0,
                "n_active": 6,
            }
        )
    outcome = decide(rows, {2: 0.0})
    assert outcome["status"] == "apoyo_descriptivo"
    assert outcome["delta_mean_ari"] >= MARGIN
    assert outcome["statistical_claim"] is False
    assert is_eligible(rows[0])
