"""Cromosoma, distancia ponderada, fuga y preprocesamiento."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from plena.chromosome import decode, equal_weight_vector, n_eff
from plena.config import SchemaConfig, load_schema
from plena.data import banned_columns, feature_frame
from plena.distance import ColumnMap, transform_weighted, weighted_squared_distances
from plena.preprocessing import build_preprocessor, fit_route_a


def test_decode_constraints():
    rng = np.random.default_rng(4)
    for _ in range(25):
        genes = rng.uniform(-1, 1, size=8)
        mask, weights = decode(genes, tau=0.0, w_max=0.3, min_active=3)
        assert len(weights) == 8
        assert np.isfinite(weights).all()
        assert abs(weights.sum() - 1) < 1e-6
        assert weights.max() <= 0.3 + 1e-6
        assert weights.min() >= -1e-12
        assert int(mask.sum()) >= 3
        assert np.all(weights[~mask] == 0)


def test_decode_activates_highest_genes_when_needed():
    genes = np.array([-0.8, -0.4, -0.2, 0.9, -0.7, -0.6])
    mask, _weights = decode(genes, tau=0.0, w_max=0.5, min_active=3)
    assert mask[3]
    assert int(mask.sum()) >= 3


def test_effective_number_of_equal_weights():
    weights = equal_weight_vector(4)
    assert abs(n_eff(weights) - 4) < 1e-8


def test_weighted_distance_matches_hand_calculation():
    matrix = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    column_map = ColumnMap(
        transformed_names=["a", "b0", "b1"],
        variable_of=["a", "b", "b"],
        variable_names=["a", "b"],
    )
    weights = np.array([0.25, 0.75])
    distances = weighted_squared_distances(matrix, matrix, weights, column_map)
    # w_a (0-1)^2 + (w_b / 2) * ((0-1)^2 + (1-0)^2) = 0.25 + 0.75
    assert abs(distances[0, 1] - 1.0) < 1e-8
    scaled = transform_weighted(matrix, weights, column_map)
    manual = np.sum((scaled[0] - scaled[1]) ** 2)
    assert abs(manual - 1.0) < 1e-8


def _tiny_schema(path: Path) -> SchemaConfig:
    payload = {
        "dataset_path": "unused.csv",
        "target": "depressed",
        "id_columns": ["survey_id"],
        "group_column": None,
        "features": {
            "age": {"type": "continuous"},
            "city": {"type": "nominal"},
            "huge": {"type": "continuous", "scaler": "robust"},
        },
        "exclusions": [{"column": "survey_id", "reason": "identificador"}],
    }
    file_path = path / "schema.yaml"
    file_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return load_schema(file_path)


def test_no_leakage(tmp_path: Path):
    schema = _tiny_schema(tmp_path)
    frame = pd.DataFrame(
        {
            "survey_id": [1, 2, 3, 4],
            "age": [20, 30, 40, 50],
            "city": ["a", "b", "a", "c"],
            "huge": [1e9, 1e9 + 5, 2e9, 3e9],
            "depressed": [0, 1, 0, 1],
        }
    )
    banned = banned_columns(schema.target, schema.id_columns, [item.column for item in schema.exclusions])
    features = feature_frame(frame, schema.feature_names, banned)
    assert "depressed" not in features.columns
    assert "survey_id" not in features.columns


def test_preprocessing_fits_on_train_only(tmp_path: Path):
    schema = _tiny_schema(tmp_path)
    train = pd.DataFrame(
        {
            "age": [10.0, 12.0, 11.0, 13.0, np.nan],
            "city": ["a", "b", "a", "b", "a"],
            "huge": [1.0e7, 1.2e7, 0.8e7, 1.1e7, 1.05e7],
        }
    )
    val = pd.DataFrame({"age": [1e6, -1e6], "city": ["z", "a"], "huge": [1.3e7, 0.7e7]})
    test = pd.DataFrame({"age": [11.0, 12.0], "city": ["b", "missing"], "huge": [900.0, 1100.0]})
    transformer, steps = build_preprocessor(schema, "robust")
    column_map, x_train, x_val, x_test = fit_route_a(
        transformer, steps, schema.feature_names, train, val, test
    )
    assert x_train.shape[1] == x_val.shape[1] == x_test.shape[1]
    assert set(column_map.block_sizes) == set(schema.feature_names)
    assert sum(column_map.block_sizes.values()) == x_train.shape[1]
    assert np.isfinite(x_train).all()
    assert np.isfinite(x_val).all()
    assert np.isfinite(x_test).all()
    age_step = transformer.named_transformers_["v0"].named_steps["imputer"]
    assert abs(float(age_step.statistics_[0]) - np.nanmedian([10, 12, 11, 13])) < 1e-8
    huge_column = [i for i, variable in enumerate(column_map.variable_of) if variable == "huge"][0]
    assert np.nanmax(np.abs(x_train[:, huge_column])) < 100
