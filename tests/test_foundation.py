"""Pruebas de semilla, nombres, esquema y particiones."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from plena.config import load_experiment, load_schema, validate_schema
from plena.data import load_table, prevalence, split_indices
from plena.names import normalize_column_name
from plena.paths import project_root
from plena.seed import set_seed


def test_set_seed_returns_generator():
    first = set_seed(7)
    second = set_seed(7)
    assert first.random() == second.random()


def test_altered_column_names(tmp_path: Path):
    assert normalize_column_name("*Age") == "age"
    assert normalize_column_name("*&#x4D;arried") == "married"
    csv_path = tmp_path / "names.csv"
    csv_path.write_text("*Age,*&#x4D;arried,depressed\n1,0,0\n", encoding="utf-8")
    frame, mapping = load_table(csv_path)
    assert list(frame.columns) == ["age", "married", "depressed"]
    assert mapping["*Age"] == "age"
    assert mapping["*&#x4D;arried"] == "married"


def test_schema_and_experiment_load():
    root = project_root()
    schema = load_schema(root / "config" / "schema.yaml")
    experiment = load_experiment(root / "config" / "experiment.yaml")
    assert schema.target == "depressed"
    assert "depressed" not in schema.feature_names
    assert experiment.force_k == 3
    assert experiment.k_values == [2, 3, 4, 5]


def test_schema_rejects_overlap(tmp_path: Path):
    payload = {
        "dataset_path": "data.csv",
        "target": "depressed",
        "id_columns": ["survey_id"],
        "group_column": None,
        "features": {"depressed": {"type": "binary"}, "age": {"type": "continuous"}},
        "exclusions": [],
    }
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_schema(path)


def test_schema_rejects_unconfirmed(tmp_path: Path):
    payload = {
        "dataset_path": "data.csv",
        "target": "depressed",
        "id_columns": [],
        "features": {"age": {"type": "continuous", "status": "UNCONFIRMED"}},
        "exclusions": [],
    }
    path = tmp_path / "draft.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_schema(path)
    loaded = load_schema(path, allow_unconfirmed=True)
    validate_schema(loaded, allow_unconfirmed=True)


def test_splits_are_reproducible_and_disjoint():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=600)
    first = split_indices(y, train=0.6, val=0.2, test=0.2, seed=3)
    second = split_indices(y, train=0.6, val=0.2, test=0.2, seed=3)
    for name in ("train", "val", "test"):
        assert np.array_equal(first[name], second[name])
    union = np.concatenate([first["train"], first["val"], first["test"]])
    assert len(union) == len(np.unique(union)) == 600
    rates = [prevalence(y[first[name]]) for name in ("train", "val", "test")]
    assert max(rates) - min(rates) < 0.08


def test_grouped_split_keeps_groups_together():
    y = np.array([0, 0, 1, 1, 0, 1] * 40)
    groups = np.repeat(np.arange(len(y) // 2), 2)
    parts = split_indices(y, train=0.6, val=0.2, test=0.2, seed=1, groups=groups)
    for name in ("train", "val", "test"):
        seen = groups[parts[name]]
        for group_id in np.unique(seen):
            rows = np.flatnonzero(groups == group_id)
            assert set(rows).issubset(set(parts[name]))
