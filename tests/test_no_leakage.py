"""Ninguna columna prohibida entra en la matriz de características."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from plena.config import load_schema
from plena.data import banned_columns, feature_frame


def test_target_cannot_be_a_feature(tmp_path: Path):
    payload = {
        "dataset_path": "unused.csv",
        "target": "depressed",
        "id_columns": ["survey_id"],
        "group_column": None,
        "features": {
            "age": {"type": "continuous"},
            "depressed": {"type": "binary"},
        },
        "exclusions": [{"column": "survey_id", "reason": "identificador"}],
    }
    path = tmp_path / "schema.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_schema(path)


def test_feature_frame_rejects_banned_name(tmp_path: Path):
    payload = {
        "dataset_path": "unused.csv",
        "target": "depressed",
        "id_columns": ["survey_id"],
        "features": {"age": {"type": "continuous"}},
        "exclusions": [{"column": "survey_id", "reason": "identificador"}],
    }
    path = tmp_path / "schema.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    schema = load_schema(path)
    frame = pd.DataFrame(
        {"survey_id": [1, 2], "age": [30, 40], "depressed": [0, 1]}
    )
    banned = banned_columns(
        schema.target,
        schema.id_columns,
        [item.column for item in schema.exclusions],
    )
    features = feature_frame(frame, schema.feature_names, banned)
    assert "depressed" not in features.columns
    assert "survey_id" not in features.columns
    assert list(features.columns) == ["age"]
