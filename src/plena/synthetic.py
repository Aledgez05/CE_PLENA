"""Dataset sintético con clusters definidos por un subconjunto conocido de variables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from plena.seed import set_seed

INFORMATIVE = ["signal_x", "signal_y", "signal_flag"]
NOISE_CONTINUOUS = [f"noise_c{index}" for index in range(12)]
NOISE_BINARY = ["noise_b0", "noise_b1"]
NOISE_DISCRETE = ["noise_d0", "noise_d1"]
NOISE_OTHER = ["noise_nominal", "noise_huge"]


def generate_synthetic(
    n: int = 800,
    seed: int = 42,
    k: int = 3,
    missing_rate: float = 0.05,
) -> tuple[pd.DataFrame, list[str]]:
    """Genera D>=20 variables. Solo tres llevan una señal débil.

    ``depressed`` se asocia de forma incompleta con el cluster generativo y
    no es una variable de agrupamiento. ``generative_cluster`` se conserva
    para medir ARI y queda fuera de X.
    """
    if k != 3:
        raise ValueError("Este generador planta tres clusters.")
    rng = set_seed(seed)
    centers = np.array([[0.0, 0.0], [2.4, 0.4], [4.8, 1.3]])
    cluster = rng.integers(0, k, size=n)
    signal = centers[cluster] + rng.normal(0.0, 0.55, size=(n, 2))
    flag_probability = np.array([0.15, 0.85, 0.4])[cluster]
    signal_flag = (rng.random(n) < flag_probability).astype(int)
    columns: dict[str, object] = {
        "survey_id": np.arange(n),
        "signal_x": signal[:, 0],
        "signal_y": signal[:, 1],
        "signal_flag": signal_flag,
    }
    for name in NOISE_CONTINUOUS:
        values = rng.normal(0.0, 1.0, size=n).astype(float)
        values[rng.random(n) < missing_rate] = np.nan
        columns[name] = values
    for name in NOISE_BINARY:
        columns[name] = rng.integers(0, 2, size=n)
    for name in NOISE_DISCRETE:
        columns[name] = rng.integers(0, 8, size=n)
    columns["noise_nominal"] = [f"nivel_{level}" for level in rng.integers(0, 25, size=n)]
    columns["noise_huge"] = rng.normal(5.0e7, 2.0e6, size=n)
    depressed = (cluster == 2).astype(int)
    flip = rng.random(n) < 0.35
    depressed[flip] = 1 - depressed[flip]
    columns["depressed"] = depressed
    columns["generative_cluster"] = cluster
    return pd.DataFrame(columns), list(INFORMATIVE)


def schema_document(dataset_path: str) -> dict:
    features = {
        "signal_x": {"type": "continuous", "status": "confirmed", "justification": "Señal plantada, continua."},
        "signal_y": {"type": "continuous", "status": "confirmed", "justification": "Señal plantada, continua."},
        "signal_flag": {"type": "binary", "status": "confirmed", "justification": "Señal plantada, binaria."},
    }
    for name in NOISE_CONTINUOUS:
        features[name] = {
            "type": "continuous",
            "status": "confirmed",
            "justification": "Ruido continuo.",
        }
    for name in NOISE_BINARY:
        features[name] = {"type": "binary", "status": "confirmed", "justification": "Ruido binario."}
    for name in NOISE_DISCRETE:
        features[name] = {
            "type": "discrete",
            "status": "confirmed",
            "justification": "Ruido discreto, tratado como numérico.",
        }
    features["noise_nominal"] = {
        "type": "nominal",
        "status": "confirmed",
        "justification": "Nominal de 25 niveles, ruido.",
    }
    features["noise_huge"] = {
        "type": "continuous",
        "scaler": "standard",
        "status": "confirmed",
        "justification": "Ruido de escala enorme; StandardScaler, sin RobustScaler.",
    }
    return {
        "dataset_path": dataset_path,
        "target": "depressed",
        "id_columns": ["survey_id"],
        "group_column": None,
        "informative_variables": list(INFORMATIVE),
        "sensitive_columns": [],
        "ablation_groups": {"financial": ["noise_huge"], "geographic": []},
        "features": features,
        "exclusions": [
            {"column": "survey_id", "reason": "identificador", "status": "confirmed"},
            {
                "column": "generative_cluster",
                "reason": "cluster generativo; usarlo sería fuga de la estructura plantada",
                "status": "confirmed",
            },
        ],
    }


def write_synthetic_bundle(root: Path, n: int = 800, seed: int = 42) -> Path:
    """Guarda el CSV, el esquema y la lista de variables informativas."""
    destination = root / "data" / "synthetic"
    destination.mkdir(parents=True, exist_ok=True)
    frame, informative = generate_synthetic(n=n, seed=seed)
    csv_path = destination / "planted.csv"
    frame.to_csv(csv_path, index=False)
    (destination / "informative_variables.json").write_text(
        json.dumps(
            {
                "informative_variables": informative,
                "n_features": len(frame.columns) - 3,
                "seed": seed,
                "n": n,
                "signal": "débil",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    schema = schema_document("data/synthetic/planted.csv")
    schema_path = root / "config" / "schema.synthetic.yaml"
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(yaml.safe_dump(schema, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return csv_path


if __name__ == "__main__":
    from plena.paths import project_root

    write_synthetic_bundle(project_root())
    print("Dataset sintético escrito en data/synthetic")
