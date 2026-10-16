"""Dataset sintético con clusters definidos por un subconjunto conocido de variables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from plena.seed import set_seed

INFORMATIVE = ["signal_x", "signal_y", "signal_flag"]


def generate_synthetic(
    n: int = 800,
    seed: int = 42,
    k: int = 3,
    missing_rate: float = 0.05,
) -> tuple[pd.DataFrame, list[str]]:
    """Genera datos mixtos. Los clusters dependen solo de tres variables.

    El resto es ruido, incluida una variable de escala enorme y una nominal
    de cardinalidad alta. ``depressed`` se asocia con el cluster generativo,
    pero no es una variable de agrupamiento.
    """
    if k != 3:
        raise ValueError("Este generador planta tres clusters.")
    rng = set_seed(seed)
    centers = np.array([[0.0, 0.0], [0.35, 0.05], [9.0, 9.0]])
    cluster = rng.integers(0, k, size=n)
    signal = centers[cluster] + rng.normal(0.0, 0.2, size=(n, 2))
    flag_probability = np.array([0.02, 0.98, 0.45])[cluster]
    signal_flag = (rng.random(n) < flag_probability).astype(int)
    noise_gauss = rng.normal(0.0, 1.0, size=n)
    noise_extra = rng.normal(5.0, 2.0, size=n)
    noise_binary = rng.integers(0, 2, size=n)
    noise_nominal = [f"nivel_{level}" for level in rng.integers(0, 25, size=n)]
    noise_huge = rng.normal(5.0e7, 2.0e6, size=n)
    missing = rng.random(n) < missing_rate
    noise_gauss = noise_gauss.astype(float)
    noise_gauss[missing] = np.nan
    depressed = (cluster == 2).astype(int)
    flip = rng.random(n) < 0.08
    depressed[flip] = 1 - depressed[flip]
    frame = pd.DataFrame(
        {
            "survey_id": np.arange(n),
            "signal_x": signal[:, 0],
            "signal_y": signal[:, 1],
            "signal_flag": signal_flag,
            "noise_gauss": noise_gauss,
            "noise_extra": noise_extra,
            "noise_binary": noise_binary,
            "noise_nominal": noise_nominal,
            "noise_huge": noise_huge,
            "depressed": depressed,
            "generative_cluster": cluster,
        }
    )
    return frame, list(INFORMATIVE)


def schema_document(dataset_path: str) -> dict:
    features = {
        "signal_x": {"type": "continuous", "status": "confirmed", "justification": "Señal plantada, continua."},
        "signal_y": {"type": "continuous", "status": "confirmed", "justification": "Señal plantada, continua."},
        "signal_flag": {"type": "binary", "status": "confirmed", "justification": "Señal plantada, binaria."},
        "noise_gauss": {"type": "continuous", "status": "confirmed", "justification": "Ruido gaussiano con faltantes."},
        "noise_extra": {"type": "continuous", "status": "confirmed", "justification": "Ruido continuo."},
        "noise_binary": {"type": "binary", "status": "confirmed", "justification": "Ruido binario."},
        "noise_nominal": {"type": "nominal", "status": "confirmed", "justification": "Nominal de 25 niveles, ruido."},
        "noise_huge": {
            "type": "continuous",
            "scaler": "robust",
            "status": "confirmed",
            "justification": "Ruido de escala enorme; se escala con RobustScaler.",
        },
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
    frame.drop(columns=["generative_cluster"]).to_csv(csv_path, index=False)
    (destination / "informative_variables.json").write_text(
        json.dumps({"informative_variables": informative, "seed": seed, "n": n}, indent=2),
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
