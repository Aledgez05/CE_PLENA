"""El estudio completo, en miniatura, no toca test para decidir."""

import json
from pathlib import Path

import yaml

from plena.config import ExperimentConfig, GAConfig, SplitConfig, load_experiment, load_schema
from plena.experiments import run_study
from plena.synthetic import generate_synthetic


def test_tiny_study_touches_test_once(tmp_path: Path):
    frame, informative = generate_synthetic(n=120, seed=3)
    csv_path = tmp_path / "planted.csv"
    frame.drop(columns=["generative_cluster"]).to_csv(csv_path, index=False)
    schema_payload = {
        "dataset_path": str(csv_path),
        "target": "depressed",
        "id_columns": ["survey_id"],
        "group_column": None,
        "informative_variables": informative,
        "ablation_groups": {"financial": ["noise_huge"]},
        "features": {
            "signal_x": {"type": "continuous"},
            "signal_y": {"type": "continuous"},
            "signal_flag": {"type": "binary"},
            "noise_c0": {"type": "continuous"},
            "noise_b0": {"type": "binary"},
            "noise_nominal": {"type": "nominal"},
            "noise_huge": {"type": "continuous", "scaler": "standard"},
        },
        "exclusions": [{"column": "survey_id", "reason": "identificador"}],
    }
    schema_path = tmp_path / "schema.yaml"
    experiment_path = tmp_path / "experiment.yaml"
    schema_path.write_text(yaml.safe_dump(schema_payload), encoding="utf-8")
    experiment = ExperimentConfig(
        seed=3,
        n_seeds_ga=2,
        n_random_runs=2,
        k_values=[3],
        force_k=3,
        fitness_sample_size=60,
        split=SplitConfig(0.6, 0.2, 0.2),
        ga=GAConfig(population=6, generations=2, elitism=1, stagnation_patience=5, min_active=3, w_max=0.4),
        kmeans_n_init=2,
        n_permutations=5,
        knn_k=3,
    )
    experiment.source_path = experiment_path
    experiment_path.write_text("seed: 3\n", encoding="utf-8")
    schema = load_schema(schema_path)
    output = tmp_path / "results"
    result = run_study(schema, experiment, output)
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["test_used_for_selection"] is False
    frame_out = result["frame"]
    test_rows = frame_out[frame_out["split"] == "test"]
    assert set(test_rows["method"]) <= {"ga", "equal_weights", "random_search"}
    assert (output / "comparison.csv").exists()
    assert (output / "comparison.md").exists()
    assert (output / "cost_estimate.csv").exists()
    assert (output / "artifacts" / "model.joblib").exists()
    assert "mejora" in (output / "comparison.md").read_text(encoding="utf-8") or "empata" in (
        output / "comparison.md"
    ).read_text(encoding="utf-8") or "empeora" in (output / "comparison.md").read_text(encoding="utf-8")
    _ = load_experiment
