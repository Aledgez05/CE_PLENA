"""Auditoría sobre una tabla mínima y señal plantada."""

from pathlib import Path

import numpy as np
import pandas as pd

from plena.audit import run_audit
from plena.config import GAConfig, PenaltyConfig, load_schema
from plena.experiments import planted_signal_report
from plena.paths import project_root
from plena.synthetic import generate_synthetic, write_synthetic_bundle


def test_audit_writes_pending_section(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "survey_id": np.arange(30),
            "age": np.linspace(18, 60, 30),
            "ville_id": [1, 1, 2] * 10,
            "income": [1e7, -1, 99] * 10,
            "depressed": [0, 1, 0] * 10,
        }
    )
    output = tmp_path / "audit"
    report = run_audit(frame, "depressed", output, {"Age": "age"}, dataset_path="toy.csv", draft_path=tmp_path / "schema.draft.yaml")
    text = (output / "data_audit.md").read_text(encoding="utf-8")
    assert "Hallazgos y decisiones pendientes" in text
    assert (output / "figures" / "missing.png").exists()
    assert (output / "figures" / "prevalence.png").exists()
    draft = (tmp_path / "schema.draft.yaml").read_text(encoding="utf-8")
    assert "UNCONFIRMED" in draft
    assert report["label"]["semantics"] == "UNCONFIRMED"


def test_synthetic_is_reproducible(tmp_path: Path):
    left, informative = generate_synthetic(n=100, seed=5)
    right, _ = generate_synthetic(n=100, seed=5)
    pd.testing.assert_frame_equal(left, right)
    assert informative == ["signal_x", "signal_y", "signal_flag"]
    write_synthetic_bundle(tmp_path, n=50, seed=1)
    assert (tmp_path / "data" / "synthetic" / "planted.csv").exists()


def test_planted_signal_recovers_informative_variables():
    root = project_root()
    write_synthetic_bundle(root, n=420, seed=42)
    schema = load_schema(root / "config" / "schema.synthetic.yaml")
    from plena.config import ExperimentConfig

    experiment = ExperimentConfig(
        seed=42,
        n_seeds_ga=5,
        n_random_runs=2,
        k_values=[3],
        force_k=3,
        fitness_sample_size=250,
        ga=GAConfig(
            population=28,
            generations=14,
            elitism=2,
            tournament=3,
            mutation_sigma=0.12,
            mutation_rate=0.25,
            w_max=0.5,
            min_active=3,
            lambda_active=0.01,
            stagnation_patience=8,
            blx_alpha=0.3,
            crossover_rate=0.85,
        ),
        kmeans_n_init=3,
        penalties=PenaltyConfig(min_cluster_frac=0.05, neff_min=2.0, neff_coef=0.02),
    )
    report = planted_signal_report(schema, experiment)
    assert report["chance"] == len(schema.informative_variables) / report["n_features"]
    assert report["passed"], report
    finite_ari = [value for value in report["ari_generative_val"] if value == value]
    assert finite_ari
    assert max(finite_ari) > 0.0
