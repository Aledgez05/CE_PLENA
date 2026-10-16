"""Punto de entrada de los experimentos reproducibles."""

from __future__ import annotations

import argparse
from pathlib import Path

from plena.config import load_experiment, load_schema
from plena.experiments import planted_signal_report, run_study
from plena.logging_utils import setup_logging
from plena.paths import project_root
from plena.synthetic import write_synthetic_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimentos de PLENA")
    parser.add_argument("--schema", default="config/schema.yaml")
    parser.add_argument("--experiment", default="config/experiment.synthetic.yaml")
    parser.add_argument("--output", default="results")
    parser.add_argument("--generate-synthetic", action="store_true")
    parser.add_argument("--planted-only", action="store_true")
    args = parser.parse_args()
    setup_logging()
    root = project_root()
    if args.generate_synthetic:
        write_synthetic_bundle(root)
    schema = load_schema(root / args.schema)
    experiment = load_experiment(root / args.experiment)
    output = root / args.output
    if args.planted_only:
        report = planted_signal_report(schema, experiment)
        print(report)
        if not report["passed"]:
            raise SystemExit("La prueba de señal plantada no cumplió el criterio del 70%.")
        return
    run_study(schema, experiment, output)
    print(f"Resultados en {output}")


if __name__ == "__main__":
    main()
