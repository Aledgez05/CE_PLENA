"""Lanza la sonda de colapso en un directorio nuevo."""

from __future__ import annotations

import argparse
from pathlib import Path

from plena.collapse_probe import run
from plena.config import load_experiment, load_schema
from plena.logging_utils import setup_logging
from plena.paths import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Sonda de penalización de colapso")
    parser.add_argument("--schema", default="config/schema.yaml")
    parser.add_argument("--experiment", default="config/experiment.collapse.yaml")
    parser.add_argument("--output", default="results/collapse_probe")
    args = parser.parse_args()
    setup_logging()
    root = project_root()
    output = root / args.output
    if output.resolve() == (root / "results" / "real").resolve():
        raise SystemExit("Esta sonda no escribe en results/real.")
    report = run(
        load_schema(root / args.schema),
        load_experiment(root / args.experiment),
        output,
    )
    decision = report["decision"]
    print(
        f"Estado: {decision['status']}. k={decision.get('k')}. "
        f"Resultados en {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
