"""Punto de entrada de los experimentos reproducibles."""

from __future__ import annotations

import argparse
from pathlib import Path

from plena.config import load_experiment, load_schema
from plena.experiments import planted_signal_report, run_study
from plena.logging_utils import setup_logging
from plena.paths import project_root
from plena.synthetic import write_synthetic_bundle


def _seed_range(text: str | None) -> tuple[int, int | None]:
    if not text:
        return 0, None
    if ":" not in text:
        raise SystemExit("--seeds debe tener la forma A:B")
    left, right = text.split(":", 1)
    start = int(left) if left else 0
    end = int(right) if right else None
    return start, end


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimentos de PLENA")
    parser.add_argument("--schema", default="config/schema.yaml")
    parser.add_argument("--experiment", default="config/experiment.yaml")
    parser.add_argument("--output", default="results/real")
    parser.add_argument("--generate-synthetic", action="store_true")
    parser.add_argument("--planted-only", action="store_true")
    parser.add_argument("--k", type=int, default=None, help="Solo este k")
    parser.add_argument("--variant", choices=("v1", "v2"), default=None)
    parser.add_argument("--seeds", default=None, help="Índices A:B, semiabierto, sobre la lista de semillas")
    parser.add_argument("--finalize", action="store_true")
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
            raise SystemExit(
                "La prueba de señal no superó la proporción esperada por azar "
                f"({report.get('chance')})."
            )
        return
    seed_start, seed_end = _seed_range(args.seeds)
    variants = (args.variant,) if args.variant else ("v1", "v2")
    k_values = None if args.k is None else [args.k]
    finalize = True if args.finalize else None
    run_study(
        schema,
        experiment,
        output,
        k_values=k_values,
        variants=variants,
        seed_start=seed_start,
        seed_end=seed_end,
        measure_cost=k_values is None and args.variant is None and args.seeds is None,
        finalize=finalize,
    )
    print(f"Resultados en {output}")


if __name__ == "__main__":
    main()
