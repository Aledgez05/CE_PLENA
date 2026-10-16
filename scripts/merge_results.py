"""Une resultados parciales sin duplicar corridas."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


KEYS = ["method", "variant", "k", "seed", "split"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Une carpetas de resultados de PLENA")
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    destination = args.output
    checkpoint_dir = destination / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    frames = []
    for source in args.sources:
        origin = source / "checkpoints"
        if origin.is_dir():
            for path in origin.glob("*.json"):
                target = checkpoint_dir / path.name
                if target.exists():
                    continue
                shutil.copy2(path, target)
                copied += 1
        for name in ("comparison.csv", "comparison.partial.csv", "robustness.csv"):
            table = source / name
            if table.is_file():
                frames.append(pd.read_csv(table))
    if frames:
        merged = pd.concat(frames, ignore_index=True)
        available = [column for column in KEYS if column in merged.columns]
        if available:
            merged = merged.drop_duplicates(available, keep="first")
        merged.to_csv(destination / "comparison.merged.csv", index=False)
    print(f"Checkpoints nuevos: {copied}. Tablas unidas en {destination}")


if __name__ == "__main__":
    main()
