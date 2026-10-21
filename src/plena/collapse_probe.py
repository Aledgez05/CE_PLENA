"""Sonda corta: aptitud V1 con penalización de colapso, sin leer la prueba.

La etiqueta no entra en la aptitud. Entra solo al puntuar la validación y al
aplicar la regla de selección escrita en ``protocol.txt`` antes de optimizar.
Ese uso hace que el ARI de validación del candidato elegido no sea una
confirmación independiente.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from plena.chromosome import equal_weight_vector
from plena.collapse import collapse_report
from plena.config import ExperimentConfig, SchemaConfig, load_experiment, load_schema
from plena.distance import transform_weighted
from plena.experiments import prepare_dataset, score_weights
from plena.paths import project_root
from plena.study_loop import plan_jobs, run_jobs

MARGIN = 0.02
NEAR_COLLAPSE_MAX = 0.5
SILHOUETTE_MAX = 0.99
VARIANT = "v1"
SPLIT_ID = "collapse"


def protocol_text(n_seeds: int) -> str:
    """Regla congelada. Se escribe antes de lanzar el genético."""
    majority = n_seeds // 2 + 1
    return (
        "Sonda de aptitud con penalización de colapso.\n"
        "La prueba no se puntúa, no entra en la aptitud y no se usa para elegir.\n"
        "depressed no entra en X ni en la aptitud de train. V1 sigue siendo Silhouette\n"
        "menos las penalizaciones ya existentes, menos collapse_coef * near_collapse.\n"
        "near_collapse es la media de exp(-a / (tau * b)): a es la distancia al propio\n"
        "centroide y b la distancia a la centroide más cercana. tau = collapse_tau.\n"
        "El redondeo a 6 decimales solo cuenta filas distintas; no define la penalización.\n"
        "\n"
        "Baseline: pesos iguales, misma partición, mismo k, puntuados en validación.\n"
        "Control histórico: medias V1 ya guardadas en results/real/comparison.csv.\n"
        "No se repite esa malla y no se emparejan semillas.\n"
        "\n"
        "Elegibilidad de una corrida, medida en validación:\n"
        f"- n_unique > k\n"
        f"- near_collapse < {NEAR_COLLAPSE_MAX}\n"
        f"- silhouette < {SILHOUETTE_MAX}\n"
        "\n"
        f"Un k es viable si al menos {majority} de {n_seeds} semillas son elegibles.\n"
        "Entre los k viables se toma el de mayor ARI medio de validación.\n"
        "El individuo que se guarda es, dentro de ese k, la semilla elegible de mayor\n"
        "ARI. El desempate es mayor N_eff, menos variables activas y semilla menor.\n"
        "Esa elección usa el ARI de validación, así que ese ARI no confirma el resultado.\n"
        "\n"
        "Decisión, con el ARI medio de las semillas de ese k, no con el máximo:\n"
        f"- apoyo_descriptivo si el medio supera a pesos iguales en al menos {MARGIN}.\n"
        f"- sin_margen si lo supera en menos de {MARGIN}.\n"
        "- rechazado si no lo supera, si ningún k es viable, o si la media no mejora.\n"
        f"Con {n_seeds} semillas el apoyo descriptivo no es una mejora general.\n"
        "No se calcula un Wilcoxon para declararla.\n"
    )


def is_eligible(row: dict) -> bool:
    return bool(
        row["n_unique"] > row["k"]
        and row["near_collapse"] < NEAR_COLLAPSE_MAX
        and row["silhouette"] < SILHOUETTE_MAX
    )


def decide(ga_rows: list[dict], baseline_ari: dict[int, float]) -> dict:
    """Aplica la regla de ``protocol_text``. No lee la prueba."""
    if not ga_rows:
        return {"status": "rechazado", "reason": "No hubo corridas válidas.", "k": None}
    seeds = sorted({int(row["seed"]) for row in ga_rows})
    n_seeds = len(seeds)
    majority = n_seeds // 2 + 1
    viable = []
    by_k: dict[int, list[dict]] = {}
    for row in ga_rows:
        by_k.setdefault(int(row["k"]), []).append(row)
    for k, rows in sorted(by_k.items()):
        eligible = [row for row in rows if is_eligible(row)]
        if len(eligible) < majority:
            continue
        mean_ari = float(np.mean([row["ari"] for row in rows]))
        viable.append((k, mean_ari, eligible, rows))
    if not viable:
        return {
            "status": "rechazado",
            "reason": "Ningún k tuvo una mayoría de semillas fuera del colapso.",
            "k": None,
            "n_seeds": n_seeds,
            "statistical_claim": False,
        }
    k, mean_ari, eligible, rows = max(viable, key=lambda item: (item[1], -item[0]))
    reference = float(baseline_ari[k])
    delta = mean_ari - reference
    chosen = max(
        eligible,
        key=lambda row: (row["ari"], row["n_eff"], -row["n_active"], -row["seed"]),
    )
    if delta >= MARGIN:
        status = "apoyo_descriptivo"
        reason = (
            "La media de validación supera el margen predefinido. "
            "Cinco semillas, o las que tenga esta sonda, no sostienen una mejora general."
        )
    elif delta > 0:
        status = "sin_margen"
        reason = "La media supera a pesos iguales por debajo del margen predefinido."
    else:
        status = "rechazado"
        reason = "La media de ARI no supera a pesos iguales."
    return {
        "status": status,
        "reason": reason,
        "k": int(k),
        "n_seeds": n_seeds,
        "n_eligible": len(eligible),
        "mean_ari": mean_ari,
        "baseline_ari": reference,
        "delta_mean_ari": delta,
        "margin": MARGIN,
        "chosen_seed": int(chosen["seed"]),
        "chosen_ari": float(chosen["ari"]),
        "statistical_claim": False,
        "validation_ari_used_for_selection": True,
    }


def _validation_geometry(bundle, weights: np.ndarray, k: int, experiment: ExperimentConfig) -> dict:
    """Puntúa solo validación. Rechaza cualquier otra partición."""
    metrics = score_weights(bundle, weights, k, experiment, "val")
    weighted = transform_weighted(bundle.x_val, weights, bundle.column_map)
    geometry = collapse_report(weighted, metrics["labels_eval"], tau=experiment.penalties.collapse_tau)
    train_weighted = transform_weighted(bundle.x_train, weights, bundle.column_map)
    train_geometry = collapse_report(
        train_weighted, metrics["labels_train"], tau=experiment.penalties.collapse_tau
    )
    return {
        "silhouette": float(metrics["silhouette"]),
        "ari": float(metrics["ari"]),
        "ri": float(metrics["ri"]),
        "n_active": int(metrics["n_active"]),
        "n_eff": float(metrics["n_eff"]),
        "n_unique": int(geometry["n_unique"]),
        "near_collapse": float(geometry["near_collapse"]),
        "enumerated_fraction": float(geometry["enumerated_fraction"]),
        "mean_intra": float(geometry["mean_intra"]),
        "min_gap": float(geometry["min_gap"]),
        "train_n_unique": int(train_geometry["n_unique"]),
        "train_near_collapse": float(train_geometry["near_collapse"]),
    }


def _historical_v1(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    validation = frame[(frame["split"] == "val") & (frame["variant"] == "v1")]
    validation = validation[validation["method"].isin(["ga", "equal_weights"])]
    summary = (
        validation.groupby(["method", "k"], as_index=False)
        .agg(
            n=("ari", "size"),
            ari_mean=("ari", "mean"),
            silhouette_mean=("silhouette", "mean"),
            n_active_mean=("n_active", "mean"),
            n_eff_mean=("n_eff", "mean"),
        )
    )
    summary.insert(0, "source", "results/real/comparison.csv")
    summary.insert(1, "role", "control_historico_v1")
    return summary


def _split_check(bundle, historical: Path) -> dict:
    payload = json.loads(historical.read_text(encoding="utf-8"))
    stored = payload["indices"]
    train_ok = np.array_equal(np.sort(bundle.splits["train"]), np.sort(np.asarray(stored["train"])))
    val_ok = np.array_equal(np.sort(bundle.splits["val"]), np.sort(np.asarray(stored["val"])))
    return {
        "train_matches_historical": bool(train_ok),
        "val_matches_historical": bool(val_ok),
        "n_train": int(len(bundle.splits["train"])),
        "n_val": int(len(bundle.splits["val"])),
        "note": "La prueba no se compara ni se puntúa en esta sonda.",
    }


def run(schema: SchemaConfig, experiment: ExperimentConfig, output_dir: Path) -> dict:
    """Ejecuta la sonda. Si los checkpoints existen, no los repite."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "protocol.txt").write_text(protocol_text(experiment.n_seeds_ga), encoding="utf-8")
    if experiment.penalties.collapse_coef <= 0:
        raise ValueError("La sonda exige collapse_coef > 0. V1 histórico no debe alterarse.")
    bundle = prepare_dataset(schema, experiment)
    historical_splits = project_root() / "results" / "real" / "splits.json"
    check = _split_check(bundle, historical_splits)
    (output_dir / "split_check.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
    if not check["train_matches_historical"] or not check["val_matches_historical"]:
        raise RuntimeError("La partición de train o validación no coincide con results/real/splits.json.")
    jobs = plan_jobs(
        experiment,
        k_values=list(experiment.k_values),
        variants=(VARIANT,),
        seed_start=0,
        seed_end=None,
        random_for_k={},
        split_id=SPLIT_ID,
        methods=("ga",),
    )
    _records, stored, histories = run_jobs(bundle, experiment, output_dir, jobs)
    rows = []
    for item in stored:
        weights = np.asarray(item["weights"], dtype=float)
        scored = _validation_geometry(bundle, weights, int(item["k"]), experiment)
        rows.append(
            {
                "method": "ga_collapse",
                "variant": VARIANT,
                "k": int(item["k"]),
                "seed": int(item["seed"]),
                "opt_seconds": None,
                "n_evaluations": int(item["n_evaluations"]),
                "eligible": None,
                **scored,
                "weights": weights.tolist(),
            }
        )
    opt_by_key = {}
    for record in _records:
        opt_by_key[(int(record["k"]), int(record["seed"]))] = float(record["opt_seconds"])
    for row in rows:
        row["opt_seconds"] = opt_by_key.get((row["k"], row["seed"]))
        row["eligible"] = is_eligible(row)
    baseline_rows = []
    baseline_ari = {}
    for k in experiment.k_values:
        weights = equal_weight_vector(len(bundle.feature_names))
        scored = _validation_geometry(bundle, weights, int(k), experiment)
        baseline_ari[int(k)] = scored["ari"]
        baseline_rows.append(
            {
                "method": "equal_weights",
                "variant": VARIANT,
                "k": int(k),
                "seed": int(experiment.seed),
                "opt_seconds": 0.0,
                "n_evaluations": 0,
                "eligible": is_eligible({"k": int(k), **scored}),
                **scored,
                "weights": weights.tolist(),
            }
        )
    decision = decide(rows, baseline_ari)
    chosen = None
    if decision.get("k") is not None:
        chosen = next(
            row
            for row in rows
            if row["k"] == decision["k"] and row["seed"] == decision["chosen_seed"]
        )
    frame = pd.DataFrame(rows + baseline_rows)
    export = frame.drop(columns=["weights"])
    export.to_csv(output_dir / "validation.csv", index=False)
    pd.DataFrame(baseline_rows).drop(columns=["weights"]).to_csv(output_dir / "equal_weights.csv", index=False)
    historical = _historical_v1(project_root() / "results" / "real" / "comparison.csv")
    if not historical.empty:
        historical.to_csv(output_dir / "historical_v1_validation.csv", index=False)
    (output_dir / "ga_history.json").write_text(json.dumps(histories), encoding="utf-8")
    (output_dir / "decision.json").write_text(
        json.dumps({"decision": decision, "chosen": chosen}, indent=2),
        encoding="utf-8",
    )
    if schema.source_path:
        shutil.copy(schema.source_path, output_dir / "schema.used.yaml")
    if experiment.source_path:
        shutil.copy(experiment.source_path, output_dir / "experiment.used.yaml")
    return {"decision": decision, "chosen": chosen, "n_runs": len(rows)}
