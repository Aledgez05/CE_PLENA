"""Carga y validación de ``schema.yaml`` y ``experiment.yaml``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from plena.paths import resolve_path

VALID_TYPES = {
    "continuous",
    "discrete",
    "binary",
    "nominal",
    "ordinal",
    "text",
    "date",
}
NUMERIC_TYPES = {"continuous", "discrete"}
ROUTE_A_TYPES = {"continuous", "discrete", "binary", "nominal", "ordinal"}


@dataclass
class FeatureSpec:
    name: str
    type: str
    order: list[str] | None = None
    missing_indicator: bool = False
    scaler: str | None = None
    status: str = "confirmed"
    justification: str = ""
    suspected_leakage: bool = False


@dataclass
class Exclusion:
    column: str
    reason: str
    status: str = "confirmed"


@dataclass
class SchemaConfig:
    dataset_path: Path
    target: str
    id_columns: list[str]
    group_column: str | None
    features: dict[str, FeatureSpec]
    exclusions: list[Exclusion]
    informative_variables: list[str] = field(default_factory=list)
    ablation_groups: dict[str, list[str]] = field(default_factory=dict)
    sensitive_columns: list[str] = field(default_factory=list)
    source_path: Path | None = None

    @property
    def feature_names(self) -> list[str]:
        return list(self.features.keys())


@dataclass
class GAConfig:
    population: int = 40
    generations: int = 40
    elitism: int = 2
    tournament: int = 3
    crossover: str = "blend"
    mutation: str = "gaussian"
    mutation_sigma: float = 0.2
    mutation_rate: float = 0.2
    tau: float = 0.0
    w_max: float = 0.3
    min_active: int = 3
    lambda_active: float = 0.02
    stagnation_patience: int = 10
    seed_equal_weights: bool = False
    blx_alpha: float = 0.5
    crossover_rate: float = 0.8


@dataclass
class PenaltyConfig:
    min_cluster_frac: float = 0.05
    neff_min: float = 3.0
    small_cluster_coef: float = 1.0
    neff_coef: float = 0.25
    invalid_score: float = -1.0


@dataclass
class SplitConfig:
    train: float = 0.6
    val: float = 0.2
    test: float = 0.2


@dataclass
class ExperimentConfig:
    seed: int = 42
    n_seeds_ga: int = 10
    n_random_runs: int = 30
    k_values: list[int] = field(default_factory=lambda: [2, 3, 4, 5])
    force_k: int = 3
    fitness_sample_size: int = 3000
    split: SplitConfig = field(default_factory=SplitConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    v2_alpha: float = 0.5
    penalties: PenaltyConfig = field(default_factory=PenaltyConfig)
    kmeans_n_init: int = 10
    kmeans_random_state: int = 42
    n_permutations: int = 100
    knn_k: int = 15
    scaler: str = "robust"
    budget_seconds: int | None = None
    source_path: Path | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} no contiene un mapa YAML.")
    return data


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "si", "sí"}
    return bool(value)


def load_schema(path: str | Path, *, allow_unconfirmed: bool = False) -> SchemaConfig:
    """Lee un esquema y, si ``allow_unconfirmed`` es falso, rechaza ``UNCONFIRMED``."""
    source = Path(path)
    raw = _load_yaml(source)
    if "dataset_path" not in raw or "target" not in raw:
        raise ValueError("El esquema necesita dataset_path y target.")
    features: dict[str, FeatureSpec] = {}
    for name, spec in (raw.get("features") or {}).items():
        spec = spec or {}
        if not isinstance(spec, dict):
            raise ValueError(f"La característica {name} debe ser un mapa.")
        ftype = str(spec.get("type", "")).strip().lower()
        order = spec.get("order")
        if order is not None:
            order = [str(item) for item in order]
        features[str(name)] = FeatureSpec(
            name=str(name),
            type=ftype,
            order=order,
            missing_indicator=_as_bool(spec.get("missing_indicator", False)),
            scaler=spec.get("scaler"),
            status=str(spec.get("status", "confirmed")),
            justification=str(spec.get("justification", "")),
            suspected_leakage=_as_bool(spec.get("suspected_leakage", False)),
        )
    exclusions = []
    for item in raw.get("exclusions") or []:
        exclusions.append(
            Exclusion(
                column=str(item["column"]),
                reason=str(item.get("reason", "")),
                status=str(item.get("status", "confirmed")),
            )
        )
    group = raw.get("group_column")
    schema = SchemaConfig(
        dataset_path=resolve_path(raw["dataset_path"]),
        target=str(raw["target"]),
        id_columns=[str(col) for col in (raw.get("id_columns") or [])],
        group_column=None if group in (None, "null", "") else str(group),
        features=features,
        exclusions=exclusions,
        informative_variables=[str(col) for col in (raw.get("informative_variables") or [])],
        ablation_groups={
            str(key): [str(col) for col in cols]
            for key, cols in (raw.get("ablation_groups") or {}).items()
        },
        sensitive_columns=[str(col) for col in (raw.get("sensitive_columns") or [])],
        source_path=source,
    )
    validate_schema(schema, allow_unconfirmed=allow_unconfirmed, require_features=True)
    return schema


def load_schema_pointer(path: str | Path) -> SchemaConfig:
    """Lee un YAML de auditoría que puede no declarar características."""
    source = Path(path)
    raw = _load_yaml(source)
    group = raw.get("group_column")
    schema = SchemaConfig(
        dataset_path=resolve_path(raw["dataset_path"]),
        target=str(raw["target"]),
        id_columns=[str(col) for col in (raw.get("id_columns") or [])],
        group_column=None if group in (None, "null", "") else str(group),
        features={},
        exclusions=[],
        source_path=source,
    )
    return schema


def load_experiment(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    raw = _load_yaml(source)
    split_raw = raw.get("split") or {}
    ga_raw = raw.get("ga") or {}
    pen_raw = raw.get("penalties") or {}
    variants = raw.get("variants") or {}
    kmeans = raw.get("kmeans") or {}
    experiment = ExperimentConfig(
        seed=int(raw.get("seed", 42)),
        n_seeds_ga=int(raw.get("n_seeds_ga", 10)),
        n_random_runs=int(raw.get("n_random_runs", 30)),
        k_values=[int(k) for k in raw.get("k_values", [2, 3, 4, 5])],
        force_k=int(raw.get("force_k", 3)),
        fitness_sample_size=int(raw.get("fitness_sample_size", 3000)),
        split=SplitConfig(
            train=float(split_raw.get("train", 0.6)),
            val=float(split_raw.get("val", 0.2)),
            test=float(split_raw.get("test", 0.2)),
        ),
        ga=GAConfig(
            population=int(ga_raw.get("population", 40)),
            generations=int(ga_raw.get("generations", 40)),
            elitism=int(ga_raw.get("elitism", 2)),
            tournament=int(ga_raw.get("tournament", 3)),
            crossover=str(ga_raw.get("crossover", "blend")),
            mutation=str(ga_raw.get("mutation", "gaussian")),
            mutation_sigma=float(ga_raw.get("mutation_sigma", 0.2)),
            mutation_rate=float(ga_raw.get("mutation_rate", 0.2)),
            tau=float(ga_raw.get("tau", 0.0)),
            w_max=float(ga_raw.get("w_max", 0.3)),
            min_active=int(ga_raw.get("min_active", 3)),
            lambda_active=float(ga_raw.get("lambda_active", 0.02)),
            stagnation_patience=int(ga_raw.get("stagnation_patience", 10)),
            seed_equal_weights=_as_bool(ga_raw.get("seed_equal_weights", False)),
            blx_alpha=float(ga_raw.get("blx_alpha", 0.5)),
            crossover_rate=float(ga_raw.get("crossover_rate", 0.8)),
        ),
        v2_alpha=float(variants.get("v2_alpha", 0.5)),
        penalties=PenaltyConfig(
            min_cluster_frac=float(pen_raw.get("min_cluster_frac", 0.05)),
            neff_min=float(pen_raw.get("neff_min", 3)),
            small_cluster_coef=float(pen_raw.get("small_cluster_coef", 1.0)),
            neff_coef=float(pen_raw.get("neff_coef", 0.25)),
            invalid_score=float(pen_raw.get("invalid_score", -1.0)),
        ),
        kmeans_n_init=int(kmeans.get("n_init", raw.get("kmeans_n_init", 10))),
        kmeans_random_state=int(kmeans.get("random_state", raw.get("seed", 42))),
        n_permutations=int(raw.get("n_permutations", 100)),
        knn_k=int(raw.get("knn_k", 15)),
        scaler=str(raw.get("scaler", "robust")),
        budget_seconds=None if raw.get("budget_seconds") is None else int(raw["budget_seconds"]),
        source_path=source,
    )
    validate_experiment(experiment)
    return experiment


def validate_schema(
    schema: SchemaConfig,
    columns: list[str] | None = None,
    *,
    allow_unconfirmed: bool = False,
    require_features: bool = True,
) -> None:
    """Comprueba tipos, solapes y, si se pasan, que las columnas existan."""
    if require_features and not schema.features:
        raise ValueError("El esquema no declara características.")
    names = list(schema.features)
    if len(names) != len(set(names)):
        raise ValueError("Hay características repetidas.")
    banned = set(schema.id_columns) | {schema.target}
    overlap = banned & set(names)
    if overlap:
        raise ValueError(
            "Estas columnas están a la vez en características y en identificador o etiqueta: "
            + ", ".join(sorted(overlap))
        )
    if schema.group_column and schema.group_column in names:
        raise ValueError("group_column no puede ser una característica de entrada.")
    exclusion_names = [item.column for item in schema.exclusions]
    if len(exclusion_names) != len(set(exclusion_names)):
        raise ValueError("Hay exclusiones repetidas.")
    excluded_overlap = set(exclusion_names) & set(names)
    if excluded_overlap:
        raise ValueError(
            "Una columna excluida también está en características: "
            + ", ".join(sorted(excluded_overlap))
        )
    for spec in schema.features.values():
        if spec.type not in VALID_TYPES:
            raise ValueError(f"Tipo no válido para {spec.name}: {spec.type}")
        if spec.type == "ordinal" and not spec.order:
            raise ValueError(f"{spec.name} es ordinal y no documenta el orden.")
        if spec.scaler not in (None, "robust", "standard"):
            raise ValueError(f"Escalador no válido para {spec.name}: {spec.scaler}")
        if spec.type in {"text", "date"}:
            raise ValueError(
                f"{spec.name} es {spec.type}. Este avance solo modela variables tabulares."
            )
    for name in schema.informative_variables:
        if name not in schema.features:
            raise ValueError(f"La variable informativa {name} no está en características.")
    for group_name, cols in schema.ablation_groups.items():
        missing = [col for col in cols if col not in schema.features]
        if missing:
            raise ValueError(
                f"El grupo de ablación {group_name} cita columnas que no son características: {missing}"
            )
    if columns is not None:
        available = set(columns)
        needed = set(names) | set(schema.id_columns) | {schema.target}
        if schema.group_column:
            needed.add(schema.group_column)
        missing = sorted(needed - available)
        if missing:
            raise ValueError("Faltan columnas en el dataset: " + ", ".join(missing))
    if not allow_unconfirmed:
        pending = [
            spec.name
            for spec in schema.features.values()
            if spec.status.upper() == "UNCONFIRMED"
        ]
        pending += [
            item.column
            for item in schema.exclusions
            if item.status.upper() == "UNCONFIRMED"
        ]
        if pending:
            raise ValueError(
                "El esquema tiene elementos UNCONFIRMED y no puede usarse para entrenar: "
                + ", ".join(pending)
            )


def validate_experiment(experiment: ExperimentConfig) -> None:
    ratios = (experiment.split.train, experiment.split.val, experiment.split.test)
    if any(part <= 0 for part in ratios):
        raise ValueError("Las proporciones de la partición deben ser positivas.")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Las proporciones train/val/test deben sumar 1.")
    if experiment.force_k not in experiment.k_values:
        raise ValueError("force_k debe pertenecer a k_values.")
    if any(k < 2 for k in experiment.k_values):
        raise ValueError("k debe ser al menos 2.")
    ga = experiment.ga
    if ga.population < 2 or ga.generations < 1:
        raise ValueError("Población y generaciones deben ser positivas.")
    if not 0 <= ga.tau < 1:
        raise ValueError("tau debe estar en [0, 1).")
    if not 0 < ga.w_max <= 1:
        raise ValueError("w_max debe estar en (0, 1].")
    if ga.min_active < 1:
        raise ValueError("min_active debe ser al menos 1.")
    if ga.elitism < 0 or ga.elitism >= ga.population:
        raise ValueError("El elitismo debe ser menor que la población.")
    if ga.tournament < 1:
        raise ValueError("El torneo debe tener al menos 1 participante.")
    if experiment.ga.crossover not in {"blend", "arithmetic"}:
        raise ValueError("El cruce debe ser blend o arithmetic.")
    if experiment.ga.mutation != "gaussian":
        raise ValueError("La mutación implementada es gaussian.")
    if not 0 <= experiment.v2_alpha <= 1:
        raise ValueError("v2_alpha debe estar en [0, 1].")
    if experiment.scaler not in {"robust", "standard"}:
        raise ValueError("El escalador debe ser robust o standard.")
    if experiment.n_seeds_ga < 1 or experiment.n_random_runs < 1:
        raise ValueError("El número de semillas y de búsquedas aleatorias debe ser positivo.")


def assert_weights_feasible(n_features: int, experiment: ExperimentConfig) -> None:
    """Comprueba que el tope de peso y el mínimo de activas caben en D."""
    ga = experiment.ga
    if ga.min_active > n_features:
        raise ValueError("min_active es mayor que el número de variables.")
    if ga.w_max * n_features < 1 - 1e-9:
        raise ValueError(
            "w_max es demasiado bajo para que los pesos sumen 1 con las variables disponibles."
        )
