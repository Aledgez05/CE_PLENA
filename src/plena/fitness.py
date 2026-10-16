"""Aptitud V1 y V2, penalizaciones y caché por cromosoma."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score

from plena.chromosome import decode_uncapped, n_eff
from plena.clustering import fit_kmeans
from plena.config import ExperimentConfig
from plena.distance import ColumnMap, transform_weighted


@dataclass
class FitnessBreakdown:
    score: float
    silhouette: float
    ari_train: float | None
    n_active: int
    n_eff: float
    penalties: dict[str, float]
    valid: bool
    weights: np.ndarray | None = None
    weights_before_cap: np.ndarray | None = None
    mask: np.ndarray | None = None
    labels: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "score": self.score,
            "silhouette": self.silhouette,
            "ari_train": self.ari_train,
            "n_active": self.n_active,
            "n_eff": self.n_eff,
            "penalties": self.penalties,
            "valid": self.valid,
            "notes": self.notes,
        }


class FitnessEvaluator:
    """Evalúa cromosomas en una submuestra fija de train.

    K-Means usa un ``random_state`` fijo, así que el mismo cromosoma y la misma
    configuración producen el mismo valor. La etiqueta solo entra en la variante V2.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        y: np.ndarray | None,
        column_map: ColumnMap,
        experiment: ExperimentConfig,
        *,
        k: int,
        variant: str,
    ) -> None:
        if variant not in {"v1", "v2"}:
            raise ValueError("La variante debe ser v1 o v2.")
        self.matrix = np.asarray(matrix, dtype=float)
        self.y = None if y is None else np.asarray(y)
        self.column_map = column_map
        self.experiment = experiment
        self.k = k
        self.variant = variant
        self.cache: dict[tuple, FitnessBreakdown] = {}
        self.calls = 0
        self.cache_hits = 0

    def __call__(self, genes: np.ndarray) -> FitnessBreakdown:
        self.calls += 1
        key = (self.variant, self.k, tuple(np.round(np.asarray(genes, dtype=float), 6)))
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        breakdown = self._evaluate(np.asarray(genes, dtype=float))
        self.cache[key] = breakdown
        return breakdown

    def _invalid(self, note: str, genes: np.ndarray) -> FitnessBreakdown:
        penalties = {"invalid": 1.0, "small_cluster": 0.0, "neff": 0.0}
        return FitnessBreakdown(
            score=float(self.experiment.penalties.invalid_score),
            silhouette=float("nan"),
            ari_train=None,
            n_active=0,
            n_eff=0.0,
            penalties=penalties,
            valid=False,
            weights=None,
            mask=None,
            notes=[note],
        )

    def _evaluate(self, genes: np.ndarray) -> FitnessBreakdown:
        ga = self.experiment.ga
        penalties_cfg = self.experiment.penalties
        try:
            mask, weights, before_cap = decode_uncapped(
                genes, tau=ga.tau, w_max=ga.w_max, min_active=ga.min_active
            )
            weighted = transform_weighted(self.matrix, weights, self.column_map)
            model = fit_kmeans(
                weighted,
                self.k,
                random_state=self.experiment.kmeans_random_state,
                n_init=self.experiment.kmeans_n_init,
            )
            labels = model.labels
            n_clusters = len(np.unique(labels))
            if n_clusters < 2 or n_clusters >= len(labels):
                return self._invalid("agrupamiento inválido", genes)
            silhouette = float(silhouette_score(weighted, labels, metric="euclidean"))
        except Exception as exc:  # una solución inválida no debe romper la corrida
            return self._invalid(str(exc), genes)

        counts = np.bincount(labels, minlength=self.k).astype(float)
        fractions = counts / counts.sum()
        present = fractions[fractions > 0]
        small_deficit = float(np.sum(np.maximum(0.0, penalties_cfg.min_cluster_frac - present)))
        effective = n_eff(weights)
        neff_deficit = float(max(0.0, penalties_cfg.neff_min - effective))
        penalty_small = penalties_cfg.small_cluster_coef * small_deficit
        penalty_neff = penalties_cfg.neff_coef * neff_deficit
        n_active = int(mask.sum())
        dimension = len(weights)
        regularizer = ga.lambda_active * (n_active / dimension)
        ari_value = None
        if self.variant == "v1":
            base = silhouette
        else:
            if self.y is None:
                return self._invalid("V2 necesita la etiqueta de train", genes)
            ari_value = float(adjusted_rand_score(self.y, labels))
            base = self.experiment.v2_alpha * silhouette + (1.0 - self.experiment.v2_alpha) * ari_value
        score = float(base - regularizer - penalty_small - penalty_neff)
        return FitnessBreakdown(
            score=score,
            silhouette=silhouette,
            ari_train=ari_value,
            n_active=n_active,
            n_eff=effective,
            penalties={
                "invalid": 0.0,
                "small_cluster": float(penalty_small),
                "neff": float(penalty_neff),
                "regularizer": float(regularizer),
            },
            valid=True,
            weights=weights,
            weights_before_cap=before_cap,
            mask=mask,
            labels=labels,
        )
