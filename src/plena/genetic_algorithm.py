"""Operadores y ciclo del algoritmo genético de dominio real."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plena.chromosome import n_eff
from plena.config import GAConfig
from plena.fitness import FitnessBreakdown, FitnessEvaluator
from plena.seed import set_seed


@dataclass
class GAResult:
    best_genes: np.ndarray
    best: FitnessBreakdown
    history: list[dict]
    n_evaluations: int
    generations_ran: int
    seed: int
    seed_equal_weights: bool


def tournament_select(population: np.ndarray, scores: np.ndarray, rng: np.random.Generator, k: int) -> np.ndarray:
    """Selección por torneo. Devuelve una copia del ganador."""
    k = min(k, len(population))
    contestants = rng.choice(len(population), size=k, replace=False)
    winner = contestants[int(np.argmax(scores[contestants]))]
    return population[winner].copy()


def blend_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
    alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Cruce BLX-α recortado a [-1, 1]."""
    low = np.minimum(parent_a, parent_b)
    high = np.maximum(parent_a, parent_b)
    span = high - low
    lower = low - alpha * span
    upper = high + alpha * span
    child_a = rng.uniform(np.minimum(lower, upper), np.maximum(lower, upper))
    child_b = rng.uniform(np.minimum(lower, upper), np.maximum(lower, upper))
    return np.clip(child_a, -1.0, 1.0), np.clip(child_b, -1.0, 1.0)


def arithmetic_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    weight = float(rng.random())
    child_a = weight * parent_a + (1.0 - weight) * parent_b
    child_b = weight * parent_b + (1.0 - weight) * parent_a
    return np.clip(child_a, -1.0, 1.0), np.clip(child_b, -1.0, 1.0)


def gaussian_mutation(
    genes: np.ndarray,
    rng: np.random.Generator,
    sigma: float,
    rate: float,
) -> np.ndarray:
    """Mutación gaussiana por gen, recortada a [-1, 1]."""
    child = genes.copy()
    hits = rng.random(len(child)) < rate
    if np.any(hits):
        child[hits] = child[hits] + rng.normal(0.0, sigma, size=int(hits.sum()))
    return np.clip(child, -1.0, 1.0)


def initialize_population(
    n_genes: int,
    config: GAConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    population = rng.uniform(-1.0, 1.0, size=(config.population, n_genes))
    if config.seed_equal_weights:
        population[0] = np.full(n_genes, 0.5)
    return population


def next_generation(
    population: np.ndarray,
    scores: np.ndarray,
    config: GAConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Elitismo intacto y el resto por torneo, cruce y mutación."""
    n_people, n_genes = population.shape
    n_elite = min(config.elitism, n_people)
    order = np.argsort(-scores, kind="mergesort")
    new_population = np.empty_like(population)
    new_population[:n_elite] = population[order[:n_elite]]
    fill = n_elite
    while fill < n_people:
        parent_a = tournament_select(population, scores, rng, config.tournament)
        parent_b = tournament_select(population, scores, rng, config.tournament)
        if rng.random() < config.crossover_rate:
            if config.crossover == "arithmetic":
                child_a, child_b = arithmetic_crossover(parent_a, parent_b, rng)
            else:
                child_a, child_b = blend_crossover(parent_a, parent_b, rng, config.blx_alpha)
        else:
            child_a, child_b = parent_a.copy(), parent_b.copy()
        child_a = gaussian_mutation(child_a, rng, config.mutation_sigma, config.mutation_rate)
        child_b = gaussian_mutation(child_b, rng, config.mutation_sigma, config.mutation_rate)
        new_population[fill] = child_a
        fill += 1
        if fill < n_people:
            new_population[fill] = child_b
            fill += 1
    return new_population


def _diversity(population: np.ndarray) -> float:
    return float(np.mean(np.std(population, axis=0)))


def run_ga(
    evaluator: FitnessEvaluator,
    n_genes: int,
    config: GAConfig,
    seed: int,
) -> GAResult:
    """Ciclo evolutivo. No se detiene por superar un baseline."""
    rng = set_seed(seed)
    population = initialize_population(n_genes, config, rng)
    history: list[dict] = []
    best_score = -np.inf
    best_genes = population[0].copy()
    best_detail: FitnessBreakdown | None = None
    stall = 0
    generations_ran = 0
    for generation in range(config.generations):
        generations_ran = generation + 1
        details = [evaluator(individual) for individual in population]
        scores = np.array([detail.score for detail in details], dtype=float)
        winner = int(np.argmax(scores))
        if scores[winner] > best_score + 1e-9:
            best_score = float(scores[winner])
            best_genes = population[winner].copy()
            best_detail = details[winner]
            stall = 0
        else:
            stall += 1
        winner_detail = details[winner]
        history.append(
            {
                "generation": generation,
                "best": float(scores[winner]),
                "mean": float(np.mean(scores)),
                "diversity": _diversity(population),
                "n_eff": float(winner_detail.n_eff),
                "n_active": int(winner_detail.n_active),
                "n_evaluations": int(evaluator.calls),
            }
        )
        if stall >= config.stagnation_patience:
            break
        if generation < config.generations - 1:
            population = next_generation(population, scores, config, rng)
    if best_detail is None:
        best_detail = evaluator(best_genes)
    return GAResult(
        best_genes=best_genes,
        best=best_detail,
        history=history,
        n_evaluations=int(evaluator.calls),
        generations_ran=generations_ran,
        seed=seed,
        seed_equal_weights=config.seed_equal_weights,
    )


def population_diversity(population: np.ndarray) -> float:
    return _diversity(population)
