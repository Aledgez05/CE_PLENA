"""FAISS k-means con k = 2 y el genético de pesos reales."""

import faiss
import numpy as np
from sklearn.metrics import balanced_accuracy_score, rand_score, silhouette_score

from plena.settings import GENERATIONS, K, KEEP, MUTATION, POPULATION, SIGMA

faiss.omp_set_num_threads(1)


def weigh(x, w):
    return np.ascontiguousarray(x * np.asarray(w, np.float32), np.float32)


def kmeans(x, w, seed):
    z = weigh(x, w)
    model = faiss.Kmeans(
        z.shape[1], K, niter=30, nredo=3, seed=int(seed), verbose=False, gpu=False,
        min_points_per_centroid=1, max_points_per_centroid=len(x),
    )
    model.train(z)
    return model


def assign(model, x, w):
    return model.index.search(weigh(x, w), 1)[1].ravel().astype(int)


def safe_silhouette(x, labels):
    if len(np.unique(labels)) < 2:
        return np.nan
    return float(silhouette_score(x, labels))


def cluster_to_class(clusters, y):
    """Elige, con train, cuál de los dos clusters se llama depresión."""
    direct = balanced_accuracy_score(y, clusters)
    flipped = balanced_accuracy_score(y, 1 - clusters)
    mapping = np.array([0, 1] if direct >= flipped else [1, 0])
    prevalence = float(y.mean())
    probs = np.array([
        (y[clusters == k].sum() + 2 * prevalence) / (np.sum(clusters == k) + 2) for k in range(K)
    ])
    return mapping, probs


def _normalize(population):
    population = np.clip(np.asarray(population, np.float64), -1.0, 1.0)
    scale = np.max(np.abs(population), axis=1, keepdims=True)
    return population / np.maximum(scale, 1e-12)


def run_ga(x_train, x_val, y_val, objective, seed):
    """El costo es la métrica de validación, con signo negativo. La prueba no entra."""
    rng = np.random.default_rng(seed)
    population = _normalize(rng.uniform(-1.0, 1.0, size=(POPULATION, x_train.shape[1])))
    population[0] = 1.0
    cache, history = {}, []

    def cost(w):
        key = np.asarray(w, np.float64).tobytes()
        if key in cache:
            return cache[key]
        if np.max(np.abs(w)) < 1e-8:
            return 2.0
        model = kmeans(x_train, w, seed)
        train_c, val_c = assign(model, x_train, w), assign(model, x_val, w)
        if min(np.bincount(train_c, minlength=K)) < 2 or min(np.bincount(val_c, minlength=K)) < 2:
            value = 2.0
        else:
            metric = safe_silhouette(weigh(x_val, w), val_c) if objective == "silhouette" else rand_score(y_val, val_c)
            value = -metric if np.isfinite(metric) else 2.0
        cache[key] = float(value)
        return cache[key]

    for generation in range(GENERATIONS + 1):
        costs = np.array([cost(w) for w in population])
        order = np.argsort(costs, kind="stable")
        population, costs = population[order], costs[order]
        history.append([generation, costs[0]])
        if generation == GENERATIONS:
            break
        n_parents = max(2, int(np.ceil(len(population) * KEEP)))
        parents = population[:n_parents]
        probs = np.arange(n_parents, 0, -1, dtype=float)
        probs /= probs.sum()
        n_children = POPULATION - 2
        mother = parents[rng.choice(n_parents, n_children, p=probs)]
        father = parents[rng.choice(n_parents, n_children, p=probs)]
        alpha = rng.uniform(0.0, 1.0, size=mother.shape)
        children = alpha * mother + (1 - alpha) * father
        mask = rng.random(children.shape) < MUTATION
        children = _normalize(children + mask * rng.normal(0.0, SIGMA, size=children.shape))
        population = np.vstack([population[:2], children])
    if costs[0] >= 2.0:
        raise RuntimeError("El genético no encontró una partición válida")
    return population[0].copy(), np.asarray(history)
