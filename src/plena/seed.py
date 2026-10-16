"""Semillas y generadores. Toda la aleatoriedad del proyecto pasa por aquí."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    """Fija las semillas globales y devuelve un ``Generator`` independiente.

    El generador devuelto es el que deben usar los operadores. Las semillas
    globales cubren bibliotecas que todavía leen el estado de ``random`` o
    de NumPy legado. scikit-learn recibe ``random_state`` de forma explícita.
    """
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("La semilla debe ser un entero.")
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
