"""Constantes del protocolo. Un solo lugar para la partición, el genético y la salida."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "b_depressed.csv"
OUT = ROOT / "results" / "cuaderno"
CSV_SHA256 = "84ab2f5d77d5c9bcdca5d37b894d28517211b22b8995da07a06f4b4714c074d7"

SEED_SPLIT = 2026
SEEDS = (17, 42, 91)
POPULATION, GENERATIONS = 20, 25
KEEP, MUTATION, SIGMA = 0.35, 0.15, 0.15
K = 2

METHODS = [
    "Clase mayoritaria",
    "FAISS base",
    "FAISS + GA Silhouette",
    "FAISS + GA Rand",
    "Regresión logística",
    "Red neuronal ligera",
]
COLORS = ["#999999", "#4477aa", "#66ccee", "#228833", "#cc6677", "#aa3377"]
