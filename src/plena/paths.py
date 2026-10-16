"""Rutas del repositorio."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Directorio que contiene ``pyproject.toml``."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("No se encontró la raíz del proyecto (pyproject.toml).")


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    """Resuelve una ruta relativa respecto a la raíz del proyecto."""
    raw = Path(path)
    if raw.is_absolute():
        return raw
    base = root or project_root()
    return (base / raw).resolve()
