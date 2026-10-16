"""Normalización de nombres de columna."""

from __future__ import annotations

import html
import re


def normalize_column_name(name: str) -> str:
    """Quita entidades HTML, asteriscos, espacios y mayúsculas.

    ``*&#x4D;arried`` pasa a ``married`` y ``*Age`` pasa a ``age``.
    """
    text = html.unescape(str(name))
    text = text.replace("*", "")
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    return text
