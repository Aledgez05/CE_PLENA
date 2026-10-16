"""Auditoría automática del dataset. No corrige datos y no confirma significados."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, pointbiserialr, skew
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency

from plena.config import load_schema_pointer
from plena.data import load_table
from plena.paths import project_root

SENTINELS = {-1, -9, 99, 999, 9999, "", "NA", "?", "na", "NaN"}
LEAKAGE_HINTS = ("depress", "diagnos", "phq", "cesd", "score_dep", "suicid")
SENSITIVE_HINTS = ("age", "sex", "gender", "married", "ville", "city", "region")
GEO_HINTS = ("ville", "city", "region", "geo", "location")


def run_audit(
    frame: pd.DataFrame,
    target: str,
    output_dir: Path,
    column_map: dict[str, str] | None = None,
    *,
    dataset_path: str = "dataset.csv",
    draft_path: Path | None = None,
    ignore_sentinel_columns: list[str] | None = None,
) -> dict:
    """Genera el informe, el JSON, los gráficos y el borrador de esquema."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if column_map:
        (output_dir / "column_map.json").write_text(
            json.dumps(column_map, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    report = {
        "structure": _structure(frame),
        "quality": _quality(frame, ignore_sentinel_columns or []),
        "distributions": _distributions(frame),
        "label": _label_section(frame, target),
        "leakage": _leakage(frame, target),
        "groups": _groups(frame, target),
        "computation": _computation(frame),
        "bias": _bias(frame, target),
        "sufficiency": _sufficiency(frame, target),
    }
    draft = propose_schema(frame, target, report, dataset_path)
    report["draft_schema"] = draft
    report["pending"] = pending_findings(frame, target, report, draft)
    _write_markdown(output_dir / "data_audit.md", report, target)
    (output_dir / "data_audit.json").write_text(
        json.dumps(_json_ready(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if draft_path is not None:
        _write_yaml(draft_path, draft)
    _write_yaml(output_dir / "schema.draft.yaml", draft)
    write_data_dictionary(draft, report, output_dir / "data_dictionary.md", confirmed=False)
    _figures(frame, target, output_dir / "figures")
    return report


def _structure(frame: pd.DataFrame) -> dict:
    n, d = frame.shape
    id_like = []
    for column in frame.columns:
        ratio = frame[column].nunique(dropna=True) / max(n, 1)
        if ratio >= 0.98:
            id_like.append({"column": column, "unique_ratio": float(ratio)})
    duplicated_rows = int(frame.duplicated().sum())
    return {
        "n_rows": int(n),
        "n_columns": int(d),
        "memory_bytes": int(frame.memory_usage(deep=True).sum()),
        "pandas_dtypes": {col: str(dtype) for col, dtype in frame.dtypes.items()},
        "inferred_types": {col: _infer_type(frame[col]) for col in frame.columns},
        "id_like": id_like,
        "duplicated_rows": duplicated_rows,
    }


def _infer_type(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return "empty"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    nunique = int(non_null.nunique())
    if nunique <= 1:
        return "constant"
    if nunique == 2:
        return "binary"
    numeric = pd.to_numeric(non_null, errors="coerce")
    numeric_ratio = float(numeric.notna().mean())
    if numeric_ratio < 0.9:
        text_length = non_null.astype(str).str.len().mean()
        return "text" if text_length > 30 else "nominal"
    if nunique <= 15 and np.all(np.isclose(numeric.dropna(), np.round(numeric.dropna()))):
        return "discrete"
    return "continuous"


def _quality(frame: pd.DataFrame, ignore_sentinel_columns: list[str] | None = None) -> dict:
    n = len(frame)
    missing_col = {col: int(frame[col].isna().sum()) for col in frame.columns}
    missing_row = frame.isna().sum(axis=1)
    sentinels = {}
    ignored = set(ignore_sentinel_columns or [])
    for column in frame.columns:
        if column in ignored:
            continue
        as_text = frame[column].map(lambda value: value if pd.notna(value) else None)
        count = 0
        for value in as_text:
            if value in SENTINELS:
                count += 1
                continue
            try:
                if int(value) in {-1, -9, 99, 999, 9999} and float(value) == int(value):
                    count += 1
            except (TypeError, ValueError):
                continue
        if count:
            sentinels[column] = int(count)
    constant = []
    near_constant = []
    for column in frame.columns:
        non_null = frame[column].dropna()
        if non_null.empty:
            continue
        top = float(non_null.value_counts(normalize=True).iloc[0])
        if top == 1.0:
            constant.append(column)
        elif top >= 0.99:
            near_constant.append({"column": column, "top_fraction": top})
    cardinality = {}
    rare = {}
    for column in frame.columns:
        if _infer_type(frame[column]) in {"nominal", "binary", "discrete", "text"}:
            counts = frame[column].value_counts(dropna=True)
            cardinality[column] = int(counts.shape[0])
            rare_levels = (counts / max(counts.sum(), 1) < 0.01).sum()
            if rare_levels:
                rare[column] = int(rare_levels)
    near_duplicates = _near_duplicates(frame)
    missing_pairs = _missing_patterns(frame)
    return {
        "missing_by_column": missing_col,
        "missing_row_mean": float(missing_row.mean()),
        "rows_with_any_missing": int((missing_row > 0).sum()),
        "sentinels": sentinels,
        "constant": constant,
        "near_constant": near_constant,
        "cardinality": cardinality,
        "rare_levels": rare,
        "near_duplicates": near_duplicates,
        "missing_patterns": missing_pairs,
        "n": int(n),
    }


def _near_duplicates(frame: pd.DataFrame) -> list[dict]:
    pairs = []
    numeric_cols = [col for col in frame.columns if pd.api.types.is_numeric_dtype(frame[col])]
    if len(numeric_cols) >= 2:
        corr = frame[numeric_cols].corr(numeric_only=True).abs()
        for i, left in enumerate(numeric_cols):
            for right in numeric_cols[i + 1 :]:
                value = corr.loc[left, right]
                if pd.notna(value) and value > 0.95:
                    pairs.append({"left": left, "right": right, "association": float(value), "kind": "correlation"})
    categorical = [col for col in frame.columns if _infer_type(frame[col]) in {"nominal", "binary"}]
    for i, left in enumerate(categorical):
        for right in categorical[i + 1 :]:
            if frame[left].astype(str).equals(frame[right].astype(str)):
                pairs.append({"left": left, "right": right, "association": 1.0, "kind": "identical_categorical"})
    return pairs


def _missing_patterns(frame: pd.DataFrame) -> list[dict]:
    indicators = frame.isna().astype(float)
    active = [col for col in indicators.columns if indicators[col].sum() > 0]
    pairs = []
    if len(active) < 2:
        return pairs
    corr = indicators[active].corr()
    for i, left in enumerate(active):
        for right in active[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and value > 0.8:
                pairs.append({"left": left, "right": right, "missing_correlation": float(value)})
    return pairs


def _distributions(frame: pd.DataFrame) -> dict:
    summary = {}
    suggestions = []
    for column in frame.columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().mean() < 0.9 or _infer_type(frame[column]) in {"nominal", "binary", "text"}:
            continue
        values = numeric.dropna().to_numpy(dtype=float)
        if len(values) < 5:
            continue
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            outlier_frac = 0.0
        else:
            outlier_frac = float(np.mean((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)))
        item = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "skew": float(skew(values)),
            "kurtosis": float(kurtosis(values)),
            "outlier_frac_iqr": outlier_frac,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        summary[column] = item
        span = item["max"] - item["min"]
        median = abs(item["median"]) if item["median"] != 0 else 1.0
        if item["max"] > 1e6 or abs(item["skew"]) > 2 or span / median > 1000:
            suggestions.append(
                {
                    "column": column,
                    "suggestion": "Considerar log1p o RobustScaler. Es una sugerencia, no una transformación aplicada.",
                }
            )
    return {"continuous": summary, "scale_suggestions": suggestions}


def _label_section(frame: pd.DataFrame, target: str) -> dict:
    if target not in frame.columns:
        return {"present": False}
    series = frame[target]
    values = series.dropna()
    counts = {str(key): int(value) for key, value in values.value_counts(dropna=False).items()}
    return {
        "present": True,
        "unique": [str(value) for value in values.unique().tolist()],
        "counts": counts,
        "missing": int(series.isna().sum()),
        "prevalence_of_one": float(np.mean(values == 1)) if set(map(str, values.unique())) <= {"0", "1"} else None,
        "semantics": "UNCONFIRMED",
        "note": "No se infiere qué significa cada código solo por el nombre de la columna.",
    }


def _leakage(frame: pd.DataFrame, target: str) -> dict:
    if target not in frame.columns:
        return {}
    y = frame[target]
    valid = y.notna()
    y_valid = y[valid]
    binary = set(map(str, y_valid.unique())) <= {"0", "1", "0.0", "1.0"}
    rows = []
    alerts = []
    name_suspects = []
    for column in frame.columns:
        if column == target:
            continue
        lowered = column.lower()
        if any(hint in lowered for hint in LEAKAGE_HINTS):
            name_suspects.append(column)
        series = frame.loc[valid, column]
        association = _association(series, y_valid, binary)
        association["column"] = column
        rows.append(association)
        auc = association.get("auc")
        if auc is not None and np.isfinite(auc) and (auc > 0.85 or auc < 0.15):
            alerts.append(
                {
                    "column": column,
                    "auc": auc,
                    "message": "Asociación univariada alta con la etiqueta. Revisar fuga antes de usarla.",
                }
            )
    return {
        "associations": rows,
        "alerts": alerts,
        "name_suspects": name_suspects,
        "same_instrument": "UNCONFIRMED: no hay metadatos que indiquen qué variables salen de la misma encuesta que la etiqueta.",
    }


def _association(series: pd.Series, y: pd.Series, binary: bool) -> dict:
    result = {"auc": None, "mutual_info": None, "cramers_v": None, "point_biserial": None}
    paired = pd.DataFrame({"x": series, "y": y}).dropna()
    if paired.empty or not binary:
        return result
    y_num = pd.to_numeric(paired["y"], errors="coerce")
    x_num = pd.to_numeric(paired["x"], errors="coerce")
    if x_num.notna().mean() > 0.9 and y_num.notna().all() and x_num.nunique() > 1:
        try:
            auc = float(roc_auc_score(y_num, x_num))
            result["auc"] = max(auc, 1 - auc)
        except ValueError:
            result["auc"] = None
        try:
            result["point_biserial"] = float(pointbiserialr(y_num, x_num.fillna(x_num.median())).statistic)
        except ValueError:
            result["point_biserial"] = None
        discrete = x_num.nunique() < 15
        mi = mutual_info_classif(x_num.fillna(x_num.median()).to_numpy().reshape(-1, 1), y_num.astype(int), discrete_features=discrete, random_state=0)
        result["mutual_info"] = float(mi[0])
    else:
        result["cramers_v"] = _cramers_v(paired["x"].astype(str), y_num.astype(str))
        # AUC de una codificación de frecuencias aprendida en la misma muestra: solo como alerta, no como modelo.
        rates = y_num.groupby(paired["x"].astype(str)).mean()
        scores = paired["x"].astype(str).map(rates)
        if scores.nunique() > 1:
            try:
                result["auc"] = float(roc_auc_score(y_num, scores))
            except ValueError:
                result["auc"] = None
    return result


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x, y)
    if table.size == 0 or min(table.shape) < 2:
        return float("nan")
    chi2 = chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def _groups(frame: pd.DataFrame, target: str) -> dict:
    notes = []
    columns = []
    for column in frame.columns:
        ratio = frame[column].nunique(dropna=True) / max(len(frame), 1)
        lowered = column.lower()
        if ratio > 0.5 or any(hint in lowered for hint in ("id", "ville", "hogar", "household", "person")):
            duplicated = int(frame[column].duplicated().sum())
            columns.append(
                {
                    "column": column,
                    "unique": int(frame[column].nunique(dropna=True)),
                    "duplicated_values": duplicated,
                    "unique_ratio": float(ratio),
                }
            )
    geo = {}
    if target in frame.columns:
        y = frame[target]
        for column in frame.columns:
            if any(hint in column.lower() for hint in GEO_HINTS):
                rates = (
                    pd.DataFrame({"g": frame[column], "y": y})
                    .dropna()
                    .groupby("g")["y"]
                    .agg(["count", "mean"])
                    .sort_values("count", ascending=False)
                    .head(10)
                )
                geo[column] = [
                    {"value": str(idx), "n": int(row["count"]), "label_mean": float(row["mean"])}
                    for idx, row in rates.iterrows()
                ]
    if not any(item["duplicated_values"] == 0 and "id" in item["column"] for item in columns):
        notes.append("No se observó un identificador de persona con repeticiones. Sigue UNCONFIRMED si el identificador de persona no está en la tabla.")
    return {"candidate_keys": columns, "geographic": geo, "notes": notes}


def _computation(frame: pd.DataFrame) -> dict:
    n, d = frame.shape
    numeric = frame.select_dtypes(include="number")
    matrix = numeric.drop(columns=[col for col in numeric.columns if "depress" in col.lower()], errors="ignore")
    matrix = matrix.fillna(matrix.median(numeric_only=True))
    sample_n = min(len(matrix), 1000)
    elapsed = None
    if matrix.shape[1] >= 1 and sample_n >= 10:
        sample = StandardScaler().fit_transform(matrix.iloc[:sample_n])
        k = 3 if sample_n > 3 else 2
        start = time.perf_counter()
        labels = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(sample)
        if len(np.unique(labels)) > 1:
            silhouette_score(sample, labels)
        elapsed = float(time.perf_counter() - start)
    nominal_extra = 0
    high_cardinality_codes = []
    for column in frame.columns:
        inferred = _infer_type(frame[column])
        nunique = int(frame[column].nunique(dropna=True))
        if inferred == "nominal":
            nominal_extra += nunique
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if inferred in {"continuous", "discrete"} and 15 < nunique < max(n * 0.5, 16):
            maximum = float(numeric.max()) if numeric.notna().any() else 0.0
            if maximum <= max(nunique * 3, 500):
                high_cardinality_codes.append(
                    {
                        "column": column,
                        "nunique": nunique,
                        "one_hot_columns_if_nominal": nunique,
                        "note": "UNCONFIRMED: si se trata como nominal, el one-hot añade estas columnas.",
                    }
                )
    dense_dim = int(matrix.shape[1] + nominal_extra)
    distance_bytes = n * n * 8
    return {
        "n": int(n),
        "d_raw": int(d),
        "d_numeric": int(matrix.shape[1]),
        "d_after_one_hot_estimate": dense_dim,
        "high_cardinality_codes": high_cardinality_codes,
        "pairwise_memory_bytes": int(distance_bytes),
        "fitness_sample_recommendation": int(min(n, 1500 if elapsed and elapsed > 1 else n)),
        "one_fitness_seconds_probe": elapsed,
        "probe_note": "Sonda aproximada sobre columnas numéricas imputadas, no sobre el preprocesamiento final.",
    }


def _bias(frame: pd.DataFrame, target: str) -> dict:
    columns = [col for col in frame.columns if any(hint in col.lower() for hint in SENSITIVE_HINTS)]
    report = {"heuristic_columns": columns, "status": "UNCONFIRMED", "subgroups": {}}
    if target not in frame.columns:
        return report
    y = pd.to_numeric(frame[target], errors="coerce")
    for column in columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 12:
            bins = pd.qcut(series, q=4, duplicates="drop")
            grouped = pd.DataFrame({"g": bins, "y": y}).dropna().groupby("g", observed=True)["y"]
        else:
            grouped = pd.DataFrame({"g": series, "y": y}).dropna().groupby("g")["y"]
        report["subgroups"][column] = [
            {"value": str(idx), "n": int(values.count()), "label_mean": float(values.mean())}
            for idx, values in grouped
        ]
    return report


def _sufficiency(frame: pd.DataFrame, target: str) -> dict:
    n, d = frame.shape
    minority = None
    if target in frame.columns:
        counts = frame[target].value_counts(dropna=True)
        if not counts.empty:
            minority = int(counts.min())
    return {
        "n_over_d": float(n / max(d, 1)),
        "minority_class_n": minority,
        "split_note": "El tamaño por partición oficial se calcula en el paso 1.6, después de confirmar el esquema. No se usa para elegir modelos.",
    }


def propose_schema(frame: pd.DataFrame, target: str, report: dict, dataset_path: str) -> dict:
    """Borrador: cada columna una vez, con tipo propuesto y justificación."""
    id_like = {item["column"] for item in report["structure"]["id_like"]}
    features = {}
    exclusions = []
    for column in frame.columns:
        if column == target:
            continue
        inferred = report["structure"]["inferred_types"][column]
        if column in id_like or column in report["quality"]["constant"]:
            reason = "identificador (relación de únicos cercana a 1)" if column in id_like else "columna constante"
            exclusions.append(
                {
                    "column": column,
                    "reason": reason,
                    "status": "UNCONFIRMED",
                }
            )
            continue
        proposed = {
            "constant": "continuous",
            "binary": "binary",
            "discrete": "discrete",
            "nominal": "nominal",
            "text": "nominal",
            "date": "nominal",
            "continuous": "continuous",
            "empty": "continuous",
        }.get(inferred, "continuous")
        features[column] = {
            "type": proposed,
            "status": "UNCONFIRMED",
            "justification": (
                f"Tipo inferido por la auditoría: {inferred}. "
                "La semántica y la codificación quedan UNCONFIRMED hasta confirmación humana."
            ),
        }
        if any(hint in column.lower() for hint in LEAKAGE_HINTS):
            features[column]["suspected_leakage"] = True
            features[column]["justification"] += " El nombre sugiere una posible derivación de la etiqueta."
    return {
        "dataset_path": dataset_path,
        "target": target,
        "id_columns": [item["column"] for item in report["structure"]["id_like"] if item["column"] != target],
        "group_column": None,
        "features": features,
        "exclusions": exclusions,
        "note": "BORRADOR. No usar para entrenar mientras haya UNCONFIRMED.",
    }


def pending_findings(frame: pd.DataFrame, target: str, report: dict, draft: dict) -> list[str]:
    findings = [
        "El significado de los códigos de la etiqueta está UNCONFIRMED.",
        "Los tipos del borrador son propuestas automáticas, no un diccionario confirmado.",
    ]
    if report["quality"]["sentinels"]:
        findings.append("Hay códigos centinela sospechosos. No se convirtieron a faltantes.")
    if report["leakage"]["alerts"]:
        findings.append("Hay variables con asociación univariada alta con la etiqueta.")
    if report["leakage"]["name_suspects"]:
        findings.append("Hay nombres que podrían corresponder a instrumentos de la etiqueta.")
    if report["groups"]["geographic"]:
        findings.append("Hay columnas geográficas candidatas. Su uso queda UNCONFIRMED.")
    for item in report["computation"].get("high_cardinality_codes", []):
        findings.append(
            f"{item['column']} tiene {item['nunique']} valores. Tratarla como nominal dispararía el one-hot."
        )
    findings.append("No se confirma si hay observaciones repetidas por persona u hogar.")
    return findings


def write_data_dictionary(draft: dict, report: dict, path: Path, confirmed: bool) -> None:
    lines = [
        "# Diccionario de datos",
        "",
        "Estado: confirmado." if confirmed else "Estado: BORRADOR. Los significados siguen UNCONFIRMED.",
        "",
        "| Columna | Significado | Tipo | Faltantes | Decisión | Motivo |",
        "|---|---|---|---|---|---|",
    ]
    missing = report.get("quality", {}).get("missing_by_column", {})
    target = draft["target"]
    lines.append(
        f"| {target} | UNCONFIRMED | etiqueta | {missing.get(target, 'n/d')} | excluida de X | solo evaluación externa |"
    )
    for item in draft.get("exclusions", []):
        lines.append(
            f"| {item['column']} | UNCONFIRMED | exclusión | {missing.get(item['column'], 'n/d')} | excluida | {item['reason']} |"
        )
    for name, spec in draft.get("features", {}).items():
        lines.append(
            f"| {name} | UNCONFIRMED | {spec['type']} | {missing.get(name, 'n/d')} | propuesta de entrada | {spec.get('justification', '')} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload: dict) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_markdown(path: Path, report: dict, target: str) -> None:
    structure = report["structure"]
    quality = report["quality"]
    label = report["label"]
    comp = report["computation"]
    lines = [
        "# Auditoría de datos",
        "",
        "Informe automático. No confirma el significado de las columnas.",
        "",
        "## Estructura",
        "",
        f"- Filas: {structure['n_rows']}",
        f"- Columnas: {structure['n_columns']}",
        f"- Memoria: {structure['memory_bytes']} bytes",
        f"- Filas duplicadas: {structure['duplicated_rows']}",
        f"- Candidatas a identificador: {structure['id_like']}",
        "",
        "## Calidad",
        "",
        f"- Faltantes por columna: {quality['missing_by_column']}",
        f"- Filas con algún faltante: {quality['rows_with_any_missing']}",
        f"- Códigos centinela (solo reportados): {quality['sentinels']}",
        f"- Constantes: {quality['constant']}",
        f"- Casi constantes: {quality['near_constant']}",
        f"- Casi duplicadas: {quality['near_duplicates']}",
        f"- Patrones de faltantes: {quality['missing_patterns']}",
        "",
        "## Etiqueta",
        "",
        f"- Columna: `{target}`",
        f"- Valores: {label.get('unique')}",
        f"- Conteos: {label.get('counts')}",
        f"- Faltantes: {label.get('missing')}",
        f"- Semántica: {label.get('semantics')}",
        "",
        "## Fuga",
        "",
        f"- Alertas AUC: {report['leakage'].get('alerts')}",
        f"- Sospechosas por nombre: {report['leakage'].get('name_suspects')}",
        f"- Misma encuesta: {report['leakage'].get('same_instrument')}",
        "",
        "## Grupos y geografía",
        "",
        f"- Claves candidatas: {report['groups']['candidate_keys']}",
        f"- Geografía: {list(report['groups']['geographic'])}",
        "",
        "## Viabilidad",
        "",
        f"- D estimada tras one-hot (solo nominales ya inferidas): {comp['d_after_one_hot_estimate']}",
        f"- Códigos de cardinalidad alta, si se one-hot: {comp.get('high_cardinality_codes')}",
        f"- Memoria de una matriz n×n: {comp['pairwise_memory_bytes']} bytes",
        f"- Segundos de una sonda de fitness: {comp['one_fitness_seconds_probe']}",
        f"- Tamaño de submuestra sugerido: {comp['fitness_sample_recommendation']}",
        "",
        "## Sesgo y suficiencia",
        "",
        f"- Columnas sensibles heurísticas: {report['bias']['heuristic_columns']}",
        f"- n/D: {report['sufficiency']['n_over_d']}",
        f"- Clase minoritaria: {report['sufficiency']['minority_class_n']}",
        "",
        "## Hallazgos y decisiones pendientes",
        "",
    ]
    for item in report["pending"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "### Preguntas para confirmar el esquema",
            "",
            "1. ¿Qué significan los valores de la etiqueta y cómo se obtuvo?",
            "2. ¿Qué columnas son nominales, ordinales (con qué orden), binarias, continuas o identificadores?",
            "3. ¿Hay columnas derivadas de la etiqueta o del mismo instrumento?",
            "4. ¿Los códigos sospechosos son faltantes?",
            "5. ¿La columna geográfica se usa, se excluye o se reserva para ablación?",
            "6. ¿Hay repeticiones por persona u hogar que obliguen a particionar por grupo?",
            "7. ¿Cuántas semillas y qué presupuesto de cómputo hay?",
            "8. ¿Se activa NSGA-II si sobra tiempo?",
            "9. ¿Este dataset es el definitivo?",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _figures(frame: pd.DataFrame, target: str, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    missing = frame.isna().mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(missing.index.astype(str), missing.values)
    ax.set_title("Proporción de faltantes")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(directory / "missing.png", dpi=120)
    plt.close(fig)
    numeric = frame.select_dtypes(include="number").columns[:6]
    if len(numeric):
        fig, axes = plt.subplots(len(numeric), 1, figsize=(6, 2.2 * len(numeric)))
        if len(numeric) == 1:
            axes = [axes]
        for axis, column in zip(axes, numeric):
            axis.hist(frame[column].dropna(), bins=30)
            axis.set_title(column)
        fig.tight_layout()
        fig.savefig(directory / "distributions.png", dpi=120)
        plt.close(fig)
    if target in frame.columns:
        counts = frame[target].value_counts(dropna=False)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_title(f"Prevalencia de {target}")
        fig.tight_layout()
        fig.savefig(directory / "prevalence.png", dpi=120)
        plt.close(fig)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditoría de un CSV de PLENA")
    parser.add_argument("--config", default=None, help="YAML con dataset_path y target")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--output", default="results/audit")
    args = parser.parse_args()
    ignore_sentinels: list[str] = []
    if args.config:
        pointer = load_schema_pointer(args.config)
        csv_path = pointer.dataset_path
        target = pointer.target
        ignore_sentinels = list(pointer.id_columns)
        if pointer.group_column:
            ignore_sentinels.append(pointer.group_column)
    else:
        if not args.csv or not args.target:
            raise SystemExit("Indica --config o bien --csv y --target.")
        csv_path = Path(args.csv)
        target = args.target
    frame, mapping = load_table(csv_path)
    output = project_root() / args.output if not Path(args.output).is_absolute() else Path(args.output)
    draft_path = project_root() / "config" / "schema.draft.yaml"
    try:
        dataset_label = str(csv_path.resolve().relative_to(project_root()))
    except ValueError:
        dataset_label = str(csv_path)
    run_audit(
        frame,
        target,
        output,
        mapping,
        dataset_path=dataset_label.replace("\\", "/"),
        draft_path=draft_path,
        ignore_sentinel_columns=ignore_sentinels,
    )
    print(f"Auditoría escrita en {output}")


if __name__ == "__main__":
    main()
