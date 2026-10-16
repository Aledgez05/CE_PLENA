"""Preprocesamiento de la ruta A y mapeo columna transformada → variable."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler

from plena.config import ROUTE_A_TYPES, FeatureSpec, SchemaConfig
from plena.distance import ColumnMap


@dataclass
class PreparedMatrices:
    transformer: ColumnTransformer
    column_map: ColumnMap
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    x_train: object
    x_val: object
    x_test: object
    y_train: object
    y_val: object
    y_test: object


class Log1pNonNegative(BaseEstimator, TransformerMixin):
    """log1p solo si no hay negativos. Un negativo detiene el ajuste."""

    def fit(self, X, y=None):
        self._refuse_negatives(X)
        return self

    def transform(self, X):
        self._refuse_negatives(X)
        values = np.asarray(X, dtype=float)
        return np.log1p(values)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("log1p necesita los nombres de entrada.")
        return np.asarray(input_features, dtype=object)

    @staticmethod
    def _refuse_negatives(X) -> None:
        values = np.asarray(X, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size and np.any(finite < 0):
            raise ValueError(
                "Hay valores negativos en una columna con transformación log1p. "
                "No se aplica log1p hasta revisarlo."
            )


def _scaler(name: str):
    if name == "robust":
        return RobustScaler()
    if name == "standard":
        return StandardScaler()
    raise ValueError(f"Escalador desconocido: {name}")


def build_preprocessor(
    schema: SchemaConfig,
    default_scaler: str = "robust",
) -> tuple[ColumnTransformer, dict[str, str]]:
    """Arma un ``ColumnTransformer`` que se ajusta solo con el conjunto que se le pase.

    Cada paso se llama ``var__<nombre>`` para recuperar la variable original
    aunque el nombre tenga guiones bajos.
    """
    transformers = []
    step_to_variable: dict[str, str] = {}
    for index, (name, spec) in enumerate(schema.features.items()):
        if spec.type not in ROUTE_A_TYPES:
            raise ValueError(f"{name} no entra en la ruta A.")
        step = f"v{index}"
        step_to_variable[step] = name
        transformers.append((step, _pipeline_for(spec, default_scaler), [name]))
    transformer = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)
    return transformer, step_to_variable


def _pipeline_for(spec: FeatureSpec, default_scaler: str) -> Pipeline:
    scaler_name = spec.scaler or default_scaler
    if spec.type in {"continuous", "discrete"}:
        steps = [("imputer", SimpleImputer(strategy="median", add_indicator=spec.missing_indicator))]
        if spec.transform == "log1p":
            steps.append(("log1p", Log1pNonNegative()))
        steps.append(("scaler", _scaler(scaler_name)))
        return Pipeline(steps)
    if spec.type == "binary":
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent", add_indicator=spec.missing_indicator),
                ),
                ("scaler", StandardScaler()),
            ]
        )
    if spec.type == "nominal":
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent", add_indicator=spec.missing_indicator),
                ),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
    if spec.type == "ordinal":
        categories = [float(item) for item in (spec.order or [])]
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent", add_indicator=spec.missing_indicator),
                ),
                (
                    "ordinal",
                    OrdinalEncoder(
                        categories=[categories],
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
        )
    raise ValueError(f"Tipo no soportado: {spec.type}")


def column_map_from_transformer(
    transformer: ColumnTransformer,
    step_to_variable: dict[str, str],
    variable_names: list[str],
) -> ColumnMap:
    transformed = list(transformer.get_feature_names_out())
    variable_of = []
    for name in transformed:
        step = name.split("__", 1)[0]
        if step not in step_to_variable:
            raise ValueError(f"No se pudo mapear la columna transformada {name}.")
        variable_of.append(step_to_variable[step])
    column_map = ColumnMap(transformed, variable_of, list(variable_names))
    if set(column_map.block_sizes) != set(variable_names):
        raise ValueError("El mapeo no cubre todas las variables.")
    if any(size < 1 for size in column_map.block_sizes.values()):
        raise ValueError("Hay una variable sin columnas transformadas.")
    return column_map


def fit_route_a(
    transformer: ColumnTransformer,
    step_to_variable: dict[str, str],
    variable_names: list[str],
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[ColumnMap, object, object, object]:
    """Ajusta el preprocesamiento solo con train y transforma las tres partes."""
    trained = transformer.fit(x_train)
    column_map = column_map_from_transformer(trained, step_to_variable, variable_names)
    return (
        column_map,
        trained.transform(x_train),
        trained.transform(x_val),
        trained.transform(x_test),
    )
