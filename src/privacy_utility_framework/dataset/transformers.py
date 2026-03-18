"""
Module: src/privacy_utility_framework/dataset/transformers.py
Description: Column transformers for preprocessing of the data of one single feature. \
    The transformed values can be reverted to the original space, but depending on the case \
        it may or may not be a bijective mapping.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Literal, TypeAlias, get_args

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_integer_dtype
from scipy.stats import norm
from sklearn.preprocessing import (
    MinMaxScaler,
    QuantileTransformer,
)
from sklearn.preprocessing import (
    OneHotEncoder as SklearnOneHotEncoder,
)

ECDFSide: TypeAlias = Literal["right", "left"]
VALID_ECDF_SIDES = get_args(ECDFSide)


def _get_num_rows(data):
    if isinstance(data, (pd.Series | pd.DataFrame | np.ndarray)):
        return len(data)

    return np.asarray(data).shape[0]


def _ensure_2d(data):
    if isinstance(data, pd.DataFrame):
        return data.to_numpy()
    if isinstance(data, pd.Series):
        return data.to_numpy().reshape(-1, 1)
    if isinstance(data, np.ndarray) and data.ndim == 1:
        return data.reshape(-1, 1)

    return np.asarray(data)


def _as_series(data, column):
    if isinstance(data, pd.DataFrame):
        return data[column]
    if isinstance(data, pd.Series):
        return data

    array_data = np.asarray(data)
    if array_data.ndim == 2:
        if array_data.shape[1] != 1:
            raise ValueError("Expected 1D data or a single-column 2D array.")
        array_data = array_data.reshape(-1)

    return pd.Series(array_data, name=column)


class ColumnTransformer:
    """Minimal single-column transformer contract used by the framework."""

    INPUT_SDTYPE = None
    SUPPORTED_SDTYPES: list[str] = []
    OUTPUT_SDTYPES = {"value": "numerical"}

    def __init__(self):
        self.column: str | None = None
        self.columns: list[str] = []
        self._original_dtype = None
        self._output_columns: list[str] = []
        self._fitted = False

    def fit(self, data, column):
        series = _as_series(data, column)
        self.column = column
        self.columns = [column]
        self._original_dtype = series.dtype
        self._output_columns = []
        self._fitted = False
        self._fit(series)
        if not self._output_columns:
            self._output_columns = self._build_output_columns()
        self._fitted = True
        return self

    def transform(self, data):
        assert self._fitted, f"{type(self).__name__} must be fitted before calling transform."
        input_data = _as_series(data, self.column) if isinstance(data, pd.DataFrame) else data
        transformed = self._transform(input_data)
        return self._wrap_forward_output(transformed, index=getattr(data, "index", None))

    def reverse_transform(self, data):
        assert self._fitted, (
            f"{type(self).__name__} must be fitted before calling reverse_transform."
        )
        if isinstance(data, pd.DataFrame):
            input_data = data.loc[:, self.get_output_columns()]
        else:
            input_data = data

        reversed_data = self._reverse_transform(input_data)
        return self._wrap_reverse_output(reversed_data, index=getattr(data, "index", None))

    def fit_transform(self, data, column):
        self.fit(data, column=column)
        return self.transform(data)

    def get_output_columns(self):
        if self._output_columns:
            return list(self._output_columns)
        if self.column is None:
            return []
        return self._build_output_columns()

    def _build_output_columns(self):
        keys = list(self.OUTPUT_SDTYPES.keys())
        if keys == ["value"]:
            return [self.column]

        return [f"{self.column}.{key}" for key in keys]

    def _wrap_forward_output(self, transformed, index=None):
        if isinstance(transformed, pd.DataFrame):
            return transformed.copy()

        array_data = _ensure_2d(transformed)
        return pd.DataFrame(array_data, columns=self.get_output_columns(), index=index)

    def _wrap_reverse_output(self, reversed_data, index=None):
        if isinstance(reversed_data, pd.DataFrame):
            result = reversed_data.copy()
        else:
            array_data = _ensure_2d(reversed_data)
            result = pd.DataFrame(array_data, columns=[self.column], index=index)

        if self.column in result.columns:
            result[self.column] = self._restore_original_dtype(result[self.column])

        return result

    def _restore_original_dtype(self, series: pd.Series) -> pd.Series:
        if self._original_dtype is None:
            return series

        if is_integer_dtype(self._original_dtype):
            rounded = np.rint(pd.to_numeric(series, errors="raise").to_numpy())
            return pd.Series(rounded, index=series.index, name=series.name).astype(
                self._original_dtype
            )

        if is_bool_dtype(self._original_dtype):
            return series.astype(self._original_dtype)

        if is_datetime64_any_dtype(self._original_dtype):
            return pd.to_datetime(series).astype(self._original_dtype)

        if isinstance(self._original_dtype, pd.CategoricalDtype):
            return series.astype(self._original_dtype)

        return series.astype(self._original_dtype)

    def _fit(self, data):
        raise NotImplementedError

    def _transform(self, data):
        raise NotImplementedError

    def _reverse_transform(self, data):
        raise NotImplementedError


class IdentityTransformer(ColumnTransformer):
    """Pass-through transformer used for unsupported or already-compatible columns."""

    OUTPUT_SDTYPES = {"value": "unknown"}

    def _fit(self, data):
        return None

    def _transform(self, data):
        return _ensure_2d(data)

    def _reverse_transform(self, data):
        return _ensure_2d(data)


class MinMaxScalerTransformer(ColumnTransformer):
    """Single-column wrapper around sklearn's MinMaxScaler."""

    INPUT_SDTYPE = "numerical"
    SUPPORTED_SDTYPES = ["numerical"]
    OUTPUT_SDTYPES = {"value": "numerical"}

    def __init__(self, feature_range=(0, 1), clip=True):
        super().__init__()
        self._scaler = MinMaxScaler(feature_range=feature_range, clip=clip)

    def _fit(self, data):
        self._scaler.fit(_ensure_2d(data))

    def _transform(self, data):
        return self._scaler.transform(_ensure_2d(data))

    def _reverse_transform(self, data):
        return self._scaler.inverse_transform(_ensure_2d(data))


class QuantileColTransformer(ColumnTransformer):
    """Single-column wrapper around sklearn's QuantileTransformer."""

    INPUT_SDTYPE = "numerical"
    SUPPORTED_SDTYPES = ["numerical"]
    OUTPUT_SDTYPES = {"value": "numerical"}

    def __init__(
        self,
        n_quantiles: int | None = 0,
        output_distribution: str = "uniform",
        subsample: int | None = None,
        random_state=None,
        **q_transformer_kwargs,
    ):
        super().__init__()
        self._n_quantiles = n_quantiles
        self._output_distribution = output_distribution
        self._subsample = subsample
        self._random_state = random_state
        self._qtransformer = None
        self._qkwargs = q_transformer_kwargs

    def from_quantile_transformer(self, qtransformer):
        self._qtransformer = qtransformer
        self._n_quantiles = qtransformer.n_quantiles
        self._output_distribution = qtransformer.output_distribution
        self._subsample = qtransformer.subsample
        self._random_state = qtransformer.random_state
        self._fitted = hasattr(qtransformer, "quantiles_")
        return self

    def _build_transformer(self, data):
        n_rows = _get_num_rows(data)
        n_quantiles = self._n_quantiles if self._n_quantiles and self._n_quantiles > 0 else n_rows
        kwargs = {
            "n_quantiles": n_quantiles,
            "output_distribution": self._output_distribution,
            "random_state": self._random_state,
            **self._qkwargs,
        }
        if self._subsample is not None:
            kwargs["subsample"] = self._subsample if self._subsample > 0 else n_rows

        self._qtransformer = QuantileTransformer(**kwargs)

    def _fit(self, data):
        data = _ensure_2d(data)
        self._build_transformer(data)
        self._qtransformer.fit(data)

    def _transform(self, data):
        return self._qtransformer.transform(_ensure_2d(data))

    def _reverse_transform(self, data):
        return self._qtransformer.inverse_transform(_ensure_2d(data))


class GaussianNormalizer(QuantileColTransformer):
    """Approximate Gaussian normalizer based on quantile-to-normal mapping."""

    _SUPPORTED_DISTRIBUTIONS = {
        "truncated_gaussian",
        "gaussian",
        "gaussian_kde",
    }

    def __init__(
        self,
        distribution: str = "truncated_gaussian",
        n_quantiles: int | None = 0,
        subsample: int | None = None,
        random_state=None,
        **kwargs,
    ):
        if distribution not in self._SUPPORTED_DISTRIBUTIONS:
            raise ValueError(
                f"Unsupported distribution '{distribution}'. "
                f"Supported values are: {sorted(self._SUPPORTED_DISTRIBUTIONS)}."
            )
        self.distribution = distribution
        self._sorted_values = None
        self._sorted_scores = None
        super().__init__(
            n_quantiles=n_quantiles,
            output_distribution="normal",
            subsample=subsample,
            random_state=random_state,
            **kwargs,
        )

    def _build_reference_values(self, values):
        values = np.asarray(values).reshape(-1)
        n_rows = len(values)

        if self._subsample is not None and 0 < self._subsample < n_rows:
            indices = np.random.RandomState(self._random_state).choice(
                n_rows,
                size=self._subsample,
                replace=False,
            )
            reference_values = values[indices]
        else:
            reference_values = values

        reference_values = np.sort(reference_values)
        n_quantiles = self._n_quantiles if self._n_quantiles and self._n_quantiles > 0 else None
        if n_quantiles is not None and n_quantiles < len(reference_values):
            quantile_grid = np.linspace(0.0, 1.0, n_quantiles)
            reference_values = np.quantile(reference_values, quantile_grid)

        return np.asarray(reference_values, dtype=float).reshape(-1)

    def _fit(self, data):
        values = np.asarray(data).reshape(-1)
        self._sorted_values = self._build_reference_values(values)
        probabilities = (np.arange(len(self._sorted_values), dtype=float) + 0.5) / len(
            self._sorted_values
        )
        self._sorted_scores = norm.ppf(probabilities)

    def _transform(self, data):
        values = np.asarray(data).reshape(-1)
        transformed = np.empty(len(values), dtype=float)
        series = pd.Series(values)

        for value, index in series.groupby(series, dropna=False).groups.items():
            group_indices = np.asarray(index, dtype=int)
            left = np.searchsorted(self._sorted_values, value, side="left")
            right = np.searchsorted(self._sorted_values, value, side="right")

            if left == right:
                probability = (left + 0.5) / len(self._sorted_values)
                transformed[group_indices] = norm.ppf(np.clip(probability, 1e-12, 1.0 - 1e-12))
                continue

            scores = self._sorted_scores[left:right]
            if len(group_indices) == 1:
                transformed[group_indices] = scores.mean()
                continue

            if len(scores) == 1:
                transformed[group_indices] = scores[0]
                continue

            transformed[group_indices] = np.interp(
                np.linspace(0.0, 1.0, len(group_indices) + 2)[1:-1],
                np.linspace(0.0, 1.0, len(scores)),
                scores,
            )

        return transformed.reshape(-1, 1)

    def _reverse_transform(self, data):
        values = np.asarray(data).reshape(-1)
        indices = np.searchsorted(self._sorted_scores, values, side="left")
        indices = np.clip(indices, 0, len(self._sorted_values) - 1)
        return self._sorted_values[indices].reshape(-1, 1)


class OneHotEncoder(ColumnTransformer):
    """Single-column categorical one-hot encoder."""

    INPUT_SDTYPE = "categorical"
    SUPPORTED_SDTYPES = ["categorical"]

    def __init__(self, handle_unknown="ignore"):
        super().__init__()
        self.handle_unknown = handle_unknown
        self._encoder = SklearnOneHotEncoder(
            sparse_output=False,
            handle_unknown=handle_unknown,
        )

    def _fit(self, data):
        values = _ensure_2d(_as_series(data, self.column))
        self._encoder.fit(values)
        self._output_columns = [
            f"{self.column}__{idx}" for idx in range(len(self._encoder.categories_[0]))
        ]

    def _transform(self, data):
        values = _ensure_2d(_as_series(data, self.column))
        return self._encoder.transform(values)

    def _reverse_transform(self, data):
        return self._encoder.inverse_transform(_ensure_2d(data))


class UniformEncoder(ColumnTransformer):
    """Encode categories into disjoint intervals whose mixture is uniform on [0, 1]."""

    INPUT_SDTYPE = "categorical"
    SUPPORTED_SDTYPES = ["categorical"]
    OUTPUT_SDTYPES = {"value": "numerical"}

    def __init__(self, handle_unknown: str = "error"):
        super().__init__()
        if handle_unknown not in {"error"}:
            raise ValueError("handle_unknown must be 'error'.")

        self.handle_unknown = handle_unknown
        self._categories = []
        self._interval_edges = np.array([0.0, 1.0])

    @staticmethod
    def _value_mask(series, value):
        return series.isna() if pd.isna(value) else series == value

    def _fit(self, data):
        series = _as_series(data, self.column)
        value_counts = series.value_counts(sort=False, dropna=False)
        self._categories = list(value_counts.index)
        probabilities = value_counts.to_numpy(dtype=float) / float(len(series))
        self._interval_edges = np.concatenate(([0.0], np.cumsum(probabilities)))

    def _transform(self, data):
        series = _as_series(data, self.column)
        transformed = np.full(len(series), np.nan, dtype=float)

        for idx, category in enumerate(self._categories):
            mask = self._value_mask(series, category).to_numpy()
            count = int(mask.sum())
            if count == 0:
                continue

            start = self._interval_edges[idx]
            end = self._interval_edges[idx + 1]
            transformed[mask] = start + ((np.arange(count, dtype=float) + 0.5) / count) * (
                end - start
            )

        if np.isnan(transformed).any():
            known_categories = pd.Index(self._categories)
            unknown_mask = ~series.isin(known_categories)
            if known_categories.isna().any():
                unknown_mask &= ~series.isna()

            unknown_categories = list(pd.unique(series[unknown_mask]))
            raise ValueError(
                f"UniformEncoder encountered unknown categories in column '{self.column}': "
                f"{unknown_categories}."
            )

        return transformed.reshape(-1, 1)

    def _reverse_transform(self, data):
        values = np.asarray(data).reshape(-1)
        clipped = np.clip(values, 0.0, np.nextafter(1.0, 0.0))
        indices = np.searchsorted(self._interval_edges, clipped, side="right") - 1
        indices = np.clip(indices, 0, len(self._categories) - 1)
        recovered = np.asarray([self._categories[idx] for idx in indices], dtype=object)
        return recovered.reshape(-1, 1)


class ECDFTransformer(ColumnTransformer):
    """Single-column empirical CDF transformer."""

    INPUT_SDTYPE = "numerical"
    SUPPORTED_SDTYPES = ["numerical"]
    OUTPUT_SDTYPES = {"value": "numerical"}

    def __init__(self, subsample: int = 0, random_state=None, side: ECDFSide = "right"):
        super().__init__()
        self._subsample = subsample
        self._random_state = random_state
        self.side = side
        self._sorted_values = None
        self._n_samples = None

    def _fit(self, data):
        values = np.asarray(data if not isinstance(data, pd.Series) else data.to_numpy())

        if values.ndim != 1:
            raise ValueError(
                "ECDFTransformer expects 1D data (single column). "
                f"Got data with shape {values.shape}. "
                "Pass individual columns via fit(data, column='col_name')."
            )

        n_samples = len(values)
        if self._subsample > 0 and self._subsample < n_samples:
            indices = np.random.RandomState(self._random_state).choice(
                n_samples,
                size=self._subsample,
                replace=False,
            )
            sampled_values = values[indices]
            self._n_samples = self._subsample
        else:
            sampled_values = values
            self._n_samples = n_samples

        self._sorted_values = np.sort(sampled_values)

    def _transform(self, data, side: ECDFSide | None = None):
        values = np.asarray(data if not isinstance(data, pd.Series) else data.to_numpy())

        if values.ndim != 1:
            raise ValueError(
                "ECDFTransformer expects 1D data (single column). "
                f"Got data with shape {values.shape}."
            )

        side = self._side if side is None else side
        ranks = np.searchsorted(self._sorted_values, values, side=side)
        transformed = ranks / self._n_samples
        return transformed.reshape(-1, 1)

    def _reverse_transform(self, data, side: ECDFSide | None = None):
        if side is not None:
            self._check_side(side)
        else:
            side = self._side

        ecdf_values = np.asarray(data if not isinstance(data, pd.Series) else data.to_numpy())
        ecdf_values = ecdf_values.reshape(-1)

        if side == "right":
            indices = np.floor(ecdf_values * self._n_samples).astype(int) - 1
        else:
            indices = np.floor(ecdf_values * self._n_samples).astype(int)

        indices = np.clip(indices, 0, self._n_samples - 1)
        recovered_values = self._sorted_values[indices]
        return recovered_values.reshape(-1, 1)

    @property
    def side(self) -> ECDFSide:
        return self._side

    @staticmethod
    def _check_side(value: ECDFSide) -> None:
        if value not in VALID_ECDF_SIDES:
            raise ValueError(f"side must be 'right' or 'left', got '{value}'.")

    @side.setter
    def side(self, value: ECDFSide) -> None:
        self._check_side(value)
        self._side = value


TRANSFORMER_NAME_TO_CLASS = {
    "ColumnTransformer": ColumnTransformer,
    "IdentityTransformer": IdentityTransformer,
    "MinMaxScalerTransformer": MinMaxScalerTransformer,
    "QuantileRDTransformer": QuantileColTransformer,
    "GaussianNormalizer": GaussianNormalizer,
    "OneHotEncoder": OneHotEncoder,
    "UniformEncoder": UniformEncoder,
    "ECDFTransformer": ECDFTransformer,
}


def create_transformer(name: str, parameters: dict | None = None):
    try:
        transformer_cls = TRANSFORMER_NAME_TO_CLASS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown transformer '{name}'.") from exc

    return transformer_cls(**(parameters or {}))


DEFAULT_TRANSFORMERS = {
    "numerical": MinMaxScalerTransformer(),
    "categorical": OneHotEncoder(),
    "datetime": IdentityTransformer(),
    "other": IdentityTransformer(),
}


def get_default_transformer(sdtype):
    return deepcopy(DEFAULT_TRANSFORMERS.get(sdtype, IdentityTransformer()))


def get_default_transformers():
    return {sdtype: deepcopy(transformer) for sdtype, transformer in DEFAULT_TRANSFORMERS.items()}
