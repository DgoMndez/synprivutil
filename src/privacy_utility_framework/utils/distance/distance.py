"""
Module: synprivutil/src/privacy_utility_framework/metrics/privacy_metrics/distance/util.py
Description: Utilities for calculating metric distances for privacy metrics.

Author: Domingo Méndez García
Email: domingo.mendezg@um.es
Date: 27/02/2026
"""

# DONE 1. Integrate strategies into privacy/similarity metrics.
# DONE 2. QuantileDistanceStrategy test
# DONE 3. cdist Matrix bug
# DONE 4. ECDF-transformer with tests

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.spatial import distance

from privacy_utility_framework.dataset.tabletransformer import TableTransformer
from privacy_utility_framework.dataset.transformers import ECDFTransformer, QuantileColTransformer


def _to_2d_array(data):
    """Normalize 1D or 2D array-like input into a 2D NumPy array."""
    array_data = np.asarray(data)
    if array_data.ndim == 1:
        array_data = array_data.reshape(-1, 1)
    if array_data.ndim != 2:
        raise ValueError("Input data must be 1D or 2D.")

    return array_data


def _to_dataframe(data, columns):
    """Return input data as a dataframe aligned to the expected feature columns."""
    if isinstance(data, pd.DataFrame):
        if columns is not None:
            missing_columns = [column for column in columns if column not in data.columns]
            if missing_columns:
                raise ValueError(
                    "Input DataFrame is missing expected columns from TableTransformer: "
                    f"{missing_columns}."
                )
            data = data.loc[:, columns]

        return data

    array_data = _to_2d_array(data)
    if columns is not None and array_data.shape[1] != len(columns):
        raise ValueError(f"Input data has {array_data.shape[1]} columns, expected {len(columns)}.")

    if columns is None:
        columns = [f"column_{idx}" for idx in range(array_data.shape[1])]

    return pd.DataFrame(array_data, columns=columns)


def _get_quantile_tabletransformer(
    original_data, qt_factory=QuantileColTransformer, output_distribution="uniform", **kwargs
):
    """
    Fit a table transformer that applies the configured quantile transform to numerical columns.
    """
    transformer = qt_factory(output_distribution=output_distribution, **kwargs)
    tabletransformer = TableTransformer()
    tabletransformer._learn_config(original_data)
    tabletransformer.update_transformers_by_sdtype(transformer=transformer, sdtype="numerical")
    tabletransformer.fit(original_data)
    return tabletransformer


def _get_ecdf_tabletransformer(original_data, ecdf_factory=ECDFTransformer, **kwargs):
    """Fit a table transformer that applies the configured ECDF transform to numerical columns."""
    transformer = ecdf_factory(**kwargs)
    tabletransformer = TableTransformer()
    tabletransformer._learn_config(original_data)
    tabletransformer.update_transformers_by_sdtype(transformer=transformer, sdtype="numerical")
    tabletransformer.fit(original_data)
    return tabletransformer


def _build_ecdf_references(original_data, ecdf_factory=ECDFTransformer, **kwargs):
    """Fit one ECDF transformer per column and extract sorted reference values."""
    original_data = _to_dataframe(original_data, getattr(original_data, "columns", None))

    references = []
    for column in original_data.columns:
        transformer = ecdf_factory(**kwargs)
        transformer.fit(data=original_data, column=column)

        sorted_values = getattr(transformer, "_sorted_values", None)
        n_samples = getattr(transformer, "_n_samples", None)
        if sorted_values is None or n_samples is None:
            raise ValueError(
                "ECDF transformer must expose fitted '_sorted_values' and '_n_samples' "
                "attributes to be used in ECDFDistance."
            )

        references.append((np.asarray(sorted_values), int(n_samples)))

    return list(original_data.columns), references


def _ecdf_bounds_from_references(data, columns, references):
    """Transform each column into its left- and right-ECDF bounds."""
    data = _to_dataframe(data, columns)
    values = data.to_numpy()

    left = np.empty(values.shape, dtype=float)
    right = np.empty(values.shape, dtype=float)

    for idx, (sorted_values, n_samples) in enumerate(references):
        column_values = values[:, idx]
        left[:, idx] = np.searchsorted(sorted_values, column_values, side="left") / n_samples
        right[:, idx] = np.searchsorted(sorted_values, column_values, side="right") / n_samples

    return left, right


def _ecdf_distance_matrix_from_bounds(
    left_A,
    right_A,
    left_B,
    right_B,
    base_metric="euclidean",
    *,
    out=None,
    **kwargs,
):
    """Compute ECDF interval distances from precomputed left/right bounds."""
    nA, n_features = left_A.shape
    nB = left_B.shape[0]
    components = np.empty((nA * nB, n_features), dtype=float)
    for idx in range(n_features):
        u = left_B[:, idx][None, :] - right_A[:, idx][:, None]
        v = left_A[:, idx][:, None] - right_B[:, idx][None, :]
        components[:, idx] = np.maximum(0.0, np.maximum(u, v)).reshape(-1)

    metric = base_metric or "euclidean"
    distances = distance.cdist(
        components,
        np.zeros((1, n_features), dtype=float),
        metric=metric,
        **kwargs,
    ).reshape(nA, nB)

    if out is not None:
        out[...] = distances
        return out
    return distances


def transformed_dist(
    u, v, *, tabletransformer: TableTransformer, base_metric: str | Callable = "euclidean", **kwargs
):
    """
    Compute distance between two 1-D arrays after applying a tabletransformer.

    Parameters
    ----------
    u : array_like
        First 1-D sample vector.
    v : array_like
        Second 1-D sample vector.
    tabletransformer : TableTransformer
        Fitted TableTransformer to apply to the samples.
    base_metric : str or Callable, optional
        Distance metric used by ``scipy.spatial.distance.cdist`` in transformed
        space.
    **kwargs
        Additional keyword arguments forwarded to ``scipy.spatial.distance.cdist``.

    Returns
    -------
    float
        Distance between ``u`` and ``v`` in transformed space.
    """
    X_A, X_B = _transform_samples(u, v, tabletransformer)
    return distance.cdist(X_A, X_B, metric=base_metric, **kwargs)[0][0]


def transformed_cdist(
    XA,
    XB,
    tabletransformer: TableTransformer,
    base_metric: str | Callable = "euclidean",
    *,
    out=None,
    **kwargs,
):
    """
    Calculate pairwise distances between two samples after applying a tabletransformer.

    Parameters
    ----------
    XA : array_like, shape (m_A, n_features) or (n_features,)
        First sample (original).
    XB : array_like, shape (m_B, n_features) or (n_features,)
        Second sample (synthetic).
    tabletransformer : TableTransformer
        Fitted TableTransformer to apply to the samples.
    base_metric : str or Callable, optional
        Distance metric used by scipy.spatial.distance.cdist in transformed
        space (default: 'euclidean').
    **kwargs
        Additional keyword arguments forwarded to scipy.spatial.distance.cdist.

    Returns
    -------
    ndarray
        Pairwise distances between rows of XA and XB, computed in transformed space.
    """

    XA_transformed, XB_transformed = _transform_samples(XA, XB, tabletransformer)

    return distance.cdist(XA_transformed, XB_transformed, metric=base_metric, out=out, **kwargs)


def _transform_samples(XA, XB, tabletransformer):
    assert tabletransformer._fitted, (
        "TableTransformer must be fitted before calling transformed_cdist."
    )

    input_columns = list(getattr(tabletransformer, "_input_columns", [])) or None
    XA = _to_dataframe(XA, input_columns)
    XB = _to_dataframe(XB, input_columns)

    assert XA.shape[1] == XB.shape[1], "XA and XB must have the same number of columns."

    XA_transformed = tabletransformer.transform(XA)
    XB_transformed = tabletransformer.transform(XB)
    return XA_transformed, XB_transformed


def quantile_dist(
    u,
    v,
    *,
    original_data,
    base_metric: str | Callable = "euclidean",
    output_distribution="uniform",
    qt_factory=QuantileColTransformer,
    **kwargs,
):
    """
    Compute distance between two 1-D arrays after quantile transformation.

    Parameters
    ----------
    u : array_like
        First 1-D sample vector.
    v : array_like
        Second 1-D sample vector.
    original_data : pd.DataFrame
        Reference data used to fit the quantile transformer.
    base_metric : str or Callable, optional
        Distance metric used by ``scipy.spatial.distance.cdist`` in transformed
        space.
    output_distribution : str, optional
        Desired target distribution of transformed data ('uniform' or 'normal'). \
            Default is 'uniform'.
    qt_factory : class, optional
        Factory for creating the QuantileRDTransformer instance. Default is QuantileRDTransformer.
    **kwargs
        Additional keyword arguments forwarded to ``scipy.spatial.distance.cdist``.

    Returns
    -------
    float
        Distance between ``u`` and ``v`` in quantile-transformed space.
    """
    return quantile_cdist(
        np.asarray([u]),
        np.asarray([v]),
        base_metric=base_metric,
        output_distribution=output_distribution,
        original_data=original_data,
        qt_factory=qt_factory,
        **kwargs,
    )[0][0]


def quantile_pdist(
    X,
    *,
    original_data,
    base_metric: str | Callable = "euclidean",
    output_distribution="uniform",
    qt_factory=QuantileColTransformer,
    **kwargs,
):
    """
    Compute pairwise distances between rows of X after quantile transformation.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Input data.
    original_data : pd.DataFrame
        Reference data used to fit the quantile transformer.
    base_metric : str or Callable, optional
        Distance metric used by ``scipy.spatial.distance.pdist`` in transformed
        space (default: 'euclidean').
    output_distribution : str, optional
        Desired target distribution of transformed data ('uniform' or 'normal'). \
            Default is 'uniform'.
    qt_factory : class, optional
        Factory for creating the QuantileRDTransformer instance. Default is QuantileRDTransformer.
    **kwargs
        Additional keyword arguments forwarded to ``scipy.spatial.distance.pdist``.

    Returns
    -------
    ndarray
        Pairwise distances between rows of X in quantile-transformed space.
    """
    if original_data is None:
        assert isinstance(X, pd.DataFrame), (
            "X must be a DataFrame if original_data is not provided."
        )
        original_data = X
    X = _to_dataframe(X, original_data.columns)
    assert X.shape[1] == original_data.shape[1], (
        "X and original_data must have the same number of columns."
    )
    transformer = qt_factory(output_distribution=output_distribution)
    tabletransformer = TableTransformer()
    tabletransformer._learn_config(original_data)
    tabletransformer.update_transformers_by_sdtype(transformer=transformer, sdtype="numerical")
    tabletransformer.fit(original_data)
    X_transformed = tabletransformer.transform(X)
    return distance.pdist(X_transformed, metric=base_metric, **kwargs)


def quantile_cdist(
    XA,
    XB,
    base_metric="euclidean",
    output_distribution="uniform",
    original_data=None,
    qt_factory=QuantileColTransformer,
    *,
    out=None,
    **kwargs,
):
    """
    Calculate distances after a copula-style marginal Cumulative Distribution Function (CDF) or \
        Probability Integral Transform (PIT), a.k.a. Quantile transformation, i.e.:
    
    d(x,y) = d(\\Phi^-1(F(x)), \\Phi^-1(F(y))) where F is the column-wise estimated CDF of the \
        pooled data and \\Phi^-1 is the column-wise inverse of the CDF of the target distribution.

    It fits a QuantileRDTransformer on original data, applies the transformation to both \
        dataframe arguments and then computes the distance in the transformed space.
    
    Parameters
    ----------
    XA : array_like, shape (m_A, n_features) or (n_features,)
        First sample (original).
    XB : array_like, shape (m_B, n_features) or (n_features,)
        Second sample (synthetic).
    base_metric : str or Callable, optional
        Distance metric used by scipy.spatial.distance.cdist in transformed
        space (default: 'euclidean').
    output_distribution : str, optional
        Desired target distribution of transformed data ('uniform' or 'normal'). \
            Default is 'uniform'.
    original_data : pd.DataFrame, optional
        Original data to fit the QuantileRDTransformer. If None, XA is used only when
        XA is a DataFrame. Otherwise, it must be explicitly provided as a DataFrame.
    qt_factory : class, optional
        Factory for creating the QuantileRDTransformer instance. Default is QuantileRDTransformer.
    **kwargs
        Additional keyword arguments forwarded to scipy.spatial.distance.cdist.

    Returns
    -------
    y : ndarray
        Pairwise distances between rows of XA and XB, computed in copula space. \
        y[i, j] is the distance between XA[i] and XB[j].
    """
    if original_data is None:
        if isinstance(XA, pd.DataFrame):
            original_data = XA
        else:
            raise ValueError("XA must be a pandas DataFrame when original_data = None.")

    XA = _to_dataframe(XA, original_data.columns)
    XB = _to_dataframe(XB, original_data.columns)

    assert XA.shape[1] == XB.shape[1] == original_data.shape[1], (
        "XA, XB, and original_data must have the same number of columns."
    )

    tabletransformer = _get_quantile_tabletransformer(
        original_data=original_data, qt_factory=qt_factory, output_distribution=output_distribution
    )
    XA_transformed = tabletransformer.transform(XA)
    XB_transformed = tabletransformer.transform(XB)
    return distance.cdist(XA_transformed, XB_transformed, metric=base_metric, out=out, **kwargs)


def ecdf_dist(
    u,
    v,
    *,
    original_data,
    base_metric: str | Callable = "euclidean",
    ecdf_factory=ECDFTransformer,
    **kwargs,
):
    """
    Compute distance between two 1-D arrays using the ECDF distance.

    The distance compares the ECDF intervals induced by the reference data rather than relying on
    a point-valued marginal transform.
    """
    return ecdf_cdist(
        np.asarray([u]),
        np.asarray([v]),
        base_metric=base_metric,
        original_data=original_data,
        ecdf_factory=ecdf_factory,
        **kwargs,
    )[0][0]


def ecdf_pdist(
    X,
    *,
    original_data,
    base_metric: str | Callable = "euclidean",
    ecdf_factory=ECDFTransformer,
    **kwargs,
):
    """
    Compute pairwise distances between rows of ``X`` using the ECDF distance.

    The ECDF reference is fitted on ``original_data`` when provided, or on ``X`` itself.
    """
    if original_data is None:
        assert isinstance(X, pd.DataFrame), (
            "X must be a DataFrame if original_data is not provided."
        )
        original_data = X

    X = _to_dataframe(X, original_data.columns)
    assert X.shape[1] == original_data.shape[1], (
        "X and original_data must have the same number of columns."
    )
    distances = ecdf_cdist(
        X,
        X,
        base_metric=base_metric,
        original_data=original_data,
        ecdf_factory=ecdf_factory,
        **kwargs,
    )
    return distance.squareform(distances, force="tovector", checks=False)


def ecdf_cdist(
    XA,
    XB,
    base_metric="euclidean",
    original_data=None,
    ecdf_factory=ECDFTransformer,
    *,
    out=None,
    **kwargs,
):
    """
    Calculate distances using ECDF interval gaps per column.

    Instead of transforming each value to a single ECDF score, this metric represents each value
    by its left/right ECDF bounds and measures how far those intervals are from one another across
    columns.

    Parameters
    ----------
    XA : array_like, shape (m_A, n_features) or (n_features,)
        First sample.
    XB : array_like, shape (m_B, n_features) or (n_features,)
        Second sample.
    base_metric : str or Callable, optional
        Base metric used to aggregate per-column ECDF interval gaps.
    original_data : pd.DataFrame, optional
        Reference data used to fit the ECDF per column. If ``None``, ``XA`` must be a DataFrame
        and is used as the reference data.
    ecdf_factory : class, optional
        Factory used to create the per-column ECDF transformers.
    out : ndarray, optional
        Output buffer receiving the pairwise distance matrix.
    **kwargs
        Additional keyword arguments forwarded to the base distance computation.

    Returns
    -------
    ndarray
        Pairwise ECDF distances between rows of ``XA`` and ``XB``.
    """
    if original_data is None:
        if isinstance(XA, pd.DataFrame):
            original_data = XA
        else:
            raise ValueError("XA must be a pandas DataFrame when original_data = None.")

    XA = _to_dataframe(XA, original_data.columns)
    XB = _to_dataframe(XB, original_data.columns)

    assert XA.shape[1] == XB.shape[1] == original_data.shape[1], (
        "XA, XB, and original_data must have the same number of columns."
    )
    columns, references = _build_ecdf_references(original_data, ecdf_factory=ecdf_factory)
    left_A, right_A = _ecdf_bounds_from_references(XA, columns, references)
    left_B, right_B = _ecdf_bounds_from_references(XB, columns, references)
    return _ecdf_distance_matrix_from_bounds(
        left_A,
        right_A,
        left_B,
        right_B,
        base_metric=base_metric,
        out=out,
        **kwargs,
    )


# Registry of custom metrics
_METRIC_INFOS = [
    {
        "canonical_name": "transformed",
        "aka": {"transformed"},
        "dist_func": transformed_dist,
        "cdist_func": transformed_cdist,
        "pdist_func": None,
    },
    {
        "canonical_name": "quantile",
        "aka": {"quantile"},
        "dist_func": quantile_dist,
        "cdist_func": quantile_cdist,
        "pdist_func": quantile_pdist,
    },
    {
        "canonical_name": "ecdf",
        "aka": {"ecdf"},
        "dist_func": ecdf_dist,
        "cdist_func": ecdf_cdist,
        "pdist_func": ecdf_pdist,
    },
]

METRICS = {info["canonical_name"]: info for info in _METRIC_INFOS}
METRIC_ALIAS = {alias: info for info in _METRIC_INFOS for alias in info["aka"]}

METRIC_NAMES = list(METRICS.keys())


def custom_dist(u, v, metric: str | Callable = "euclidean", **kwargs):
    """
    Compute distance between two 1-D arrays using a custom or built-in metric.

    Parameters
    ----------
    u : array_like
        First 1-D sample vector.
    v : array_like
        Second 1-D sample vector.
    metric : str or Callable
        Distance metric to use. Registered project metrics such as ``"quantile"``,
        ``"transformed"``, and ``"ecdf"`` are supported in addition to scipy metric names.
        If a callable is provided, it is used directly.
    **kwargs
        Additional keyword arguments forwarded to the distance function.

    Returns
    -------
    float
        Distance between ``u`` and ``v`` according to the specified metric.
    """
    if isinstance(metric, str):
        if metric in METRIC_ALIAS:
            dist_func = METRIC_ALIAS[metric]["dist_func"]
            if dist_func is None:
                raise ValueError(f"Custom metric '{metric}' does not support dist.")
        else:
            return distance.cdist([u], [v], metric=metric, **kwargs)[0][0]
    elif callable(metric):
        dist_func = metric

    return dist_func(u, v, **kwargs)


def custom_pdist(X, metric: str | Callable = "euclidean", **kwargs):
    """
    Compute pairwise distances between rows of X using a custom or built-in metric.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Input data.
    metric : str or Callable
        Distance metric to use. Registered project metrics such as ``"quantile"``,
        ``"transformed"``, and ``"ecdf"`` are supported in addition to scipy metric names.
        If a callable is provided, it is used directly.
    **kwargs
        Additional keyword arguments forwarded to the distance function.

    Returns
    -------
    ndarray
        Pairwise distances between rows of X.
    """
    if isinstance(metric, str):
        if metric in METRIC_ALIAS:
            dist_func = METRIC_ALIAS[metric]["pdist_func"]
            if dist_func is None:
                raise ValueError(f"Custom metric '{metric}' does not support pdist.")
        else:
            return distance.pdist(X, metric=metric, **kwargs)
    elif callable(metric):
        dist_func = metric

    return dist_func(X, **kwargs)


def custom_cdist(XA, XB, metric: str | Callable = "euclidean", *, out=None, **kwargs):
    """
    Compute pairwise distances between two samples using a custom or built-in metric.

    Parameters
    ----------
    XA : array_like, shape (m_A, n_features) or (n_features,)
        First sample (original).
    XB : array_like, shape (m_B, n_features) or (n_features,)
        Second sample (synthetic).
    metric : str or Callable
        Distance metric to use. Registered project metrics such as ``"quantile"``,
        ``"transformed"``, and ``"ecdf"`` are supported in addition to scipy metric names.
        If a callable is provided, it is used directly.
    **kwargs
        Additional keyword arguments forwarded to the distance function.

    Returns
    -------
    ndarray
        Pairwise distances between rows of XA and XB.
    """
    if isinstance(metric, str):
        if metric in METRIC_ALIAS:
            dist_func = METRIC_ALIAS[metric]["cdist_func"]
            if dist_func is None:
                raise ValueError(f"Custom metric '{metric}' does not support cdist.")
        else:
            return distance.cdist(XA, XB, metric=metric, out=out, **kwargs)
    elif callable(metric):
        dist_func = metric

    return dist_func(XA, XB, out=out, **kwargs)
