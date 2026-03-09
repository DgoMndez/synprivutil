"""
Module: synprivutil/src/privacy_utility_framework/metrics/privacy_metrics/distance/util.py
Description: Utilities for calculating metric distances for privacy metrics.

Author: Domingo Méndez García
Email: domingo.mendezg@um.es
Date: 27/02/2026
"""

# TODO:
# DONE 1. Integrate strategies into privacy/similarity metrics.
# 2. QuantileDistanceStrategy test
# 3. cdist Matrix bug
# 4. ECDF-transformer with tests

from collections.abc import Callable

import numpy as np
import pandas as pd
from rdt import HyperTransformer
from scipy.spatial import distance

from privacy_utility_framework.dataset.transformers import QuantileRDTransformer


def _to_2d_array(data):
    array_data = np.asarray(data)
    if array_data.ndim == 1:
        array_data = array_data.reshape(-1, 1)
    if array_data.ndim != 2:
        raise ValueError("Input data must be 1D or 2D.")

    return array_data


def _to_dataframe(data, columns):
    if isinstance(data, pd.DataFrame):
        if columns is not None:
            missing_columns = [column for column in columns if column not in data.columns]
            if missing_columns:
                raise ValueError(
                    "Input DataFrame is missing expected columns from HyperTransformer: "
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


def _get_quantile_hypertransformer(
    original_data, qt_factory=QuantileRDTransformer, output_distribution="uniform", **kwargs
):
    transformer = qt_factory(output_distribution=output_distribution, **kwargs)
    hypertransformer = HyperTransformer()
    hypertransformer._learn_config(original_data)
    hypertransformer.update_transformers_by_sdtype(transformer=transformer, sdtype="numerical")
    hypertransformer.fit(original_data)
    return hypertransformer


def transformed_dist(
    u, v, *, hypertransformer: HyperTransformer, base_metric: str | Callable = "euclidean", **kwargs
):
    """
    Compute distance between two 1-D arrays after applying a hypertransformer.

    Parameters
    ----------
    u : array_like
        First 1-D sample vector.
    v : array_like
        Second 1-D sample vector.
    hypertransformer : HyperTransformer
        Fitted HyperTransformer to apply to the samples.
    base_metric : str or callable, optional
        Distance metric used by ``scipy.spatial.distance.cdist`` in transformed
        space.
    **kwargs
        Additional keyword arguments forwarded to ``scipy.spatial.distance.cdist``.

    Returns
    -------
    float
        Distance between ``u`` and ``v`` in transformed space.
    """
    X_A, X_B = _transform_samples(u, v, hypertransformer)
    return distance.cdist(X_A, X_B, metric=base_metric, **kwargs)[0][0]


def transformed_cdist(
    XA,
    XB,
    hypertransformer: HyperTransformer,
    base_metric: str | Callable = "euclidean",
    *,
    out=None,
    **kwargs,
):
    """
    Calculate pairwise distances between two samples after applying a hypertransformer.

    Parameters
    ----------
    XA : array_like, shape (m_A, n_features) or (n_features,)
        First sample (original).
    XB : array_like, shape (m_B, n_features) or (n_features,)
        Second sample (synthetic).
    hypertransformer : HyperTransformer
        Fitted HyperTransformer to apply to the samples.
    base_metric : str or callable, optional
        Distance metric used by scipy.spatial.distance.cdist in transformed
        space (default: 'euclidean').
    **kwargs
        Additional keyword arguments forwarded to scipy.spatial.distance.cdist.

    Returns
    -------
    ndarray
        Pairwise distances between rows of XA and XB, computed in transformed space.
    """

    XA_transformed, XB_transformed = _transform_samples(XA, XB, hypertransformer)

    return distance.cdist(XA_transformed, XB_transformed, metric=base_metric, out=out, **kwargs)


def _transform_samples(XA, XB, hypertransformer):
    assert hypertransformer._fitted, (
        "HyperTransformer must be fitted before calling transformed_cdist."
    )

    input_columns = list(getattr(hypertransformer, "_input_columns", [])) or None
    XA = _to_dataframe(XA, input_columns)
    XB = _to_dataframe(XB, input_columns)

    assert XA.shape[1] == XB.shape[1], "XA and XB must have the same number of columns."

    XA_transformed = hypertransformer.transform(XA)
    XB_transformed = hypertransformer.transform(XB)
    return XA_transformed, XB_transformed


def quantile_dist(
    u,
    v,
    *,
    original_data,
    base_metric: str | Callable = "euclidean",
    output_distribution="uniform",
    qt_factory=QuantileRDTransformer,
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
    base_metric : str or callable, optional
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
    qt_factory=QuantileRDTransformer,
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
    base_metric : str or callable, optional
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
    hypertransformer = HyperTransformer()
    hypertransformer._learn_config(original_data)
    hypertransformer.update_transformers_by_sdtype(transformer=transformer, sdtype="numerical")
    hypertransformer.fit(original_data)
    X_transformed = hypertransformer.transform(X)
    return distance.pdist(X_transformed, metric=base_metric, **kwargs)


def quantile_cdist(
    XA,
    XB,
    base_metric="euclidean",
    output_distribution="uniform",
    original_data=None,
    qt_factory=QuantileRDTransformer,
    *,
    out=None,
    **kwargs,
):
    """
    Calculate distances after a copula-style marginal Cumulative Distribution Function (CDF) or \
        Probability Integral Transform (PIT), a.k.a. Quantile transformation, i.e.:
    
    d(x,y) = d(\Phi^-1(F(x)), \Phi^-1(F(y))) where F is the column-wise estimated CDF of the \
        pooled data and \Phi^-1 is the column-wise inverse of the CDF of the target distribution.

    It fits a QuantileRDTransformer on original data, applies the transformation to both \
        dataframe arguments and then computes the distance in the transformed space.
    
    Parameters
    ----------
    XA : array_like, shape (m_A, n_features) or (n_features,)
        First sample (original).
    XB : array_like, shape (m_B, n_features) or (n_features,)
        Second sample (synthetic).
    base_metric : str or callable, optional
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

    hypertransformer = _get_quantile_hypertransformer(
        original_data=original_data, qt_factory=qt_factory, output_distribution=output_distribution
    )
    XA_transformed = hypertransformer.transform(XA)
    XB_transformed = hypertransformer.transform(XB)
    return distance.cdist(XA_transformed, XB_transformed, metric=base_metric, out=out, **kwargs)


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
]

METRICS = {info["canonical_name"]: info for info in _METRIC_INFOS}
METRIC_ALIAS = {alias: info for info in _METRIC_INFOS for alias in info["aka"]}

METRIC_NAMES = list(METRICS.keys())


def custom_dist(u, v, metric: str | Callable = "euclidean", **kwargs):
    """
    Compute distance between two 1-D arrays using a custom metric.

    Parameters
    ----------
    u : array_like
        First 1-D sample vector.
    v : array_like
        Second 1-D sample vector.
    metric : str or callable
        Distance metric to use. If a string, it must be one of the registered custom metrics.
        If a callable, it should have the same signature as scipy.spatial.distance.cdist.
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
    Compute pairwise distances between rows of X using a custom metric.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Input data.
    metric : str or callable
        Distance metric to use. If a string, it must be one of the registered custom metrics.
        If a callable, it should have the same signature as scipy.spatial.distance.pdist.
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
    Compute pairwise distances between two samples using a custom metric.

    Parameters
    ----------
    XA : array_like, shape (m_A, n_features) or (n_features,)
        First sample (original).
    XB : array_like, shape (m_B, n_features) or (n_features,)
        Second sample (synthetic).
    metric : str or callable
        Distance metric to use. If a string, it must be one of the registered custom metrics.
        If a callable, it should have the same signature as scipy.spatial.distance.cdist.
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
