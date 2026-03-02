"""
Module: synprivutil/src/privacy_utility_framework/metrics/privacy_metrics/distance/util.py
Description: Utilities for calculating metric distances for privacy metrics.

Author: Domingo Méndez García
Email: domingo.mendezg@um.es
Date: 27/02/2026
"""

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


def transformed_cdist(
    XA, XB, hypertransformer: HyperTransformer, metric="euclidean", *, out=None, **kwargs
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
    metric : str or callable, optional
        Distance metric used by scipy.spatial.distance.cdist in transformed
        space (default: 'euclidean').
    **kwargs
        Additional keyword arguments forwarded to scipy.spatial.distance.cdist.

    Returns
    -------
    ndarray
        Pairwise distances between rows of XA and XB, computed in transformed space.
    """

    assert hypertransformer._fitted, (
        "HyperTransformer must be fitted before calling transformed_cdist."
    )

    input_columns = list(getattr(hypertransformer, "_input_columns", [])) or None
    XA = _to_dataframe(XA, input_columns)
    XB = _to_dataframe(XB, input_columns)

    assert XA.shape[1] == XB.shape[1], "XA and XB must have the same number of columns."

    XA_transformed = hypertransformer.transform(XA)
    XB_transformed = hypertransformer.transform(XB)

    return distance.cdist(XA_transformed, XB_transformed, metric=metric, out=out, **kwargs)


def cdf_cdist(
    XA,
    XB,
    metric="euclidean",
    output_distribution="uniform",
    original_data=None,
    *,
    out=None,
    **kwargs,
):
    """
    Calculate distances after a copula-style marginal Cumulative Distribution Function (CDF) or \
        Probability Integral Transform (PIT), a.k.a. Quantile transformation, i.e.:
    
    d(x,y) = d(\Phi^-1(F(x)), \Phi^-1(F(y))) where F is the column-wise empirical CDF of the \
        pooled data and \Phi^-1 is the column-wise inverse of the CDF of the target distribution.

    It fits a QuantileRDTransformer on original data, applies the transformation to both \
        dataframe arguments and then computes the distance in the transformed space.
    
    Parameters
    ----------
    XA : array_like, shape (m_A, n_features) or (n_features,)
        First sample (original).
    XB : array_like, shape (m_B, n_features) or (n_features,)
        Second sample (synthetic).
    metric : str or callable, optional
        Distance metric used by scipy.spatial.distance.cdist in transformed
        space (default: 'euclidean').
    output_distribution : str, optional
        Desired target distribution of transformed data ('uniform' or 'normal'). \
            Default is 'uniform'.
    original_data : pd.DataFrame, optional
        Original data to fit the QuantileRDTransformer. If None, XA is used only when
        XA is a DataFrame. Otherwise, it must be explicitly provided as a DataFrame.
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

    transformer = QuantileRDTransformer(output_distribution=output_distribution)
    hypertransformer = HyperTransformer()
    hypertransformer._learn_config(original_data)
    hypertransformer.update_transformers_by_sdtype(transformer=transformer, sdtype="numerical")
    hypertransformer.fit(original_data)
    XA_transformed = hypertransformer.transform(XA)
    XB_transformed = hypertransformer.transform(XB)
    return distance.cdist(XA_transformed, XB_transformed, metric=metric, out=out, **kwargs)
