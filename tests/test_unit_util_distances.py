from unittest.mock import Mock

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from privacy_utility_framework.dataset.transformers import QuantileRDTransformer
from privacy_utility_framework.utils.distance.distance import (
    ecdf_cdist,
    quantile_cdist,
    transformed_cdist,
)


def test_identity_transformed_cdist():
    # Create a simple dataset
    XA = np.array([[1, 2], [3, 4], [5, 6]])
    XB = np.array([[1, 2], [3, 4], [5, 6]])
    XC = np.array([[2, 3], [4, 5], [6, 7]])

    # Create a simple HyperTransformer that does nothing (identity)
    class IdentityTransformer:
        def fit(self, X):
            pass

        def transform(self, X):
            return X

        _fitted = True

    hypertransformer = IdentityTransformer()

    # Calculate distances using transformed_cdist
    distances = transformed_cdist(XA, XB, hypertransformer)

    # The distance between identical points should be zero
    assert np.all(distances.diagonal() == 0), "Distances between identical points should be zero."

    distances = transformed_cdist(XA, XC, hypertransformer)
    true_distances = cdist(XA, XC, metric="euclidean")

    assert np.allclose(distances, true_distances), (
        "Transformed distances should match true distances for identity transformer."
    )


def test_linear_transformed_cdist():
    # Create a simple dataset
    XA = np.array([[1, 2], [3, 4], [5, 6]])
    XB = np.array([[2, 4], [6, 8], [10, 12]])

    # Create a simple HyperTransformer that scales by a factor of 2
    class LinearTransformer:
        def fit(self, X):
            pass

        def transform(self, X):
            return X * 2

        _fitted = True

    hypertransformer = LinearTransformer()

    # Calculate distances using transformed_cdist
    distances = transformed_cdist(XA, XB, hypertransformer)

    # The distance should be scaled by a factor of 2
    true_distances = cdist(XA * 2, XB * 2, metric="euclidean")

    assert np.allclose(distances, true_distances), (
        "Transformed distances should match true distances for linear transformer."
    )


def test_quantile_cdist():
    # Create a simple dataset
    XA = np.array(
        [np.linspace(0, 100, 10, endpoint=False), np.linspace(100, 200, 10, endpoint=False)]
    ).T
    XB = np.array(
        [np.linspace(0, 50, 10, endpoint=False), np.linspace(0, 150, 10, endpoint=False)]
    ).T

    column1 = np.linspace(10, 100, 10)
    column2 = np.linspace(110, 200, 10)
    original_data = pd.DataFrame({"col1": column1, "col2": column2})

    class RankQuantileMockTransformer(QuantileRDTransformer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._sorted_values = None
            self._n_samples = None

        def _fit(self, data):
            values = np.asarray(data).reshape(-1)
            self._sorted_values = np.sort(values)
            self._n_samples = len(values)
            self._fitted = True

        def _transform(self, data):
            assert self._fitted, "RankQuantileMockTransformer must be fitted before transform."
            values = np.asarray(data).reshape(-1)
            ranks = np.searchsorted(self._sorted_values, values, side="right") - 1
            ranks = np.clip(ranks, 0, self._n_samples - 1)
            transformed = (ranks + 1) / self._n_samples
            return transformed.reshape(-1, 1)

    qt_factory = Mock(side_effect=lambda **kwargs: RankQuantileMockTransformer(**kwargs))

    # Calculate distances using cdf_cdist
    distances = quantile_cdist(
        XA,
        XB,
        base_metric="euclidean",
        output_distribution="uniform",
        original_data=original_data,
        qt_factory=qt_factory,
    )

    def rank_transform(values, reference_values):
        sorted_values = np.sort(np.asarray(reference_values).reshape(-1))
        n_samples = len(sorted_values)
        ranks = np.searchsorted(sorted_values, np.asarray(values).reshape(-1), side="right") - 1
        ranks = np.clip(ranks, 0, n_samples - 1)
        return (ranks + 1) / n_samples

    YA = np.column_stack(
        [
            rank_transform(XA[:, 0], original_data["col1"].to_numpy()),
            rank_transform(XA[:, 1], original_data["col2"].to_numpy()),
        ]
    )
    YB = np.column_stack(
        [
            rank_transform(XB[:, 0], original_data["col1"].to_numpy()),
            rank_transform(XB[:, 1], original_data["col2"].to_numpy()),
        ]
    )
    true_distances = cdist(YA, YB, metric="euclidean")

    qt_factory.assert_called_once_with(output_distribution="uniform")

    assert np.allclose(distances, true_distances), (
        "CDF-transformed distances should match true distances in uniform space."
    )


def test_ecdf_cdist_matches_interval_gap_definition():
    original_data = pd.DataFrame(
        {
            "col1": np.array([1.0, 1.0, 3.0, 5.0]),
            "col2": np.array([10.0, 20.0, 20.0, 40.0]),
        }
    )
    XA = np.array([[1.0, 20.0], [5.0, 10.0]])
    XB = np.array([[3.0, 40.0], [1.0, 10.0]])

    distances = ecdf_cdist(XA, XB, original_data=original_data, base_metric="euclidean")

    expected = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.25],
        ]
    )

    assert np.allclose(distances, expected)
