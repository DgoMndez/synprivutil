import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from privacy_utility_framework.metrics.privacy_metrics.distance.util import (
    cdf_cdist,
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


def test_cdf_cdist():
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

    # Calculate distances using cdf_cdist
    distances = cdf_cdist(
        XA, XB, metric="euclidean", output_distribution="uniform", original_data=original_data
    )

    YA = np.array([np.linspace(0, 1, 10, endpoint=False), np.linspace(0, 1, 10, endpoint=False)]).T
    YB = np.array(
        [np.linspace(0, 0.5, 10, endpoint=False), np.linspace(0, 0.75, 10, endpoint=False)]
    ).T
    YB[1::2, :] = YB[0::2, :]
    true_distances = cdist(YA, YB, metric="euclidean")

    assert np.allclose(distances, true_distances), (
        "CDF-transformed distances should match true distances in uniform space."
    )
