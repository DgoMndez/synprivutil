"""
Module: synprivutil/tests/test_unit_quantile_transformer.py
Description: Unit test for QuantileRDTransformer.

Author: Domingo Méndez García
Email: domingo.mendezg@um.es
Date: 02/03/2026
"""

import numpy as np
import pandas as pd

from privacy_utility_framework.dataset.transformers import QuantileRDTransformer


def test_quantile_rd_transformer():
    # Create a simple dataset
    N = 10
    data = pd.DataFrame({"A": np.linspace(10, 100, N)})

    # Initialize the QuantileRDTransformer
    transformer = QuantileRDTransformer(
        n_quantiles=0, subsample=None, output_distribution="uniform"
    )

    # Fit the transformer to the data
    transformer.fit(data, column="A")

    # Transform the data
    transformed_data = transformer.transform(data)

    print(transformed_data["A"])

    # ECDF expected values: [0.1, 0.2, ..., 1.0] = [(r_i + 1 )/ N]
    # Sklearn's QuantileTransformer expected values:
    # [0, 0.11, 0.22, ..., 1.0] = [r_i / (N-1)]

    y = np.linspace(0, 1, N)
    print(y)

    # Check that the transformed data is approximately equal to the expected quantiles
    assert np.allclose(transformed_data["A"], y, atol=1e-2), (
        "Transformed values for column A do not match expected quantiles."
    )


def test_revert_data():
    x = pd.DataFrame({"A": np.array([10, 20, 30, 40, 50])})
    # Check that the inverse transform recovers the original
    transformer = QuantileRDTransformer()
    y = transformer.fit_transform(x, "A")
    inverse_transformed_data = transformer.reverse_transform(y)
    assert np.allclose(inverse_transformed_data["A"], x["A"], atol=1e-2), (
        "Inverse transformed values for column A do not match original data."
    )
