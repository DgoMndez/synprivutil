"""
Module: synprivutil/tests/test_unit_quantile_transformer.py
Description: Unit test for QuantileRDTransformer.

Author: Domingo Méndez García
Email: domingo.mendezg@um.es
Date of creation: 02/03/2026
"""

import numpy as np
import pandas as pd

from privacy_utility_framework.dataset.transformers import QuantileColTransformer


def test_quantile_rd_transformer():
    # Create a simple dataset
    for i in range(1, 10):
        N = 1 << (i + 1)
        x = np.linspace(10, 100, N) * i + np.linspace(0, N - 1, N) * i
        data = pd.DataFrame({"A": x})

        # Initialize the QuantileRDTransformer
        transformer = QuantileColTransformer(
            n_quantiles=0, subsample=None, output_distribution="uniform"
        )

        # Fit the transformer to the data
        transformer.fit(data, column="A")

        # Transform the data
        transformed_data = transformer.transform(data)

        # ECDF expected values: [0.1, 0.2, ..., 1.0] = [(r_i + 1 )/ N]
        # Sklearn's QuantileTransformer expected values (interpolation):
        # [0, 0.11, 0.22, ..., 1.0] = [r_i / (N-1)]

        y = np.linspace(0, 1, N)

        # Check that the transformed data is approximately equal to the expected quantiles
        assert np.allclose(transformed_data["A"], y, atol=1e-2), (
            "Transformed values for column A do not match expected quantiles."
        )


def test_revert_data():
    transformer = QuantileColTransformer()
    for i in range(10):
        n = 1 << (i + 1)
        r = np.random.rand(n, 2 * i) * 100
        x = pd.DataFrame(r, columns=[f"col{j}" for j in range(2 * i)])

        # Check that the inverse transform recovers the original
        for c in x.columns:
            y = transformer.fit_transform(x, column=c)
            inverse_transformed_data = transformer.reverse_transform(y)
            assert np.allclose(inverse_transformed_data[c], x[c], atol=1e-2), (
                f"Inverse transformed values for column {c} do not match original data."
            )
