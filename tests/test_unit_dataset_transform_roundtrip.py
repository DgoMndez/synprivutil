import os

import numpy as np
import pandas as pd

from privacy_utility_framework.dataset.dataset import Dataset


def test_dataset_transform_roundtrip():
    filepath = os.path.join(
        os.path.dirname(__file__), "../datasets/original/insurance.csv"
    )  # mixed

    original = pd.read_csv(filepath)

    dataset = Dataset(original, name="test_dataset")
    dataset.set_transformer()
    dataset.transform()

    assert dataset.transformed_data is not None
    assert len(dataset.transformed_data) == len(original)

    reversed_data = dataset.transformer.reverse_transform(dataset.transformed_data)

    assert list(reversed_data.columns) == list(original.columns)
    numeric_cols = original.select_dtypes(include=[float, int]).columns
    for col in numeric_cols:
        assert np.allclose(
            reversed_data[col].to_numpy(dtype=float), original[col].to_numpy(dtype=float)
        )
    categorical_cols = original.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        assert reversed_data[col].astype(str).tolist() == original[col].astype(str).tolist()

    rest_cols = original.columns.difference(numeric_cols.union(categorical_cols))
    for col in rest_cols:
        assert reversed_data[col].equals(original[col])
