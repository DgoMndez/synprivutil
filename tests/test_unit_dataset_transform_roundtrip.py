import os

import numpy as np
import pandas as pd
from rdt import HyperTransformer

from privacy_utility_framework.dataset.dataset import Dataset, DatasetManager

DATASETS_PATH = os.path.join(os.path.dirname(__file__), "../datasets/")


def test_dataset_transform_roundtrip():
    filepath = os.path.join(DATASETS_PATH, "original/insurance.csv")  # mixed

    original = pd.read_csv(filepath)

    dataset = Dataset(original, name="test_dataset")
    dataset.fit_hypertransformer()
    dataset.transform()

    assert dataset.transformed_data is not None
    assert len(dataset.transformed_data) == len(original)

    reversed_data = dataset.hypertransformer.reverse_transform(dataset.transformed_data)

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


def test_custom_hypertransformer():
    original = pd.read_csv(os.path.join(DATASETS_PATH, "original/insurance.csv"))
    synthetic = pd.read_csv(
        os.path.join(DATASETS_PATH, "synthetic/insurance_datasets/ctgan_sample.csv")
    )
    dm = DatasetManager.from_dataframes(original, synthetic)
    hypertransformer = HyperTransformer()

    hypertransformer.update_transformers_by_sdtype(sdtype="numerical", transformer="gaussian")
    hypertransformer.update_transformers_by_sdtype(sdtype="categorical", transformer="uniform")
    dm.set_hypertransformer(hypertransformer)
    dm.transform_datasets()

    # TODO: test normal distribution
