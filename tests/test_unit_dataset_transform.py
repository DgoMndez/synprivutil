import os

import numpy as np
import pandas as pd
import scipy
from rdt import HyperTransformer
from rdt.transformers import BaseTransformer
from scipy import stats

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


def normality_report(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 20:
        raise ValueError("Need at least 20 samples.")

    mean = x.mean()
    std = x.std(ddof=1)
    skew = stats.skew(x, bias=False)
    ex_kurt = stats.kurtosis(x, fisher=True, bias=False)  # 0 for normal

    xs = np.sort(x)
    p = (np.arange(1, n + 1) - 0.5) / n
    q = stats.norm.ppf(p)
    qq_rmse = np.sqrt(np.mean((xs - q) ** 2))

    # Optional p-values (diagnostic only)
    dagostino_p = stats.normaltest(x).pvalue if n >= 20 else np.nan
    shapiro_p = stats.shapiro(x).pvalue if n <= 5000 else np.nan
    anderson = stats.anderson(x, dist="norm")  # statistic + critical values

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "skew": skew,
        "excess_kurtosis": ex_kurt,
        "qq_rmse": qq_rmse,
        "dagostino_p": dagostino_p,
        "shapiro_p": shapiro_p,
        "anderson_stat": anderson.statistic,
        "anderson_5pct_crit": anderson.critical_values[2],  # 5%
    }


def is_approximately_standard_normal(r):
    return (
        abs(r["mean"]) < 0.05
        and abs(r["std"] - 1.0) < 0.2
        and abs(r["skew"]) < 0.10
        and abs(r["excess_kurtosis"]) < 0.20
        and r["qq_rmse"] < 0.05
    )


def test_custom_hypertransformer():
    # Insurance: mixed types, 4 numeric, 3 categorical
    original = pd.read_csv(os.path.join(DATASETS_PATH, "original/insurance.csv"))
    synthetic = pd.read_csv(
        os.path.join(DATASETS_PATH, "synthetic/insurance_datasets/ctgan_sample.csv")
    )
    dm = DatasetManager.from_dataframes(original, synthetic)
    hypertransformer = HyperTransformer()
    hypertransformer._learn_config(original)
    hypertransformer.update_transformers_by_sdtype(
        sdtype="numerical",
        transformer_name="GaussianNormalizer",
        transformer_parameters={"distribution": "gaussian_kde"},
    )
    hypertransformer.update_transformers_by_sdtype(
        sdtype="categorical", transformer_name="UniformEncoder"
    )

    assert not hypertransformer._fitted, "HyperTransformer should not be fitted yet."
    dm.set_hypertransformer(hypertransformer)
    assert hypertransformer._fitted, "HyperTransformer should be fitted after set."

    dm.transform_datasets()

    transformed_original = dm.get_original_dataset().get_transformed_data()
    transformed_synthetic = dm.get_synthetic_dataset().get_transformed_data()

    # Transformed datasets are not identical dataframes
    assert not transformed_original.equals(transformed_synthetic), (
        "Transformed original and synthetic datasets should not be identical."
    )

    # Test normalization N(0,1) for numerical columns (not statistically)

    numerical_cols = original.select_dtypes(include=[float, int]).columns

    for col in numerical_cols:
        mu = transformed_original[col].mean()
        sigma = transformed_original[col].std()
        assert abs(mu) < 0.1, f"Mean of transformed column {col} is not close to 0 (mu={mu})."
        assert abs(sigma - 1.0) < 0.2, (
            f"Std of transformed column {col} is not close to 1 (sigma={sigma})."
        )

        _, p = stats.normaltest(transformed_original[col])
        # Not necessarily normal, would fail with integer columns
        assert p > 1e-25, f"Transformed column {col} is not normally distributed (p={p})."

    # Test uniform distribution (Kolmogorov-Smirnov test) for categorical columns
    cat_cols = original.select_dtypes(include=["object", "category"]).columns
    output_cat_cols = []
    for col in cat_cols:
        t: BaseTransformer = hypertransformer.field_transformers[col]
        aux = t.get_output_columns()
        output_cat_cols.extend(aux)

    assert len(output_cat_cols) > 0, "No output columns found for categorical features."

    p_values = np.zeros((len(output_cat_cols),))
    for i, col in enumerate(output_cat_cols):
        _, p_values[i] = scipy.stats.kstest(transformed_original[col], "uniform")

    assert np.all(p_values > 0.01), (
        "Not all categorical columns are uniformly distributed after transformation."
    )
