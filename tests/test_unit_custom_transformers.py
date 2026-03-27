import numpy as np
import pandas as pd
import pytest

from privacy_utility_framework.dataset.transformers import GaussianNormalizer, UniformEncoder


def test_uniform_encoder_raises_on_unknown_category():
    transformer = UniformEncoder()
    train = pd.DataFrame({"category": ["a", "b", "a"]})
    test = pd.DataFrame({"category": ["a", "z", "b"]})

    transformer.fit(train, column="category")

    with pytest.raises(ValueError, match="unknown categories"):
        transformer.transform(test)


def test_gaussian_normalizer_honors_n_quantiles_and_random_state():
    data = pd.DataFrame({"value": np.linspace(0.0, 99.0, 100)})

    first = GaussianNormalizer(
        distribution="gaussian_kde",
        subsample=20,
        random_state=7,
        n_quantiles=5,
    )
    second = GaussianNormalizer(
        distribution="gaussian_kde",
        subsample=20,
        random_state=7,
        n_quantiles=5,
    )
    third = GaussianNormalizer(
        distribution="gaussian_kde",
        subsample=20,
        random_state=8,
        n_quantiles=5,
    )

    first.fit(data, column="value")
    second.fit(data, column="value")
    third.fit(data, column="value")

    assert len(first._sorted_values) == 5
    assert np.allclose(first._sorted_values, second._sorted_values)
    assert not np.allclose(first._sorted_values, third._sorted_values)


def test_gaussian_normalizer_distribution_changes_transformation_strategy():
    data = pd.DataFrame({"value": [0.0, 0.0, 0.0, 1.0, 2.0, 10.0, 10.0, 25.0]})

    kde = GaussianNormalizer(distribution="gaussian_kde")
    gaussian = GaussianNormalizer(distribution="gaussian")

    kde.fit(data, column="value")
    gaussian.fit(data, column="value")

    kde_values = kde.transform(data)["value"].to_numpy()
    gaussian_values = gaussian.transform(data)["value"].to_numpy()

    assert not np.allclose(kde_values, gaussian_values)
