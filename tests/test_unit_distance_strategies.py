import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from privacy_utility_framework.dataset.transformers import ECDFTransformer, QuantileRDTransformer
from privacy_utility_framework.utils.distance.distance import ecdf_cdist, quantile_cdist
from privacy_utility_framework.utils.distance.strategies import (
    CustomDistanceStrategy,
    ECDFDistanceStrategy,
    QuantileDistanceStrategy,
    ScipyDistanceStrategy,
    TransformedDistanceStrategy,
)
from privacy_utility_framework.utils.distance.strategy_factory import DistanceStrategyFactory


class IdentityHyperTransformer:
    _fitted = True

    def __init__(self, columns):
        self._input_columns = list(columns)

    def transform(self, X):
        return X.copy()


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


class RankECDFMockTransformer(ECDFTransformer):
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
        assert self._fitted, "RankECDFMockTransformer must be fitted before transform."
        values = np.asarray(data).reshape(-1)
        ranks = np.searchsorted(self._sorted_values, values, side="right")
        transformed = ranks / self._n_samples
        return transformed.reshape(-1, 1)


def test_strategy_factory_creates_appropiate_strategy():
    original_data = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [2.0, 3.0, 4.0]})

    def func(u, v, **kwargs):
        return np.linalg.norm(u - v, axis=1)

    strategies = {
        "scipy": ScipyDistanceStrategy,
        "transformed": TransformedDistanceStrategy,
        "quantile": QuantileDistanceStrategy,
        "ecdf": ECDFDistanceStrategy,
        "custom": CustomDistanceStrategy,
        "euclidean": ScipyDistanceStrategy,
        func: CustomDistanceStrategy,
    }

    additional_args = {
        "transformed": {
            "hypertransformer": IdentityHyperTransformer(columns=original_data.columns)
        },
        "quantile": {"original_data": original_data, "qt_factory": RankQuantileMockTransformer},
        "ecdf": {"original_data": original_data, "ecdf_factory": RankECDFMockTransformer},
        "custom": {"metric": "euclidean"},
    }

    for metric, expected_cls in strategies.items():
        kwargs = additional_args.get(metric, {})
        strategy = DistanceStrategyFactory.create(
            strategy=metric, default_args={"original_data": original_data}, **kwargs
        )
        assert isinstance(strategy, expected_cls)


def test_scipy_strategy_batched_aggregate_matches_direct():
    for k in range(1, 6):
        aux = 1 if k % 2 == 0 else -1
        XA = np.random.rand(10 * (k + 1 + aux), 5 * k) * 100
        XB = np.random.rand(10 * (k + 1 - aux), 5 * k) * 100

        direct = ScipyDistanceStrategy(metric="euclidean")
        for method in ["mean", "max", "min"]:
            expected = direct.aggregate_cdist(XA, XB, method=method)

            for batch_size in [1, 2, 3, 4]:
                batched = ScipyDistanceStrategy(metric="euclidean")
                batched.max_size = (batch_size * len(XB)) << 4
                result = batched.aggregate_cdist(XA, XB, method=method)

                assert np.allclose(result, expected)


def test_custom_strategy_nearest_neighbors_returns_indices():
    X_source = np.array([[0.0], [3.0], [10.0]])
    X_target = np.array([[1.0], [8.0]])

    strategy = CustomDistanceStrategy(metric="euclidean")
    distances, indices = strategy.nearest_neighbors(X_source, X_target, k=1)

    assert distances.shape == (2, 1)
    assert indices.shape == (2, 1)
    assert np.allclose(distances[:, 0], np.array([1.0, 2.0]))
    assert np.array_equal(indices[:, 0], np.array([0, 2]))


def test_transformed_strategy_identity_matches_scipy_cdist_for_dataframe():
    for k in range(1, 6):
        n = 10 * k
        m = 5 * k
        aux = 1 if k % 2 == 0 else -1
        XA = pd.DataFrame(
            np.random.randn(int(n * (1 + 0 * 0.5 * aux)), m) * 100,
            columns=[f"y_{i}" for i in range(m)],
        )
        XB = pd.DataFrame(
            np.random.randn(int(n * (1 - 0.5 * aux)), m) * 100, columns=[f"y_{i}" for i in range(m)]
        )

        strategy = TransformedDistanceStrategy(
            hypertransformer=IdentityHyperTransformer(columns=XA.columns),
            base_metric="euclidean",
        )

        distances = strategy.cdist(XA, XB)
        expected = cdist(XA.to_numpy(), XB.to_numpy(), metric="euclidean")

        assert np.allclose(distances, expected)


def test_quantile_strategy_matches_quantile_cdist():
    for k in range(1, 6):
        n = 10 * k
        m = 5 * k
        aux = 1 if k % 2 == 0 else -1

        XA = pd.DataFrame(
            np.random.randn(int(n * (1 + 0 * 0.5 * aux)), m) * 100,
            columns=[f"y_{i}" for i in range(m)],
        )
        XB = pd.DataFrame(
            np.random.randn(int(n * (1 - 0.5 * aux)), m) * 100, columns=[f"y_{i}" for i in range(m)]
        )

        original_data = pd.DataFrame(
            np.random.randn(int(m + 0.5 * aux), m) * 100, columns=[f"y_{i}" for i in range(m)]
        )

        strategy = QuantileDistanceStrategy(
            original_data=original_data,
            base_metric="euclidean",
            output_distribution="uniform",
            qt_factory=RankQuantileMockTransformer,
        )

        distances = strategy.cdist(XA, XB)
        expected = quantile_cdist(
            XA,
            XB,
            base_metric="euclidean",
            output_distribution="uniform",
            original_data=original_data,
            qt_factory=RankQuantileMockTransformer,
        )

        assert np.allclose(distances, expected)


def test_ecdf_strategy_matches_ecdf_cdist():
    for k in range(1, 6):
        n = 10 * k
        m = 5 * k
        aux = 1 if k % 2 == 0 else -1

        XA = pd.DataFrame(
            np.random.randn(int(n * (1 + 0 * 0.5 * aux)), m) * 100,
            columns=[f"y_{i}" for i in range(m)],
        )
        XB = pd.DataFrame(
            np.random.randn(int(n * (1 - 0.5 * aux)), m) * 100,
            columns=[f"y_{i}" for i in range(m)],
        )

        original_data = pd.DataFrame(
            np.random.randn(int(m + 0.5 * aux), m) * 100,
            columns=[f"y_{i}" for i in range(m)],
        )

        strategy = ECDFDistanceStrategy(
            original_data=original_data,
            base_metric="euclidean",
            ecdf_factory=RankECDFMockTransformer,
        )

        distances = strategy.cdist(XA, XB)
        expected = ecdf_cdist(
            XA,
            XB,
            base_metric="euclidean",
            original_data=original_data,
            ecdf_factory=RankECDFMockTransformer,
        )

        assert np.allclose(distances, expected)
