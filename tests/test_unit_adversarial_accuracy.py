import numpy as np
import pandas as pd

from privacy_utility_framework.metrics.privacy.distance import (
    AdversarialAccuracyCalculator,
)
from privacy_utility_framework.utils.distance.strategies import DistanceStrategy


class RecordingDistanceStrategy(DistanceStrategy):
    def __init__(self):
        super().__init__()
        self.min_calls = []
        self.cdist_calls = []
        self.nn_calls = []

    def _cdist(self, XA, XB, *, out=None, **kwargs):
        self.cdist_calls.append((len(XA), len(XB)))
        return np.zeros((len(XA), len(XB)))

    def min_cdist(self, XA, XB, same=False, bidirectional=False, **kwargs):
        self.min_calls.append((len(XA), len(XB), same))
        return np.zeros(len(XA))

    def nearest_neighbors(self, X_source, X_target=None, k=1, **kwargs):
        n_target = len(X_target) if X_target is not None else len(X_source)
        self.nn_calls.append((len(X_source), n_target, k, X_target is None))
        return np.zeros((n_target, k)), np.zeros((n_target, k), dtype=int)


class SupportsSklearnDistanceStrategy(RecordingDistanceStrategy):
    @property
    def supports_sklearn_nn(self) -> bool:
        return True


def test_adversarial_accuracy_uses_full_data_when_nn_sampling_disabled():
    original = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    synthetic = pd.DataFrame({"x": [0.5, 1.5, 2.5]})
    strategy = RecordingDistanceStrategy()

    calculator = AdversarialAccuracyCalculator(
        original,
        synthetic,
        distance_strategy=strategy,
        nn_samples=0,
    )

    calculator._calculate_min_distances()

    assert strategy.min_calls == [
        (4, 4, True),
        (4, 3, False),
        (3, 4, False),
        (3, 3, True),
    ]
    assert strategy.cdist_calls == []


def test_adversarial_accuracy_samples_only_target_rows_when_requested():
    original = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    synthetic = pd.DataFrame({"x": [0.5, 1.5, 2.5, 3.5, 4.5]})
    strategy = RecordingDistanceStrategy()

    calculator = AdversarialAccuracyCalculator(
        original,
        synthetic,
        distance_strategy=strategy,
        nn_samples=2,
        nn_random_state=7,
    )

    calculator._calculate_min_distances()

    assert strategy.min_calls == [
        (2, 5, False),
        (2, 4, False),
    ]
    assert strategy.cdist_calls == [
        (2, 4),
        (2, 5),
    ]


def test_adversarial_accuracy_uses_nn_backend_when_requested():
    original = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    synthetic = pd.DataFrame({"x": [0.5, 1.5, 2.5]})
    strategy = RecordingDistanceStrategy()

    calculator = AdversarialAccuracyCalculator(
        original,
        synthetic,
        distance_strategy=strategy,
        backend="nn",
        nn_samples=0,
    )

    calculator._calculate_min_distances()

    assert strategy.nn_calls == [
        (4, 4, 1, True),
        (3, 4, 1, False),
        (4, 3, 1, False),
        (3, 3, 1, True),
    ]
    assert strategy.min_calls == []


def test_adversarial_accuracy_auto_backend_switches_to_nn_when_matrix_is_large():
    original = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    synthetic = pd.DataFrame({"x": [0.5, 1.5, 2.5]})
    strategy = RecordingDistanceStrategy()
    strategy.max_size = 16  # Force auto backend to avoid brute pairwise matrices.

    calculator = AdversarialAccuracyCalculator(
        original,
        synthetic,
        distance_strategy=strategy,
        backend="auto",
        nn_samples=0,
    )

    calculator._calculate_min_distances()

    assert strategy.nn_calls != []
    assert strategy.min_calls == []


def test_adversarial_accuracy_auto_backend_prefers_nn_for_sklearn_ready_strategy():
    original = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    synthetic = pd.DataFrame({"x": [0.5, 1.5, 2.5]})
    strategy = SupportsSklearnDistanceStrategy()
    strategy.max_size = 1 << 40  # Large enough to allow brute, but auto should still pick nn.

    calculator = AdversarialAccuracyCalculator(
        original,
        synthetic,
        distance_strategy=strategy,
        backend="auto",
        nn_samples=0,
    )

    calculator._calculate_min_distances()

    assert strategy.nn_calls != []
    assert strategy.min_calls == []
