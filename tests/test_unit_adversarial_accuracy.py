import numpy as np
import pandas as pd

from privacy_utility_framework.metrics.privacy_metrics.distance.adversarial_accuracy_class import (
    AdversarialAccuracyCalculator,
)
from privacy_utility_framework.utils.distance.strategies import DistanceStrategy


class RecordingDistanceStrategy(DistanceStrategy):
    def __init__(self):
        super().__init__()
        self.min_calls = []
        self.cdist_calls = []

    def _cdist(self, XA, XB, *, out=None, **kwargs):
        self.cdist_calls.append((len(XA), len(XB)))
        return np.zeros((len(XA), len(XB)))

    def min_cdist(self, XA, XB, same=False, bidirectional=False, **kwargs):
        self.min_calls.append((len(XA), len(XB), same))
        return np.zeros(len(XA))


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
