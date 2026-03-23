from unittest.mock import Mock, patch

import pandas as pd

from privacy_utility_framework.metrics.privacy_metrics import PrivacyMetricCalculator
from privacy_utility_framework.metrics.privacy_metrics.distance import (
    DistancePrivacyMetricCalculator,
)


class DummyPrivacyMetricCalculator(PrivacyMetricCalculator):
    def evaluate(self):
        original = self._get_comparison_data(self.original)
        synthetic = self._get_comparison_data(self.synthetic)
        return original.shape[1], synthetic.shape[1]


class DummyDistancePrivacyMetricCalculator(DistancePrivacyMetricCalculator):
    def evaluate(self):
        return 0.0


def test_privacy_metric_calculator_leaves_data_untransformed_by_default():
    original = pd.DataFrame(
        {
            "category": ["a", "b", "a"],
            "value": [1.0, 2.0, 3.0],
        }
    )
    synthetic = pd.DataFrame(
        {
            "category": ["a", "b", "b"],
            "value": [1.5, 2.5, 3.5],
        }
    )

    calculator = DummyPrivacyMetricCalculator(original, synthetic)

    assert calculator.original.transformed_data is None
    assert calculator.synthetic.transformed_data is None
    assert calculator.evaluate() == (2, 2)


def test_privacy_metric_calculator_can_opt_into_internal_preprocessing():
    original = pd.DataFrame(
        {
            "category": ["a", "b", "a"],
            "value": [1.0, 2.0, 3.0],
        }
    )
    synthetic = pd.DataFrame(
        {
            "category": ["a", "b", "b"],
            "value": [1.5, 2.5, 3.5],
        }
    )

    calculator = DummyPrivacyMetricCalculator(original, synthetic, preprocess=True)

    assert calculator.original.transformed_data is not None
    assert calculator.synthetic.transformed_data is not None
    assert calculator.evaluate()[0] > original.shape[1]


def test_distance_calculator_keeps_preprocessing_out_of_strategy_kwargs():
    original = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    synthetic = pd.DataFrame({"x": [1.5, 2.5, 3.5]})

    with patch(
        "privacy_utility_framework.metrics.privacy_metrics.distance.distance_privacy_metric_calculator.DistanceStrategyFactory.create"
    ) as create:
        create.return_value = Mock()

        DummyDistancePrivacyMetricCalculator(
            original,
            synthetic,
            distance_strategy="minkowski",
            preprocess=True,
            p=1,
        )

        create.assert_called_once_with(strategy="minkowski", p=1)
