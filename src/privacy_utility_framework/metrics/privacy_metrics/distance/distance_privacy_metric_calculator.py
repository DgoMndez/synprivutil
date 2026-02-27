import pandas as pd

from ..privacy_metric_calculator import PrivacyMetricCalculator


class DistancePrivacyMetricCalculator(PrivacyMetricCalculator):
    """
    Abstract base class for distance-based privacy metric calculators, extending
    the PrivacyMetricCalculator with distance metric on initialization.

    Parameters
    ----------
    original : pd.DataFrame
        The original dataset to compare against the synthetic data.
    synthetic : pd.DataFrame
        The synthetic dataset generated to resemble the original data.
    original_name : str, optional
        Name for the original dataset.
    synthetic_name : str, optional
        Name for the synthetic dataset.
    distance_metric : str or callable, optional
        The distance metric to use for calculations (default: 'euclidean').
    """

    def __init__(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        distance_metric: str | callable = "euclidean",
        original_name: str = None,
        synthetic_name: str = None,
    ):
        super().__init__(original, synthetic, original_name, synthetic_name)
        self.distance_metric = distance_metric
