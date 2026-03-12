from collections.abc import Callable

import pandas as pd

from ....utils.distance.strategies import DistanceStrategy
from ....utils.distance.strategy_factory import DistanceStrategyFactory
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
    distance_strategy : str | Callable | DistanceStrategy, optional
        The distance strategy to use for calculations (default: 'euclidean').
    **kwargs : dict, optional
        Keyword arguments forwarded the distance strategy creation.
    """

    def __init__(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        distance_strategy: str | Callable | DistanceStrategy = "euclidean",
        original_name: str = None,
        synthetic_name: str = None,
        **kwargs,
    ):
        super().__init__(original, synthetic, original_name, synthetic_name)
        self.distance_strategy = self._resolve_distance_strategy(distance_strategy, **kwargs)

    @staticmethod
    def _resolve_distance_strategy(
        distance_strategy: str | Callable | DistanceStrategy,
        **kwargs,
    ) -> DistanceStrategy:
        if isinstance(distance_strategy, DistanceStrategy):
            return distance_strategy

        if callable(distance_strategy):
            return DistanceStrategyFactory.create(
                strategy=distance_strategy,
                default_args=kwargs or None,
            )

        return DistanceStrategyFactory.create(strategy=distance_strategy, **kwargs)

    def cdist(self, XA, XB, *, out=None, **kwargs):
        """
        Compute pairwise distances using the configured distance metric.

        Parameters
        ----------
        XA : array_like
            First sample matrix.
        XB : array_like
            Second sample matrix.
        out : ndarray, optional
            Output array as in ``scipy.spatial.distance.cdist``.
        **kwargs
            Extra keyword arguments for the metric calculation.

        Returns
        -------
        ndarray
            Pairwise distance matrix.
        """
        return self.distance_strategy.cdist(XA, XB, out=out, **kwargs)

    def set_metric(self, distance_strategy: str | Callable | DistanceStrategy, **kwargs):
        """
        Sets or updates the distance metric.

        Parameters:
            distance_strategy (str or Callable or DistanceStrategy): The distance strategy to use.
            **kwargs (dict, optional): Additional keyword arguments forwarded to \
                the distance strategy creation if `distance_strategy` is a string or `Callable`.
        """
        self.distance_strategy = self._resolve_distance_strategy(distance_strategy, **kwargs)
