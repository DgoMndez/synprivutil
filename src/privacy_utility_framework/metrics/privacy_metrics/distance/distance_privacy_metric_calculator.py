from collections.abc import Callable

import pandas as pd

from ..privacy_metric_calculator import PrivacyMetricCalculator
from .util import custom_cdist


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
    distance_metric_args : dict, optional
        Keyword arguments forwarded to ``custom_cdist``.
    """

    def __init__(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        distance_metric: str | Callable = "euclidean",
        distance_metric_args: dict | None = None,
        original_name: str = None,
        synthetic_name: str = None,
    ):
        super().__init__(original, synthetic, original_name, synthetic_name)
        assert distance_metric is not None, "Parameter 'distance_metric' is required."
        self.distance_metric = distance_metric

        metric_args = distance_metric_args
        self.distance_metric_args = metric_args.copy() if metric_args else {}

    def compute_cdist(self, XA, XB, *, out=None, **kwargs):
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
            Extra keyword arguments merged on top of ``distance_metric_args``.

        Returns
        -------
        ndarray
            Pairwise distance matrix.
        """
        metric_kwargs = {**self.distance_metric_args, **kwargs}
        return custom_cdist(XA, XB, metric=self.distance_metric, out=out, **metric_kwargs)

    def set_metric(self, metric: str | Callable, metric_args: dict | None = None):
        """
        Sets or updates the distance metric.

        Parameters:
            metric (str or callable): The distance metric to use in DCR calculation.
            metric_args (dict, optional): Additional keyword arguments forwarded to \
                the distance metric function.
        """
        self.distance_metric = metric
        self.distance_metric_args = metric_args if metric_args else {}
