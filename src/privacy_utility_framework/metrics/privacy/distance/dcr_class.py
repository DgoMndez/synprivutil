import numpy as np
import pandas as pd

from privacy_utility_framework.dataset.tabletransformer import TableTransformer

from ....utils.distance.strategies import DistanceStrategy
from .distance_privacy_metric_calculator import DistancePrivacyMetricCalculator


class DCRCalculator(DistancePrivacyMetricCalculator):
    def __init__(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        original_name: str = None,
        synthetic_name: str = None,
        preprocess: bool = False,
        preprocessor: TableTransformer | None = None,
        distance_strategy: str | DistanceStrategy = "euclidean",
        weights: np.ndarray = None,
        **kwargs,
    ):
        """
        Initializes the DCRCalculator with datasets and a specified distance metric.

        Parameters:
            original (pd.DataFrame): Original dataset.
            synthetic (pd.DataFrame): Synthetic dataset.
            original_name (str, optional): Name for the original dataset (default: None).
            synthetic_name (str, optional): Name for the synthetic dataset (default: None).
            preprocess (bool, optional): Whether to preprocess both datasets before evaluation.
            preprocessor (TableTransformer, optional): Optional transformer to reuse when
                preprocessing is enabled.
            distance_strategy (str or DistanceStrategy): The strategy for calculating distances \
                ('euclidean', 'cityblock', etc.).
            weights (np.ndarray, optional): Array of weights for each feature in the datasets.
            **kwargs (dict, optional): Extra keyword arguments forwarded to \
            the distance strategy creation for custom metrics.
        """
        # Initialize the superclass with datasets and settings
        super().__init__(
            original,
            synthetic,
            distance_strategy=distance_strategy,
            original_name=original_name,
            synthetic_name=synthetic_name,
            preprocess=preprocess,
            preprocessor=preprocessor,
            **kwargs,
        )

        # Define feature weights for calculations
        comparison_data = self._get_comparison_data(self.original)
        transformed_feature_count = comparison_data.shape[1]
        if weights is None:
            self.weights = np.ones(transformed_feature_count)
        else:
            self.weights = np.asarray(weights).reshape(-1)
            if self.weights.shape[0] != transformed_feature_count:
                raise ValueError(
                    "Length of 'weights' must match the number of transformed features "
                    f"({transformed_feature_count}), got {self.weights.shape[0]}."
                )

    def evaluate(self) -> float:
        """
        Computes the Distance to Closest Record (DCR) between the synthetic and original datasets.

        Returns:
            float: The average minimum distance from each synthetic record to the closest original \
                record.
        """
        # Retrieve the representation selected for metric evaluation.
        original = self._get_comparison_data(self.original)
        synthetic = self._get_comparison_data(self.synthetic)

        # TODO: Check if this is correct (I think not)
        # Apply feature weights to both datasets
        weighted_original_data = original * self.weights
        weighted_synthetic_data = synthetic * self.weights

        # Compute minimum distances between synthetic and original data
        min_distances = self.distance_strategy.min_cdist(
            weighted_synthetic_data, weighted_original_data
        )

        return np.mean(min_distances)
