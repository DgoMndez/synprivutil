import numpy as np
import pandas as pd

from privacy_utility_framework.dataset.tabletransformer import TableTransformer
from privacy_utility_framework.utils.distance.strategies import DistanceStrategy

from .distance_privacy_metric_calculator import DistancePrivacyMetricCalculator


class NNDRCalculator(DistancePrivacyMetricCalculator):
    def __init__(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        original_name: str = None,
        synthetic_name: str = None,
        preprocess: bool = False,
        preprocessor: TableTransformer | None = None,
        distance_strategy: str | DistanceStrategy = "euclidean",
        **kwargs,
    ):
        """
        Initialize the NNDRCalculator with original and synthetic datasets and a distance metric.

        Parameters:
            original (pd.DataFrame): Original dataset.
            synthetic (pd.DataFrame): Synthetic dataset.
            original_name (str, optional): Name for the original dataset (default: None).
            synthetic_name (str, optional): Name for the synthetic dataset (default: None).
            preprocess (bool, optional): Whether to preprocess both datasets before evaluation.
            preprocessor (TableTransformer, optional): Optional transformer to reuse when
                preprocessing is enabled.
            distance_strategy (str or DistanceStrategy): The strategy for calculating distances.
            **kwargs (dict, optional): Extra keyword arguments forwarded to
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

    def evaluate(self) -> float:
        """
        Calculate the Nearest Neighbor Distance Ratio (NNDR) for synthetic data compared to \
            original data.

        Returns:
            float: The mean NNDR value for the synthetic dataset.
        """

        # Use transformed data when available; otherwise compare the user-provided data directly.
        original = self._get_comparison_data(self.original)
        synthetic = self._get_comparison_data(self.synthetic)

        # Compute distances from each synthetic record to all original records
        distances, indices = self.distance_strategy.nearest_neighbors(synthetic, original, k=2)

        # Find the nearest and second nearest distances for each synthetic record
        nearest_distances = distances[:, 0]
        second_nearest_distances = distances[:, 1]

        # Calculate the NNDR for each synthetic record
        nndr_list = nearest_distances / (second_nearest_distances + 1e-16)  # Avoid division by zero

        # Return the average NNDR across all synthetic records
        return np.mean(nndr_list)
