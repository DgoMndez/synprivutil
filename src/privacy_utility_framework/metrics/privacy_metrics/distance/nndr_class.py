import numpy as np
import pandas as pd

from privacy_utility_framework.utils.distance.strategies import DistanceStrategy

from .distance_privacy_metric_calculator import DistancePrivacyMetricCalculator


class NNDRCalculator(DistancePrivacyMetricCalculator):
    def __init__(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        original_name: str = None,
        synthetic_name: str = None,
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
            **kwargs,
        )

    def evaluate(self) -> float:
        """
        Calculate the Nearest Neighbor Distance Ratio (NNDR) for synthetic data compared to \
            original data.

        Returns:
            float: The mean NNDR value for the synthetic dataset.
        """

        # Use transformed and normalized data for NNDR calculation
        original = self.original.transformed_data
        synthetic = self.synthetic.transformed_data

        # Compute distances from each synthetic record to all original records
        distances = self.compute_cdist(synthetic, original)

        # Find the nearest and second nearest distances for each synthetic record
        partitioned_distances = np.partition(distances, 1, axis=1)[:, :2]
        nearest_distances = partitioned_distances[:, 0]
        second_nearest_distances = partitioned_distances[:, 1]

        # Calculate the NNDR for each synthetic record
        nndr_list = nearest_distances / (second_nearest_distances + 1e-16)  # Avoid division by zero

        # Return the average NNDR across all synthetic records
        return np.mean(nndr_list)
