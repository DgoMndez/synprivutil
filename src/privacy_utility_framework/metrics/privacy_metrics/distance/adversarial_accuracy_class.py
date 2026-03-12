import numpy as np
import pandas as pd
from tqdm import tqdm

from privacy_utility_framework.utils.distance.strategies import DistanceStrategy

from .distance_privacy_metric_calculator import (
    DistancePrivacyMetricCalculator,
)


class AdversarialAccuracyCalculator(DistancePrivacyMetricCalculator):
    """
    Calculate nearest neighbors and adversarial accuracy metrics for original \
        and synthetic datasets.
    """

    def __init__(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        distance_strategy: str | DistanceStrategy = "euclidean",
        original_name: str = None,
        synthetic_name: str = None,
        **kwargs,
    ):
        """
        Initializes the AdversarialAccuracyCalculator with original and synthetic datasets \
            and a distance metric.
        reference paper: https://github.com/yknot/ESANN2019/blob/master/metrics/nn_adversarial_accuracy.py

        Parameters:
            original (pd.DataFrame): Original dataset.
            synthetic (pd.DataFrame): Synthetic dataset.
            distance_strategy (str or DistanceStrategy): The distance strategy to use.
            original_name (str, optional): Name for the original dataset (default: None).
            synthetic_name (str, optional): Name for the synthetic dataset (default: None).
            nn_samples (int, optional): Number of samples used in mean nearest neighbor distance \
                stimations. If 0 or less, all samples are used (default: 0).
            nn_random_state (int, optional): Random state for sampling in nearest neighbor \
                calculations (default: None).
            **kwargs (dict, optional): Extra keyword arguments forwarded to
                the distance strategy creation.
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

    def evaluate(self):
        """
        Calculate the Nearest Neighbor Adversarial Accuracy (NNAA).

        Returns:
            float: The calculated NNAA.
        """
        # Calculate minimum distances between records in original and synthetic data
        min_orig, min_syn = self._calculate_min_distances()

        # Compute NNAA based on distances within and between datasets
        term1 = np.mean(min_orig[1] > min_orig[0])
        term2 = np.mean(min_syn[0] > min_syn[1])

        nnaa_value = 0.5 * (term1 + term2)

        return nnaa_value

    def _calculate_min_distances(self):
        """
        Calculate minimum distances for nearest neighbor adversarial accuracy.
        
        Returns:
            tuple: (M_0, M_1) of arrays forming (M_{i,j,k} : i=0,1; j=0,1; 0<=k<n_i) st:
                - Datasets are represented by indexes i and j: 0 = original, 1 = synthetic.
                - Index k represents the k-th record in dataset i.
                - M_{i,j,k} is the distance from record k in dataset i \
                    to its nearest neighbor in dataset j.
        """
        # The transformed and normalized data is used for the NNAA

        original = self.original.transformed_data
        synthetic = self.synthetic.transformed_data

        aux_list = [original, synthetic]
        len_list = [original.shape[0], synthetic.shape[0]]
        min_distances = [np.empty((2, len_list[0])), np.empty((2, len_list[1]))]

        # Calculate pairwise distances between original and synthetic datasets
        # i = 0 for original, i = 1 for synthetic
        # d[i][j] will hold the minimum distances from dataset i to dataset j

        for i in range(2):
            for j in range(2):
                min_distances[i][j, :] = self.distance_strategy.min_cdist(
                    aux_list[i], aux_list[j], same=(i == j)
                )
        return min_distances[0], min_distances[1]


class AdversarialAccuracyCalculator_NN(DistancePrivacyMetricCalculator):
    """
    Calculate nearest neighbors and adversarial accuracy metrics for original and synthetic \
        datasets using Nearest Neighbors (may be faster in some cases).
    """

    def __init__(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        distance_strategy: str | DistanceStrategy = "euclidean",
        original_name: str = None,
        synthetic_name: str = None,
        nn_samples: int = 0,
        nn_random_state: int = None,
        **kwargs,
    ):
        """
        Initialize the AdversarialAccuracyCalculator_NN with datasets and distance strategy.

        Parameters:
            original (pd.DataFrame): Original dataset.
            synthetic (pd.DataFrame): Synthetic dataset.
            distance_metric (str or Callable): The distance metric to use for nearest neighbor \
                calculations. It must be admissible by sklearn's NearestNeighbors.
            nn_samples (int, optional): Number of samples used in mean nearest neighbor distance \
                stimations. If 0 or less, all samples are used (default: 0).
            nn_random_state (int, optional): Random state for sampling in nearest neighbor \
                calculations (default: None).
            **kwargs: Extra keyword arguments forwarded to Nearest Neighbors for custom metrics.
        """
        super().__init__(
            original,
            synthetic,
            distance_strategy=distance_strategy,
            original_name=original_name,
            synthetic_name=synthetic_name,
            **kwargs,
        )

        # Define data for calculation
        self.data = {
            "original": self.original.transformed_data,
            "synthetic": self.synthetic.transformed_data,
        }
        self.dists = {}
        self.nn_samples = nn_samples
        self.random_state = nn_random_state

    def _nearest_neighbors(self, t, s):
        """
        Calculate nearest neighbors between two datasets (t and s).

        Parameters:
            t (str): Target dataset name ('original' or 'synthetic').
            s (str): Source dataset name ('original' or 'synthetic').

        Returns:
            distances (ndarray): Array of nearest neighbor distances from dataset t to dataset s.
        """
        if self.nn_samples <= 0 or self.data[s].shape[0] <= self.nn_samples:
            if t == s:
                d, _ = self.distance_strategy.nearest_neighbors(self.data[s], None)
            else:
                d, _ = self.distance_strategy.nearest_neighbors(self.data[s], self.data[t])
        else:
            X_target = self.data[t].sample(n=self.nn_samples, random_state=self.random_state)
            if t == s:
                aux_d, _ = self.distance_strategy.nearest_neighbors(X_target, self.data[s], k=2)
                d = np.partition(aux_d, 1, axis=1)[:, 1]  # Get the second smallest distance
            else:
                d, _ = self.distance_strategy.nearest_neighbors(X_target, self.data[t])

        return d

    def _compute_nn(self):
        """Compute nearest neighbors for all pairs of original and synthetic datasets."""
        pairs = [
            ("original", "original"),
            ("original", "synthetic"),
            ("synthetic", "synthetic"),
            ("synthetic", "original"),
        ]
        for t, s in tqdm(pairs):
            d = self._nearest_neighbors(t, s)
            self.dists[(t, s)] = d

    def _adversarial_accuracy(self):
        """Calculate the adversarial accuracy score based on nearest neighbor distances."""
        orig_vs_synth = np.mean(
            self.dists[("original", "synthetic")] > self.dists[("original", "original")]
        )
        synth_vs_orig = np.mean(
            self.dists[("synthetic", "original")] > self.dists[("synthetic", "synthetic")]
        )
        return 0.5 * (orig_vs_synth + synth_vs_orig)

    def evaluate(self):
        """Run nearest neighbor computation and calculate adversarial accuracy."""
        self._compute_nn()
        return self._adversarial_accuracy()
