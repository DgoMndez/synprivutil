from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from .distance_privacy_metric_calculator import (
    DistancePrivacyMetricCalculator,
)
from .util import METRIC_ALIAS


class AdversarialAccuracyCalculator(DistancePrivacyMetricCalculator):
    """
    Calculate nearest neighbors and adversarial accuracy metrics for original \
        and synthetic datasets.
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
        """
        Initializes the AdversarialAccuracyCalculator with original and synthetic datasets \
            and a distance metric.
        reference paper: https://github.com/yknot/ESANN2019/blob/master/metrics/nn_adversarial_accuracy.py

        Parameters:
            original (pd.DataFrame): Original dataset.
            synthetic (pd.DataFrame): Synthetic dataset.
            distance_metric (str or callable): The metric for calculating distances.
            distance_metric_args (dict, optional): Extra keyword arguments forwarded to
                ``custom_cdist`` for custom string/callable metrics.
        """

        # Initialize the superclass with datasets and settings
        super().__init__(
            original,
            synthetic,
            distance_metric=distance_metric,
            distance_metric_args=distance_metric_args,
            original_name=original_name,
            synthetic_name=synthetic_name,
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
                d = self.compute_cdist(aux_list[i], aux_list[j])
                if i == j:
                    np.fill_diagonal(d, np.inf)  # Ignore self-distances for same dataset
                min_distances[i][j, :] = np.min(d, axis=1)

        return min_distances[0], min_distances[1]


class AdversarialAccuracyCalculator_NN(AdversarialAccuracyCalculator):
    """
    Calculate nearest neighbors and adversarial accuracy metrics for original and synthetic \
        datasets using Nearest Neighbors (may be faster in some cases).
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
        """
        Initialize the AdversarialAccuracyCalculator_NN with datasets and distance metric.

        Parameters:
            original (pd.DataFrame): Original dataset.
            synthetic (pd.DataFrame): Synthetic dataset.
            distance_metric (str or callable): Metric for distance calculation.
            distance_metric_args (dict, optional): Extra keyword arguments forwarded to
                ``custom_cdist`` for custom string/callable metrics.
        """
        super().__init__(
            original,
            synthetic,
            distance_metric=distance_metric,
            distance_metric_args=distance_metric_args,
            original_name=original_name,
            synthetic_name=synthetic_name,
        )

        # Define data for calculation
        self.data = {
            "original": self.original.transformed_data,
            "synthetic": self.synthetic.transformed_data,
        }
        self.dists = {}

    def _nearest_neighbors(self, t, s):
        """
        Calculate nearest neighbors between two datasets (t and s).

        Parameters:
            t (str): Target dataset name ('original' or 'synthetic').
            s (str): Source dataset name ('original' or 'synthetic').

        Returns:
            tuple: (target dataset, source dataset, distances).
        """
        is_custom_registered_metric = (
            isinstance(self.distance_metric, str) and self.distance_metric in METRIC_ALIAS
        )

        if is_custom_registered_metric:
            d = self.compute_cdist(self.data[t], self.data[s])
            if t == s:
                np.fill_diagonal(d, np.inf)
            d = np.min(d, axis=1).reshape(-1, 1)
        else:
            nn_s = NearestNeighbors(
                n_neighbors=1,
                metric=self.distance_metric,
                metric_params=self.distance_metric_args,
            ).fit(self.data[s])
            if t == s:
                # Find distances within the same dataset
                d = nn_s.kneighbors()[0]
            else:
                # Find distances between different datasets
                d = nn_s.kneighbors(self.data[t])[0]

        return t, s, d

    def _compute_nn(self):
        """Compute nearest neighbors for all pairs of original and synthetic datasets."""
        pairs = [
            ("original", "original"),
            ("original", "synthetic"),
            ("synthetic", "synthetic"),
            ("synthetic", "original"),
        ]
        for t, s in tqdm(pairs):
            t, s, d = self._nearest_neighbors(t, s)
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
