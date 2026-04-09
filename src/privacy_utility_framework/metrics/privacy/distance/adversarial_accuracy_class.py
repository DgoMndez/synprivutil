import numpy as np
import pandas as pd

from privacy_utility_framework.dataset.tabletransformer import TableTransformer
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
        preprocess: bool = False,
        preprocessor: TableTransformer | None = None,
        backend: str = "auto",
        nn_samples: int = 0,
        nn_random_state: int = None,
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
            preprocess (bool, optional): Whether to preprocess both datasets before evaluation.
            preprocessor (TableTransformer, optional): Optional transformer to reuse when
                preprocessing is enabled.
            backend (str, optional): Backend used to compute nearest-neighbor distances:
                "brute" (cdist/min_cdist), "nn" (nearest_neighbors), or
                "auto" (select based on estimated matrix size).
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
            preprocess=preprocess,
            preprocessor=preprocessor,
            **kwargs,
        )
        backend = backend.lower().strip()
        if backend not in {"brute", "nn", "auto"}:
            raise ValueError("'backend' must be one of {'brute', 'nn', 'auto'}.")
        self.backend = backend
        self.nn_samples = nn_samples
        self.random_state = nn_random_state

    def evaluate(self):
        """
        Calculate the Nearest Neighbor Adversarial Accuracy (NNAA).

        Returns:
            float: The calculated NNAA.
        """
        # Calculate minimum distances between records in original and synthetic data.
        min_orig, min_syn = self._calculate_min_distances()

        # Compute NNAA based on distances within and between datasets
        term1 = np.mean(min_orig[1] > min_orig[0])
        term2 = np.mean(min_syn[0] > min_syn[1])

        nnaa_value = 0.5 * (term1 + term2)

        return nnaa_value

    def _calculate_min_distances(self):
        backend = self._resolve_backend()
        if backend == "nn":
            return self._calculate_min_distances_nn()
        return self._calculate_min_distances_brute()

    def _resolve_backend(self) -> str:
        if self.backend in {"brute", "nn"}:
            return self.backend

        if getattr(self.distance_strategy, "supports_sklearn_nn", False):
            return "nn"

        # Auto mode picks brute when the largest required pairwise matrix fits max_size.
        original = self._get_comparison_data(self.original)
        synthetic = self._get_comparison_data(self.synthetic)
        n_orig = len(original)
        n_syn = len(synthetic)
        max_required_bytes = max(n_orig * n_orig, n_orig * n_syn, n_syn * n_syn) << 4
        max_size = getattr(self.distance_strategy, "max_size", 1 << 30)
        return "brute" if max_required_bytes <= max_size else "nn"

    def _calculate_min_distances_brute(self):
        """
        Calculate minimum distances for nearest neighbor adversarial accuracy.
        
        Returns:
            tuple: (M_0, M_1) of arrays forming (M_{i,j,k} : i=0,1; j=0,1; 0<=k<n_i) st:
                - Datasets are represented by indexes i and j: 0 = original, 1 = synthetic.
                - Index k represents the k-th record in dataset i.
                - M_{i,j,k} is the distance from record k in dataset i \
                    to its nearest neighbor in dataset j.
        """
        # Use transformed data when available; otherwise compare the user-provided data directly.
        original = self._get_comparison_data(self.original)
        synthetic = self._get_comparison_data(self.synthetic)

        aux_list = [original, synthetic]
        sampled_data = [
            self._sample_for_nn(original),
            self._sample_for_nn(synthetic),
        ]
        sampled_list = [item[0] for item in sampled_data]
        sampled_indices = [item[1] for item in sampled_data]
        len_list = [sampled_list[0].shape[0], sampled_list[1].shape[0]]
        min_distances = [np.empty((2, len_list[0])), np.empty((2, len_list[1]))]

        # Calculate pairwise distances between original and synthetic datasets
        # i = 0 for original, i = 1 for synthetic
        # d[i][j] will hold the minimum distances from dataset i to dataset j

        for i in range(2):
            for j in range(2):
                if (
                    i == j
                    and len(sampled_list[i]) == len(aux_list[j])
                    and np.array_equal(sampled_indices[i], np.arange(len(aux_list[j]), dtype=int))
                ):
                    min_distances[i][j, :] = self.distance_strategy.min_cdist(
                        sampled_list[i], aux_list[j], same=True
                    )
                else:
                    min_distances[i][j, :] = self._min_distance_to_source(
                        sampled_list[i],
                        aux_list[j],
                        same_indices=sampled_indices[i] if i == j else None,
                    )
        return min_distances[0], min_distances[1]

    def _calculate_min_distances_nn(self):
        """Calculate minimum distances using nearest-neighbor backend."""
        original = self._get_comparison_data(self.original)
        synthetic = self._get_comparison_data(self.synthetic)

        aux_list = [original, synthetic]
        sampled_data = [
            self._sample_for_nn(original),
            self._sample_for_nn(synthetic),
        ]
        sampled_list = [item[0] for item in sampled_data]
        sampled_indices = [item[1] for item in sampled_data]

        min_distances = [
            np.empty((2, len(sampled_list[0])), dtype=float),
            np.empty((2, len(sampled_list[1])), dtype=float),
        ]

        for i in range(2):
            for j in range(2):
                source = aux_list[j]
                target = sampled_list[i]

                if i != j:
                    d, _ = self.distance_strategy.nearest_neighbors(source, target, k=1)
                    min_distances[i][j, :] = d[:, 0]
                    continue

                full_self_sample = len(target) == len(source) and np.array_equal(
                    sampled_indices[i], np.arange(len(source), dtype=int)
                )
                if full_self_sample:
                    d, _ = self.distance_strategy.nearest_neighbors(source, None, k=1)
                    min_distances[i][j, :] = d[:, 0]
                else:
                    d, _ = self.distance_strategy.nearest_neighbors(source, target, k=2)
                    min_distances[i][j, :] = np.partition(d, 1, axis=1)[:, 1]

        return min_distances[0], min_distances[1]

    def _sample_for_nn(self, data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Return sampled rows and their positions in the original dataset."""
        if self.nn_samples is None or self.nn_samples <= 0 or len(data) <= self.nn_samples:
            return data, np.arange(len(data), dtype=int)

        sampled = data.sample(n=self.nn_samples, random_state=self.random_state)
        sampled_indices = data.index.get_indexer(sampled.index)
        return sampled, sampled_indices

    def _min_distance_to_source(
        self,
        X_target: pd.DataFrame,
        X_source: pd.DataFrame,
        same_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute row-wise minimum distances, masking exact self-pairs when requested."""
        if same_indices is None:
            return self.distance_strategy.min_cdist(X_target, X_source, same=False)

        max_size = getattr(self.distance_strategy, "max_size", 1 << 30)
        batch_size = max(1, (max_size >> 4) // max(1, len(X_source)))
        min_distances = np.empty(len(X_target), dtype=float)

        for start in range(0, len(X_target), batch_size):
            stop = min(start + batch_size, len(X_target))
            batch_distances = self.distance_strategy.cdist(X_target.iloc[start:stop], X_source)
            row_indices = np.arange(stop - start)
            batch_distances[row_indices, same_indices[start:stop]] = np.inf
            min_distances[start:stop] = np.min(batch_distances, axis=1)

        return min_distances
