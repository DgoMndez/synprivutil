"""
Module: src/privacy_utility_framework/utils/distance/strategies.py
Description: DistanceStrategy class for unified customizable distance computations over \
    original or synthetic samples of datasets.

Author: Domingo Méndez García
Email: domingo.mendezg@um.es
Date: 20/03/2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
import pandas as pd
from humanize import naturalsize
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

from privacy_utility_framework.dataset.tabletransformer import TableTransformer
from privacy_utility_framework.dataset.transformers import ECDFTransformer, QuantileColTransformer

from .distance import (
    _build_ecdf_references,
    _ecdf_bounds_from_references,
    _ecdf_distance_matrix_from_bounds,
    _get_ecdf_tabletransformer,
    _get_quantile_tabletransformer,
    custom_cdist,
    ecdf_pdist,
    transformed_cdist,
)


class DistanceStrategy(ABC):
    """Strategy interface for distance computation (single samples, pairwise or matrix)."""

    canonical_name = None
    _max_size = 1 << 30  # Default maximum size for cdist distance matrices: 1 GiB

    def __init__(self, default_args: dict | None = None):
        """
        Constructor for DistanceStrategy.
        
        Args:
            default_args (dict | None, optional): Default arguments for the distance metric in \
                dist, cdist and pdist. Defaults to None.
        """
        self.metric_args = default_args.copy() if default_args else {}

    @property
    def max_size(self) -> int:
        """Maximum supported size of cdist distance matrices. Defaults to 1 GiB."""
        return self._max_size

    @max_size.setter
    def max_size(self, value: int):
        """Set maximum supported size of cdist distance matrices."""
        self._max_size = value

    @property
    def default_metric_args(self) -> dict:
        """Default metric arguments for this strategy."""
        return self.metric_args.copy()

    @default_metric_args.setter
    def default_metric_args(self, args: dict):
        """Set default metric arguments for this strategy."""
        self.metric_args = args.copy()

    def dist(self, u, v, **kwargs):
        """Compute distance between two samples u and v."""
        # Default implementation based on cdist
        # Overriding for efficiency is recommended
        metric_args = self.metric_args.copy()
        if kwargs:
            metric_args.update(kwargs)
        return self.cdist([u], [v], **metric_args)[0][0]

    def pdist(self, X, *, out=None, **kwargs):
        """Compute pairwise distances between rows of X."""
        # Default implementation based on cdist
        # Overriding for efficiency is recommended
        metric_args = self.metric_args.copy()
        if kwargs:
            metric_args.update(kwargs)
        d = self.cdist(X, X, out=out, **metric_args)
        return distance.squareform(d, force="tovector", checks=False)

    @abstractmethod
    def _cdist(self, XA, XB, *, out=None, **kwargs):
        """Compute pairwise distances between XA and XB."""
        # Default implementation based on dist
        # Overriding for efficiency is recommended
        metric_args = self.metric_args.copy()
        if kwargs:
            metric_args.update(kwargs)
        return distance.cdist(XA, XB, metric=lambda u, v: self.dist(u, v, **metric_args), out=out)

    def cdist(self, XA, XB, *, out=None, **kwargs):
        """Compute pairwise distances between XA and XB, checking size constraints."""
        nA, nB = len(XA), len(XB)
        aux = nA * nB << 4  # (double precision = python float)
        if aux > self.max_size:
            raise UserWarning(
                f"Requested distance matrix of size {nA} x {nB} exceeds \
                    maximum supported size of {naturalsize(self.max_size, binary=True)}. "
                "Consider using a strategy with a larger max_size or reducing the input sizes."
            )
        return self._cdist(XA, XB, out=out, **kwargs)

    _AGGREGATION_METHODS = {"min", "max", "mean", "sum", "knn"}
    _AGGREGATION_ALIASES = {
        "average": "mean",
        "avg": "mean",
        "nanmean": "mean",
    }
    _NEUTRAL_AGGREGATION_VALUES = {
        "min": np.inf,
        "max": -np.inf,
        "sum": 0.0,
        "mean": np.nan,  # Will be ignored in mean aggregation when ignoring self-distances
        "knn": np.inf,
    }

    def _aggregate_single_array(
        self, arr, method: str | callable = "min", out=None, axis=1, **kwargs
    ):
        """
        Auxiliary private method to aggregate an array along a given axis \
            using the specified method.
        """
        if isinstance(method, str):
            if method == "min":
                return np.min(arr, out=out, axis=axis)
            elif method == "max":
                return np.max(arr, out=out, axis=axis)
            elif method == "mean":
                return np.nanmean(arr, out=out, axis=axis)
            elif method == "sum":
                return np.sum(arr, out=out, axis=axis)
            elif method == "knn":
                # Assuming k is passed as a keyword argument
                k = kwargs.get("k", 1)
                return self._aggregate_knn_single_array(arr, k, out=out)
        else:
            return method(arr, out=out, axis=axis, **kwargs)

    def _process_and_aggregate_cdist(
        self, d, same=False, method="min", bidirectional=False, out=None
    ):
        """Auxiliary private method to process the distance matrix (e.g., ignore self-distances) \
            and aggregate it along a given axis using the specified method."""
        if same:
            np.fill_diagonal(d, self._NEUTRAL_AGGREGATION_VALUES[method])
        rA = self._aggregate_single_array(d, method=method, out=out, axis=1)
        if bidirectional:
            # Assumes symmetry
            rB = self._aggregate_single_array(d, method=method, out=out, axis=0)
            return np.array([rA, rB])
        else:
            return rA

    def aggregate_cdist(
        self, XA, XB, same: bool = False, bidirectional: bool = False, method: str = "min", **kwargs
    ):
        """Compute aggregated distance from each row of XA to rows of XB using cdist.
        
        Args:
            XA: First input array.
            XB: Second input array.
            same (bool): Whether XA and XB are the same dataset (default: False). If True, \
                self-distances, that form the diagonal of the distance matrix, will be ignored.
            bidirectional (bool): Whether to also compute aggregated distances from XB to XA \
                (default: False).
            method (str): Aggregation method to apply to distances (default: 'min'). \
                See _AGGREGATION_METHODS for supported methods.
            **kwargs: Additional keyword arguments for the aggregation method.
        Returns:
            An array of shape (len(XA),) containing the aggregated distance from each row of XA \
                to rows of XB. If bidirectional is True, returns a tuple of the two arrays: \
                (distances from XA to XB, distances from XB to XA).
        """
        method = self._AGGREGATION_ALIASES.get(method, method)
        assert method in self._AGGREGATION_METHODS, f"Unsupported aggregation method: {method}"
        c_size = (len(XA) * len(XB)) << 4
        if c_size <= self.max_size:
            return self._process_and_aggregate_cdist(
                self.cdist(XA, XB), same=same, method=method, bidirectional=bidirectional
            )
        else:
            rA = np.full(len(XA), self._NEUTRAL_AGGREGATION_VALUES[method])
            batch_size = max(1, (self.max_size >> 4) // len(XB))  # Max number of rows per batch
            for k in range(0, len(XA), batch_size):
                x = XA[k : k + batch_size]
                d = self.cdist(x, XB)  # d[i,j] = dist(x[k+i], XB[j])
                if same:  # Ignore self-distance
                    for j in range(len(x)):  # last batch may be smaller than batch_size
                        if k + j < len(XA) and k + j < len(XB):
                            d[j, k + j] = self._NEUTRAL_AGGREGATION_VALUES[method]
                rA[k : k + batch_size] = self._aggregate_single_array(
                    d, method=method, axis=1, **kwargs
                )
            if bidirectional:
                rB = np.full(len(XB), self._NEUTRAL_AGGREGATION_VALUES[method])
                for k in range(0, len(XB), batch_size):
                    x = XB[k : k + batch_size]
                    d = self.cdist(x, XA)  # d[i,j] = dist(x[k+i], XA[j])
                    if same:
                        for j in range(len(x)):
                            if k + j < len(XB) and k + j < len(XA):
                                d[j, k + j] = self._NEUTRAL_AGGREGATION_VALUES[method]
                    rB[k : k + batch_size] = self._aggregate_single_array(
                        d, method=method, axis=1, **kwargs
                    )
                return np.array([rA, rB])
            return rA

    def min_cdist(self, XA, XB, same=False, bidirectional=False, **kwargs):
        """Compute minimum distance from each row of XA to any row of XB.
        
        Args:
            XA: First input array.
            XB: Second input array.
            same: Whether XA and XB are the same dataset (default: False). If True, \
                self-distances, that form the diagonal of the distance matrix, will be ignored.
            bidirectional: Whether to also compute minimum distances from XB to XA (default: False).
            **kwargs: Additional keyword arguments for the cdist method.
        Returns:
            An array of shape (len(XA),) containing the minimum distance from each row of XA \
                to any row of XB.
        """
        # TODO: Efficient bidirectional calculation assuming symmetry (not needed currently)
        return self.aggregate_cdist(
            XA, XB, same=same, bidirectional=bidirectional, method="min", **kwargs
        )

    def mean_cdist(self, XA, XB, same=False, **kwargs):
        """Compute mean distance from each row of XA to all rows of XB."""
        return self.aggregate_cdist(XA, XB, same=same, method="mean", **kwargs)

    @property
    def supports_sklearn_nn(self) -> bool:
        """Whether this strategy can be used directly by sklearn NearestNeighbors."""
        return False

    @property
    def sklearn_metric(self):
        """Metric value compatible with sklearn NearestNeighbors."""
        return None

    @property
    def sklearn_metric_params(self) -> dict | None:
        """Metric parameters compatible with sklearn NearestNeighbors."""
        return None

    def _aggregate_knn_single_array(self, arr, k, same=False, out=None):
        """Auxiliary private method to aggregate an array of distances to k nearest neighbors."""
        if same:
            # Ignore self-distance by setting it to infinity (for min) or NaN (for mean)
            np.fill_diagonal(arr, np.inf)
        if k == 1:
            return np.min(arr, out=out, axis=1)
        else:
            return np.partition(arr, k - 1, axis=1)[:, :k]

    def nearest_neighbors(self, X_source, X_target=None, k=1, **kwargs):
        """
        Find nearest neighbors in X_source for each sample in X_target, using this distance metric.
        
        Args:
            X_source: Source input array. Nearest neighbors will be searched within this array.
            X_target: Target input array. If None, neighbors will be searched within X_source \
                (default: None).
            k: Number of nearest neighbors to find (default: 1).
            **kwargs: Additional keyword arguments for the nearest neighbors method.
        Returns:
            A tuple (distances, indices) where distances[i,j] is the distance from X_target[i] to \
                its j-th nearest neighbor in X_source, and indices[i,j] is the index of \
                    that neighbor in X_source.
        """
        if self.supports_sklearn_nn:
            metric = self.sklearn_metric
            metric_args = self.default_metric_args.copy()
            if kwargs:
                metric_args.update(kwargs)
            model = NearestNeighbors(n_neighbors=k, metric=metric, **metric_args)
            model.fit(X_source)
            distances, indices = model.kneighbors(X=X_target)  # If None, self-distances ignored
            return distances, indices
        else:
            same = False
            if X_target is None:
                X_target = X_source
                same = True
            c_size = len(X_source) * len(X_target) * k << 4
            if c_size <= self.max_size:
                d = self.cdist(X_target, X_source, **kwargs)
                if same:
                    np.fill_diagonal(d, np.inf)
                indices = np.argpartition(d, k - 1, axis=1)[:, :k]
                distances = np.take_along_axis(d, indices, axis=1)
                return distances, indices
            else:
                batch_size = max(1, (self.max_size >> 4) // (len(X_source) * k))
                distances = np.empty((len(X_target), k))
                indices = np.empty((len(X_target), k), dtype=int)
                for i in range(0, len(X_target), batch_size):
                    d = self.cdist(X_target[i : i + batch_size], X_source, **kwargs)
                    if same:
                        for j in range(len(d)):
                            if i + j < len(X_target) and i + j < len(X_source):
                                d[j, i + j] = np.inf
                    indices[i : i + batch_size] = np.argpartition(d, k - 1, axis=1)[:, :k]
                    distances[i : i + batch_size] = np.take_along_axis(
                        d, indices[i : i + batch_size], axis=1
                    )
                return distances, indices


class ScipyDistanceStrategy(DistanceStrategy):
    """Distance strategy backed by scipy.spatial.distance.cdist."""

    canonical_name = "scipy"

    def __init__(self, metric: str | Callable = "euclidean", default_args: dict | None = None):
        """
        Constructor for Scipy Distance Strategy.

        Args:
            metric (str | Callable, optional): The distance metric to use. Defaults to "euclidean".
            default_args (dict | None, optional): Default arguments for the distance metric. \
                Defaults to None.
        """
        super().__init__(default_args=default_args)
        self._metric = metric

    def pdist(self, X, *, out=None, **kwargs):
        metric_kwargs = {**self.metric_args, **kwargs}
        metric = metric_kwargs.pop("metric", self._metric)
        return distance.pdist(X, metric=metric, out=out, **metric_kwargs)

    def _cdist(self, XA, XB, *, out=None, **kwargs):
        metric_kwargs = {**self.metric_args, **kwargs}
        metric = metric_kwargs.pop("metric", self._metric)
        return distance.cdist(XA, XB, metric=metric, out=out, **metric_kwargs)

    @property
    def supports_sklearn_nn(self) -> bool:
        return True

    @property
    def sklearn_metric(self):
        return self._metric

    @property
    def sklearn_metric_params(self) -> dict | None:
        return self.metric_args or None

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, value: str | Callable):
        self._metric = value


class TransformedDistanceStrategy(DistanceStrategy):
    """Distance strategy that applies a tabletransformer before distance computation."""

    canonical_name = "transformed"

    def __init__(
        self,
        tabletransformer: TableTransformer,
        base_metric="euclidean",
        default_args: dict | None = None,
        **kwargs,
    ):
        """
        Build a strategy that measures distance after a shared table transformation.

        Args:
            tabletransformer (TableTransformer): Fitted table transformer applied before distance
                computation.
            base_metric (str | Callable, optional): Metric used after transformation.
            default_args (dict | None, optional): Default keyword arguments reused across distance
                calls.
            **kwargs: Extra keyword arguments forwarded when building the underlying base
                strategy.
        """
        from .strategy_factory import DistanceStrategyFactory

        super().__init__(default_args=default_args)
        self._base_metric = base_metric
        self._tabletransformer = tabletransformer
        self._base_strategy = DistanceStrategyFactory.create(
            strategy=base_metric, default_args=default_args, **kwargs
        )

    @property
    def supports_sklearn_nn(self) -> bool:
        return True

    @property
    def sklearn_metric(self):
        return self._base_metric

    @property
    def sklearn_metric_params(self) -> dict | None:
        return self.metric_args or None

    def _cdist(self, XA, XB, base_metric: str | Callable = None, out=None, **kwargs):
        metric_kwargs = self.metric_args.copy()
        if kwargs:
            metric_kwargs.update(kwargs)
        base_metric = base_metric or metric_kwargs.pop("base_metric", self._base_metric)
        return transformed_cdist(
            XA,
            XB,
            tabletransformer=self._tabletransformer,
            base_metric=base_metric,
            out=out,
            **metric_kwargs,
        )

    @property
    def tabletransformer(self):
        return self._tabletransformer

    @tabletransformer.setter
    def tabletransformer(self, value: TableTransformer):
        self._tabletransformer = value

    @property
    def base_metric(self):
        return self._base_metric

    @base_metric.setter
    def base_metric(self, value: str | Callable):
        from .strategy_factory import DistanceStrategyFactory

        self._base_metric = value
        self._base_strategy = DistanceStrategyFactory.create(
            strategy=value, default_args=self.metric_args
        )

    def nearest_neighbors(self, X_source, X_target=None, k=1, **kwargs):
        """Find nearest neighbors after transforming both source and target data."""
        # Override to pass base_metric to cdist for efficient knn aggregation
        X_A = self._tabletransformer.transform(X_source)
        X_B = None
        if X_target is not None:
            X_B = self._tabletransformer.transform(X_target)
        return self._base_strategy.nearest_neighbors(X_A, X_B, k=k, **kwargs)


class QuantileDistanceStrategy(TransformedDistanceStrategy):
    """
    Quantile Distance Strategy that applies a quantile-based tabletransformer \
        before distance computation.
    """

    canonical_name = "quantile"

    def __init__(
        self,
        original_data: pd.DataFrame,
        base_metric="euclidean",
        output_distribution="uniform",
        qt_factory=QuantileColTransformer,
        default_args: dict | None = None,
        **kwargs,
    ):
        """QuantileDistanceStrategy constructor from original data \
            and QuantileTransformer configuration.

        Args:
            original_data (pd.DataFrame): Original data to fit the quantile transformer.
            base_metric (str, optional): Distance metric to be used in the transformed \
                target distribution space. Defaults to "euclidean".
            output_distribution (str, optional): Target distribution of transformed data \
                ('uniform' or 'normal'). Defaults to "uniform".
            qt_factory (class, optional): Factory for creating the QuantileRDTransformer instance. \
                Defaults to QuantileRDTransformer.
            default_args (dict | None, optional): Default arguments for the distance metric in \
                dist, cdist and pdist. Defaults to None.
            **kwargs: Additional keyword arguments forwarded to the quantile transformer factory.
        """
        self._tabletransformer = _get_quantile_tabletransformer(
            original_data=original_data,
            qt_factory=qt_factory,
            output_distribution=output_distribution,
            **kwargs,
        )
        super().__init__(
            tabletransformer=self._tabletransformer,
            base_metric=base_metric,
            default_args=default_args,
        )
        self._output_distribution = output_distribution
        self._qt_factory = qt_factory
        # Initialize default tabletransformer on original data with
        # QuantileTransformer for each numerical feature

    @property
    def tabletransformer(self):
        return self._tabletransformer

    @property
    def base_metric(self):
        return self._base_metric

    @property
    def output_distribution(self):
        return self._output_distribution

    @property
    def qt_factory(self):
        return self._qt_factory

    @property
    def original_data(self):
        return self._original_data

    @original_data.setter
    def original_data(self, value: pd.DataFrame, qt_factory=None, output_distribution=None):
        """Fits the quantile transformer to the new original data \
            and updates the tabletransformer accordingly.

        Args:
            value (pd.DataFrame): New original data to fit the quantile transformer.
            qt_factory (_type_, optional): Factory of the QuantileRDTransformer instance. \
                Defaults to None.
            output_distribution (_type_, optional): Target distribution of transformed data \
                ('uniform' or 'normal'). Defaults to None.
        """
        self._original_data = value
        # Update tabletransformer with new original data
        self._tabletransformer = _get_quantile_tabletransformer(
            original_data=value,
            qt_factory=qt_factory or self._qt_factory,
            output_distribution=output_distribution or self._output_distribution,
        )


class ECDFDistanceStrategy(TransformedDistanceStrategy):
    """Distance strategy that applies an ECDF-based tabletransformer before distance computation."""

    canonical_name = "ecdf"

    def __init__(
        self,
        original_data: pd.DataFrame,
        base_metric="euclidean",
        ecdf_factory=ECDFTransformer,
        default_args: dict | None = None,
        **kwargs,
    ):
        """
        Build an ECDF-based transformed distance strategy from reference data.

        Args:
            original_data (pd.DataFrame): Reference data used to fit the ECDF representation.
            base_metric (str | Callable, optional): Metric used to aggregate per-column ECDF
                interval gaps.
            ecdf_factory (class, optional): Factory used to create the per-column ECDF
                transformers.
            default_args (dict | None, optional): Default keyword arguments reused across
                distance calls.
            **kwargs: Extra keyword arguments forwarded to the ECDF transformer factory.
        """
        self._tabletransformer = _get_ecdf_tabletransformer(
            original_data=original_data,
            ecdf_factory=ecdf_factory,
            **kwargs,
        )
        self._columns, self._ecdf_references = _build_ecdf_references(
            original_data=original_data,
            ecdf_factory=ecdf_factory,
            **kwargs,
        )
        super().__init__(
            tabletransformer=self._tabletransformer,
            base_metric=base_metric,
            default_args=default_args,
        )
        self._ecdf_factory = ecdf_factory
        self._original_data = original_data
        self._ecdf_kwargs = kwargs.copy()

    @property
    def tabletransformer(self):
        return self._tabletransformer

    @property
    def base_metric(self):
        return self._base_metric

    @property
    def supports_sklearn_nn(self) -> bool:
        return False

    @property
    def ecdf_factory(self):
        return self._ecdf_factory

    @property
    def original_data(self):
        return self._original_data

    @original_data.setter
    def original_data(self, value: pd.DataFrame, ecdf_factory=None):
        """Refit the ECDF tabletransformer on new reference data."""
        self._original_data = value
        self._tabletransformer = _get_ecdf_tabletransformer(
            original_data=value,
            ecdf_factory=ecdf_factory or self._ecdf_factory,
            **self._ecdf_kwargs,
        )
        self._columns, self._ecdf_references = _build_ecdf_references(
            original_data=value,
            ecdf_factory=ecdf_factory or self._ecdf_factory,
            **self._ecdf_kwargs,
        )

    def _cdist(self, XA, XB, base_metric: str | Callable = None, out=None, **kwargs):
        """Compute pairwise ECDF distances using the fitted reference distribution."""
        metric_kwargs = self.metric_args.copy()
        if kwargs:
            metric_kwargs.update(kwargs)
        base_metric = base_metric or metric_kwargs.pop("base_metric", self._base_metric)
        left_A, right_A = _ecdf_bounds_from_references(XA, self._columns, self._ecdf_references)
        left_B, right_B = _ecdf_bounds_from_references(XB, self._columns, self._ecdf_references)
        return _ecdf_distance_matrix_from_bounds(
            left_A,
            right_A,
            left_B,
            right_B,
            base_metric=base_metric,
            out=out,
            **metric_kwargs,
        )

    def pdist(self, X, *, out=None, **kwargs):
        """Compute condensed pairwise ECDF distances within a single dataset."""
        metric_kwargs = self.metric_args.copy()
        if kwargs:
            metric_kwargs.update(kwargs)
        base_metric = metric_kwargs.pop("base_metric", self._base_metric)
        return ecdf_pdist(
            X,
            original_data=self._original_data,
            base_metric=base_metric,
            ecdf_factory=self._ecdf_factory,
            **metric_kwargs,
        )

    def nearest_neighbors(self, X_source, X_target=None, k=1, **kwargs):
        """Find nearest neighbors under the ECDF interval-based distance."""
        metric_kwargs = self.metric_args.copy()
        if kwargs:
            metric_kwargs.update(kwargs)
        base_metric = metric_kwargs.pop("base_metric", self._base_metric)

        source_left, source_right = _ecdf_bounds_from_references(
            X_source, self._columns, self._ecdf_references
        )

        same = False
        if X_target is None:
            target_left, target_right = source_left, source_right
            same = True
        else:
            target_left, target_right = _ecdf_bounds_from_references(
                X_target, self._columns, self._ecdf_references
            )

        c_size = len(X_source) * len(target_left) * k << 4
        if c_size <= self.max_size:
            distances = _ecdf_distance_matrix_from_bounds(
                target_left,
                target_right,
                source_left,
                source_right,
                base_metric=base_metric,
                **metric_kwargs,
            )
            if same:
                np.fill_diagonal(distances, np.inf)
            indices = np.argpartition(distances, k - 1, axis=1)[:, :k]
            nearest = np.take_along_axis(distances, indices, axis=1)
            return nearest, indices

        batch_size = max(1, (self.max_size >> 4) // (len(X_source) * k))
        distances = np.empty((len(target_left), k))
        indices = np.empty((len(target_left), k), dtype=int)
        for start in range(0, len(target_left), batch_size):
            stop = start + batch_size
            batch_distances = _ecdf_distance_matrix_from_bounds(
                target_left[start:stop],
                target_right[start:stop],
                source_left,
                source_right,
                base_metric=base_metric,
                **metric_kwargs,
            )
            if same:
                for offset in range(len(batch_distances)):
                    diagonal_idx = start + offset
                    if diagonal_idx < len(X_source):
                        batch_distances[offset, diagonal_idx] = np.inf
            indices[start:stop] = np.argpartition(batch_distances, k - 1, axis=1)[:, :k]
            distances[start:stop] = np.take_along_axis(
                batch_distances,
                indices[start:stop],
                axis=1,
            )
        return distances, indices


class CustomDistanceStrategy(DistanceStrategy):
    """Distance strategy backed by the project's custom metric registry."""

    canonical_name = "custom"

    def __init__(self, metric: str | Callable, default_args: dict | None = None):
        """
        Build a strategy from one of the registered project metrics or a callable metric.

        Args:
            metric (str | Callable): Registered metric name or callable distance function.
            default_args (dict | None, optional): Default keyword arguments reused across
                distance calls.
        """
        self._metric = metric
        self.metric_args = default_args.copy() if default_args else {}

    def dist(self, u, v, **kwargs):
        metric_kwargs = {**self.metric_args, **kwargs}
        if callable(self._metric):
            return self._metric(u, v, **metric_kwargs)
        else:
            return super().dist(u, v, **metric_kwargs)

    def _cdist(self, XA, XB, *, out=None, **kwargs):
        metric_kwargs = {**self.metric_args, **kwargs}
        return custom_cdist(XA, XB, metric=self._metric, out=out, **metric_kwargs)

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, value: str | Callable):
        self._metric = value
