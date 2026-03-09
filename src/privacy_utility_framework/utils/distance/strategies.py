from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import pandas as pd
from rdt import HyperTransformer
from scipy.spatial import distance

from privacy_utility_framework.dataset.transformers import QuantileRDTransformer

from .distance import (
    _get_quantile_hypertransformer,
    custom_cdist,
    transformed_cdist,
)


class DistanceStrategy(ABC):
    """Strategy interface for distance computation (single samples, pairwise or matrix)."""

    canonical_name = None

    def __init__(self, default_args: dict | None = None):
        """
        Constructor for DistanceStrategy.
        
        Args:
            default_args (dict | None, optional): Default arguments for the distance metric in \
                dist, cdist and pdist. Defaults to None.
        """
        self.metric_args = default_args.copy() if default_args else {}

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
    def cdist(self, XA, XB, *, out=None, **kwargs):
        """Compute pairwise distances between XA and XB."""
        # Default implementation based on dist
        # Overriding for efficiency is recommended
        metric_args = self.metric_args.copy()
        if kwargs:
            metric_args.update(kwargs)
        return distance.cdist(XA, XB, metric=lambda u, v: self.dist(u, v, **metric_args), out=out)

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

    def cdist(self, XA, XB, *, out=None, **kwargs):
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
    """Distance strategy that applies a hypertransformer before distance computation."""

    canonical_name = "transformed"

    def __init__(
        self,
        hypertransformer: HyperTransformer,
        base_metric="euclidean",
        default_args: dict | None = None,
    ):
        super().__init__(default_args=default_args)
        self._base_metric = base_metric
        self._hypertransformer = hypertransformer

    def cdist(self, XA, XB, base_metric: str | Callable = None, out=None, **kwargs):
        metric_kwargs = self.metric_args.copy()
        if kwargs:
            metric_kwargs.update(kwargs)
        base_metric = base_metric or metric_kwargs.pop("base_metric", self._base_metric)
        return transformed_cdist(
            XA,
            XB,
            hypertransformer=self._hypertransformer,
            base_metric=base_metric,
            out=out,
            **metric_kwargs,
        )

    @property
    def hypertransformer(self):
        return self._hypertransformer

    @hypertransformer.setter
    def hypertransformer(self, value: HyperTransformer):
        self._hypertransformer = value

    @property
    def base_metric(self):
        return self._base_metric

    @base_metric.setter
    def base_metric(self, value: str | Callable):
        self._base_metric = value


class QuantileDistanceStrategy(DistanceStrategy):
    """
    Quantile Distance Strategy that applies a quantile-based hypertransformer \
        before distance computation.
    """

    canonical_name = "quantile"

    def __init__(
        self,
        original_data: pd.DataFrame,
        base_metric="euclidean",
        output_distribution="uniform",
        qt_factory=QuantileRDTransformer,
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

        super().__init__(default_args=default_args)
        self._original_data = original_data
        self._base_metric = base_metric
        self._output_distribution = output_distribution
        self._qt_factory = qt_factory
        # Initialize default hypertransformer on original data with
        # QuantileTransformer for each numerical feature
        self._hypertransformer = _get_quantile_hypertransformer(
            original_data=original_data,
            qt_factory=qt_factory,
            output_distribution=output_distribution,
            **kwargs,
        )

    def cdist(self, XA, XB, base_metric: str | Callable = None, out=None, **kwargs):
        metric_kwargs = {**self.metric_args, **kwargs}
        base_metric = base_metric or metric_kwargs.pop("base_metric", self._base_metric)

        return transformed_cdist(
            XA,
            XB,
            hypertransformer=self._hypertransformer,
            base_metric=base_metric,
            out=out,
            **metric_kwargs,
        )

    @property
    def hypertransformer(self):
        return self._hypertransformer

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
            and updates the hypertransformer accordingly.

        Args:
            value (pd.DataFrame): New original data to fit the quantile transformer.
            qt_factory (_type_, optional): Factory of the QuantileRDTransformer instance. \
                Defaults to None.
            output_distribution (_type_, optional): Target distribution of transformed data \
                ('uniform' or 'normal'). Defaults to None.
        """
        self._original_data = value
        # Update hypertransformer with new original data
        self._hypertransformer = _get_quantile_hypertransformer(
            original_data=value,
            qt_factory=qt_factory or self._qt_factory,
            output_distribution=output_distribution or self._output_distribution,
        )


class CustomDistanceStrategy(DistanceStrategy):
    """Distance strategy backed by the project's custom metric registry."""

    canonical_name = "custom"

    def __init__(self, metric: str | Callable, default_args: dict | None = None):
        self._metric = metric
        self.metric_args = default_args.copy() if default_args else {}

    def dist(self, u, v, **kwargs):
        metric_kwargs = {**self.metric_args, **kwargs}
        if callable(self._metric):
            return self._metric(u, v, **metric_kwargs)
        else:
            return super().dist(u, v, **metric_kwargs)

    def cdist(self, XA, XB, *, out=None, **kwargs):
        metric_kwargs = {**self.metric_args, **kwargs}
        return custom_cdist(XA, XB, metric=self._metric, out=out, **metric_kwargs)

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, value: str | Callable):
        self._metric = value
