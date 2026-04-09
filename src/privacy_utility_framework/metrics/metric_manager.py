from __future__ import annotations

from privacy_utility_framework.metrics.base_metric_calculator import BaseMetricCalculator
from privacy_utility_framework.metrics.metric_factory import MetricFactory


class MetricManager:
    """
    Container for metric calculator instances executed as a group.

    This manager supports incremental metric registration and a unified
    `evaluate_all` call that returns all metric outputs in a single mapping.
    """

    def __init__(self):
        self.metric_instances: list[BaseMetricCalculator] = []

    def add_metric(
        self,
        metric_instance: BaseMetricCalculator | list[BaseMetricCalculator],
    ):
        """Add one metric calculator or a list of calculator instances."""
        if isinstance(metric_instance, list):
            for metric in metric_instance:
                self._add_single_metric(metric)
        else:
            self._add_single_metric(metric_instance)

    def add_metric_by_name(self, metric_name: str, *args, **kwargs) -> BaseMetricCalculator:
        """Create a metric by registry name, store it, and return the created instance."""
        metric = MetricFactory.create(metric_name, *args, **kwargs)
        self._add_single_metric(metric)
        return metric

    def _add_single_metric(self, metric_instance: BaseMetricCalculator):
        """Internal helper to validate and append a single metric instance."""
        if not isinstance(metric_instance, BaseMetricCalculator):
            raise TypeError("Metric instance must be a subclass of BaseMetricCalculator.")
        self.metric_instances.append(metric_instance)

    def evaluate_all(self) -> dict[str, float | dict]:
        """Evaluate all stored metrics and return results keyed by metric and dataset names."""
        results = {}
        for metric in self.metric_instances:
            metric_name = metric.__class__.__name__
            datasets_names = f"{metric.original.name, metric.synthetic.name}"
            results[metric_name + datasets_names] = metric.evaluate()
        return results
