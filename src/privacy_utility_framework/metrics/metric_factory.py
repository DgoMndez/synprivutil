from __future__ import annotations

from collections.abc import Sequence

from privacy_utility_framework.metrics.base_metric_calculator import BaseMetricCalculator
from privacy_utility_framework.metrics.metric_registry import MetricRegistry


class MetricFactory:
    """
    Factory for standardized metric creation and evaluation.

    Callers can instantiate/evaluate metrics either from calculator classes
    or from registry keys, enabling configuration-driven metric execution.
    """

    @staticmethod
    def create(
        metric: str | type[BaseMetricCalculator],
        *args,
        discover: bool = True,
        discovery_modules: tuple[str, ...] | list[str] | None = None,
        **kwargs,
    ) -> BaseMetricCalculator:
        """Create a metric calculator instance from a class or registered metric name."""
        if isinstance(metric, type):
            if not issubclass(metric, BaseMetricCalculator):
                raise TypeError("Metric class must inherit from BaseMetricCalculator.")
            return metric(*args, **kwargs)

        if discover:
            MetricRegistry.discover(modules=discovery_modules)

        metric_cls = MetricRegistry.get(metric)
        return metric_cls(*args, **kwargs)

    @staticmethod
    def evaluate(
        metric: str | type[BaseMetricCalculator],
        *args,
        discover: bool = True,
        discovery_modules: tuple[str, ...] | list[str] | None = None,
        **kwargs,
    ):
        """Create and evaluate a single metric in one call."""
        calculator = MetricFactory.create(
            metric,
            *args,
            discover=discover,
            discovery_modules=discovery_modules,
            **kwargs,
        )
        return calculator.evaluate()

    @staticmethod
    def evaluate_many(
        metric_specs: Sequence[dict],
        *,
        discover: bool = True,
        discovery_modules: tuple[str, ...] | list[str] | None = None,
    ) -> dict[str, float | dict]:
        """
        Evaluate a batch of metrics from declarative metric specifications.

        Spec shape:
            {
                "name": "privacy.dcr" | "basic_stats" | "ClassName",
                "args": [...],
                "kwargs": {...},
                "id": "optional-result-key"
            }
        """
        if discover:
            MetricRegistry.discover(modules=discovery_modules)

        results: dict[str, float | dict] = {}
        for spec in metric_specs:
            metric_name = spec["name"]
            args = spec.get("args", [])
            kwargs = spec.get("kwargs", {})
            result_key = spec.get("id", str(metric_name))
            results[result_key] = MetricFactory.evaluate(
                metric_name,
                *args,
                discover=False,
                **kwargs,
            )
        return results
