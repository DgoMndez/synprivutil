from __future__ import annotations

import importlib
import inspect
import pkgutil
import re

from privacy_utility_framework.metrics.base_metric_calculator import BaseMetricCalculator


class MetricRegistry:
    """
    Global registry of metric calculator classes.

    The registry maps human-friendly metric names (for example
    ``basic_stats`` or ``privacy.dcr``) to calculator classes and supports
    optional package discovery so callers can create metrics by name without
    importing every metric module manually.
    """

    DEFAULT_DISCOVERY_MODULES = (
        "privacy_utility_framework.metrics.utility",
        "privacy_utility_framework.metrics.privacy",
    )

    _registry: dict[str, type[BaseMetricCalculator]] = {}

    @classmethod
    def register(cls, metric_cls: type[BaseMetricCalculator], *aliases: str):
        """Register a metric class under canonical names and optional aliases."""
        if not issubclass(metric_cls, BaseMetricCalculator):
            raise TypeError("Registered class must inherit from BaseMetricCalculator.")

        names = {metric_cls.__name__, cls._canonical_name(metric_cls.__name__)}
        names.update(aliases)

        module = metric_cls.__module__
        names.add(f"{module}.{metric_cls.__name__}")

        if ".utility." in module:
            names.add(f"utility.{cls._canonical_name(metric_cls.__name__)}")
        if ".privacy." in module:
            names.add(f"privacy.{cls._canonical_name(metric_cls.__name__)}")

        for name in names:
            normalized = cls._normalize_key(name)
            existing = cls._registry.get(normalized)
            if existing is not None and existing is not metric_cls:
                raise ValueError(f"Registry key '{name}' is already bound to {existing.__name__}.")
            cls._registry[normalized] = metric_cls

    @classmethod
    def get(cls, name: str) -> type[BaseMetricCalculator]:
        """Return the metric class registered for a given key."""
        metric_cls = cls._registry.get(cls._normalize_key(name))
        if metric_cls is None:
            raise KeyError(f"Metric '{name}' is not registered.")
        return metric_cls

    @classmethod
    def list_registered(cls) -> list[str]:
        """List all normalized registry keys currently available."""
        return sorted(cls._registry.keys())

    @classmethod
    def clear(cls):
        """Remove all registered metric entries."""
        cls._registry.clear()

    @classmethod
    def discover(cls, modules: tuple[str, ...] | list[str] | None = None):
        """Import configured metric packages and register all concrete calculators."""
        modules = tuple(modules or cls.DEFAULT_DISCOVERY_MODULES)

        for module_name in modules:
            module = importlib.import_module(module_name)
            module_path = getattr(module, "__path__", None)
            if module_path is None:
                continue

            for _, discovered, _ in pkgutil.walk_packages(module_path, prefix=f"{module_name}."):
                importlib.import_module(discovered)

        for metric_cls in cls._all_metric_subclasses(BaseMetricCalculator):
            if inspect.isabstract(metric_cls):
                continue
            cls.register(metric_cls)

    @classmethod
    def _all_metric_subclasses(
        cls, base: type[BaseMetricCalculator]
    ) -> set[type[BaseMetricCalculator]]:
        """Recursively collect all subclasses of a base metric calculator type."""
        subclasses: set[type[BaseMetricCalculator]] = set(base.__subclasses__())
        for subcls in list(subclasses):
            subclasses.update(cls._all_metric_subclasses(subcls))
        return subclasses

    @staticmethod
    def _canonical_name(name: str) -> str:
        """Convert a calculator class name into its canonical snake_case key."""
        base = re.sub(r"Calculator$", "", name)
        snake = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", base)
        snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", snake).lower()
        return re.sub(r"_calculator$", "", snake)

    @staticmethod
    def _normalize_key(name: str) -> str:
        """Normalize registry lookup keys for case-insensitive matching."""
        return name.strip().lower()
