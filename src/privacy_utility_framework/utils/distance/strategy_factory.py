from __future__ import annotations

from collections.abc import Callable

from .strategies import (
    DistanceStrategy,
    ScipyDistanceStrategy,
)

_SCIPY_METRICS = [
    "braycurtis",
    "canberra",
    "chebyshev",
    "cityblock",
    "correlation",
    "cosine",
    "dice",
    "euclidean",
    "hamming",
    "jaccard",
    "jensenshannon",
    "kulczynski1",
    "mahalanobis",
    "matching",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
]

# Factory software design pattern for distance strategies.


class DistanceStrategyFactory:
    """Factory that resolves distance strategies from their canonical names."""

    @staticmethod
    def _all_subclasses(cls):
        subclasses = set()
        for subcls in cls.__subclasses__():
            subclasses.add(subcls)
            subclasses.update(DistanceStrategyFactory._all_subclasses(subcls))
        return subclasses

    @classmethod
    def _strategies(cls):
        return {
            subcls.canonical_name: subcls
            for subcls in cls._all_subclasses(DistanceStrategy)
            if getattr(subcls, "canonical_name", None)
        }

    @classmethod
    def create(
        cls, strategy: str | Callable = "euclidean", default_args: dict | None = None, **kwargs
    ) -> DistanceStrategy:
        """
        Creates a DistanceStrategy object from the name of its class, default arguments of the \
            metric and other constructor arguments.

        Args:
            strategy (str | Callable): The canonical name of the distance strategy to create, \
                or a `Callable` if `CustomDistanceStrategy` is to be used. \
                This must match the __canonical_name attribute of the strategy class. \
                Available options are: "scipy", "transformed", "quantile", "ecdf" \
                and "custom". \
                Additionally, any distance in scipy can be used by specifying its name.
            default_args (dict | None, optional): Default arguments for the distance metric \
                calculator in the strategy. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the strategy constructor.

        Returns:
            DistanceStrategy: The created distance strategy object.
        """
        strategies = cls._strategies()

        if isinstance(strategy, str):
            if strategy in _SCIPY_METRICS:
                return ScipyDistanceStrategy(metric=strategy, default_args=default_args, **kwargs)

            assert strategy in strategies, (
                f"Unknown distance strategy '{strategy}'. "
                f"Available options are: {list(strategies.keys())}."
            )
            return strategies[strategy](default_args=default_args, **kwargs)
        else:
            metric_args = default_args.copy() if default_args else {}
            if kwargs:
                metric_args.update(kwargs)

            return strategies["custom"](
                metric=strategy,
                default_args=metric_args or None,
            )
