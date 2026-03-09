from __future__ import annotations

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

    _STRATEGIES = {SubDist.canonical_name: SubDist for SubDist in DistanceStrategy.__subclasses__()}

    @classmethod
    def create(
        cls, strategy: str = "euclidean", default_args: dict | None = None, **kwargs
    ) -> DistanceStrategy:
        """
        Creates a DistanceStrategy object from the name of its class, default arguments of the \
            metric and other constructor arguments.

        Args:
            strategy (str): The canonical name of the distance strategy to create. \
                This must match the __canonical_name attribute of the strategy class. \
                Available options are: "scipy", "transformed", "quantile" and "custom". \
                Additionally, any distance in scipy can be used by specifying its name.
            default_args (dict | None, optional): Default arguments for the distance metric \
                calculator in the strategy. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the strategy constructor.

        Returns:
            DistanceStrategy: The created distance strategy object.
        """
        if strategy in _SCIPY_METRICS:
            return ScipyDistanceStrategy(metric=strategy, default_args=default_args, **kwargs)

        assert strategy in cls._STRATEGIES, (
            f"Unknown distance strategy '{strategy}'. "
            f"Available options are: {list(cls._STRATEGIES.keys())}."
        )
        return cls._STRATEGIES[strategy](default_args=default_args, **kwargs)
