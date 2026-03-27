"""Public synthesizer APIs exposed by the framework."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from typing import TYPE_CHECKING

from .core import (
    BaseModel,
    GaussianMixtureModel,
    RandomModel,
)

if TYPE_CHECKING:
    from .sdv import (
        CopulaGANModel,
        CTGANModel,
        GaussianCopulaModel,
        TVAEModel,
    )

_CORE_EXPORTS = [
    BaseModel,
    GaussianMixtureModel,
    RandomModel,
]
_SDV_EXPORTS = [
    CTGANModel,
    CopulaGANModel,
    GaussianCopulaModel,
    TVAEModel,
]

__all__ = list(_CORE_EXPORTS)
if find_spec("sdv") is not None:
    __all__.extend(_SDV_EXPORTS)


def __getattr__(name: str):
    if name in _SDV_EXPORTS:
        try:
            module = import_module(".sdv", __name__)
        except ImportError as exc:
            raise ImportError(
                f"{name} requires the optional SDV dependencies. "
                "Install privacy_utility_framework with the 'sdv' extra."
            ) from exc
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_CORE_EXPORTS) | set(_SDV_EXPORTS))
