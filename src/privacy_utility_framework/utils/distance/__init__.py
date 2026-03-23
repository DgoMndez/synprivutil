"""
Module: synprivutil/src/privacy_utility_framework/utils/distance/__init__.py
Description: Distance utilities for synthetic data evaluation, distance strategies and custom cdist.

Author: Domingo Méndez García
Email: domingo.mendezg@um.es
Creation date: 09/03/2026
"""

__author__ = "Domingo Méndez García"

from .strategy_factory import DistanceStrategyFactory

__all__ = [
    "DistanceStrategyFactory",
    "metrics",
    "strategies",
]
