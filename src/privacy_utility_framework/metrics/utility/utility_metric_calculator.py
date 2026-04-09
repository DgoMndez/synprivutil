import pandas as pd

from privacy_utility_framework.dataset.tabletransformer import TableTransformer
from privacy_utility_framework.metrics.base_metric_calculator import BaseMetricCalculator


class UtilityMetricCalculator(BaseMetricCalculator):
    def __init__(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        original_name: str = None,
        synthetic_name: str = None,
        preprocess: bool = True,
        preprocessor: TableTransformer | None = None,
        **kwargs,
    ):
        super().__init__(
            original=original,
            synthetic=synthetic,
            original_name=original_name,
            synthetic_name=synthetic_name,
            preprocess=preprocess,
            preprocessor=preprocessor,
            **kwargs,
        )
