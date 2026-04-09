from privacy_utility_framework.metrics.base_metric_calculator import BaseMetricCalculator

# DONE 1: Add support for callable distance metrics on all distance-based privacy metrics.
# DONE 2: Admit Datasets apart from DataFrames in the constructor (flexibility).
# TODO 2.5: accept DatasetManager directly in the constructor and \
# leave dataframes for another cls method
# DONE 3: Implement CDF-based distance metrics.
# TODO 4: Decide whether preprocessing is left to the user or not, and wheter to use
# default transformer like in the previous version or configurable transformer


class PrivacyMetricCalculator(BaseMetricCalculator):
    """
    Abstract base class for privacy metric calculators, providing data validation
    and transformation methods for original and synthetic datasets.

    Parameters
    ----------
    original : pd.DataFrame or Dataset
        The original dataset to compare against the synthetic data.
    synthetic : pd.DataFrame or Dataset
        The synthetic dataset generated to resemble the original data.
    original_name : str, optional
        Name for the original dataset.
    synthetic_name : str, optional
        Name for the synthetic dataset.
    """

    pass
