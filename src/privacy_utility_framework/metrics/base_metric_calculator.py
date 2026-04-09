from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from privacy_utility_framework.dataset.dataset import Dataset, DatasetManager
from privacy_utility_framework.dataset.tabletransformer import TableTransformer


class BaseMetricCalculator(ABC):
    """
    Shared abstract base for all metric calculators.

    This class standardizes how metric calculators receive input datasets,
    optionally preprocess them, validate comparison compatibility, and expose
    the original/synthetic datasets through a common interface.

    Concrete metric calculators implement only the metric-specific scoring
    logic in `evaluate`.
    """

    def __init__(
        self,
        original: pd.DataFrame | Dataset,
        synthetic: pd.DataFrame | Dataset,
        original_name: str | None = None,
        synthetic_name: str | None = None,
        preprocess: bool = False,
        preprocessor: TableTransformer | None = None,
        **kwargs,
    ):
        self._dm: DatasetManager | None = None

        # Keep signature extensible for concrete metrics that forward extra kwargs.
        _ = kwargs

        self._prepare_datasets(
            original,
            synthetic,
            original_name,
            synthetic_name,
            preprocess=preprocess,
            preprocessor=preprocessor,
        )
        self._validate_data()

    @classmethod
    def from_dataframes(
        cls,
        original_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        original_name: str | None = None,
        synthetic_name: str | None = None,
        preprocess: bool = False,
        preprocessor: TableTransformer | None = None,
        **kwargs,
    ):
        """Create a calculator directly from two pandas DataFrames."""
        return cls(
            original=original_df,
            synthetic=synthetic_df,
            original_name=original_name,
            synthetic_name=synthetic_name,
            preprocess=preprocess,
            preprocessor=preprocessor,
            **kwargs,
        )

    @classmethod
    def from_datasetmanager(cls, dataset_manager: DatasetManager):
        """Create a calculator from an already prepared DatasetManager."""
        calculator = cls.__new__(cls)
        calculator._dm = dataset_manager
        calculator._validate_data()
        return calculator

    @staticmethod
    def _build_dataset_manager(
        original: pd.DataFrame | Dataset,
        synthetic: pd.DataFrame | Dataset,
        original_name: str | None = None,
        synthetic_name: str | None = None,
    ) -> DatasetManager:
        """Build a DatasetManager from DataFrame or Dataset inputs."""
        if isinstance(original, pd.DataFrame) and isinstance(synthetic, pd.DataFrame):
            return DatasetManager.from_dataframes(
                original,
                synthetic,
                original_name,
                synthetic_name,
            )

        if isinstance(original, Dataset) and isinstance(synthetic, Dataset):
            return DatasetManager.from_datasets(original, synthetic)

        raise TypeError(
            "'original' and 'synthetic' must both be pandas DataFrames or both be Dataset "
            f"instances. Got original={type(original).__name__}, "
            f"synthetic={type(synthetic).__name__}."
        )

    @abstractmethod
    def evaluate(self):
        """Compute the metric score."""
        raise NotImplementedError("Subclasses must implement the evaluate method.")

    def _validate_data(self):
        """Validate that original and synthetic data are comparable for metric evaluation."""
        if set(self.original.data.columns) != set(self.synthetic.data.columns):
            raise ValueError("Column names do not match between original and synthetic datasets.")

        if len(self.original.data.columns) != len(self.synthetic.data.columns):
            raise ValueError(
                "Number of columns do not match between original and synthetic datasets."
            )

        assert not self.original.data.isnull().any().any(), (
            "Original dataset contains missing values."
        )
        assert not self.synthetic.data.isnull().any().any(), (
            "Synthetic dataset contains missing values."
        )

        for col in self.original.data.columns:
            if self.original.data[col].dtype != self.synthetic.data[col].dtype:
                raise ValueError(f"Data type mismatch in column '{col}'.")

    def _prepare_datasets(
        self,
        original: pd.DataFrame | Dataset,
        synthetic: pd.DataFrame | Dataset,
        original_name: str | None,
        synthetic_name: str | None,
        preprocess: bool = False,
        preprocessor: TableTransformer | None = None,
    ):
        """Initialize and optionally transform datasets before metric evaluation."""
        self._dm = self._build_dataset_manager(
            original=original,
            synthetic=synthetic,
            original_name=original_name,
            synthetic_name=synthetic_name,
        )

        if preprocessor is not None and not isinstance(preprocessor, TableTransformer):
            raise TypeError("'preprocessor' must be a TableTransformer.")

        if preprocessor is not None:
            preprocess = True

        if not preprocess:
            return

        if preprocessor is not None:
            self._dm.set_tabletransformer(transformer=preprocessor)
        elif isinstance(original, Dataset):
            self._dm.set_tabletransformer(transformer=original.get_tabletransformer())
        else:
            self._dm.set_tabletransformer()

        self._dm.transform_datasets()

    @staticmethod
    def _get_comparison_data(dataset: Dataset) -> pd.DataFrame:
        """Return transformed data when available, otherwise return raw data."""
        return dataset.transformed_data if dataset.transformed_data is not None else dataset.data

    @property
    def dataset_manager(self) -> DatasetManager:
        """Dataset manager backing this calculator instance."""
        return self._dm

    @property
    def original(self) -> Dataset:
        """Original dataset used as reference for metric computation."""
        return self._dm.original_dataset

    @property
    def synthetic(self) -> Dataset:
        """Synthetic dataset evaluated against the original dataset."""
        return self._dm.synthetic_dataset
