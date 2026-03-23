from abc import ABC, abstractmethod

import pandas as pd

from privacy_utility_framework.dataset.dataset import Dataset, DatasetManager
from privacy_utility_framework.dataset.tabletransformer import TableTransformer

# DONE 1: Add support for callable distance metrics on all distance-based privacy metrics.
# DONE 2: Admit Datasets apart from DataFrames in the constructor (flexibility).
# TODO 2.5: accept DatasetManager directly in the constructor and \
# leave dataframes for another cls method
# DONE 3: Implement CDF-based distance metrics.
# TODO 4: Decide whether preprocessing is left to the user or not, and wheter to use
# default transformer like in the previous version or configurable transformer


class PrivacyMetricCalculator(ABC):
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

    def __init__(
        self,
        original: pd.DataFrame | Dataset,
        synthetic: pd.DataFrame | Dataset,
        original_name: str = None,
        synthetic_name: str = None,
        preprocess: bool = False,
        preprocessor: TableTransformer | None = None,
        **kwargs,
    ):
        # Initialize attributes for original and synthetic data
        self._dm = None

        # Build datasets and optionally preprocess them for comparison.
        self._prepare_datasets(
            original,
            synthetic,
            original_name,
            synthetic_name,
            preprocess=preprocess,
            preprocessor=preprocessor,
        )
        # Perform data validation to ensure compatibility between datasets
        self._validate_data()

    # TODO: from_dataframes
    @classmethod
    def from_dataframes(
        cls,
        original_df,
        synthetic_df,
        original_name=None,
        synthetic_name=None,
        preprocess=False,
        preprocessor=None,
    ):
        """
        Alternative constructor to create a PrivacyMetricCalculator directly from pandas DataFrames.

        Args:
            original_df (pd.DataFrame): The original dataset as a DataFrame.
            synthetic_df (pd.DataFrame): The synthetic dataset as a DataFrame.
            original_name (str, optional): Name for the original dataset. Defaults to None.
            synthetic_name (str, optional): Name for the synthetic dataset. Defaults to None.
            preprocess (bool, optional): Whether to preprocess both datasets before evaluation.
            preprocessor (TableTransformer, optional): Optional transformer to reuse when
                ``preprocess`` is enabled.
        """
        raise NotImplementedError("Not implemented yet")

    # TODO: from_datasetmanager
    @classmethod
    def from_datasetmanager(cls, dataset_manager: DatasetManager):
        """
        Alternative constructor to create a PrivacyMetricCalculator directly from Dataset objects.

        Args:
            original_dataset (Dataset): The original dataset as a Dataset object.
            synthetic_dataset (Dataset): The synthetic dataset as a Dataset object.
        """
        calculator = cls.__new__(cls)  # Create an uninitialized instance
        calculator._dm = dataset_manager  # Directly assign the provided DatasetManager
        calculator._validate_data()  # Validate the datasets
        return calculator

    @staticmethod
    def _build_dataset_manager(
        original: pd.DataFrame | Dataset,
        synthetic: pd.DataFrame | Dataset,
        original_name: str = None,
        synthetic_name: str = None,
    ) -> DatasetManager:
        """
        Build a DatasetManager from either DataFrame inputs or Dataset inputs.

        Raises
        ------
        TypeError
            If input types are mixed or unsupported.
        """
        if isinstance(original, pd.DataFrame) and isinstance(synthetic, pd.DataFrame):
            return DatasetManager.from_dataframes(
                original, synthetic, original_name, synthetic_name
            )

        if isinstance(original, Dataset) and isinstance(synthetic, Dataset):
            return DatasetManager.from_datasets(original, synthetic)

        raise TypeError(
            "'original' and 'synthetic' must both be pandas DataFrames or both be Dataset "
            f"instances. Got original={type(original).__name__}, "
            f"synthetic={type(synthetic).__name__}."
        )

    @abstractmethod
    def evaluate(self) -> float:
        """
        Abstract method for metric evaluation. Must be implemented in subclasses.

        Returns
        -------
        float
            The calculated privacy metric score.
        """
        raise NotImplementedError("Subclasses must implement the evaluate method.")

    def _validate_data(self):
        """
        Validates that the original and synthetic datasets have the same structure,
        no missing values, and compatible data types.

        Raises
        ------
        ValueError
            If the datasets do not match in structure or have incompatible data types.
        """
        # Check that column names match between original and synthetic datasets
        if set(self.original.data.columns) != set(self.synthetic.data.columns):
            raise ValueError("Column names do not match between original and synthetic datasets.")

        # Check that the number of columns matches
        if len(self.original.data.columns) != len(self.synthetic.data.columns):
            raise ValueError(
                "Number of columns do not match between original and synthetic datasets."
            )

        # Ensure no missing values in either dataset
        assert not self.original.data.isnull().any().any(), (
            "Original dataset contains missing values."
        )
        assert not self.synthetic.data.isnull().any().any(), (
            "Synthetic dataset contains missing values."
        )

        # Confirm data types are consistent across columns in both datasets
        for col in self.original.data.columns:
            if self.original.data[col].dtype != self.synthetic.data[col].dtype:
                raise ValueError(f"Data type mismatch in column '{col}'.")

    def _prepare_datasets(
        self,
        original: pd.DataFrame | Dataset,
        synthetic: pd.DataFrame | Dataset,
        original_name: str,
        synthetic_name: str,
        preprocess: bool = False,
        preprocessor: TableTransformer | None = None,
    ):
        """
        Build the dataset manager and optionally preprocess both datasets for comparison.

        Parameters
        ----------
        original : pd.DataFrame or Dataset
            The original dataset to prepare.
        synthetic : pd.DataFrame or Dataset
            The synthetic dataset to prepare.
        original_name : str
            The name of the original dataset.
        synthetic_name : str
            The name of the synthetic dataset.
        preprocess : bool, optional
            Whether to preprocess both datasets inside the calculator before evaluation.
        preprocessor : TableTransformer, optional
            Transformer to fit on the original dataset and reuse on the synthetic dataset when
            preprocessing is enabled.
        """
        # Initialize DatasetManager from DataFrames or Dataset objects
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

        # Respect a caller-provided transformer first. Otherwise, if a Dataset already carries
        # a configured transformer, reuse it; fall back to the default dataset transformer only
        # for the opt-in preprocessing path.
        if preprocessor is not None:
            self._dm.set_tabletransformer(transformer=preprocessor)
        elif isinstance(original, Dataset):
            self._dm.set_tabletransformer(transformer=original.get_tabletransformer())
        else:
            self._dm.set_tabletransformer()

        self._dm.transform_datasets()

    @staticmethod
    def _get_comparison_data(dataset: Dataset) -> pd.DataFrame:
        """
        Return the representation to use for metric evaluation.

        Transformed data is used when available; otherwise metrics operate on the original data
        supplied by the user.
        """
        return dataset.transformed_data if dataset.transformed_data is not None else dataset.data

    @property
    def dataset_manager(self) -> DatasetManager:
        """
        Accessor for the internal DatasetManager instance.

        Returns
        -------
        DatasetManager
            The DatasetManager instance managing the original and synthetic datasets.
        """
        return self._dm

    @property
    def original(self) -> Dataset:
        """
        Accessor for the original dataset.

        Returns
        -------
        Dataset
            The original dataset after transformation and normalization.
        """
        return self._dm.original_dataset

    @property
    def synthetic(self) -> Dataset:
        """
        Accessor for the synthetic dataset.

        Returns
        -------
        Dataset
            The synthetic dataset after transformation and normalization.
        """
        return self._dm.synthetic_dataset
