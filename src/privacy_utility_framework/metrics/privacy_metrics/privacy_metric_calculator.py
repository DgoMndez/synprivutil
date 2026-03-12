from abc import ABC, abstractmethod

import pandas as pd

from privacy_utility_framework.dataset.dataset import Dataset, DatasetManager

# DONE 1: Add support for callable distance metrics on all distance-based privacy metrics.
# DONE 2: Admit Datasets apart from DataFrames in the constructor (flexibility).
# TODO 2.5: accept DatasetManager directly in the constructor and \
# leave dataframes for another cls method
# DONE 3: Implement CDF-based distance metrics.


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
        **kwargs,
    ):
        # Initialize attributes for original and synthetic data
        self._dm = None

        # Perform data transformation and normalization
        self._transform(original, synthetic, original_name, synthetic_name)
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
        hypertransformer=None,
    ):
        """
        Alternative constructor to create a PrivacyMetricCalculator directly from pandas DataFrames.

        Args:
            original_df (pd.DataFrame): The original dataset as a DataFrame.
            synthetic_df (pd.DataFrame): The synthetic dataset as a DataFrame.
            original_name (str, optional): Name for the original dataset. Defaults to None.
            synthetic_name (str, optional): Name for the synthetic dataset. Defaults to None.
            hypertransformer (HyperTransformer, optional): An optional HyperTransformer to apply \
                to both datasets. Defaults to None.
        """
        raise NotImplementedError("Not implemented yet")

    # TODO: from_datasets
    @classmethod
    def from_datasets(cls, original_dataset, synthetic_dataset):
        """
        Alternative constructor to create a PrivacyMetricCalculator directly from Dataset objects.

        Args:
            original_dataset (Dataset): The original dataset as a Dataset object.
            synthetic_dataset (Dataset): The synthetic dataset as a Dataset object.
        """
        raise NotImplementedError("Not implemented yet")

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

    def _transform(
        self,
        original: pd.DataFrame | Dataset,
        synthetic: pd.DataFrame | Dataset,
        original_name: str,
        synthetic_name: str,
    ):
        """
        Transforms both the original and synthetic datasets, \
            applying encoding or normalization for each column as needed.

        Parameters
        ----------
        original : pd.DataFrame or Dataset
            The original dataset to transform and normalize.
        synthetic : pd.DataFrame or Dataset
            The synthetic dataset to transform and normalize.
        original_name : str
            The name of the original dataset.
        synthetic_name : str
            The name of the synthetic dataset.
        """
        # Initialize DatasetManager from DataFrames or Dataset objects
        self._dm = self._build_dataset_manager(
            original=original,
            synthetic=synthetic,
            original_name=original_name,
            synthetic_name=synthetic_name,
        )

        # TODO 4: Check if data transformation should be done here or left to the user
        # Configure the transformer and scaler to apply transformations to datasets
        if isinstance(original, Dataset):
            # Respect the transformer already configured on the original Dataset.
            # This preserves custom transformer choices instead of resetting to defaults.
            self._dm.set_hypertransformer(transformer=original.get_hypertransformer())
        else:
            self._dm.set_hypertransformer()

        # Perform transformation and normalization on both datasets
        self._dm.transform_datasets()

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
