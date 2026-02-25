import numpy as np
import pandas as pd
from rdt import HyperTransformer
from rdt.transformers import BaseTransformer, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


class MinMaxScalerTransformer(BaseTransformer):
    r"""
    RDT-compatible wrapper for sklearn's MinMaxScaler.

    This transformer scales numeric features to a fixed range [0, 1] using
    sklearn's MinMaxScaler, wrapped in the RDT Transformer interface for
    compatibility with HyperTransformer and other RDT pipelines.

    Overrides protected methods (_fit, _transform, _reverse_transform) following
    RDT's template method pattern, where the public methods (fit, transform, etc.)
    in BaseTransformer handle common logic and invoke these protected hooks.

    Attributes:
        _scaler (MinMaxScaler): The underlying sklearn MinMaxScaler instance.
    """

    def __init__(self, feature_range=(0, 1), clip=True):
        r"""
        Initialize the MinMaxScalerTransformer.

        Args:
            feature_range (tuple): Target range for scaled features. Default is (0, 1).
            clip (bool): Whether to clip transformed data to feature_range. Default is True.
        """
        super().__init__()
        self._scaler = MinMaxScaler(feature_range=feature_range, clip=clip)

    def _fit(self, data):
        r"""
        Fit the scaler to the data. (Protected method following RDT pattern)

        Args:
            data: Input data (pandas Series, numpy array, or DataFrame column).
        """
        # Ensure data is 2D for sklearn
        if isinstance(data, pd.Series):
            data = data.values.reshape(-1, 1)
        elif isinstance(data, np.ndarray) and data.ndim == 1:
            data = data.reshape(-1, 1)

        self._scaler.fit(data)

    def _transform(self, data):
        r"""
        Transform the data using the fitted scaler. (Protected method following RDT pattern)

        Args:
            data: Input data to transform.

        Returns:
            Scaled data as numpy array (1D or 2D depending on input).
        """
        # Ensure data is 2D for sklearn
        if isinstance(data, pd.Series):
            data = data.values.reshape(-1, 1)
        elif isinstance(data, np.ndarray) and data.ndim == 1:
            data = data.reshape(-1, 1)

        return self._scaler.transform(data)

    def _reverse_transform(self, data):
        r"""
        Reverse the scaling transformation (inverse transform). \
            (Protected method following RDT pattern)

        Args:
            data: Scaled data to inverse transform back to original scale.

        Returns:
            Data in original scale as numpy array.
        """
        # Ensure data is 2D for sklearn
        if isinstance(data, pd.Series):
            data = data.values.reshape(-1, 1)
        elif isinstance(data, np.ndarray) and data.ndim == 1:
            data = data.reshape(-1, 1)

        return self._scaler.inverse_transform(data)


class Dataset:
    def __init__(self, data: pd.DataFrame, name=""):
        """
        Initialize a dataset with data and an optional name.

        Args:
            data (pd.DataFrame): The dataset to be managed.
            name (str): Optional name for the dataset.
        """
        self.data = data
        self.name = name
        self.transformer = None
        self.transformed_data = None

    def set_transformer(self):
        """
        Sets up and fits the transformer for the dataset.
        Configures a transformer for both categorical encoding (OneHotEncoder) and numeric scaling \
            (MinMaxScalerTransformer).
        """
        # Initialize transformer and configure it for categorical and numeric columns
        self.transformer = HyperTransformer()
        self.transformer.detect_initial_config(data=self.data)
        config = self.transformer.get_config()

        categorical_columns = self.data.select_dtypes(include=["object", "category"]).columns
        numeric_columns = self.data.select_dtypes(include=[float, int]).columns

        # Add OneHotEncoder for categorical columns
        config["transformers"].update({col: OneHotEncoder() for col in categorical_columns})

        # Add MinMaxScalerTransformer for numeric columns
        config["transformers"].update({col: MinMaxScalerTransformer() for col in numeric_columns})

        # Fit transformer to the original dataset
        self.transformer.fit(self.data)

    def transform(self):
        """
        Applies the transformation using the fitted transformer.
        This includes both categorical encoding and numeric scaling.
        Raises an error if the transformer is not set.
        """
        if self.transformer is None:
            raise RuntimeError("Transformer must be set before transformation.")
        self.transformed_data = self.transformer.transform(self.data)

    def set_transformer_from(self, other_dataset):
        """
        Sets the transformer from another dataset to ensure consistent transformation and scaling.

        Args:
            other_dataset (Dataset): The dataset from which to copy the transformer.
        """
        self.transformer = other_dataset.transformer


class DatasetManager:
    def __init__(self, original, synthetic, original_name=None, synthetic_name=None):
        """
        Initialize with original and synthetic Dataset objects.

        Args:
            original (pd.DataFrame): The original dataset.
            synthetic (pd.DataFrame): The synthetic dataset.
        """
        original_name = original_name if original_name is not None else "Original_Dataset"
        synthetic_name = synthetic_name if synthetic_name is not None else "Synthetic_Dataset"
        self.original_dataset = Dataset(original, name=original_name)
        self.synthetic_dataset = Dataset(synthetic, name=synthetic_name)

    def set_transformer(self):
        """
        Sets up and fits the transformer and scaler on the original dataset,
        then applies the same transformer and scaler to the synthetic dataset.
        """
        # Set and fit transformer and scaler for the original dataset
        self.original_dataset.set_transformer()

        # Copy the fitted transformer and scaler to the synthetic dataset
        self.synthetic_dataset.set_transformer_from(self.original_dataset)

    def transform_datasets(self):
        """
        Transforms both datasets using the same transformer.
        Ensures consistent transformation (including scaling) between both datasets.
        """
        # Set up and copy transformer from original to synthetic dataset
        self.set_transformer()

        # Transform both datasets using the same transformer
        self.original_dataset.transform()
        self.synthetic_dataset.transform()
