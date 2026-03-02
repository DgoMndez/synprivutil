from copy import deepcopy
from functools import lru_cache

import pandas as pd
import rdt.transformers
from rdt import HyperTransformer
from rdt.transformers import OneHotEncoder

from privacy_utility_framework.dataset.transformers import MinMaxScalerTransformer


class Dataset:
    def __init__(self, data: pd.DataFrame, name="", transformer: HyperTransformer = None):
        """
        Initialize a dataset with data and an optional name.

        Args:
            data (pd.DataFrame): The dataset to be managed.
            name (str): Optional name for the dataset.
        """
        self.data = data
        self.name = name
        self.hypertransformer: HyperTransformer = transformer
        if transformer is None:
            self.set_hypertransformer(fit=False)
        self.transformed_data = None

    def get_hypertransformer(self):
        """
        Get the current transformer associated with the dataset.

        Returns:
            HyperTransformer: The current transformer of the dataset.
        """
        return self.hypertransformer

    def set_hypertransformer(self, transformer: HyperTransformer = None, fit=True):
        """
        Set the transformer for the dataset, optionally fitting it to the data.
        
        This transformer, from RDT HyperTransformer, may transform every feature in the dataset, \
            whether numerical or categorical, based on its configuration.
        
        If a transformer is provided, it will be used; otherwise, a default HyperTransformer \
        will be initialized and fitted to the data, that consist of a OneHotEncoder for \
        categorical columns, a MinMaxScalerTransformer for numeric columns \
        and the default transformer in RDT for other data types.

        Args:
            transformer (HyperTransformer): The transformer to be set for the dataset.
            fit (bool): Whether to fit the transformer to the dataset's data. Default is True.
        """
        if transformer is not None:
            self.hypertransformer = transformer
        else:
            self.hypertransformer = HyperTransformer()
            self.hypertransformer._learn_config(data=self.data)
            cat_cols = self.data.select_dtypes(include=["object", "category"]).columns
            num_cols = self.data.select_dtypes(include=[float, int]).columns
            sdtypes = {}
            sdtypes.update({col: "categorical" for col in cat_cols})
            sdtypes.update({col: "numerical" for col in num_cols})
            transformers = {}
            transformers.update({col: OneHotEncoder() for col in cat_cols})
            transformers.update({col: MinMaxScalerTransformer() for col in num_cols})
            self.hypertransformer.update_sdtypes(sdtypes)
            self.hypertransformer.update_transformers(transformers)
        if fit:
            self.fit_hypertransformer()

    def fit_hypertransformer(self):
        """
        Fits the already configured hypertransformer for the dataset.
        """
        # Fit hypertransformer to the original dataset
        self.hypertransformer.fit(self.data)

    def fit_transform(self):
        """
        Fits the hypertransformer to the dataset and then applies the transformation.
        """
        self.fit_hypertransformer()
        self.transform()

    def transform(self):
        """
        Applies the transformation using the fitted hypertransformer.
        This includes both categorical encoding and numeric scaling.
        Raises an error if the hypertransformer is not fitted.
        """
        if not self.is_hypertransformer_fitted():
            raise RuntimeError("Hypertransformer must be set or fitted before transformation.")
        self.transformed_data = self.hypertransformer.transform(self.data)

    def set_hypertransformer_from(self, other_dataset):
        """
        Sets the hypertransformer from another dataset to ensure consistent transformation \
            and scaling.

        Args:
            other_dataset (Dataset): The dataset from which to copy the hypertransformer.
        """
        self.set_hypertransformer(transformer=other_dataset.get_hypertransformer(), fit=False)

    def is_hypertransformer_fitted(self):
        """
        Check if the hypertransformer is fitted.

        Returns:
            bool: True if the transformer is fitted, False otherwise.
        """
        return self.hypertransformer._fitted

    _DEFAULT_TRANSFORMERS = {"numerical": MinMaxScalerTransformer(), "categorical": OneHotEncoder()}

    _LRU_MAXSIZE = len(HyperTransformer._get_supported_sdtypes())

    @classmethod
    @lru_cache(maxsize=_LRU_MAXSIZE)
    def get_default_transformer(cls, sdtype):
        """
        Return the default transformer for a given sdtype.
        """
        val = cls._DEFAULT_TRANSFORMERS.get(sdtype)
        if val is None:
            return rdt.transformers.get_default_transformer(sdtype)
        else:
            return deepcopy(val)

    @classmethod
    @lru_cache(maxsize=1)
    def get_default_transformers(cls):
        """
        Return the default transformers for all supported sdtypes.
        """
        transformers = rdt.transformers.get_default_transformers()
        transformers.update(cls._DEFAULT_TRANSFORMERS)
        return transformers


class DatasetManager:
    def __init__(self, original: Dataset, synthetic: Dataset):
        """
        Initialize with original and synthetic Dataset objects.

        Args:
            original (Dataset): The original dataset.
            synthetic (Dataset): The synthetic dataset.
        """
        self.original_dataset = original
        self.synthetic_dataset = synthetic

    @classmethod
    def from_dataframes(cls, original_df, synthetic_df, original_name=None, synthetic_name=None):
        """
        Alternative constructor to create a DatasetManager directly from pandas DataFrames.

        Args:
            original_df (pd.DataFrame): The original dataset as a DataFrame.
            synthetic_df (pd.DataFrame): The synthetic dataset as a DataFrame.
            original_name (str, optional): Name for the original dataset. Defaults to None.
            synthetic_name (str, optional): Name for the synthetic dataset. Defaults to None.
        Returns:
            DatasetManager: An instance of DatasetManager initialized with the provided DataFrames.
        """
        original_name = original_name if original_name is not None else "Original Dataset"
        synthetic_name = synthetic_name if synthetic_name is not None else "Synthetic Dataset"
        original = Dataset(original_df, name=original_name)
        synthetic = Dataset(synthetic_df, name=synthetic_name)
        return cls(original, synthetic)

    @classmethod
    def from_datasets(cls, original_dataset, synthetic_dataset):
        """
        Alternative constructor to create a DatasetManager directly from Dataset objects.

        Args:
            original_dataset (Dataset): The original dataset as a Dataset object.
            synthetic_dataset (Dataset): The synthetic dataset as a Dataset object.
        Returns:
            DatasetManager: An instance of DatasetManager initialized with the provided \
                Dataset objects.
        """
        return cls(original_dataset, synthetic_dataset)

    def set_hypertransformer(self, transformer: HyperTransformer = None):
        """
        Sets up the hypertransformer on the original dataset and fits it if not already fitted,
        then applies the same hypertransformer to the synthetic dataset.
        """
        # Set and fit hypertransformer for the original dataset

        self.original_dataset.set_hypertransformer(transformer=transformer)
        if not self.original_dataset.is_hypertransformer_fitted():
            self.original_dataset.fit_hypertransformer()

        # Copy the fitted hypertransformer to the synthetic dataset
        self.synthetic_dataset.set_hypertransformer_from(self.original_dataset)

    def transform_datasets(self):
        """
        Transforms both datasets using the same transformer.
        Ensures consistent transformation (including scaling) between both datasets.
        """
        # Set up and copy transformer from original to synthetic dataset
        self.set_hypertransformer()

        # Transform both datasets using the same transformer
        self.original_dataset.transform()
        self.synthetic_dataset.transform()
