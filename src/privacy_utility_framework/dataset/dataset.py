"""Dataset wrappers and helpers for keeping tabular transformations consistent."""

from copy import deepcopy
from functools import lru_cache

import pandas as pd

from privacy_utility_framework.dataset.tabletransformer import TableTransformer
from privacy_utility_framework.dataset.transformers import (
    MinMaxScalerTransformer,
    OneHotEncoder,
    get_default_transformer,
    get_default_transformers,
)


class Dataset:
    """Wrap a dataframe together with the transformer used to encode it."""

    def __init__(self, data: pd.DataFrame, name="", transformer: TableTransformer = None):
        """
        Initialize a dataset with data and an optional name.

        Args:
            data (pd.DataFrame): The dataset to be managed.
            name (str): Optional name for the dataset.
            transformer (TableTransformer | None): Preconfigured transformer to attach to the
                dataset. When omitted, ``set_tabletransformer`` can create a default one later.
        """
        self.data = data
        self.name = name
        self.tabletransformer: TableTransformer = transformer
        if transformer is None:
            self.set_tabletransformer(fit=False)
        self.transformed_data = None

    def get_tabletransformer(self):
        """
        Get the current transformer associated with the dataset.

        Returns:
            TableTransformer: Transformer used to fit/transform this dataset.
        """
        return self.tabletransformer

    def set_tabletransformer(self, transformer: TableTransformer = None, fit=True):
        """
        Set the transformer for the dataset, optionally fitting it to the data.

        This transformer may transform every feature in the dataset, \
            whether numerical or categorical, based on its configuration.
        
        If a transformer is provided, it will be used; otherwise, a default \
        :class:`TableTransformer` will be initialized and fitted to the data. By default, it uses \
        a OneHotEncoder for categorical columns, a MinMaxScalerTransformer for numeric columns, \
            and the framework default transformer for other data types.

        Args:
            transformer (TableTransformer | None): Transformer instance to reuse. Passing an
                existing transformer is useful when multiple datasets must share the same fitted
                preprocessing pipeline.
            fit (bool): Whether to fit the resulting transformer on ``self.data`` immediately.
        """
        if transformer is not None:
            self.tabletransformer = transformer
        else:
            self.tabletransformer = TableTransformer()
            self.tabletransformer._learn_config(data=self.data)
            cat_cols = self.data.select_dtypes(include=["object", "category"]).columns
            num_cols = self.data.select_dtypes(include=[float, int]).columns
            sdtypes = {}
            sdtypes.update({col: "categorical" for col in cat_cols})
            sdtypes.update({col: "numerical" for col in num_cols})
            transformers = {}
            transformers.update({col: OneHotEncoder() for col in cat_cols})
            transformers.update({col: MinMaxScalerTransformer() for col in num_cols})
            # Explicit per-column overrides win over generic defaults inferred by TableTransformer.
            self.tabletransformer.update_sdtypes(sdtypes)
            self.tabletransformer.update_transformers(transformers)
        if fit:
            self.fit_tabletransformer()

    def fit_tabletransformer(self):
        """
        Fits the already configured :class:`TableTransformer` for the dataset.
        """
        self.tabletransformer.fit(self.data)

    def fit_transform(self):
        """
        Fits the :class:`TableTransformer` to the dataset and then applies the transformation.
        """
        self.fit_tabletransformer()
        self.transform()

    def transform(self):
        """
        Applies the transformation using the fitted :class:`TableTransformer`.
        This includes both categorical encoding and numeric scaling.
        Raises an error if the :class:`TableTransformer` is not fitted.
        """
        if not self.is_tabletransformer_fitted():
            raise RuntimeError("Tabletransformer must be set or fitted before transformation.")
        self.transformed_data = self.tabletransformer.transform(self.data)

    def set_tabletransformer_from(self, other_dataset: "Dataset"):
        """
        Sets the :class:`TableTransformer` from another dataset to ensure consistent \
            transformation and scaling.

        Args:
            other_dataset (Dataset): Dataset whose transformer should be shared.
        """
        self.set_tabletransformer(transformer=other_dataset.get_tabletransformer(), fit=False)

    def is_tabletransformer_fitted(self):
        """
        Check if the :class:`TableTransformer` is fitted.

        Returns:
            bool: ``True`` when the transformer can be used for ``transform``.
        """
        return self.tabletransformer._fitted

    _DEFAULT_TRANSFORMERS = {"numerical": MinMaxScalerTransformer(), "categorical": OneHotEncoder()}

    _LRU_MAXSIZE = len(TableTransformer.get_supported_sdtypes())

    @classmethod
    @lru_cache(maxsize=_LRU_MAXSIZE)
    def get_default_transformer(cls, sdtype):
        """
        Return a fresh default transformer for a semantic dtype.

        A deepcopy is returned so callers can mutate the transformer safely without affecting the
        cached defaults shared by the class.
        """
        val = cls._DEFAULT_TRANSFORMERS.get(sdtype)
        if val is None:
            return get_default_transformer(sdtype)

        return deepcopy(val)

    @classmethod
    @lru_cache(maxsize=1)
    def get_default_transformers(cls):
        """
        Return the default transformer mapping for all supported semantic dtypes.
        """
        transformers = get_default_transformers()
        transformers.update(cls._DEFAULT_TRANSFORMERS)
        return transformers

    def get_data(self):
        """
        Get the original data of the dataset.

        Returns:
            pd.DataFrame: Untransformed source dataframe.
        """
        return self.data

    def get_transformed_data(self):
        """
        Get the transformed data of the dataset.

        Returns:
            pd.DataFrame | None: Encoded dataframe or ``None`` when ``transform`` has not been
                called yet.
        """
        return self.transformed_data


class DatasetManager:
    """Coordinate transformation setup for an original/synthetic dataset pair."""

    def __init__(self, original: Dataset, synthetic: Dataset):
        """
        Initialize with original and synthetic Dataset objects.

        Args:
            original (Dataset): Reference dataset used to fit the shared transformer.
            synthetic (Dataset): Dataset transformed with the same fitted preprocessing pipeline.
        """
        self.original_dataset = original
        self.synthetic_dataset = synthetic
        self.tabletransformer = None
        self._set_tabletransformer = False

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
            DatasetManager: Manager wrapping the created :class:`Dataset` instances.
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
            DatasetManager: Manager initialized with the provided datasets.
        """
        return cls(original_dataset, synthetic_dataset)

    def set_tabletransformer(self, transformer: TableTransformer = None):
        """
        Sets up the :class:`TableTransformer` on the original dataset and fits it if not already \
            fitted, then applies the same :class:`TableTransformer` to the synthetic dataset.

        The transformer is always fitted on the original dataset first, then the fitted instance
        is attached to the synthetic dataset so both datasets are encoded in the same feature
        space.
        """
        self.original_dataset.set_tabletransformer(transformer=transformer)
        if not self.original_dataset.is_tabletransformer_fitted():
            self.original_dataset.fit_tabletransformer()

        # Share the fitted transformer so category ordering and scaling remain aligned.
        self.synthetic_dataset.set_tabletransformer_from(self.original_dataset)
        self._set_tabletransformer = True

    def transform_datasets(self):
        """
        Transforms both datasets using the same transformer.

        This method assumes ``set_tabletransformer`` has already been called so both datasets use
        the same preprocessing rules and output schema.
        """
        assert self._set_tabletransformer, (
            "TableTransformer must be set before transforming datasets."
        )

        self.original_dataset.transform()
        self.synthetic_dataset.transform()

    def get_original_dataset(self):
        """
        Get the original dataset.

        Returns:
            Dataset: The original dataset.
        """
        return self.original_dataset

    def get_synthetic_dataset(self):
        """
        Get the synthetic dataset.

        Returns:
            Dataset: The synthetic dataset.
        """
        return self.synthetic_dataset

    def get_datasets(self):
        """
        Return both managed datasets as a pair.

        Returns:
            tuple: A tuple containing the original and synthetic datasets.
        """
        return self.original_dataset, self.synthetic_dataset
