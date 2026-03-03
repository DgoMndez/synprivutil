"""
Module: synprivutil/src/privacy_utility_framework/dataset/transformers.py
Description: RDT transformer wrappers.

Author: Domingo Méndez García
Email: domingo.mendezg@um.es
Date: 02/03/2026
"""

import numpy as np
import pandas as pd
from rdt.transformers import BaseTransformer
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer


class MinMaxScalerTransformer(BaseTransformer):
    r"""
    RDT-compatible wrapper for sklearn's MinMaxScaler.

    This transformer scales numeric features to a fixed range [0, 1] using
    sklearn's MinMaxScaler, wrapped in the RDT Transformer interface for
    compatibility with HyperTransformer and other RDT pipelines.

    Attributes:
        _scaler (MinMaxScaler): The underlying sklearn MinMaxScaler instance.
    """

    INPUT_SDTYPE = "numerical"
    SUPPORTED_SDTYPES = ["numerical"]
    OUTPUT_SDTYPES = {"value": "numerical"}

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


class QuantileRDTransformer(BaseTransformer):
    """
    RDT-compatible wrapper for sklearn's QuantileTransformer.

    This transformer applies a quantile transformation to numeric features, mapping
    them to a specified distribution (e.g., uniform or normal).

    Attributes:
        _qtransformer (QuantileTransformer): The underlying sklearn QuantileTransformer instance, \
            if fitted.
        _n_quantiles (int): Number of quantiles to compute for the transformation.
        _output_distribution (str): Desired target distribution of transformed data \
            ('uniform' or 'normal').
        _subsample (int): Maximum number of samples to use for fitting the quantile transformer.
        _random_state (int or None): Random state for reproducibility when subsampling.

    """

    INPUT_SDTYPE = "numerical"
    SUPPORTED_SDTYPES = ["numerical"]
    OUTPUT_SDTYPES = {"value": "numerical"}

    def __init__(
        self,
        n_quantiles: int | None = 0,
        output_distribution: str = "uniform",
        subsample: int | None = None,
        random_state=None,
        **q_transformer_kwargs,
    ):
        """
        Initialize the QuantileRDTransformer.

        Args:
            n_quantiles (int or None): Number of quantiles to compute. If None or <= 0,
                the number of quantiles is lazily set to the number of rows when fitted. \
                    Default is 0.
            output_distribution (str): Desired target distribution of transformed data \
                ('uniform' or 'normal'). Default is 'uniform'.
            subsample (int): Maximum number of samples to use for fitting.  \
                If <=0, it will be lazily set to the number of rows when fitted. \
                    If None, no subsampling is performed.Default is None. \
               
             **q_transformer_kwargs: Additional keyword arguments forwarded to the \
                underlying QuantileTransformer.
        """
        super().__init__()
        self._n_quantiles = n_quantiles
        self._output_distribution = output_distribution
        self._subsample = subsample
        self._random_state = random_state
        self._qtransformer = None
        self._qkwargs = q_transformer_kwargs
        self._fitted = False

    def from_quantile_transformer(self, qtransformer):
        """
        Initialize the QuantileRDTransformer from an existing sklearn QuantileTransformer.

        Args:
            qtransformer (QuantileTransformer): A QuantileTransformer instance.
        """
        self._qtransformer = qtransformer
        self._n_quantiles = qtransformer.n_quantiles
        self._output_distribution = qtransformer.output_distribution
        self._subsample = qtransformer.subsample
        self._random_state = qtransformer.random_state
        self._fitted = False

    @staticmethod
    def _get_num_rows(data):
        if isinstance(data, (pd.Series | pd.DataFrame | np.ndarray)):
            return len(data)

        return np.asarray(data).shape[0]

    @staticmethod
    def _ensure_2d(data):
        if isinstance(data, pd.Series):
            return data.values.reshape(-1, 1)
        if isinstance(data, np.ndarray) and data.ndim == 1:
            return data.reshape(-1, 1)

        return data

    def _build_transformer(self, data):
        n_rows = self._get_num_rows(data)
        if self._n_quantiles is None or self._n_quantiles <= 0:
            self._n_quantiles = n_rows
        if self._subsample is not None and self._subsample <= 0:
            self._subsample = n_rows
        self._qtransformer = QuantileTransformer(
            n_quantiles=self._n_quantiles,
            output_distribution=self._output_distribution,
            subsample=self._subsample,
            random_state=self._random_state,
            **self._qkwargs,
        )

    def _fit(self, data):
        data = self._ensure_2d(data)
        self._build_transformer(data)
        self._qtransformer.fit(data)
        self._fitted = True

    def _transform(self, data):
        assert self._fitted, "QuantileRDTransformer must be fitted before calling transform."
        data = self._ensure_2d(data)
        return self._qtransformer.transform(data)

    def _reverse_transform(self, data):
        assert self._fitted, (
            "QuantileRDTransformer must be fitted before calling reverse_transform."
        )
        data = self._ensure_2d(data)
        return self._qtransformer.inverse_transform(data)
