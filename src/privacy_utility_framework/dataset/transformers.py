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


def _get_num_rows(data):
    if isinstance(data, (pd.Series | pd.DataFrame | np.ndarray)):
        return len(data)

    return np.asarray(data).shape[0]


def _ensure_2d(data):
    if isinstance(data, pd.Series):
        return data.values.reshape(-1, 1)
    if isinstance(data, np.ndarray) and data.ndim == 1:
        return data.reshape(-1, 1)

    return data


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

    def _build_transformer(self, data):
        n_rows = _get_num_rows(data)
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
        data = _ensure_2d(data)
        self._build_transformer(data)
        self._qtransformer.fit(data)
        self._fitted = True

    def _transform(self, data):
        assert self._fitted, "QuantileRDTransformer must be fitted before calling transform."
        data = _ensure_2d(data)
        return self._qtransformer.transform(data)

    def _reverse_transform(self, data):
        assert self._fitted, (
            "QuantileRDTransformer must be fitted before calling reverse_transform."
        )
        data = _ensure_2d(data)
        return self._qtransformer.inverse_transform(data)


class ECDFTransformer(BaseTransformer):
    """
    RDT-compatible wrapper for Empirical Cumulative Distribution Function (ECDF) transformation.

    This transformer maps numeric features to their ECDF values, transforming data into a \
        uniform distribution on [0, 1]. For each value x_j in the fitted column:
    
    y_j = F^(x_j) = r_j / N,
    
    where r_j is the rank of x_j in the fitted data, and N is the total number of samples.

    This transformer is designed to work on single columns following RDT's pattern \
        (data is passed as pd.Series when called via fit(data, column="col_name")).

    Attributes:
        _sorted_values (numpy array): Sorted values from the fitted column.
        _n_samples (int): Number of samples used during fitting.
    """

    INPUT_SDTYPE = "numerical"
    SUPPORTED_SDTYPES = ["numerical"]
    OUTPUT_SDTYPES = {"value": "numerical"}

    def __init__(self, subsample: int = 0, random_state=None, side: str = "right"):
        r"""
        Initialize the ECDFTransformer.

        Args:
            subsample (int): If > 0, use subsampling for fitting. Default is 0 (no subsampling).
            random_state (int or None): Random seed for reproducibility during subsampling.
            side (str): Which side of the ECDF to approximate.

                - ``"right"`` (default): right-continuous ECDF,
                  :math:`F(x) = P(X \leq x)`.
                - ``"left"``: left-continuous ECDF,
                  :math:`F(x^-) = P(X < x)`.

        Raises:
            ValueError: If ``side`` is not ``"right"`` or ``"left"``.
        """
        super().__init__()
        if side not in ("right", "left"):
            raise ValueError(f"side must be 'right' or 'left', got '{side}'.")
        self._subsample = subsample
        self._random_state = random_state
        self._side = side
        self._sorted_values = None
        self._n_samples = None
        self._fitted = False

    def _fit(self, data):
        r"""
        Fit the ECDF transformer to a single column of data.

        Args:
            data: pandas Series or numpy array (1D) containing the column values.

        Raises:
            ValueError: If data is multidimensional.
        """
        # Convert Series to array if needed
        if isinstance(data, pd.Series):
            values = data.values
        else:
            values = np.asarray(data)

        # Validate 1D input
        if values.ndim != 1:
            raise ValueError(
                f"ECDFTransformer expects 1D data (single column). "
                f"Got data with shape {values.shape}. "
                f"Pass individual columns via fit(data, column='col_name')."
            )

        n_samples = len(values)

        # Apply subsampling if specified
        if self._subsample > 0 and self._subsample < n_samples:
            # Subsample values
            indices = np.random.RandomState(self._random_state).choice(
                n_samples, size=self._subsample, replace=False
            )
            sampled_values = values[indices]
            self._n_samples = self._subsample
        else:
            sampled_values = values
            self._n_samples = n_samples

        # Sort and store unique values for ECDF computation
        self._sorted_values = np.sort(sampled_values)
        self._fitted = True

    def _transform(self, data):
        r"""
        Transform data using the fitted ECDF.

        Args:
            data: pandas Series or numpy array (1D) to transform.

        Returns:
            numpy array of shape (n_samples, 1) with ECDF values in [0, 1].

        Raises:
            AssertionError: If transformer has not been fitted.
            ValueError: If data is multidimensional.
        """
        assert self._fitted, "ECDFTransformer must be fitted before calling transform."

        # Convert Series to array if needed
        if isinstance(data, pd.Series):
            values = data.values
        else:
            values = np.asarray(data)

        # Validate 1D input
        if values.ndim != 1:
            raise ValueError(
                f"ECDFTransformer expects 1D data (single column). "
                f"Got data with shape {values.shape}."
            )

        ranks = np.searchsorted(self._sorted_values, values, side=self._side)
        transformed = ranks / self._n_samples

        return transformed.reshape(-1, 1)

    def _reverse_transform(self, data):
        r"""
        Reverse the ECDF transformation to recover approximate original values.

        Args:
            data: pandas Series or numpy array with ECDF values in [0, 1].

        Returns:
            numpy array of shape (n_samples, 1) with approximate original values.

        Raises:
            AssertionError: If transformer has not been fitted.
            ValueError: If data is multidimensional.
        """
        assert self._fitted, "ECDFTransformer must be fitted before calling reverse_transform."

        # Convert Series to array if needed
        if isinstance(data, pd.Series):
            ecdf_values = data.values
        else:
            ecdf_values = np.asarray(data)

        # Validate 1D input
        if ecdf_values.ndim != 1:
            raise ValueError(
                f"ECDFTransformer expects 1D data (single column). "
                f"Got data with shape {ecdf_values.shape}."
            )

        # Convert ECDF values [0, 1] back to sorted-array indices.
        #
        # side="right": F(x) = rank/N, rank in [1, N]  =>  index = rank - 1
        #   index = floor(p * N) - 1
        # side="left":  F(x-) = rank/N, rank in [0, N-1]  =>  index = rank
        #   index = floor(p * N)
        if self._side == "right":
            indices = np.floor(ecdf_values * self._n_samples).astype(int) - 1
        else:  # side == "left"
            indices = np.floor(ecdf_values * self._n_samples).astype(int)
        # Clip to valid range [0, n_samples - 1]
        indices = np.clip(indices, 0, self._n_samples - 1)

        # Retrieve values from sorted array
        recovered_values = self._sorted_values[indices]

        return recovered_values.reshape(-1, 1)
