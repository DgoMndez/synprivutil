"""
Module: src/privacy_utility_framework/dataset/tabletransformer.py
Description: TableTransformer class for dataframe preprocessing consisting of mapping \
    a ColumnTransformer to each feature.
"""

from __future__ import annotations

from copy import deepcopy

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

from privacy_utility_framework.dataset.transformers import (
    create_transformer,
    get_default_transformer,
)


class TableTransformer:
    """
    Transformer for tabular datasets that manages a mapping of :class:`ColumnTransformer`s \
        to each feature.
    """

    def __init__(self):
        self._column_sdtypes: dict[str, str] = {}
        self._column_transformer_templates: dict[str, object] = {}
        self._transformers_by_sdtype: dict[str, object] = {}
        self.column_transformers: dict[str, object] = {}
        self._input_columns: list[str] = []
        self._output_columns: list[str] = []
        self._fitted = False

    @staticmethod
    def _coerce_dataframe(data):
        """Return a defensive dataframe copy and reject unsupported input containers."""
        if isinstance(data, pd.DataFrame):
            return data.copy()

        raise TypeError(
            "TableTransformer expects pandas DataFrame inputs for fit/transform operations."
        )

    SDTYPES = ["numerical", "categorical", "datetime", "other"]
    _SDTYPE_PREDICATES = (
        ("categorical", is_bool_dtype),
        ("numerical", is_numeric_dtype),
        ("datetime", is_datetime64_any_dtype),
        ("categorical", lambda series: isinstance(series.dtype, pd.CategoricalDtype)),
        ("categorical", is_string_dtype),
        ("categorical", is_object_dtype),
    )

    @classmethod
    def get_supported_sdtypes(cls):
        """
        List of supported semantic data types: "numerical", "categorical", "datetime", and "other".
        """
        return cls.SDTYPES

    @classmethod
    def _infer_sdtype(cls, series: pd.Series) -> str:
        for sdtype, predicate in cls._SDTYPE_PREDICATES:
            if predicate(series):
                return sdtype

        return "other"

    def _learn_config(self, data):
        """Infers semantic dtypes from the dataframe and resets any fitted state."""
        self._input_columns = list(data.columns)
        self._column_sdtypes = {
            column: self._infer_sdtype(data[column]) for column in self._input_columns
        }
        self._fitted = False
        self.column_transformers = {}
        self._output_columns = []
        return deepcopy(self._column_sdtypes)

    def update_sdtypes(self, sdtypes):
        """Overrides the inferred semantic dtype for one or more columns."""
        for column, sdtype in sdtypes.items():
            self._column_sdtypes[column] = sdtype
            if column not in self._input_columns:
                self._input_columns.append(column)

        self._fitted = False
        return self

    def update_transformers(self, transformers_dict):
        """
        Registers explicit transformer templates for specific columns.

        Args:
            transformers_dict (dict): Mapping from input column name to transformer instance.

        Returns:
            TableTransformer: The current instance.
        """
        self._column_transformer_templates.update(
            {column: deepcopy(transformer) for column, transformer in transformers_dict.items()}
        )
        self._fitted = False
        return self

    def update_transformers_by_sdtype(
        self,
        sdtype: str,
        transformer=None,
        transformer_name=None,
        transformer_parameters=None,
    ):
        """
        Sets the default transformer used for all columns with a given semantic dtype.

        Args:
            sdtype (str): Semantic dtype to target, such as ``"numerical"``.
            transformer (object, optional): Transformer instance to clone per matching column.
            transformer_name (str, optional): Registered transformer name used to build the
                transformer lazily.
            transformer_parameters (dict, optional): Parameters forwarded when
                ``transformer_name`` is used.

        Raises:
            ValueError: If both ``transformer`` and ``transformer_name`` are provided.

        Returns:
            TableTransformer: The current instance.
        """
        if transformer is not None and transformer_name is not None:
            raise ValueError("Use either 'transformer' or 'transformer_name', not both.")

        if transformer_name is not None:
            transformer = create_transformer(transformer_name, transformer_parameters)
        elif transformer is None:
            transformer = get_default_transformer(sdtype)

        self._transformers_by_sdtype[sdtype] = deepcopy(transformer)
        self._fitted = False
        return self

    def _get_transformer_for_field(self, column):
        """
        Resolves the transformer for a particular column."""
        transformer = self._column_transformer_templates.get(column)
        if transformer is not None:
            return deepcopy(transformer)

        sdtype = self._column_sdtypes.get(column, "other")
        transformer = self._transformers_by_sdtype.get(sdtype)
        if transformer is not None:
            return deepcopy(transformer)

        return get_default_transformer(sdtype)

    def fit(self, data):
        """
        Fits the transformer over the original data.
        """
        data = self._coerce_dataframe(data)
        if not self._input_columns:
            self._learn_config(data)

        missing_columns = [column for column in self._input_columns if column not in data.columns]
        if missing_columns:
            raise ValueError(f"Input data is missing configured columns: {missing_columns}.")

        self.column_transformers = {}
        self._output_columns = []

        for column in self._input_columns:
            transformer = self._get_transformer_for_field(column)
            transformer.fit(data, column=column)
            self.column_transformers[column] = transformer
            # Preserve output column order so reverse_transform can rebuild the original table.
            self._output_columns.extend(transformer.get_output_columns())

        self._fitted = True
        return self

    def transform(self, data):
        """
        Transforms the tabular data using the fitted column transformers.
        """
        assert self._fitted, ":class:`TableTransformer` must be fitted before calling transform."
        data = self._coerce_dataframe(data)
        data = data.loc[:, self._input_columns]
        transformed = [
            self.column_transformers[column].transform(data) for column in self._input_columns
        ]
        if not transformed:
            return pd.DataFrame(index=data.index)

        return pd.concat(transformed, axis=1)

    def fit_transform(self, data):
        """Fits transformer over the data and then returns the transformed data."""
        self.fit(data)
        return self.transform(data)

    def reverse_transform(self, data):
        """Reverts a transformed dataframe back into the original column space."""
        assert self._fitted, (
            ":class:`TableTransformer` must be fitted before calling reverse_transform."
        )
        data = self._coerce_dataframe(data)

        missing_columns = [column for column in self._output_columns if column not in data.columns]
        if missing_columns:
            raise ValueError(
                "Input data is missing transformed columns required for reverse transform: "
                f"{missing_columns}."
            )

        restored_columns = {}
        for column in self._input_columns:
            transformer = self.column_transformers[column]
            restored = transformer.reverse_transform(data)
            restored_columns[column] = restored.iloc[:, 0]

        return pd.DataFrame(restored_columns, columns=self._input_columns, index=data.index)
