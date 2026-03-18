"""
Module: src/privacy_utility_framework/dataset/hypertransformer.py
Description: TableTransformer class for dataframe preprocessing consisting of mapping \
    a ColumnTransformer to each feature.
"""

from __future__ import annotations

from copy import deepcopy

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from privacy_utility_framework.dataset.transformers import (
    create_transformer,
    get_default_transformer,
)


class TableTransformer:
    """Small subset of the RDT HyperTransformer API used by this project."""

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
        if isinstance(data, pd.DataFrame):
            return data.copy()

        raise TypeError(
            "HyperTransformer expects pandas DataFrame inputs for fit/transform operations."
        )

    @classmethod
    def get_supported_sdtypes(cls):
        return ("numerical", "categorical", "datetime", "other")

    @staticmethod
    def _infer_sdtype(series: pd.Series) -> str:
        if is_bool_dtype(series):
            return "categorical"
        if is_numeric_dtype(series):
            return "numerical"
        if is_datetime64_any_dtype(series):
            return "datetime"
        if str(series.dtype) in {"object", "category", "string"}:
            return "categorical"
        return "other"

    def _learn_config(self, data):
        self._input_columns = list(data.columns)
        self._column_sdtypes = {
            column: self._infer_sdtype(data[column]) for column in self._input_columns
        }
        self._fitted = False
        self.column_transformers = {}
        self._output_columns = []
        return deepcopy(self._column_sdtypes)

    def update_sdtypes(self, sdtypes):
        for column, sdtype in sdtypes.items():
            self._column_sdtypes[column] = sdtype
            if column not in self._input_columns:
                self._input_columns.append(column)

        self._fitted = False
        return self

    def update_transformers(self, transformers):
        self._column_transformer_templates.update(
            {column: deepcopy(transformer) for column, transformer in transformers.items()}
        )
        self._fitted = False
        return self

    def update_transformers_by_sdtype(
        self,
        sdtype,
        transformer=None,
        transformer_name=None,
        transformer_parameters=None,
    ):
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
        transformer = self._column_transformer_templates.get(column)
        if transformer is not None:
            return deepcopy(transformer)

        sdtype = self._column_sdtypes.get(column, "other")
        transformer = self._transformers_by_sdtype.get(sdtype)
        if transformer is not None:
            return deepcopy(transformer)

        return get_default_transformer(sdtype)

    def fit(self, data):
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
            self._output_columns.extend(transformer.get_output_columns())

        self._fitted = True
        return self

    def transform(self, data):
        assert self._fitted, "HyperTransformer must be fitted before calling transform."
        data = self._coerce_dataframe(data)
        data = data.loc[:, self._input_columns]
        transformed = [
            self.column_transformers[column].transform(data) for column in self._input_columns
        ]
        if not transformed:
            return pd.DataFrame(index=data.index)

        return pd.concat(transformed, axis=1)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def reverse_transform(self, data):
        assert self._fitted, "HyperTransformer must be fitted before calling reverse_transform."
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
