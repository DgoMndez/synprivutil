"""
Module: synprivutil/tests/test_unit_ecdf_transformer.py
Description: Comprehensive unit tests for ECDFTransformer.

Author: Domingo Méndez García
Email: domingo.mendezg@um.es
Date of creation: 12/03/2026
"""

import numpy as np
import pandas as pd
import pytest

from privacy_utility_framework.dataset.transformers import ECDFTransformer

seed = 7428


def _get_seed():
    global seed
    res = seed
    seed += 1
    return res


def _make_dataframe(values, column="values"):
    return pd.DataFrame({column: np.asarray(values)})


def _expected_ecdf(values, side="right"):
    values = np.asarray(values)
    sorted_values = np.sort(values)
    ranks = np.searchsorted(sorted_values, values, side=side)
    res = ranks / len(values)
    assert np.all((res >= 0.0) & (res <= 1.0)), "Expected ECDF values should be in [0, 1]"
    return res


class TestECDFTransformerBasicFunctionality:
    """Test basic ECDF transformation functionality."""

    def test_ecdf_trivial_unique(self):
        """Original [0..9] should map query [-1..10] to [0, 0.1, ..., 1, 1]."""
        original = _make_dataframe(np.arange(10, dtype=float))
        query = _make_dataframe(np.arange(-1, 11, dtype=float))
        transformer = ECDFTransformer()

        transformer.fit(original, column="values")
        transformed = transformer.transform(query)

        expected = np.concatenate(([0.0], np.arange(1, 11) / 10.0, [1.0]))
        transformed_values = transformed["values"].to_numpy()
        assert np.allclose(transformed_values, expected)

    def test_ecdf_trivial_repeated(self):
        """Same boundary idea as [0..9], but with repeated values in original data."""
        original = _make_dataframe(np.array([0, 1, 1, 2, 3, 5, 5, 7, 8, 9], dtype=float))
        query = _make_dataframe(np.arange(-1, 11, dtype=float))
        transformer = ECDFTransformer()

        transformer.fit(original, column="values")
        transformed = transformer.transform(query)

        expected = np.array([0, 0.1, 0.3, 0.4, 0.5, 0.5, 0.7, 0.7, 0.8, 0.9, 1, 1])
        transformed_values = transformed["values"].to_numpy()
        assert np.allclose(transformed_values, expected)

    def test_ecdf_matches_expected_ranks_for_random_unique_samples(self):
        """ECDF should match searchsorted-based expected ranks on random unique data."""
        rng = np.random.default_rng(_get_seed())

        for size in [8, 17, 64, 257]:
            values = rng.choice(np.arange(size * 20, dtype=float), size=size, replace=False)
            data = _make_dataframe(values)
            transformer = ECDFTransformer()

            transformer.fit(data, column="values")
            transformed = transformer.transform(data)

            expected = _expected_ecdf(values)
            transformed_values = transformed["values"].to_numpy()
            assert np.allclose(transformed_values, expected)

    def test_ecdf_matches_expected_ranks_for_random_integer_samples(self):
        """ECDF should handle random integer-valued samples consistently."""
        rng = np.random.default_rng(_get_seed())

        for size in [10, 25, 80]:
            values = rng.integers(-50, 50, size=size)
            data = _make_dataframe(values, column="col1")
            transformer = ECDFTransformer()

            transformer.fit(data, column="col1")
            transformed = transformer.transform(data)

            expected = _expected_ecdf(values)
            transformed_values = transformed["col1"].to_numpy()
            assert np.allclose(transformed_values, expected)

    def test_ecdf_rdt_pattern_dataframe_column(self):
        """Test ECDF with RDT pattern across several random floating-point datasets."""
        transformer = ECDFTransformer()
        rng = np.random.default_rng(_get_seed())

        for size in [7, 31, 103]:
            df = _make_dataframe(rng.normal(loc=0.0, scale=5.0, size=size))
            transformer.fit(df, column="values")
            transformed = transformer.transform(df)

            assert transformed.columns.tolist() == ["values"]
            assert len(transformed) == len(df)
            assert np.all((transformed["values"] >= 0.0) & (transformed["values"] <= 1.0))

    def test_ecdf_preserves_order(self):
        """ECDF should preserve ordering across multiple random datasets."""
        rng = np.random.default_rng(_get_seed())

        for size in [20, 50, 100]:
            values = rng.normal(size=size)
            data = _make_dataframe(values)
            transformer = ECDFTransformer()

            transformer.fit(data, column="values")
            transformed = transformer.transform(data)

            original = data["values"].to_numpy()
            transformed_flat = transformed["values"].to_numpy()
            order = np.argsort(original)
            assert np.all(transformed_flat[order][:-1] <= transformed_flat[order][1:])

    def test_ecdf_output_in_valid_range(self):
        """Test that ECDF values are in [0, 1]."""
        data = pd.DataFrame({"values": np.random.uniform(-100, 100, 1000)})
        transformer = ECDFTransformer()

        transformer.fit(data, column="values")
        transformed = transformer.transform(data)
        transformed_values = transformed["values"].to_numpy()

        assert np.all(transformed_values >= 0.0), "ECDF values should be >= 0"
        assert np.all(transformed_values <= 1.0), "ECDF values should be <= 1"


class TestECDFTransformerReverseTransform:
    """Test reverse transformation (inverse ECDF)."""

    def test_reverse_transform_recovers_random_unique_samples(self):
        """Reverse transform should recover the original values on random unique samples."""
        rng = np.random.default_rng(_get_seed())

        for size in [12, 40, 128]:
            values = rng.choice(np.arange(size * 50, dtype=float), size=size, replace=False)
            data = _make_dataframe(values)
            transformer = ECDFTransformer()

            transformer.fit(data, column="values")
            transformed = transformer.transform(data)
            reversed_data = transformer.reverse_transform(transformed)

            assert np.allclose(
                reversed_data["values"].to_numpy(),
                data["values"].to_numpy(),
                atol=1e-8,
            )

    def test_reverse_transform_recovers_sortorder(self):
        """Test that reverse transform preserves sorted order."""
        data = pd.DataFrame({"values": [5.0, 1.0, 3.0, 2.0, 4.0]})
        transformer = ECDFTransformer()

        transformer.fit(data, column="values")
        transformed = transformer.transform(data)
        reversed_data = transformer.reverse_transform(transformed)

        # The recovered data should maintain the sorted values
        sorted_original = np.sort(data["values"].to_numpy())
        sorted_recovered = np.sort(reversed_data["values"].to_numpy())
        assert np.allclose(sorted_original, sorted_recovered, atol=1.0)


class TestECDFTransformerSubsampling:
    """Test subsampling functionality."""

    def test_subsampling_smaller_sample(self):
        """Test that subsampling works with smaller dataset."""
        data = pd.DataFrame({"values": np.linspace(0, 100, 1000)})
        transformer = ECDFTransformer(subsample=100, random_state=_get_seed())

        transformer.fit(data, column="values")
        transformed = transformer.transform(data)
        transformed_values = transformed["values"].to_numpy()

        # Should still produce valid ECDF values
        assert np.all(transformed_values >= 0.0)
        assert np.all(transformed_values <= 1.0)
        assert transformed_values.shape == (1000,)

    def test_subsampling_no_effect_when_larger_than_data(self):
        """Test that subsampling has no effect when subsample > data size."""
        data = pd.DataFrame({"values": [1.0, 2.0, 3.0]})
        transformer = ECDFTransformer(subsample=1000, random_state=_get_seed())

        transformer.fit(data, column="values")
        assert transformer._n_samples == 3, "Should use all 3 samples"

    def test_subsampling_zero_uses_all_data(self):
        """Test that subsample=0 uses all data."""
        data = pd.DataFrame({"values": np.linspace(0, 100, 500)})
        transformer = ECDFTransformer(subsample=0, random_state=_get_seed())

        transformer.fit(data, column="values")
        assert transformer._n_samples == 500


class TestECDFTransformerErrorHandling:
    """Test error handling and edge cases."""

    def test_fit_without_column_raises_typeerror(self):
        """BaseTransformer contract: fit requires a column argument."""
        data = pd.DataFrame({"values": [1, 2, 3]})
        transformer = ECDFTransformer()

        with pytest.raises(TypeError):
            transformer.fit(data)

    def test_transform_before_fit_raises_error(self):
        """Test that transform before fit raises AssertionError."""
        data = pd.DataFrame({"values": [1.0, 2.0, 3.0]})
        transformer = ECDFTransformer()

        with pytest.raises((AssertionError, AttributeError)):
            transformer.transform(data)

    def test_reverse_transform_before_fit_raises_error(self):
        """Test that reverse_transform before fit raises AssertionError."""
        data = pd.DataFrame({"values": [0.2, 0.4, 0.6]})
        transformer = ECDFTransformer(random_state=_get_seed())

        with pytest.raises((AssertionError, AttributeError)):
            transformer.reverse_transform(data)

    def test_non_existing_column_raises_error(self):
        """Using an unknown column should fail through ColumnTransformer path."""
        data = pd.DataFrame({"values": [1.0, 2.0, 3.0]})
        transformer = ECDFTransformer(random_state=_get_seed())

        with pytest.raises((KeyError, ValueError)):
            transformer.fit(data, column="missing")

    def test_multidimensional_column_rejected_by_internal_fit(self):
        """Direct internal call still rejects multidimensional arrays."""
        transformer = ECDFTransformer(random_state=_get_seed())
        multidim_data = np.array([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(ValueError, match="expects 1D data"):
            transformer._fit(multidim_data)


class TestECDFTransformerDuplicateValues:
    """Test handling of duplicate values."""

    def test_duplicate_values_random_samples(self):
        """Random duplicated values should still map to valid ECDF values."""
        rng = np.random.default_rng(_get_seed())

        for size in [30, 75, 150]:
            values = rng.integers(-10, 10, size=size)
            data = _make_dataframe(values)
            transformer = ECDFTransformer(random_state=_get_seed())

            transformer.fit(data, column="values")
            transformed = transformer.transform(data)
            transformed_values = transformed["values"].to_numpy()

            expected = _expected_ecdf(values)
            assert np.all(transformed_values >= 0.0)
            assert np.all(transformed_values <= 1.0)
            assert np.allclose(transformed_values, expected)

    def test_all_same_values(self):
        """All-identical random-sized samples should map to ECDF value 1."""
        for size in [4, 19, 63]:
            data = _make_dataframe(np.full(size, 5.0))
            transformer = ECDFTransformer(random_state=_get_seed())

            transformer.fit(data, column="values")
            transformed = transformer.transform(data)
            transformed_values = transformed["values"].to_numpy()

            assert np.allclose(transformed_values, np.ones(size))


class TestECDFTransformerRandomData:
    """Test with random and edge case data."""

    def test_random_distributions_and_scales(self):
        """ECDF should behave consistently across random distributions and scales."""
        rng = np.random.default_rng(42)
        datasets = [
            rng.normal(loc=0, scale=1, size=500),
            rng.uniform(-1000, 1000, size=300),
            rng.lognormal(mean=1.0, sigma=0.8, size=200),
            rng.normal(loc=1e6, scale=1e3, size=150),
        ]

        for values in datasets:
            data = _make_dataframe(values)
            transformer = ECDFTransformer(random_state=_get_seed())

            transformer.fit(data, column="values")
            transformed = transformer.transform(data)
            transformed_values = transformed["values"].to_numpy()

            assert np.all(transformed_values >= 0.0)
            assert np.all(transformed_values <= 1.0)
            assert transformed_values.shape == (len(values),)
            assert np.allclose(transformed_values, _expected_ecdf(values))


class TestECDFTransformerMonotonicity:
    """Test monotonicity properties of ECDF."""

    def test_ecdf_is_monotonic(self):
        """Test that ECDF is a monotonically increasing function."""
        np.random.seed(42)
        fit_data = pd.DataFrame({"values": np.random.uniform(0, 100, 100)})
        test_data = pd.DataFrame({"values": np.sort(np.random.uniform(0, 100, 50))})

        transformer = ECDFTransformer()
        transformer.fit(fit_data, column="values")
        transformed = transformer.transform(test_data)

        # Check monotonicity
        transformed_flat = transformed["values"].to_numpy()
        assert np.all(transformed_flat[:-1] <= transformed_flat[1:]), (
            "ECDF should be monotonically increasing"
        )

    def test_ecdf_left_continuity(self):
        """ECDF should be monotone on random query points inside the fitted range."""
        rng = np.random.default_rng(808)

        for _ in range(5):
            fit_values = rng.normal(loc=10.0, scale=3.0, size=120)
            low, high = np.min(fit_values), np.max(fit_values)
            query_values = np.sort(rng.uniform(low, high, size=40))

            data = _make_dataframe(fit_values)
            test_points = _make_dataframe(query_values)
            transformer = ECDFTransformer()

            transformer.fit(data, column="values")
            transformed = transformer.transform(test_points)

            transformed_flat = transformed["values"].to_numpy()
            assert np.all(transformed_flat[:-1] <= transformed_flat[1:])


class TestECDFTransformerSide:
    """Test side='right' and side='left' behaviour."""

    def test_invalid_side_raises_valueerror(self):
        """Passing an unknown side should raise ValueError immediately."""

        x = ECDFTransformer(side="right")
        assert x.side == "right"
        x = ECDFTransformer(side="left")
        assert x.side == "left"

        with pytest.raises(ValueError):
            ECDFTransformer(side="center")

    def test_right_side_default_matches_explicit(self):
        """Default constructor and side='right' should produce identical results."""
        rng = np.random.default_rng(_get_seed())
        values = rng.normal(size=80)
        data = _make_dataframe(values)

        t_default = ECDFTransformer()
        t_right = ECDFTransformer(side="right")

        t_default.fit(data, column="values")
        t_right.fit(data, column="values")

        assert np.allclose(
            t_default.transform(data)["values"].to_numpy(),
            t_right.transform(data)["values"].to_numpy(),
        )

    def test_right_ecdf_matches_expected_unique(self):
        """Right ECDF should equal P(X<=x) on unique data."""
        original = _make_dataframe(np.arange(10, dtype=float))
        query = _make_dataframe(np.arange(-1, 11, dtype=float))
        transformer = ECDFTransformer(side="right")

        transformer.fit(original, column="values")
        transformed = transformer.transform(query)["values"].to_numpy()

        # Query values outside the support first and last: 0.0, then 0.1..1.0, then 1.0
        expected = np.concatenate(([0.0], np.arange(1, 11) / 10.0, [1.0]))
        assert np.allclose(transformed, expected)

    def test_left_ecdf_matches_expected_unique(self):
        """Left ECDF should equal P(X<x) on unique data."""
        original = _make_dataframe(np.arange(10, dtype=float))
        query = _make_dataframe(np.arange(-1, 11, dtype=float))
        transformer = ECDFTransformer(side="left")

        transformer.fit(original, column="values")
        transformed = transformer.transform(query)["values"].to_numpy()

        # F(-1-)=0, F(0-)=0, F(1-)=0.1, ..., F(9-)=0.9, F(10-)=1.0
        expected = np.concatenate(([0.0, 0.0], np.arange(1, 10) / 10.0, [1.0]))
        assert np.allclose(transformed, expected)

    def test_right_vs_left_differ_at_duplicates(self):
        """For data with duplicates, right and left ECDFs should differ at jump points."""
        # [0, 1, 1, 2]: duplicated 1 → P(X<=1)=0.75, P(X<1)=0.25
        original = _make_dataframe(np.array([0.0, 1.0, 1.0, 2.0]))
        query = _make_dataframe(np.array([1.0]))

        t_right = ECDFTransformer(side="right")
        t_left = ECDFTransformer(side="left")
        t_right.fit(original, column="values")
        t_left.fit(original, column="values")

        assert np.isclose(t_right.transform(query)["values"].to_numpy()[0], 0.75)
        assert np.isclose(t_left.transform(query)["values"].to_numpy()[0], 0.25)

    def test_right_vs_left_agree_between_jump_points(self):
        """Between jump points, right and left ECDFs must coincide."""
        # Unique even grid: 0, 2, 4, ..., 18 (N=10); query at odd midpoints
        original = _make_dataframe(np.arange(0, 20, 2, dtype=float))
        query = _make_dataframe(np.arange(1, 18, 2, dtype=float))

        t_right = ECDFTransformer(side="right")
        t_left = ECDFTransformer(side="left")
        t_right.fit(original, column="values")
        t_left.fit(original, column="values")

        assert np.allclose(
            t_right.transform(query)["values"].to_numpy(),
            t_left.transform(query)["values"].to_numpy(),
        )

    def test_right_ecdf_matches_expected_random_unique(self):
        """Right ECDF on random unique data should match _expected_ecdf(side='right')."""
        rng = np.random.default_rng(_get_seed())
        for size in [15, 60, 200]:
            values = rng.choice(np.arange(size * 10, dtype=float), size=size, replace=False)
            data = _make_dataframe(values)
            transformer = ECDFTransformer(side="right")

            transformer.fit(data, column="values")
            transformed = transformer.transform(data)["values"].to_numpy()

            assert np.allclose(transformed, _expected_ecdf(values, side="right"))

    def test_left_ecdf_matches_expected_random_unique(self):
        """Left ECDF on random unique data should match _expected_ecdf(side='left')."""
        rng = np.random.default_rng(_get_seed())
        for size in [15, 60, 200]:
            values = rng.choice(np.arange(size * 10, dtype=float), size=size, replace=False)
            data = _make_dataframe(values)
            transformer = ECDFTransformer(side="left")

            transformer.fit(data, column="values")
            transformed = transformer.transform(data)["values"].to_numpy()

            assert np.allclose(transformed, _expected_ecdf(values, side="left"))

    def test_right_ecdf_matches_expected_random_integers(self):
        """Right ECDF on random integers (with duplicates) matches _expected_ecdf(side='right')."""
        rng = np.random.default_rng(_get_seed())
        for size in [20, 50, 100]:
            values = rng.integers(-20, 20, size=size)
            data = _make_dataframe(values.astype(float))
            transformer = ECDFTransformer(side="right")

            transformer.fit(data, column="values")
            transformed = transformer.transform(data)["values"].to_numpy()

            assert np.allclose(transformed, _expected_ecdf(values, side="right"))

    def test_left_ecdf_matches_expected_random_integers(self):
        """Left ECDF on random integers (with duplicates) matches _expected_ecdf(side='left')."""
        rng = np.random.default_rng(_get_seed())
        for size in [20, 50, 100]:
            values = rng.integers(-20, 20, size=size)
            data = _make_dataframe(values.astype(float))
            transformer = ECDFTransformer(side="left")

            transformer.fit(data, column="values")
            transformed = transformer.transform(data)["values"].to_numpy()

            assert np.allclose(transformed, _expected_ecdf(values, side="left"))

    def test_left_ecdf_minimum_value_maps_to_zero(self):
        """The minimum value must have left-ECDF = 0 (no element is strictly less)."""
        values = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
        data = _make_dataframe(values)
        transformer = ECDFTransformer(side="left")

        transformer.fit(data, column="values")
        transformed = transformer.transform(data)["values"].to_numpy()

        assert np.all(transformed[values == values.min()] == 0.0)

    def test_right_ecdf_maximum_value_maps_to_one(self):
        """The maximum value must have right-ECDF = 1 (all elements are <= max)."""
        values = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
        data = _make_dataframe(values)
        transformer = ECDFTransformer(side="right")

        transformer.fit(data, column="values")
        transformed = transformer.transform(data)["values"].to_numpy()

        assert np.all(transformed[values == values.max()] == 1.0)

    def test_reverse_transform_right_recovers_unique_values(self):
        """Reverse transform with side='right' should recover original unique values."""
        rng = np.random.default_rng(_get_seed())
        for size in [10, 40, 120]:
            values = rng.choice(np.arange(size * 50, dtype=float), size=size, replace=False)
            data = _make_dataframe(values)
            transformer = ECDFTransformer(side="right")

            transformer.fit(data, column="values")
            recovered = transformer.reverse_transform(transformer.transform(data))

            assert np.allclose(recovered["values"].to_numpy(), data["values"].to_numpy(), atol=1e-8)

    def test_reverse_transform_left_recovers_unique_values(self):
        """Reverse transform with side='left' should recover original unique values."""
        rng = np.random.default_rng(_get_seed())
        for size in [10, 40, 120]:
            values = rng.choice(np.arange(size * 50, dtype=float), size=size, replace=False)
            data = _make_dataframe(values)
            transformer = ECDFTransformer(side="left")

            transformer.fit(data, column="values")
            recovered = transformer.reverse_transform(transformer.transform(data))

            assert np.allclose(recovered["values"].to_numpy(), data["values"].to_numpy(), atol=1e-8)


# TODO: Simplify tests by unifying left and right ECDF

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
