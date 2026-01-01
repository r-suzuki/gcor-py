# ============================================================
# Tests for discretize()
# ============================================================

import numpy as np
import pandas as pd
from gcor._core import discretize

# 1) If already categorical, return as-is.
def test_discretize_returns_as_is_when_already_categorical():
    x = pd.Series(["a", "b", "a"], dtype="category")
    y = discretize(x, k=3, max_levels=10)

    assert y is x
    assert isinstance(y.dtype, pd.CategoricalDtype)


# 2) If nunique() fails due to unhashable elements (e.g., dict/list), return as-is.
def test_discretize_returns_as_is_when_elements_are_unhashable():
    x = pd.Series([{"a": 1}, {"a": 1}, {"a": 2}])
    y = discretize(x, k=2, max_levels=10)

    assert y is x
    assert not isinstance(y.dtype, pd.CategoricalDtype)


# 3) If the number of distinct values (excluding NA) is <= k, cast to category (even if numeric).
def test_discretize_casts_to_category_when_n_unique_leq_k_even_if_numeric():
    x = pd.Series([1.0, 2.0, 1.0, 2.0])  # n_unique(dropna=True) = 2
    y = discretize(x, k=2, max_levels=100)

    assert isinstance(y.dtype, pd.CategoricalDtype)
    assert set(y.cat.categories) == {1.0, 2.0}


# 4) qcut path: if qcut is applicable and n_unique > k, return qcut result (interval categories).
def test_discretize_uses_qcut_when_possible():
    x = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # n_unique=10
    y = discretize(x, k=4, max_levels=100)

    assert isinstance(y.dtype, pd.CategoricalDtype)
    assert 1 <= y.cat.categories.size <= 4


# 5) If qcut fails and n_unique <= max_levels, cast to category.
def test_discretize_falls_back_to_category_when_qcut_fails_and_n_unique_leq_max_levels():
    x = pd.Series(list("abcdef"))  # n_unique=6, qcut should fail (non-numeric)
    y = discretize(x, k=3, max_levels=10)

    assert isinstance(y.dtype, pd.CategoricalDtype)
    assert set(y.cat.categories) == set("abcdef")


# 6) If qcut fails and n_unique > max_levels, return as-is.
def test_discretize_returns_as_is_when_qcut_fails_and_too_many_levels():
    x = pd.Series([f"id{i}" for i in range(20)])  # n_unique=20
    y = discretize(x, k=3, max_levels=10)

    assert y is x
    assert not isinstance(y.dtype, pd.CategoricalDtype)


# 7) dropna=True behavior: NA should not increase the distinct count used for branching.
def test_discretize_unique_count_excludes_na_for_branching():
    x = pd.Series([1.0, 2.0, 3.0, np.nan])  # n_unique(dropna=True)=3
    y = discretize(x, k=3, max_levels=100)

    assert isinstance(y.dtype, pd.CategoricalDtype)
    assert y.isna().sum() == 1

# 8) Automatic k detection when k=None
def test_discretize_k_none_produces_two_bins_for_small_numeric_sample():
    x = pd.Series([1, 2, 3, 4])
    y = discretize(x, k=None, max_levels=100)

    assert isinstance(y.dtype, pd.CategoricalDtype)
    assert y.cat.categories.size == 2
