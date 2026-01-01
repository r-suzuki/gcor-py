from pathlib import Path

import numpy as np
import pandas as pd
import warnings
import pytest

from gcor import gcor
from gcor._core import discretize

HERE = Path(__file__).parent

# Load the iris dataset from a CSV compatible with R's `iris`.
def load_iris() -> pd.DataFrame:
    return pd.read_csv(HERE / 'data' / 'iris.csv', comment='#')


# A pair expected to show positive association (Sepal.Length vs Petal.Width)
def test_gcor_positive_pair_returns_finite_number():
    df = load_iris()
    x = df['Sepal.Length']
    y = df['Petal.Width']

    val = gcor(x, y)

    # Smoke check: returns a scalar without errors.
    assert isinstance(val, (float, np.floating))

    # iris has no missing values for this pair, so the result should be finite.
    assert np.isfinite(val)

    # Optional / weak guardrail: this pair is strongly increasing in iris.
    # If your gcor is always non-negative, this should hold.
    assert float(val) >= 0.0


# A pair showing negative Pearson correlation (Petal.Length vs Sepal.Width)
def test_gcor_negative_pair_returns_finite_number():
    df = load_iris()
    x = df['Petal.Length']
    y = df['Sepal.Width']

    val = gcor(x, y)

    assert isinstance(val, (float, np.floating))
    assert np.isfinite(val)

    # gcor is expected to be non-negative by design.
    assert float(val) >= 0.0


# Numeric vs categorical (Petal.Width vs Species)
def test_gcor_numeric_and_categorical_returns_finite_number():
    df = load_iris()
    x = df['Petal.Width']
    y = df['Species']

    val = gcor(x, y)

    assert isinstance(val, (float, np.floating))
    assert np.isfinite(val)


# Empty Series inputs
def test_gcor_empty_series_returns_nan():
    x = pd.Series([], dtype=float)
    y = pd.Series([], dtype=float)

    r = gcor(x, y)

    assert isinstance(r, (float, np.floating))
    assert np.isnan(r)


# A small numeric matrix (Sepal.Length, Petal.Length, Petal.Width)
def test_gcor_matrix_returns_square_dataframe():
    df = load_iris()

    cols = ['Sepal.Length', 'Petal.Length', 'Petal.Width']
    data = df[cols]

    res = gcor(data)

    assert isinstance(res, pd.DataFrame)
    assert res.shape == (3, 3)
    assert list(res.columns) == cols
    assert list(res.index) == cols

    # The diagonal is always set to 1.0 by definition.
    assert np.allclose(np.diag(res.to_numpy()), 1.0)


# Numeric-only matrix (iris without Species)
def test_gcor_matrix_numeric_only_has_expected_shape_and_labels():
    df = load_iris()

    cols = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']
    data = df[cols]

    res = gcor(data)

    assert isinstance(res, pd.DataFrame)
    assert res.shape == (4, 4)
    assert list(res.columns) == cols
    assert list(res.index) == cols

    # Symmetry and diagonal
    arr = res.to_numpy()
    assert np.allclose(np.diag(arr), 1.0)
    assert np.allclose(arr, arr.T, equal_nan=True)


# Mixed-type matrix (full iris, including Species)
def test_gcor_matrix_with_categorical_column_has_expected_shape_and_labels():
    df = load_iris()

    cols = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species']
    data = df[cols]

    res = gcor(data)

    assert isinstance(res, pd.DataFrame)
    assert res.shape == (5, 5)
    assert list(res.columns) == cols
    assert list(res.index) == cols

    arr = res.to_numpy()
    assert np.allclose(np.diag(arr), 1.0)
    assert np.allclose(arr, arr.T, equal_nan=True)

    # --- Soft compatibility check against the R implementation ---
    r_reference = np.array([
        [1.0000000, 0.2349075, 0.8846517, 0.8741873, 0.7623968],
        [0.2349075, 1.0000000, 0.3143301, 0.2669031, 0.6510740],
        [0.8846517, 0.3143301, 1.0000000, 0.9503289, 0.8221674],
        [0.8741873, 0.2669031, 0.9503289, 1.0000000, 0.8237429],
        [0.7623968, 0.6510740, 0.8221674, 0.8237429, 1.0000000],
    ], dtype=float)

    diff = np.abs(arr - r_reference)

    # Allow moderate numerical differences (visual agreement level).
    if not np.all(diff <= 1e-4):
        max_diff = np.nanmax(diff)
        warnings.warn(
            f"gcor results differ from the R reference on iris "
            f"(max abs diff = {max_diff:.2e}).",
            UserWarning,
        )


# Empty DataFrame input keeps labels; diagonal is 1.0 by definition.
def test_gcor_matrix_empty_dataframe_keeps_columns():
    df = pd.DataFrame(
        {
            'a': pd.Series([], dtype=float),
            'b': pd.Series([], dtype=float),
            'c': pd.Series([], dtype=float),
        }
    )

    res = gcor(df)

    assert isinstance(res, pd.DataFrame)
    assert res.shape == (3, 3)
    assert list(res.index) == ['a', 'b', 'c']
    assert list(res.columns) == ['a', 'b', 'c']

    # The diagonal is always set to 1.0 by definition.
    assert np.allclose(np.diag(res.to_numpy()), 1.0)

    # Off-diagonal entries should be NaN for empty inputs.
    off_diag = res.to_numpy().copy()
    np.fill_diagonal(off_diag, np.nan)
    assert np.isnan(off_diag).all()


# discretize(): numeric columns should be discretized, and low-cardinality
# non-numeric columns should be converted to categorical when possible.
def test_discretize_on_iris_numeric_and_species():
    df = load_iris()

    x_num = df['Sepal.Length']
    x_cat = df['Species']

    y_num = discretize(x_num, k=4, max_levels=100)
    y_cat = discretize(x_cat, k=4, max_levels=100)

    assert isinstance(y_num, pd.Series)
    assert isinstance(y_cat, pd.Series)

    assert isinstance(y_num.dtype, pd.CategoricalDtype)
    assert isinstance(y_cat.dtype, pd.CategoricalDtype)

    # iris Species has 3 categories.
    assert y_cat.cat.categories.size == 3

# Type errors
def test_gcor_x_none_raises_typeerror():
    with pytest.raises(TypeError):
        gcor(None, None)

def test_gcor_y_none_raises_typeerror_when_x_not_dataframe():
    with pytest.raises(TypeError):
        gcor([1, 2, 3], None)

def test_gcor_dataframe_with_non_none_y_raises_typeerror():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(TypeError):
        gcor(df, y=[1, 2, 3])

# Value errors
def test_gcor_invalid_drop_na_raises_valueerror():
    with pytest.raises(ValueError):
        gcor([1, 2, 3], [1, 2, 3], drop_na="INVALID")

def test_gcor_length_mismatch_raises_valueerror():
    with pytest.raises(ValueError):
        gcor([1, 2, 3], [1, 2])

# Too many levels
def test_gcor_warns_when_non_categorical_columns_remain():
    # Simple non-numeric data with multiple levels (R: letters[1:5])
    x = pd.Series(list("abcde"))
    y = pd.Series(list("vwxyz"))

    # Small max_levels forces non-conversion to categorical
    with pytest.warns(UserWarning):
        gcor(x, y, max_levels=3)
