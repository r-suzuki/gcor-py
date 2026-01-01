import numpy as np
import pandas as pd
import pytest

from gcor import gcor


def _example_data():
    x = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y = pd.Series([1, 2, 3, 4, 5, 3, 4, 5, 6, 7])
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'z': ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd', 'e', 'e'],
    })
    return x, y, df


def test_gcor_example_scalar_regression():
    x, y, _ = _example_data()

    got = gcor(x, y)
    expected = 0.5345224838248488

    # Regression test for the default behavior (k=None).
    assert isinstance(got, (float, np.floating))
    assert float(got) == pytest.approx(expected, rel=0.0, abs=1e-12)


def test_gcor_example_matrix_regression():
    _, _, df = _example_data()

    got = gcor(df)

    # Basic shape/labels (regression for API behavior)
    assert isinstance(got, pd.DataFrame)
    assert list(got.columns) == ['x', 'y', 'z']
    assert list(got.index) == ['x', 'y', 'z']

    expected = np.array([
        [1.0,      0.534522, 0.806219],
        [0.534522, 1.0,      0.734035],
        [0.806219, 0.734035, 1.0],
    ], dtype=float)

    # Match the Examples section (printed to 6 decimals).
    np.testing.assert_allclose(got.to_numpy(), expected, rtol=0.0, atol=5e-5)


def test_gcor_example_scalar_matches_matrix_entry():
    x, y, df = _example_data()

    scalar = float(gcor(x, y))
    mat = gcor(df)

    # Consistency check: scalar result matches the corresponding matrix entry.
    assert scalar == pytest.approx(float(mat.loc['x', 'y']), rel=0.0, abs=1e-12)
