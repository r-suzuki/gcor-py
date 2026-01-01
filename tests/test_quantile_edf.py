import numpy as np
import pandas as pd
import pytest

from gcor._core import quantile_edf

# quantile_edf is basically compatible with R's quantile(..., type = 1).

def test_quantile_edf_basic_example_matches_r_type1():
    # > quantile(c(1, 1, 2, 3), probs = seq(0, 1, length.out = 5), na.rm = TRUE, type = 1)
    #   0%  25%  50%  75% 100% 
    #    1    1    1    2    3 
    
    x = pd.Series([1, 1, 2, 3])
    probs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    got = quantile_edf(x, probs)
    exp = pd.Series([1, 1, 1, 2, 3], index=probs)

    pd.testing.assert_series_equal(got, exp)


def test_quantile_edf_dropna_behavior():
    # > quantile(c(NA, 2, 1, NA, 3), probs = c(0, .5, 1), na.rm = TRUE, type = 1)
    #   0%  50% 100% 
    #    1    2    3 
    
    x = pd.Series([np.nan, 2.0, 1.0, np.nan, 3.0])
    probs = np.array([0.0, 0.5, 1.0])

    got = quantile_edf(x, probs)
    # non-NA sorted => [1,2,3], n=3 => p=0.5 -> ceil(1.5)=2 -> x_(2)=2
    exp = pd.Series([1.0, 2.0, 3.0], index=probs)

    pd.testing.assert_series_equal(got, exp)


def test_quantile_edf_all_na_returns_all_na_with_same_length_and_dtype():
    # > quantile(c(NA, NA), probs = c(0, .25, .5, 1), na.rm = TRUE, type = 1)
    #   0%  25%  50% 100% 
    #   NA   NA   NA   NA 

    x = pd.Series([np.nan, np.nan], dtype="float64")
    probs = np.array([0.0, 0.25, 0.5, 1.0])

    got = quantile_edf(x, probs)

    assert len(got) == len(probs)
    assert got.index.to_numpy().tolist() == probs.tolist()
    assert got.isna().all()


def test_quantile_edf_preserves_datetime_dtype():
    # > as.Date(c("2020-01-01", "2020-01-01", "2020-01-03", "2020-01-02")) |>
    # as.integer() |> quantile(probs = c(0, .5, 1), type = 1) |> as.Date()
    #           0%          50%         100% 
    # "2020-01-01" "2020-01-01" "2020-01-03" 

    x = pd.Series(pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-03", "2020-01-02"]))
    probs = np.array([0.0, 0.5, 1.0])

    got = quantile_edf(x, probs)

    # sorted => 2020-01-01, 2020-01-01, 2020-01-02, 2020-01-03
    exp = pd.Series(
        pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-03"]),
        index=probs,
    )

    pd.testing.assert_series_equal(got, exp)
    assert pd.api.types.is_datetime64_any_dtype(got.dtype)

def test_quantile_edf_out_of_range_probs_raises_value_error():
    # > quantile(c(10, 20, 30, 40), probs = c(-1e-12, 0, .5, 1), na.rm = TRUE, type = 1)
    # Error in `quantile.default()`:
    # ! 'probs' outside [0,1]
    # 
    # > quantile(c(10, 20, 30, 40), probs = c(0, .5, 1, 1 + 1e-12), na.rm = TRUE, type = 1)
    # Error in `quantile.default()`:
    # ! 'probs' outside [0,1]

    x = pd.Series([10, 20, 30, 40])
    probs1 = np.array([-1e-12, 0.0, 0.5, 1.0])
    probs2 = np.array([0.0, 0.5, 1.0, 1 + 1e-12])

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        quantile_edf(x, probs1)
    
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        quantile_edf(x, probs2)

def test_quantile_edf_empty_int64_series_fallback():
    # > quantile(numeric(0), probs = c(0, .5, 1), na.rm = TRUE, type = 1)
    #   0%  50% 100% 
    #   NA   NA   NA 
    
    x = pd.Series([], dtype="int64")
    probs = np.array([0.0, 0.5, 1.0])

    got = quantile_edf(x, probs)

    # length and index must match probs
    assert len(got) == len(probs)
    assert got.index.to_numpy().tolist() == probs.tolist()

    # all values must be missing
    assert got.isna().all()

    # dtype must not remain non-nullable int64
    assert got.dtype != "int64"

def test_quantile_edf_small_n_explicit():
    # > quantile(c(4, 1, 3, 2), probs = c(0, .25, .5, .75, 1), na.rm = TRUE, type = 1)
    #   0%  25%  50%  75% 100% 
    #    1    1    2    3    4 

    x = pd.Series([4, 1, 3, 2])  # sorted -> [1,2,3,4], n=4
    probs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    got = quantile_edf(x, probs)
    exp = pd.Series([1, 1, 2, 3, 4], index=probs)  # type-1: ceil(n p)

    pd.testing.assert_series_equal(got, exp)

def test_quantile_edf_many_duplicates():
    # > quantile(c(1, 1, 1, 2, 2, 3), probs = (0:6)/6, na.rm = TRUE, type = 1)
    #        0% 16.66667% 33.33333%       50% 66.66667% 83.33333%      100% 
    #         1         1         1         1         2         2         3 
    
    x = pd.Series([1, 1, 1, 2, 2, 3])  # sorted same, n=6
    probs = np.array([0.0, 1/6, 2/6, 3/6, 4/6, 5/6, 1.0])

    got = quantile_edf(x, probs)
    exp = pd.Series([1, 1, 1, 1, 2, 2, 3], index=probs)

    pd.testing.assert_series_equal(got, exp)
