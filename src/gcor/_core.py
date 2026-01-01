import numpy as np
import pandas as pd
from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
    is_timedelta64_dtype
)
from math import floor, log10, sqrt
# import warnings

# TODO: return phi, kx and ky, and leave post-processing to the caller
def gcor_cat(
    x: pd.Series,
    y: pd.Series,
) -> float:
    """
    Generalized correlation for a pair of categorical Series.
    Missing values are treated as observations of a separate category.

    Parameters
    ----------
    x, y : pandas.Series
        With the same length and the same index.
    
    Returns
    -------
    float
        numpy.nan is returned if either of the following holds:
        - x or y is not pandas.CategoricalDtype.
        - x and y are of length zero.
    """    
    if not isinstance(x.dtype, pd.CategoricalDtype) or \
        not isinstance(y.dtype, pd.CategoricalDtype):
        return np.nan

    if len(x) != len(y) or not x.index.equals(y.index):
        raise ValueError("x and y must have the same length and index.")
    
    n = len(x)
    if n == 0:
        return np.nan

    if x is y or x.equals(y):
        return 1.0
    
    xt = pd.crosstab(x, y, dropna=False)

    # avoid overflows
    xt_fl = xt.astype('float64', copy=False)

    row_sum = xt_fl.sum(axis=1)
    col_sum = xt_fl.sum(axis=0)

    row_mask = row_sum > 0
    col_mask = col_sum > 0

    nn = xt_fl.loc[row_mask, col_mask].to_numpy()
    nx = row_sum.loc[row_mask].to_numpy()
    ny = col_sum.loc[col_mask].to_numpy()

    phi = (nn ** 2 / np.outer(nx, ny)).sum()
    kx = nx.size
    ky = ny.size

    # phi should be greater or equal to 1, but estimated values
    # with some approximation could be less than 1. It is adjusted here.    
    if phi is not None and phi < 1:
        # warnings.warn(
        #     "Estimated mutual dependency < 1; adjusted to 1.",
        #     RuntimeWarning,
        #     stacklevel=2
        # )
        phi = 1.0

    # If kk == 1, both x and y is constant.
    # In this case gcor(x,y) = 1 and gdis(x,y) = 0.
    if kx == ky == 1:
        cor = 1.0
    else:
        kk = sqrt(kx) * sqrt(ky)
        r2 = 1 - 1/phi
        cor = sqrt(r2 / (1 - 1/kk))
    
    return cor

def quantile_edf(x: pd.Series, probs) -> pd.Series:
    """
    Sample quantiles based on empirical distribution function.

    Parameters
    ----------
    x : pandas.Series
        Input data.
    
    probs : 1d array-like
        Probabilities with values in [0, 1].
    
    Returns
    -------
    pandas.Series
    """
    s = x.dropna().sort_values()
    n = len(s)
    p = np.asarray(probs, dtype=float)

    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probs should be in the interval [0, 1]")

    if n == 0:
        try:
            return pd.Series([None] * len(p), dtype=x.dtype, index=p)
        except (TypeError, ValueError):
            return pd.Series([None] * len(p), index=p)
    
    idx = np.ceil(n * p).astype(int) - 1
    idx = np.clip(idx, 0, n - 1)

    out = s.iloc[idx]
    out.index = p
    return out

def qcut_edf(x, k) -> pd.Series:
    """
    Discretize values based on empirical distribution function.

    Similar to pandas.qcut, with the lowest bound closed.

    Parameters
    ----------
    x : pandas.Series
        Input data.
    
    k : int
        Number of bins.
    
    Returns
    -------
    pandas.Series
    """
    p = np.linspace(0, 1, k + 1)
    qt = quantile_edf(x, p)

    edges = np.sort(qt.unique())
    labels = [
        f'{ "[" if i == 0 else "(" }{edges[i]}, {edges[i + 1]}]'
        for i in range(len(edges) - 1)
    ]

    ret = pd.cut(x, edges, right=True, labels=labels, include_lowest=True)

    return ret

def discretize(x, k, max_levels) -> pd.Series:
    """
    Discretize (numeric) values.
    
    Return a categorical series when possible; otherwise return x unchanged.

    Parameters
    ----------
    x : pandas.Series
        Input data.
    
    k : int
        Number of quantile divisions. If None, determined automatically.
    
    max_levels : int
        The maximum number of levels allowed when converting non-numeric variables to categories.
    
    Returns
    -------
    pandas.Series
    """

    # If already categorical, return as-is.
    if isinstance(x.dtype, pd.CategoricalDtype):
        return x

    # Count distinct values (excluding NA).
    # If this fails (unhashable elements; e.g., dict/list), return as-is.
    try:
        n_unique = x.nunique(dropna=True)
    except TypeError:
        return x
    
    if is_numeric_dtype(x) or is_datetime64_any_dtype(x) or is_timedelta64_dtype(x):
        if k is None:
            k = max(2, floor((x.count() ** log10(2)) / 2))
        
        # If the number of distinct values (excluding NA) is <= k, cast to category.
        if n_unique <= k:
            return x.astype("category")

        # If qcut is applicable, return quantile bins.
        try:
            # return pd.qcut(x, k, duplicates="drop")
            return qcut_edf(x, k)
        except (TypeError, ValueError):
            pass

    # Otherwise, if distinct values (excluding NA) is <= max_levels, cast to category.
    if n_unique <= max_levels:
        return x.astype("category")

    # Fallback: return as-is.
    return x
