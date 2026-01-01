from typing import Union
import numpy as np
import pandas as pd
import warnings
from gcor._core import discretize, gcor_cat

def gcor(
    x,
    y=None,
    *,
    drop_na='none',
    k=None,
    max_levels=100,
) -> Union[float, pd.DataFrame]:
    """
    Generalized correlation measure.

    Parameters
    ----------
    x : 1d array-like or `pandas.DataFrame`
        Input data. If 1d array-like, treated as n observations of a random variable.
        If DataFrame, treated as n observations of p random variables (columns).
        Must not be None.
    
    y : 1d array-like or None, default None
        Second input data. Required if `x` is 1d array-like.
        It is treated as n observations of another random variable.
        Must be None if `x` is a DataFrame.
    
    drop_na : {'none', 'pairwise', 'casewise'}, default 'none'
        How to handle missing values:
        - 'none': Keep missing values and treat them as observations of a separate category
        - 'pairwise': For each variable pair (X, Y), drop the i-th observation
          if either X or Y is missing at position i
        - 'casewise': Drop the i-th observation for all variables,
          if any variable is missing at position i
    
    k : int or None, default None
        Number of quantile divisions. If None, determined automatically.
    
    max_levels : int, default 100
        The maximum number of levels allowed when converting non-numeric variables to categories.

    Returns
    -------
    float or pandas.DataFrame
        If `x` is 1d array-like, returns a float (or `numpy.nan`).
        If `x` is DataFrame, returns a generalized correlation matrix as a DataFrame.
        The diagonal elements are always set to 1.0.
    
    Raises
    ------
    ValueError
        If `drop_na` is invalid or if the lengths of `x` and `y` do not match.
    TypeError
        If the combination of `x` and `y` is invalid.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from gcor import gcor
    >>> x = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    >>> y = pd.Series([1, 2, 3, 4, 5, 3, 4, 5, 6, 7])
    >>> gcor(x, y)
    0.534522

    >>> df = pd.DataFrame({
    ...     'x': x,
    ...     'y': y,
    ...     'z': ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd', 'e', 'e'],
    ... })
    >>> gcor(df)
              x         y         z
    x  1.000000  0.534522  0.806219
    y  0.534522  1.000000  0.734035
    z  0.806219  0.734035  1.000000
    """
    # --- Validation ---
    valid_drop_na = {'none', 'pairwise', 'casewise'}
    if drop_na not in valid_drop_na:
        raise ValueError(
            "drop_na must be one of {'none', 'pairwise', 'casewise'} "
            f"(got {drop_na!r})."
        )
    
    if x is None:
        raise TypeError("x must not be None.")

    # --- Store inputs in a temporary DataFrame (df_tmp) ---
    return_matrix: bool = isinstance(x, pd.DataFrame)

    if return_matrix:
        if y is not None:
            raise TypeError('y must be None when x is a pandas.DataFrame.')
        df_tmp = x.copy(deep=False)
    else:
        if y is None:
            raise TypeError('y must not be None when x is not a pandas.DataFrame.')

        xs = None
        ys = None

        # If both are Series, align by position and warn when indices differ.
        if isinstance(x, pd.Series) and isinstance(y, pd.Series):
            if not x.index.equals(y.index):
                warnings.warn(
                    'x and y are pandas.Series but their indices do not match. '
                    'Resetting indices (drop=True) and matching by position.',
                    UserWarning,
                    stacklevel=2,
                )
                xs = x.reset_index(drop=True)
                ys = y.reset_index(drop=True)

        # If xs/ys are still None, convert inputs to Series (or keep as-is if already Series).
        if xs is None:
            xs = x if isinstance(x, pd.Series) else pd.Series(x)
        if ys is None:
            ys = y if isinstance(y, pd.Series) else pd.Series(y)

        # Ensure lengths match.
        if len(xs) != len(ys):
            raise ValueError(
                f'x and y must have the same length (got len(x)={len(xs)} and len(y)={len(ys)}).'
            )

        df_tmp = pd.DataFrame({'x': xs, 'y': ys})

    # --- Missing-value handling (casewise) ---
    if drop_na == 'casewise':
        df_tmp = df_tmp.dropna(how='any')

    # --- Discretize each column ---
    df = df_tmp.apply(lambda s: discretize(s, k=k, max_levels=max_levels), axis=0)

    # Precompute missing-value mask only when needed (pairwise).
    if drop_na == 'pairwise':
        na = df.isna()

    # --- Compute generalized correlation matrix ---
    ncol = df.shape[1]
    mat = np.full((ncol, ncol), np.nan, dtype=float)

    for i in range(ncol):
        for j in range(i, ncol):
            if i == j:
                mat[i, j] = 1.0
                continue

            # Missing value handling (pairwise)
            if drop_na == 'pairwise':
                mask = ~(na.iloc[:, i] | na.iloc[:, j])
                val = gcor_cat(df.iloc[mask, i], df.iloc[mask, j])
            else:
                val = gcor_cat(df.iloc[:, i], df.iloc[:, j])

            mat[i, j] = val
            mat[j, i] = val

    if return_matrix:
        return pd.DataFrame(mat, index=df.columns, columns=df.columns)
    else:
        return mat[0, 1]
