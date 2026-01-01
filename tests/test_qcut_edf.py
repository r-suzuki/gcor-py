import numpy as np
import pandas as pd

from gcor._core import qcut_edf


def test_qcut_edf_includes_lowest_in_first_bin():
    x = pd.Series([4, 1, 3, 2], name='x')  # intentionally unsorted
    ret = qcut_edf(x, k=2)

    # min must fall into the first bin
    xmin_label = ret.loc[x.idxmin()]
    assert xmin_label == ret.cat.categories[0]

    # sanity: returned as categorical with expected number of bins (may be < k if ties)
    assert hasattr(ret, 'cat')
    assert 1 <= len(ret.cat.categories) <= 2


def test_qcut_edf_label_brackets():
    x = pd.Series([1, 2, 3, 4, 5, 6], name='x')
    ret = qcut_edf(x, k=3)
    cats = list(ret.cat.categories)

    # first bin label starts with '['
    assert cats[0].startswith('[')

    # remaining bins start with '(' (if any)
    for c in cats[1:]:
        assert c.startswith('(')

    # all bins are right-closed in the label
    for c in cats:
        assert c.endswith(']')


def test_qcut_edf_ties_reduce_bins_but_still_work():
    # many ties -> quantile edges will have duplicates -> bins may collapse
    x = pd.Series([0, 0, 0, 1, 1, 1], name='x')
    ret = qcut_edf(x, k=4)

    # should not error, and should produce at least 1 bin
    assert hasattr(ret, 'cat')
    assert len(ret.cat.categories) >= 1

    # all non-missing values must be assigned
    assert ret.notna().all()


def test_qcut_edf_preserves_nan_as_nan():
    x = pd.Series([1.0, np.nan, 2.0, 3.0], name='x')
    ret = qcut_edf(x, k=2)

    assert ret.isna().sum() == 1
    assert pd.isna(ret.iloc[1])
