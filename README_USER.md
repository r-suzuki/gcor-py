# A Python library for generalized correlation


A Python implementation of generalized correlation measure. For
development status and source code, see
<https://github.com/r-suzuki/gcor-py>.

**Note that this project is in an early stage of development, so changes
may occur frequently.**

## Installation

``` bash
pip install git+https://github.com/r-suzuki/gcor-py.git
```

## Examples

**Generalized correlation measure** takes values in \[0, 1\] and can
capture both linear and nonlinear associations. It naturally handles
mixed data types, including numerical and categorical variables.

### Scalar example

``` python
import pandas as pd
from gcor import gcor

x = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
y = pd.Series([1, 2, 3, 4, 5, 3, 4, 5, 6, 7])

g = gcor(x, y)
print(g)
```

    0.5345224838248488

### Matrix example (mixed numeric and categorical data)

``` python
df = pd.DataFrame({
    "x": x,
    "y": y,
    "z": ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"],
})

gmat = gcor(df)
print(gmat)
```

              x         y         z
    x  1.000000  0.534522  0.806219
    y  0.534522  1.000000  0.734035
    z  0.806219  0.734035  1.000000
