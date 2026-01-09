# `gcor` library for Python
A Python implementation of generalized correlation measure.

**Note that this project is in an early stage of development, so changes
may occur frequently.**

## Documentation

Documentation site (user guide and API reference)
: <https://r-suzuki.github.io/gcor-py/>

User guide
: `README_USER.md`

## Development

This project uses `uv` and `Makefile` targets.

```bash
make sync     # install dependencies (incl. dev)
make test     # run tests
make docs     # build docs (pdoc + README_USER)
make build    # build sdist/wheel
make check    # full check before release
```

## References

`gcor` package for R
: <https://github.com/r-suzuki/gcor-r>
