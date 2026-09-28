# Contributing

Thanks for your interest in cleanplan. Bug reports, ideas and pull requests
are all welcome.

## Principles

Every change should keep these guarantees:

- **Nothing changes silently.** Every modification to the data is recorded in
  the `ChangeLog`, with a reason a person can understand.
- **Conservative defaults.** `inspect()` suggests; it does not destroy data.
  New default behaviour should flag rather than modify.
- **Errors never feed the statistics.** Anything learned (medians, bounds) is
  computed after invalid values and outliers have been handled.
- **Same schema for every frame.** Anything that affects output columns or
  dtypes is decided in `fit`, not per frame.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Checks

CI runs the following. Please run them before opening a pull request:

```bash
ruff format --check .
ruff check .
mypy
pytest --cov=cleanplan
```

Tests run with warnings treated as errors, so pandas deprecations are caught
early. Please add a test for every bug fix and every new option.

## Releasing (maintainers)

1. Update `__version__` in `src/cleanplan/__init__.py` and `CHANGELOG.md`.
2. Commit, then tag: `git tag v0.1.0 && git push --tags`.
3. The `release` workflow builds the package and publishes it to PyPI through
   trusted publishing. This needs a one-time setup on PyPI: add a trusted
   publisher for this repository, workflow `release.yml`, environment `pypi`.
