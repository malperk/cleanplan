# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-09-28

### Added
- `inspect()` detects exact duplicates, key conflicts, missing values,
  placeholder values, rule violations and outliers, and suggests a
  conservative `CleaningPlan`.
- `CleaningPlan` with per-column policies (`configure`), domain rules
  (`add_rule`), `fit` / `transform` / `apply`, and JSON-friendly
  `to_dict` / `from_dict`.
- Rules: `Range` (numeric bounds or bounds from other columns), `Check`
  (row-wise pandas expressions) and `Unique` (single or composite keys).
- Rule-aware imputation: imputed values are clipped into `Range` bounds.
- Outlier detection with IQR or MAD, and the actions flag, set_missing,
  clip or drop_row.
- Group-wise imputation with a minimum group size and fallback to the
  overall statistic.
- `ChangeLog` recording every changed cell and removed row with its reason.
- `ValidationReport` re-checking rules, key uniqueness and remaining gaps.
- `<column>_imputed` / `<column>_outlier` indicator columns.

[Unreleased]: https://github.com/malperk/cleanplan/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/malperk/cleanplan/releases/tag/v0.1.0
