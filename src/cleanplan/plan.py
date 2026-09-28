"""The cleaning plan: inspect, review, adjust, apply, validate."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Literal, get_args

import numpy as np
import pandas as pd

from . import detect
from .result import ChangeLog, CleaningResult, ValidationIssue, ValidationReport
from .rules import Range, Rule, Unique, integer_bounds

ImputeStrategy = Literal["median", "mean", "mode", "constant", "none"]
OutlierAction = Literal["flag", "set_missing", "clip", "drop_row", "ignore"]
OutlierMethod = Literal["iqr", "mad"]
InvalidAction = Literal["set_missing", "flag", "drop_row"]
KeyConflictAction = Literal["flag", "keep_first", "keep_last", "drop"]

#: Fill value suggested for missing categories. Keeping "missing" as its own
#: category is usually more honest than assigning the most frequent one.
DEFAULT_CATEGORY_FILL = "Unknown"

IMPUTED_SUFFIX = "_imputed"
OUTLIER_SUFFIX = "_outlier"


def _check_choice(name: str, value: Any, choices: Any) -> None:
    options = get_args(choices)
    if value not in options:
        raise ValueError(f"{name} must be one of {options}, got {value!r}")


def _as_key(key: str | Sequence[str] | None) -> tuple[str, ...] | None:
    if key is None:
        return None
    if isinstance(key, str):
        return (key,)
    return tuple(key) or None


@dataclass
class ColumnPolicy:
    """How one column is cleaned.

    Attributes:
        impute: How missing values are filled. ``median``/``mean`` need a
            numeric column; ``mode`` works for any type; ``constant`` uses
            ``fill_value``; ``none`` leaves them missing.
        fill_value: The value used by ``impute="constant"``.
        by: Learn ``median``/``mean``/``mode`` per group of this column.
        min_group_size: Groups with fewer observed values fall back to the
            overall statistic instead of trusting a tiny sample.
        outliers: What to do with statistical outliers. ``flag`` marks them
            in a ``<column>_outlier`` column and changes nothing;
            ``set_missing`` treats them as errors and imputes them; ``clip``
            caps them at the bounds (winsorizing); ``drop_row`` removes the
            row; ``ignore`` skips detection.
        outlier_method: ``iqr`` (Tukey's fences) or ``mad`` (modified z-score).
        outlier_threshold: IQR multiplier or z cut-off; defaults to 1.5 / 3.5.
        invalid: What to do with values that break a rule blaming this column:
            ``set_missing`` (then impute), ``flag`` (report only) or
            ``drop_row``.
    """

    impute: ImputeStrategy = "none"
    fill_value: Any = None
    by: str | None = None
    min_group_size: int = 5
    outliers: OutlierAction = "ignore"
    outlier_method: OutlierMethod = "iqr"
    outlier_threshold: float | None = None
    invalid: InvalidAction = "set_missing"

    def __post_init__(self) -> None:
        _check_choice("impute", self.impute, ImputeStrategy)
        _check_choice("outliers", self.outliers, OutlierAction)
        _check_choice("outlier_method", self.outlier_method, OutlierMethod)
        _check_choice("invalid", self.invalid, InvalidAction)
        if self.impute == "constant" and self.fill_value is None:
            raise ValueError("impute='constant' needs a fill_value")
        if self.by is not None and self.impute not in ("median", "mean", "mode"):
            raise ValueError("'by' only applies to impute='median', 'mean' or 'mode'")
        if self.min_group_size < 1:
            raise ValueError("min_group_size must be at least 1")
        if self.outlier_threshold is not None and self.outlier_threshold <= 0:
            raise ValueError("outlier_threshold must be positive")

    @property
    def threshold(self) -> float:
        if self.outlier_threshold is not None:
            return self.outlier_threshold
        return detect.DEFAULT_OUTLIER_THRESHOLDS[self.outlier_method]

    def describe(self) -> str:
        parts = []
        if self.impute == "constant":
            parts.append(f"impute {self.fill_value!r}")
        elif self.impute != "none":
            by = f" by {self.by}" if self.by else ""
            parts.append(f"impute {self.impute}{by}")
        if self.outliers != "ignore":
            parts.append(f"outliers: {self.outliers} ({self.outlier_method.upper()} {self.threshold:g})")
        return " | ".join(parts) or "no action"


@dataclass(frozen=True)
class OutlierFinding:
    count: int
    lower: float
    upper: float
    method: str
    threshold: float


@dataclass
class Findings:
    """What :func:`inspect` found in the data, before anything was changed."""

    n_rows: int
    n_columns: int
    key: tuple[str, ...] | None = None
    exact_duplicates: int = 0
    key_conflicts: int = 0
    missing: dict[Any, int] = field(default_factory=dict)
    sentinels: dict[Any, dict[Any, int]] = field(default_factory=dict)
    rule_violations: dict[str, int] = field(default_factory=dict)
    outliers: dict[Any, OutlierFinding] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FittedFill:
    """Imputation values learned during :meth:`CleaningPlan.fit`."""

    overall: Any
    groups: dict[Any, Any] = field(default_factory=dict)


class CleaningPlan:
    """An explicit, reviewable description of how to clean a DataFrame.

    Usually created with :func:`cleanplan.inspect`, which detects issues and
    suggests a policy per column. Review it with :meth:`report`, adjust it
    with :meth:`configure` and :meth:`add_rule`, then run it with
    :meth:`apply` (or :meth:`fit` on training data and :meth:`transform` on
    new data, so no statistics leak from evaluation data).

    Steps always run in this order, so that errors never feed the statistics
    used to fill gaps:

    1. normalize placeholder values (``"N/A"``, ``-999``) to missing
    2. drop exact duplicates and resolve key conflicts
    3. handle values that break rules
    4. handle outliers
    5. impute missing values, then clip imputed values into rule ranges
    6. validate the result
    """

    def __init__(
        self,
        policies: dict[Any, ColumnPolicy] | None = None,
        *,
        key: str | Sequence[str] | None = None,
        rules: Iterable[Rule] = (),
        drop_duplicates: bool = True,
        key_conflicts: KeyConflictAction = "flag",
        string_sentinels: Iterable[str] = detect.DEFAULT_STRING_SENTINELS,
        numeric_sentinels: Iterable[float] = detect.DEFAULT_NUMERIC_SENTINELS,
        indicators: bool = True,
        restore_integers: bool = True,
    ) -> None:
        _check_choice("key_conflicts", key_conflicts, KeyConflictAction)
        self.policies: dict[Any, ColumnPolicy] = dict(policies or {})
        self.key = _as_key(key)
        self.rules: list[Rule] = list(rules)
        self.drop_duplicates = drop_duplicates
        self.key_conflicts: KeyConflictAction = key_conflicts
        self.string_sentinels = frozenset(string_sentinels)
        self.numeric_sentinels = frozenset(numeric_sentinels)
        self.indicators = indicators
        self.restore_integers = restore_integers
        self.findings: Findings | None = None
        self._fitted = False

    # ------------------------------------------------------------------ editing

    def configure(self, column: Any, **changes: Any) -> CleaningPlan:
        """Change the policy of ``column``. Returns the plan, so calls chain::

        plan.configure("salary", impute="median", by="department")
        """
        valid = {f.name for f in fields(ColumnPolicy)}
        unknown = set(changes) - valid
        if unknown:
            raise TypeError(f"Unknown policy option(s) {sorted(unknown)}; valid options are {sorted(valid)}")
        if column not in self.policies:
            if self.findings is not None:
                raise KeyError(f"Unknown column {column!r}; the inspected columns are {list(self.policies)}")
            self.policies[column] = ColumnPolicy(**changes)
        else:
            self.policies[column] = replace(self.policies[column], **changes)
        self._fitted = False
        return self

    def add_rule(self, *rules: Rule) -> CleaningPlan:
        """Declare domain rules. Returns the plan, so calls chain."""
        for rule in rules:
            if not isinstance(rule, Rule):
                raise TypeError(f"Expected a Rule, got {type(rule).__name__}")
        self.rules.extend(rules)
        self._fitted = False
        return self

    # ----------------------------------------------------------------- fitting

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, df: pd.DataFrame) -> CleaningPlan:
        """Learn outlier bounds and imputation values from ``df``."""
        self._fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> CleaningResult:
        """Clean ``df`` with what was learned in :meth:`fit`. ``df`` is not modified."""
        if not self._fitted:
            raise RuntimeError("This plan is not fitted; call fit() or apply() first")
        self._check_input(df)
        labels = df.index
        work = _positional_copy(df)
        log = ChangeLog()
        work = self._prepare(work, log, labels)
        work, flagged, _ = self._handle_outliers(work, log, labels)
        work, imputed = self._impute(work, log, labels)
        self._repair_imputed(work, imputed, log, labels)
        if self.restore_integers:
            work = self._restore_integers(work)
        validation = self._validate(work, flagged, labels)
        work = self._add_indicators(work, imputed, flagged)
        work.index = labels.take(work.index.to_numpy())
        return CleaningResult(work, log, validation, self)

    def apply(self, df: pd.DataFrame) -> CleaningResult:
        """Fit on ``df`` and clean it. Equivalent to ``fit(df).transform(df)``."""
        return self.fit(df).transform(df)

    # ------------------------------------------------------------ presentation

    def report(self) -> str:
        """A human-readable summary of the findings and the planned actions."""
        from .report import format_report

        return format_report(self)

    def __repr__(self) -> str:
        return self.report()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the plan's decisions (not the learned values) to plain data.

        Store the result as JSON next to your pipeline to keep a record of how
        the data was cleaned; rebuild the plan with :meth:`from_dict`.
        """
        return {
            "format": 1,
            "key": list(self.key) if self.key else None,
            "drop_duplicates": self.drop_duplicates,
            "key_conflicts": self.key_conflicts,
            "indicators": self.indicators,
            "restore_integers": self.restore_integers,
            "string_sentinels": sorted(self.string_sentinels),
            "numeric_sentinels": sorted(self.numeric_sentinels),
            "rules": [rule.to_dict() for rule in self.rules],
            "policies": {column: asdict(policy) for column, policy in self.policies.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CleaningPlan:
        if data.get("format") != 1:
            raise ValueError(f"Unsupported plan format: {data.get('format')!r}")
        return cls(
            {column: ColumnPolicy(**policy) for column, policy in data["policies"].items()},
            key=data["key"],
            rules=[Rule.from_dict(rule) for rule in data["rules"]],
            drop_duplicates=data["drop_duplicates"],
            key_conflicts=data["key_conflicts"],
            string_sentinels=data["string_sentinels"],
            numeric_sentinels=data["numeric_sentinels"],
            indicators=data["indicators"],
            restore_integers=data["restore_integers"],
        )

    # ---------------------------------------------------------------- internals

    def _policy(self, column: Any) -> ColumnPolicy:
        return self.policies.get(column) or ColumnPolicy()

    def _check_input(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}")
        if df.columns.has_duplicates:
            raise ValueError(f"Column names must be unique; duplicated: {list(df.columns[df.columns.duplicated()])}")
        needed = list(self.policies) + list(self.key or ()) + [p.by for p in self.policies.values() if p.by]
        missing = [c for c in dict.fromkeys(needed) if c not in df.columns]
        if missing:
            raise KeyError(f"The plan refers to column(s) not in the data: {missing}")

    def _fit(self, df: pd.DataFrame) -> dict[Any, int]:
        """Fit the plan and return the number of outliers found per column."""
        self._check_input(df)
        labels = df.index
        scratch = ChangeLog()
        work = self._prepare(_positional_copy(df), scratch, labels)

        self.integer_columns_ = {c for c in work.columns if detect.is_integral(work[c])}
        # Columns that are integers at heart but float only because of gaps. Decided
        # here, not per frame, so every transformed frame gets the same dtypes.
        self.restore_columns_ = {
            c
            for c in self.integer_columns_
            if pd.api.types.is_integer_dtype(df[c]) or df[c].isna().any() or work[c].isna().any()
        }
        self.outlier_bounds_ = self._learn_outlier_bounds(work)
        work, flagged, counts = self._handle_outliers(work, scratch, labels)
        self.fill_values_ = self._learn_fills(work)

        # As in scikit-learn's add_indicator, indicator columns are decided at
        # fit time so that every transformed frame has the same schema.
        self.imputed_indicators_ = [c for c, p in self.policies.items() if p.impute != "none" and work[c].isna().any()]
        self.outlier_indicators_ = [c for c, mask in flagged.items() if mask.any()]
        if self.indicators:
            names = [f"{c}{IMPUTED_SUFFIX}" for c in self.imputed_indicators_]
            names += [f"{c}{OUTLIER_SUFFIX}" for c in self.outlier_indicators_]
            clashes = [n for n in names if n in df.columns]
            if clashes:
                raise ValueError(f"Indicator column(s) {clashes} already exist; rename them or pass indicators=False")
        self._fitted = True
        return counts

    def _prepare(self, work: pd.DataFrame, log: ChangeLog, labels: pd.Index) -> pd.DataFrame:
        work = self._normalize_sentinels(work, log, labels)
        work = self._handle_duplicates(work, log, labels)
        return self._handle_invalid(work, log, labels)

    def _normalize_sentinels(self, work: pd.DataFrame, log: ChangeLog, labels: pd.Index) -> pd.DataFrame:
        for column in work.columns:
            mask = detect.sentinel_mask(work[column], self.string_sentinels, self.numeric_sentinels)
            if mask.any():
                old = work.loc[mask, column].to_numpy(copy=True)
                _set_missing(work, column, mask)
                log.record(
                    "normalize_placeholder", "placeholder for a missing value", _rows(labels, mask), column, old, np.nan
                )
        return work

    def _handle_duplicates(self, work: pd.DataFrame, log: ChangeLog, labels: pd.Index) -> pd.DataFrame:
        if self.drop_duplicates:
            mask = work.duplicated(keep="first")
            if mask.any():
                log.record("drop_duplicate", "exact duplicate of an earlier row", _rows(labels, mask))
                work = work.loc[~mask].copy()
        if self.key and self.key_conflicts != "flag":
            keep: Literal["first", "last", False] = (
                "first"
                if self.key_conflicts == "keep_first"
                else "last"
                if self.key_conflicts == "keep_last"
                else False
            )
            subset = work[list(self.key)]
            mask = subset.duplicated(keep=keep) & subset.notna().all(axis=1)
            if mask.any():
                reason = f"another row has the same key ({self.key_conflicts})"
                log.record("resolve_key_conflict", reason, _rows(labels, mask))
                work = work.loc[~mask].copy()
        return work

    def _handle_invalid(self, work: pd.DataFrame, log: ChangeLog, labels: pd.Index) -> pd.DataFrame:
        blamed = [rule for rule in self.rules if rule.column is not None]
        # Evaluate every rule on the same snapshot so that rule order does not matter.
        violations = [(rule, rule.violations(work)) for rule in blamed]
        drop = pd.Series(False, index=work.index)
        for rule, mask in violations:
            column = rule.column
            mask = mask & work[column].notna()
            if not mask.any():
                continue
            action = self._policy(column).invalid
            reason = f"breaks rule '{rule.name}'"
            if action == "set_missing":
                old = work.loc[mask, column].to_numpy(copy=True)
                _set_missing(work, column, mask)
                log.record("invalid_value", reason, _rows(labels, mask), column, old, np.nan)
            elif action == "drop_row":
                log.record("drop_invalid_row", reason, _rows(labels, mask & ~drop), column)
                drop |= mask
            # "flag": leave the value; validation reports the broken rule.
        if drop.any():
            work = work.loc[~drop].copy()
        return work

    def _learn_outlier_bounds(self, work: pd.DataFrame) -> dict[Any, tuple[float, float]]:
        bounds = {}
        for column, policy in self.policies.items():
            if policy.outliers == "ignore" or not detect.is_numeric(work[column]):
                continue
            found = detect.outlier_bounds(work[column], policy.outlier_method, policy.threshold)
            if found is not None:
                bounds[column] = found
        return bounds

    def _handle_outliers(
        self, work: pd.DataFrame, log: ChangeLog, labels: pd.Index
    ) -> tuple[pd.DataFrame, dict[Any, pd.Series], dict[Any, int]]:
        flagged: dict[Any, pd.Series] = {}
        counts: dict[Any, int] = {}
        drop = pd.Series(False, index=work.index)
        for column, (lower, upper) in self.outlier_bounds_.items():
            policy = self._policy(column)
            values = work[column].astype("float64")
            mask = (values < lower) | (values > upper)
            counts[column] = int(mask.sum())
            if not mask.any():
                continue
            reason = f"outside [{lower:,.6g}, {upper:,.6g}] ({policy.outlier_method.upper()} {policy.threshold:g})"
            old = work.loc[mask, column].to_numpy(copy=True)
            if policy.outliers == "drop_row":
                log.record("drop_outlier_row", reason, _rows(labels, mask & ~drop), column, old)
                drop |= mask
                continue
            flagged[column] = mask
            if policy.outliers == "set_missing":
                _set_missing(work, column, mask)
                log.record("outlier", reason, _rows(labels, mask), column, old, np.nan)
            elif policy.outliers == "clip":
                low, high = lower, upper
                if column in self.integer_columns_:
                    low, high = float(np.ceil(lower)), float(np.floor(upper))
                _ensure_float(work, column)
                work.loc[mask, column] = values[mask].clip(low, high)
                log.record("outlier", reason, _rows(labels, mask), column, old, work.loc[mask, column].to_numpy())
        if drop.any():
            work = work.loc[~drop].copy()
            flagged = {c: m.loc[work.index] for c, m in flagged.items()}
        return work, flagged, counts

    def _learn_fills(self, work: pd.DataFrame) -> dict[Any, FittedFill]:
        fills = {}
        for column, policy in self.policies.items():
            if policy.impute in ("none", "constant"):
                continue
            series = work[column]
            if policy.impute in ("median", "mean") and not detect.is_numeric(series):
                raise TypeError(
                    f"Cannot impute {column!r} with the {policy.impute}: it is not numeric ({series.dtype}). "
                    "Use impute='mode' or impute='constant'."
                )
            statistic = _STATISTICS[policy.impute]
            observed = series.dropna()
            overall = statistic(observed) if len(observed) else np.nan
            groups: dict[Any, Any] = {}
            if policy.by is not None:
                grouped = series.groupby(work[policy.by], dropna=True, observed=True)
                sizes = grouped.count()
                values = grouped.agg(statistic)
                groups = {g: values[g] for g, n in sizes.items() if n >= policy.min_group_size}
            fills[column] = FittedFill(overall, groups)
        return fills

    def _impute(
        self, work: pd.DataFrame, log: ChangeLog, labels: pd.Index
    ) -> tuple[pd.DataFrame, dict[Any, pd.Series]]:
        # Group keys are read before anything is imputed, so a row whose group
        # is missing falls back to the overall statistic instead of borrowing
        # from whatever group its own imputation assigned.
        group_keys = {p.by: work[p.by].copy() for p in self.policies.values() if p.by is not None}
        imputed: dict[Any, pd.Series] = {}
        for column, policy in self.policies.items():
            if policy.impute == "none":
                continue
            missing = work[column].isna()
            if not missing.any():
                continue
            if policy.impute == "constant":
                values: Any = policy.fill_value
            else:
                fitted = self.fill_values_[column]
                if policy.by is not None and fitted.groups:
                    keys = group_keys[policy.by][missing]
                    values = pd.Series(
                        [fitted.groups.get(k, fitted.overall) if pd.notna(k) else fitted.overall for k in keys],
                        index=keys.index,
                        dtype=object,
                    )
                else:
                    values = fitted.overall
            series = work[column]
            if detect.is_numeric(series):
                if policy.impute == "constant" and not pd.api.types.is_number(values):
                    raise TypeError(f"Cannot fill numeric column {column!r} with the non-numeric value {values!r}")
                values = pd.to_numeric(values, errors="coerce")
                _ensure_float(work, column)
                if column in self.integer_columns_ and policy.impute != "constant":
                    values = np.round(values)
            elif detect.is_categorical(series):
                new = pd.Series(values if isinstance(values, pd.Series) else [values]).dropna().unique()
                extra = [v for v in new if v not in series.cat.categories]
                if extra:
                    work[column] = series.cat.add_categories(extra)
            try:
                work.loc[missing, column] = values
            except (TypeError, ValueError) as exc:
                raise TypeError(f"Cannot fill {column!r} ({work[column].dtype}) with {values!r}: {exc}") from exc
            filled = missing & work[column].notna()
            if filled.any():
                by = f" by {policy.by}" if policy.by else ""
                log.record(
                    "impute",
                    f"missing, filled with {policy.impute}{by}",
                    _rows(labels, filled),
                    column,
                    np.full(int(filled.sum()), np.nan, dtype=object),
                    work.loc[filled, column].to_numpy(),
                )
                imputed[column] = filled
        return work, imputed

    def _repair_imputed(
        self, work: pd.DataFrame, imputed: dict[Any, pd.Series], log: ChangeLog, labels: pd.Index
    ) -> None:
        """Clip imputed values into the ranges declared by :class:`Range` rules."""
        for rule in self.rules:
            if not isinstance(rule, Range) or rule.column not in imputed:
                continue
            column = rule.column
            lower, upper = rule.bounds(work)
            if column in self.integer_columns_:
                lower, upper = integer_bounds(lower, upper)
            values = work[column].astype("float64")
            repaired = values.where(~(values < lower), lower)
            repaired = repaired.where(~(repaired > upper), upper)
            changed = imputed[column] & values.notna() & (repaired != values)
            if changed.any():
                old = work.loc[changed, column].to_numpy(copy=True)
                work.loc[changed, column] = repaired[changed]
                reason = f"imputed value adjusted to satisfy '{rule.name}'"
                log.record("adjust_imputed", reason, _rows(labels, changed), column, old, repaired[changed].to_numpy())

    def _restore_integers(self, work: pd.DataFrame) -> pd.DataFrame:
        """Turn columns that only became float because of missing values back into integers."""
        for column in self.restore_columns_:
            series = work[column]
            if pd.api.types.is_float_dtype(series) and series.notna().all() and detect.is_integral(series):
                work[column] = series.astype("int64")
        return work

    def _validate(self, work: pd.DataFrame, flagged: dict[Any, pd.Series], labels: pd.Index) -> ValidationReport:
        report = ValidationReport()
        rules = list(self.rules)
        if self.key and not any(isinstance(r, Unique) and r.columns == self.key for r in rules):
            rules.append(Unique(*self.key, name=f"key ({', '.join(self.key)}) is unique"))
        for rule in rules:
            mask = rule.violations(work)
            if mask.any():
                report.failures.append(ValidationIssue(f"rule '{rule.name}'", _rows(labels, mask)))
        if self.drop_duplicates:
            mask = work.duplicated(keep=False)
            if mask.any():
                report.failures.append(ValidationIssue("exact duplicates", _rows(labels, mask)))
        for column, policy in self.policies.items():
            mask = work[column].isna()
            if not mask.any():
                continue
            if policy.impute != "none":
                detail = f"{column!r} still has missing values (nothing was learned to impute them with)"
                report.failures.append(ValidationIssue("missing values", _rows(labels, mask), detail))
            else:
                detail = f"{column!r} has missing values (imputation is off for this column)"
                report.warnings.append(ValidationIssue("missing values", _rows(labels, mask), detail))
        for column, mask in flagged.items():
            if self._policy(column).outliers == "flag" and mask.any():
                detail = f"{column!r} flagged, not modified"
                report.warnings.append(ValidationIssue("outliers", _rows(labels, mask), detail))
        return report

    def _add_indicators(
        self, work: pd.DataFrame, imputed: dict[Any, pd.Series], flagged: dict[Any, pd.Series]
    ) -> pd.DataFrame:
        if not self.indicators:
            return work
        no = pd.Series(False, index=work.index)
        for column in self.imputed_indicators_:
            work[f"{column}{IMPUTED_SUFFIX}"] = imputed.get(column, no).reindex(work.index, fill_value=False)
        for column in self.outlier_indicators_:
            work[f"{column}{OUTLIER_SUFFIX}"] = flagged.get(column, no).reindex(work.index, fill_value=False)
        return work

    def _scan(self, df: pd.DataFrame, max_missing: float) -> Findings:
        """Collect findings on ``df`` and fit the plan to it."""
        findings = Findings(n_rows=len(df), n_columns=df.shape[1], key=self.key)
        findings.missing = {c: int(n) for c, n in df.isna().sum().items() if n}
        for column in df.columns:
            mask = detect.sentinel_mask(df[column], self.string_sentinels, self.numeric_sentinels)
            if mask.any():
                findings.sentinels[column] = df.loc[mask, column].value_counts(dropna=False).to_dict()

        labels = df.index
        work = self._normalize_sentinels(_positional_copy(df), ChangeLog(), labels)
        findings.exact_duplicates = int(work.duplicated().sum())
        deduped = work.drop_duplicates() if self.drop_duplicates else work
        if self.key:
            subset = deduped[list(self.key)]
            findings.key_conflicts = int((subset.duplicated(keep=False) & subset.notna().all(axis=1)).sum())
        for rule in self.rules:
            findings.rule_violations[rule.name] = int(rule.violations(deduped).sum())

        counts = self._fit(df)
        for column, (lower, upper) in self.outlier_bounds_.items():
            if counts.get(column):
                policy = self._policy(column)
                findings.outliers[column] = OutlierFinding(
                    counts[column], lower, upper, policy.outlier_method, policy.threshold
                )

        findings.warnings = self._warnings(df, max_missing)
        return findings

    def _warnings(self, df: pd.DataFrame, max_missing: float) -> list[str]:
        warnings = []
        if not self.rules:
            warnings.append(
                "No rules declared. Valid ranges and cross-column constraints cannot be inferred from the data; "
                "declare them with plan.add_rule(Range(...), Check(...)) so they are enforced and imputation "
                "respects them."
            )
        if self.key is None:
            warnings.append("No key column found; pass key=... to detect rows that share an ID but differ.")
        for column in df.columns:
            series = df[column]
            ratio = series.isna().mean() if len(series) else 0.0
            if ratio > max_missing:
                warnings.append(
                    f"{column!r} is {ratio:.0%} missing; imputation is off. Consider dropping it or configuring it."
                )
            elif ratio and detect.is_datetime(series) and self._policy(column).impute == "none":
                warnings.append(f"{column!r} has missing dates; there is no default imputation for datetimes.")
            if len(series) > 1 and series.nunique(dropna=True) <= 1:
                warnings.append(f"{column!r} has a single distinct value and carries no information.")
        for column in self.key or ():
            if df[column].isna().any():
                warnings.append(f"Key column {column!r} has missing values; those rows cannot be identified.")
        return warnings


def inspect(
    df: pd.DataFrame,
    *,
    key: str | Sequence[str] | None = "auto",
    rules: Iterable[Rule] = (),
    outlier_method: OutlierMethod = "iqr",
    max_missing: float = 0.5,
    **options: Any,
) -> CleaningPlan:
    """Detect data quality issues in ``df`` and suggest a :class:`CleaningPlan`.

    Nothing is changed. Print the returned plan to review what was found and
    what would be done, adjust it, then call :meth:`CleaningPlan.apply`.

    Args:
        df: The data to inspect.
        key: Column(s) that identify a row. ``"auto"`` guesses from column
            names such as ``id`` or ``customer_id``; ``None`` disables key checks.
        rules: Domain rules (:class:`Range`, :class:`Check`, :class:`Unique`).
        outlier_method: ``"iqr"`` or ``"mad"`` for suggested outlier detection.
        max_missing: Columns missing more than this share are not imputed by
            default, because filling mostly-empty columns invents data.
        **options: Passed to :class:`CleaningPlan`, e.g. ``key_conflicts``.

    The suggestions are deliberately conservative: outliers are flagged, not
    changed, and missing categories become ``"Unknown"`` rather than the most
    frequent value.
    """
    _check_choice("outlier_method", outlier_method, OutlierMethod)
    if key == "auto":
        key = detect.guess_key(df)
    key_columns = _as_key(key) or ()
    policies = {
        column: _suggest_policy(df[column], column in key_columns, outlier_method, max_missing) for column in df.columns
    }
    plan = CleaningPlan(policies, key=key, rules=rules, **options)
    plan.findings = plan._scan(df, max_missing)
    return plan


def _suggest_policy(series: pd.Series, is_key: bool, outlier_method: OutlierMethod, max_missing: float) -> ColumnPolicy:
    if is_key:
        return ColumnPolicy()
    fillable = (series.isna().mean() if len(series) else 0.0) <= max_missing
    if detect.is_numeric(series):
        return ColumnPolicy(
            impute="median" if fillable else "none",
            outliers="flag" if detect.is_outlier_candidate(series) else "ignore",
            outlier_method=outlier_method,
        )
    if pd.api.types.is_bool_dtype(series):
        return ColumnPolicy(impute="mode" if fillable else "none")
    if (detect.is_textual(series) or detect.is_categorical(series)) and fillable:
        return ColumnPolicy(impute="constant", fill_value=DEFAULT_CATEGORY_FILL)
    return ColumnPolicy()


_STATISTICS = {
    "median": lambda s: s.median(),
    "mean": lambda s: s.mean(),
    "mode": detect.most_frequent,
}


def _positional_copy(df: pd.DataFrame) -> pd.DataFrame:
    """Copy ``df`` with a RangeIndex; original labels are restored at the end.

    Working on positions keeps masks unambiguous even when the caller's index
    has duplicate labels.
    """
    work = df.copy()
    work.index = pd.RangeIndex(len(work))
    return work


def _rows(labels: pd.Index, mask: pd.Series) -> pd.Index:
    """Original index labels of the rows selected by ``mask``."""
    return labels.take(mask.index[mask.to_numpy(dtype=bool)].to_numpy())


def _ensure_float(work: pd.DataFrame, column: Any) -> None:
    if not pd.api.types.is_float_dtype(work[column]) or isinstance(
        work[column].dtype, pd.api.extensions.ExtensionDtype
    ):
        work[column] = work[column].astype("float64")


def _set_missing(work: pd.DataFrame, column: Any, mask: pd.Series) -> None:
    series = work[column]
    if detect.is_numeric(series):
        _ensure_float(work, column)
        work.loc[mask, column] = np.nan
    elif series.dtype == bool:
        work[column] = series.astype(object)
        work.loc[mask, column] = None
    else:
        work.loc[mask, column] = None
