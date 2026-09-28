"""Low-level detection helpers shared by inspection and fitting."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

#: Strings that commonly stand in for a missing value. Matching ignores case
#: and surrounding whitespace.
DEFAULT_STRING_SENTINELS: frozenset[str] = frozenset({"", "na", "n/a", "nan", "null", "none", "?", "-"})
#: Numeric codes that commonly stand in for a missing value.
DEFAULT_NUMERIC_SENTINELS: frozenset[float] = frozenset({-999, -9999, -99999})

#: Default thresholds: IQR multiplier and modified z-score cut-off.
DEFAULT_OUTLIER_THRESHOLDS = {"iqr": 1.5, "mad": 3.5}
OUTLIER_METHODS = tuple(DEFAULT_OUTLIER_THRESHOLDS)

#: Numeric columns with fewer distinct values are treated as codes or ordinal
#: scales, where "outlier" is not a meaningful concept.
MIN_UNIQUE_FOR_OUTLIERS = 10
#: Outlier bounds are not estimated from fewer observed values than this.
MIN_VALUES_FOR_OUTLIERS = 8

_KEY_NAME = re.compile(r"(^id$)|(_id$)|([a-z]Id$)|(^id_)", re.IGNORECASE)


def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def is_textual(series: pd.Series) -> bool:
    return series.dtype == object or isinstance(series.dtype, pd.StringDtype)


def is_categorical(series: pd.Series) -> bool:
    return isinstance(series.dtype, pd.CategoricalDtype)


def is_datetime(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series)


def is_integral(series: pd.Series) -> bool:
    """True if every observed value of a numeric column is a whole number."""
    if not is_numeric(series):
        return False
    values = series.dropna()
    if values.empty:
        return False
    if pd.api.types.is_integer_dtype(series):
        return True
    array = values.to_numpy(dtype="float64")
    return bool(np.all(np.isfinite(array)) and np.all(np.mod(array, 1) == 0))


def sentinel_mask(
    series: pd.Series,
    string_sentinels: Iterable[str],
    numeric_sentinels: Iterable[float],
) -> pd.Series:
    """Boolean mask of values that are placeholders for "missing"."""
    if is_textual(series) or is_categorical(series):
        texts = {s.strip().lower() for s in string_sentinels}
        if not texts:
            return pd.Series(False, index=series.index)
        normalized = series.map(lambda v: v.strip().lower() if isinstance(v, str) else None)
        return normalized.isin(texts).astype(bool)
    if is_numeric(series):
        codes = list(numeric_sentinels)
        if not codes:
            return pd.Series(False, index=series.index)
        return series.isin(codes).fillna(False).astype(bool)
    return pd.Series(False, index=series.index)


def outlier_bounds(values: pd.Series, method: str, threshold: float) -> tuple[float, float] | None:
    """Return ``(lower, upper)`` outlier bounds, or None if they cannot be estimated.

    ``iqr``: Tukey's fences, ``Q1 - t*IQR`` and ``Q3 + t*IQR``.
    ``mad``: modified z-score (Iglewicz & Hoaglin), ``|0.6745 * (x - median) / MAD| > t``.

    Both use rank-based statistics, so the outliers being searched for do not
    distort the bounds. A classic mean/standard-deviation z-score is
    deliberately not offered for that reason.
    """
    observed = values.dropna().astype("float64")
    if len(observed) < MIN_VALUES_FOR_OUTLIERS:
        return None
    if method == "iqr":
        q1, q3 = observed.quantile([0.25, 0.75])
        spread = q3 - q1
        if spread == 0:
            return None
        return float(q1 - threshold * spread), float(q3 + threshold * spread)
    if method == "mad":
        median = observed.median()
        mad = (observed - median).abs().median()
        if mad == 0:
            return None
        half_width = threshold * mad / 0.6745
        return float(median - half_width), float(median + half_width)
    raise ValueError(f"Unknown outlier method {method!r}; expected one of {OUTLIER_METHODS}")


def is_outlier_candidate(series: pd.Series) -> bool:
    return is_numeric(series) and series.nunique(dropna=True) >= MIN_UNIQUE_FOR_OUTLIERS


def guess_key(df: pd.DataFrame, min_unique_ratio: float = 0.9) -> str | None:
    """Guess an identifier column from its name and cardinality."""
    for column in df.columns:
        if not isinstance(column, str) or not _KEY_NAME.search(column):
            continue
        observed = df[column].dropna()
        if len(observed) and observed.nunique() / len(observed) >= min_unique_ratio:
            return column
    return None


def most_frequent(series: pd.Series) -> Any:
    modes = series.mode(dropna=True)
    return modes.iloc[0] if len(modes) else np.nan
