"""Declarative data rules.

Rules carry domain knowledge that cannot be inferred from the data itself,
such as "age must be at least years_experience + 18". cleanplan uses rules in
three places:

1. to find invalid values before any statistics are learned,
2. to keep imputed values consistent with the rest of the row, and
3. to validate the cleaned result.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pandas as pd

Bound = float | int | str | None

_IDENTIFIER = re.compile(r"`([^`]+)`|([A-Za-z_][A-Za-z0-9_]*)")


def referenced_columns(expr: str, columns: pd.Index) -> list[str]:
    """Return the columns of ``columns`` that appear in a pandas ``eval`` expression."""
    names = {quoted or bare for quoted, bare in _IDENTIFIER.findall(expr)}
    return [c for c in columns if c in names]


def _require_columns(df: pd.DataFrame, columns: list[str], rule: Rule) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Rule {rule.name!r} refers to missing column(s): {missing}")


def _evaluate(df: pd.DataFrame, expr: str, rule: Rule) -> Any:
    try:
        return df.eval(expr)
    except Exception as exc:  # pandas raises a wide range of errors from eval
        raise ValueError(f"Could not evaluate {expr!r} for rule {rule.name!r}: {exc}") from exc


class Rule:
    """Base class for rules. Subclasses implement :meth:`violations`."""

    kind: ClassVar[str]
    name: str
    #: The column held responsible when the rule fails. Invalid values in this
    #: column are handled according to its ``invalid`` policy. Rules without a
    #: column are only reported.
    column: str | None

    def violations(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean Series that is True for rows breaking the rule.

        Rows whose inputs are missing are not considered violations; missing
        values are handled separately.
        """
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Rule:
        data = dict(data)
        kind = data.pop("kind")
        if kind == Range.kind:
            return Range(**data)
        if kind == Check.kind:
            return Check(**data)
        if kind == Unique.kind:
            return Unique(*data["columns"], name=data.get("name", ""))
        raise ValueError(f"Unknown rule kind: {kind!r}")


@dataclass(frozen=True)
class Range(Rule):
    """``column`` must lie within ``[min, max]`` (inclusive).

    Bounds can be numbers or pandas ``eval`` expressions over other columns::

        Range("age", min=18, max=70)
        Range("years_experience", max="age - 18")

    When a value in ``column`` is imputed, cleanplan clips it into this range
    so that imputation never creates a violation.
    """

    column: str
    min: Bound = None
    max: Bound = None
    name: str = ""

    kind: ClassVar[str] = "range"

    def __post_init__(self) -> None:
        if self.min is None and self.max is None:
            raise ValueError("Range needs at least one of min or max")
        if not self.name:
            parts = []
            if self.min is not None:
                parts.append(f"{self.column} >= {self.min}")
            if self.max is not None:
                parts.append(f"{self.column} <= {self.max}")
            object.__setattr__(self, "name", " and ".join(parts))

    def bounds(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Evaluate the lower and upper bound for every row of ``df``."""
        return self._bound(df, self.min, -math.inf), self._bound(df, self.max, math.inf)

    def _bound(self, df: pd.DataFrame, bound: Bound, default: float) -> pd.Series:
        if bound is None:
            return pd.Series(default, index=df.index, dtype="float64")
        if isinstance(bound, str):
            value = _evaluate(df, bound, self)
            if isinstance(value, pd.Series):
                return pd.to_numeric(value, errors="coerce").astype("float64")
            return pd.Series(float(value), index=df.index, dtype="float64")
        return pd.Series(float(bound), index=df.index, dtype="float64")

    def violations(self, df: pd.DataFrame) -> pd.Series:
        _require_columns(df, [self.column], self)
        column = df[self.column]
        if not pd.api.types.is_numeric_dtype(column) or pd.api.types.is_bool_dtype(column):
            raise TypeError(f"Range rule {self.name!r} needs a numeric column, got {column.dtype}")
        values = column.astype("float64")
        lower, upper = self.bounds(df)
        # Comparisons against NaN are False, so rows with a missing value or a
        # missing bound input are never reported as violations.
        return ((values < lower) | (values > upper)).astype(bool)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "column": self.column, "min": self.min, "max": self.max, "name": self.name}


@dataclass(frozen=True)
class Check(Rule):
    """An arbitrary boolean pandas ``eval`` expression that must hold for every row.

    ``column`` optionally names the column to blame when the check fails; its
    values are then treated as invalid. Without it, failures are only reported::

        Check("hire_date <= termination_date")
        Check("discount <= price", column="discount")
    """

    expr: str
    column: str | None = None
    name: str = ""

    kind: ClassVar[str] = "check"

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(self, "name", self.expr)

    def violations(self, df: pd.DataFrame) -> pd.Series:
        if self.column is not None:
            _require_columns(df, [self.column], self)
        result = _evaluate(df, self.expr, self)
        if not isinstance(result, pd.Series):
            raise TypeError(f"Check {self.name!r} must evaluate to one boolean per row")
        passed = result.astype("boolean").fillna(True).astype(bool)
        inputs = referenced_columns(self.expr, df.columns)
        incomplete = df[inputs].isna().any(axis=1) if inputs else False
        return ~passed & ~incomplete

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "expr": self.expr, "column": self.column, "name": self.name}


class Unique(Rule):
    """The combination of ``columns`` must identify at most one row.

    Rows where any of the columns is missing are ignored.
    """

    kind: ClassVar[str] = "unique"
    column = None

    def __init__(self, *columns: str, name: str = "") -> None:
        if not columns:
            raise ValueError("Unique needs at least one column")
        self.columns = tuple(columns)
        self.name = name or f"unique({', '.join(self.columns)})"

    def violations(self, df: pd.DataFrame) -> pd.Series:
        _require_columns(df, list(self.columns), self)
        subset = df[list(self.columns)]
        complete = subset.notna().all(axis=1)
        return (subset.duplicated(keep=False) & complete).astype(bool)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "columns": list(self.columns), "name": self.name}

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Unique) and (self.columns, self.name) == (other.columns, other.name)

    def __hash__(self) -> int:
        return hash((self.kind, self.columns, self.name))

    def __repr__(self) -> str:
        return f"Unique({', '.join(map(repr, self.columns))})"


def integer_bounds(lower: pd.Series, upper: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Tighten bounds so that clipping an integer column keeps it integral."""
    return np.ceil(lower), np.floor(upper)
