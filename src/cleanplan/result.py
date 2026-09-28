"""Outputs of applying a plan: the change log, the validation report and the result."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .plan import CleaningPlan

LOG_COLUMNS = ["step", "row", "column", "old", "new", "reason"]


@dataclass(frozen=True)
class _Entry:
    step: str
    reason: str
    rows: pd.Index
    column: Any = None
    old: np.ndarray | None = None
    new: np.ndarray | None = None


class ChangeLog:
    """Every change a plan made, one entry per changed cell or removed row.

    Entries are stored in batches, so logging stays cheap on large frames;
    :meth:`to_frame` expands them on demand.
    """

    def __init__(self) -> None:
        self._entries: list[_Entry] = []

    def record(
        self,
        step: str,
        reason: str,
        rows: pd.Index,
        column: Any = None,
        old: Any = None,
        new: Any = None,
    ) -> None:
        if len(rows) == 0:
            return
        old_values = None if old is None else np.asarray(old, dtype=object)
        if new is None:
            new_values = None
        elif np.ndim(new) == 0:
            new_values = np.full(len(rows), new, dtype=object)
        else:
            new_values = np.asarray(new, dtype=object)
        self._entries.append(_Entry(step, reason, pd.Index(rows), column, old_values, new_values))

    def __len__(self) -> int:
        return sum(len(entry.rows) for entry in self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)

    def to_frame(self) -> pd.DataFrame:
        """One row per change: ``step, row, column, old, new, reason``.

        ``row`` is the index label in the input frame; ``column`` is empty for
        row-level changes such as dropped duplicates.
        """
        if not self._entries:
            return pd.DataFrame(columns=LOG_COLUMNS)
        frames = []
        for entry in self._entries:
            n = len(entry.rows)
            empty = np.full(n, None, dtype=object)
            frames.append(
                pd.DataFrame(
                    {
                        "step": np.full(n, entry.step, dtype=object),
                        "row": np.asarray(entry.rows, dtype=object),
                        # A list, not np.full: column labels may be tuples.
                        "column": [entry.column] * n,
                        "old": empty if entry.old is None else entry.old,
                        "new": empty if entry.new is None else entry.new,
                        "reason": np.full(n, entry.reason, dtype=object),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    def summary(self) -> pd.DataFrame:
        """Number of changes per step, column and reason."""
        rows = [
            {
                "step": e.step,
                "column": "(row)" if e.column is None else e.column,
                "reason": e.reason,
                "count": len(e.rows),
            }
            for e in self._entries
        ]
        if not rows:
            return pd.DataFrame(columns=["step", "column", "reason", "count"])
        return (
            pd.DataFrame(rows)
            .groupby(["step", "column", "reason"], dropna=False, sort=False)["count"]
            .sum()
            .reset_index()
        )

    def __repr__(self) -> str:
        return f"ChangeLog({len(self)} changes)"


@dataclass(frozen=True)
class ValidationIssue:
    check: str
    rows: pd.Index
    detail: str = ""

    @property
    def count(self) -> int:
        return len(self.rows)

    def __str__(self) -> str:
        preview = ", ".join(map(str, list(self.rows[:5])))
        more = f", ... ({self.count} rows)" if self.count > 5 else ""
        detail = f" {self.detail}" if self.detail else ""
        return f"{self.check}:{detail} rows [{preview}{more}]"


class ValidationError(ValueError):
    """Raised by :meth:`ValidationReport.raise_if_failed`."""


@dataclass
class ValidationReport:
    """Checks run against the cleaned data.

    ``failures`` are broken rules, duplicate keys and missing values in columns
    that were supposed to be imputed. ``warnings`` are things worth a look,
    such as flagged outliers, that do not make the data wrong.
    """

    failures: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def raise_if_failed(self) -> None:
        if self.failures:
            raise ValidationError("Cleaned data failed validation:\n" + "\n".join(f"  - {f}" for f in self.failures))

    def __str__(self) -> str:
        if not self.failures and not self.warnings:
            return "Validation passed."
        lines = ["Validation passed." if self.ok else f"Validation failed ({len(self.failures)} issue(s)):"]
        lines += [f"  x {issue}" for issue in self.failures]
        lines += [f"  ! {issue}" for issue in self.warnings]
        return "\n".join(lines)

    __repr__ = __str__


@dataclass
class CleaningResult:
    """What :meth:`CleaningPlan.transform` returns."""

    data: pd.DataFrame
    log: ChangeLog
    validation: ValidationReport
    plan: CleaningPlan

    def __repr__(self) -> str:
        rows, columns = self.data.shape
        return f"CleaningResult({rows} rows x {columns} columns, {len(self.log)} changes)\n{self.validation}"
