"""cleanplan: reviewable, rule-aware data cleaning for pandas.

cleanplan detects data quality issues automatically but never decides for
you silently. It proposes a plan, you review and adjust it, and every change
it makes is logged and validated::

    import cleanplan as cp

    plan = cp.inspect(df, rules=[cp.Range("age", min=18, max=70)])
    print(plan)                                   # findings + proposed actions
    plan.configure("salary", impute="median", by="department")
    result = plan.apply(df)

    result.data            # cleaned DataFrame
    result.log.to_frame()  # every change: row, column, old, new, reason
    result.validation      # rules re-checked on the cleaned data
"""

from .plan import (
    CleaningPlan,
    ColumnPolicy,
    Findings,
    FittedFill,
    OutlierFinding,
    inspect,
)
from .result import ChangeLog, CleaningResult, ValidationError, ValidationIssue, ValidationReport
from .rules import Check, Range, Rule, Unique

__version__ = "0.1.0"

__all__ = [
    "ChangeLog",
    "Check",
    "CleaningPlan",
    "CleaningResult",
    "ColumnPolicy",
    "Findings",
    "FittedFill",
    "OutlierFinding",
    "Range",
    "Rule",
    "Unique",
    "ValidationError",
    "ValidationIssue",
    "ValidationReport",
    "__version__",
    "inspect",
]
