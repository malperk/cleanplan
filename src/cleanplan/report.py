"""Plain-text rendering of a plan, readable in a terminal and in notebooks."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .plan import CleaningPlan

LABEL_WIDTH = 18
MAX_COLUMN_WIDTH = 24
MAX_POLICY_LINES = 40


def _number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.6g}"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    return repr(value)


def _section(lines: list[str], label: str, items: list[str], width: int = LABEL_WIDTH) -> None:
    items = items or ["none"]
    lines.append(f"  {label:<{width}}{items[0]}")
    lines.extend(f"  {'':<{width}}{item}" for item in items[1:])


def _describe_column(plan: CleaningPlan, column: Any) -> str:
    policy = plan.policies[column]
    text = policy.describe()
    if plan.is_fitted and column in getattr(plan, "fill_values_", {}):
        fill = plan.fill_values_[column]
        learned = _number(fill.overall)
        if policy.by is not None:
            learned = f"{len(fill.groups)} groups, fallback {learned}"
        head, sep, tail = text.partition(" | ")
        text = f"{head} ({learned}){sep}{tail}"
    blamed = [rule.name for rule in plan.rules if rule.column == column]
    if blamed:
        text += f" | rule violations: {policy.invalid}"
    return text


def format_report(plan: CleaningPlan) -> str:
    findings = plan.findings
    lines: list[str] = []
    active = [
        c for c, p in plan.policies.items() if p.describe() != "no action" or any(r.column == c for r in plan.rules)
    ]
    longest = min(max((len(str(c)) for c in active), default=0), MAX_COLUMN_WIDTH)
    width = max(LABEL_WIDTH, longest + 2)
    if findings is not None:
        lines.append(f"CleaningPlan for {findings.n_rows:,} rows x {findings.n_columns} columns")
    else:
        lines.append("CleaningPlan")
    lines.append(f"  key: {', '.join(map(str, plan.key)) if plan.key else 'none'}")

    if findings is not None:
        lines += ["", "Findings"]
        n = max(findings.n_rows, 1)
        _section(
            lines,
            "Exact duplicates",
            [f"{findings.exact_duplicates:,} rows"] if findings.exact_duplicates else [],
            width,
        )
        if plan.key:
            conflicts = [f"{findings.key_conflicts:,} rows share a key but differ"] if findings.key_conflicts else []
            _section(lines, "Key conflicts", conflicts, width)
        _section(
            lines,
            "Missing values",
            [f"{c}: {k:,} ({k / n:.1%})" for c, k in sorted(findings.missing.items(), key=lambda i: -i[1])],
            width,
        )
        _section(
            lines,
            "Placeholders",
            [
                f"{c}: " + ", ".join(f"{v!r} x{k}" for v, k in values.items())
                for c, values in findings.sentinels.items()
            ],
            width,
        )
        if plan.rules:
            _section(
                lines,
                "Rule violations",
                [f"{name}: {k:,} rows" for name, k in findings.rule_violations.items() if k],
                width,
            )
        else:
            _section(lines, "Rule violations", ["no rules declared"], width)
        _section(
            lines,
            "Outliers",
            [
                f"{c}: {o.count:,} outside [{_number(o.lower)}, {_number(o.upper)}] "
                f"({o.method.upper()} {o.threshold:g})"
                for c, o in findings.outliers.items()
            ],
            width,
        )

    lines += ["", "Plan"]
    row_actions = []
    if plan.drop_duplicates:
        row_actions.append("drop exact duplicates (keep first)")
    if plan.key:
        conflict = "flag (reported, not resolved)" if plan.key_conflicts == "flag" else plan.key_conflicts
        row_actions.append(f"key conflicts: {conflict}")
    _section(lines, "Rows", row_actions or ["no row-level actions"], width)

    for column in active[:MAX_POLICY_LINES]:
        name = str(column)
        name = name if len(name) <= MAX_COLUMN_WIDTH else name[: MAX_COLUMN_WIDTH - 1] + "~"
        lines.append(f"  {name:<{width}}{_describe_column(plan, column)}")
    if len(active) > MAX_POLICY_LINES:
        lines.append(f"  ... {len(active) - MAX_POLICY_LINES} more (see plan.policies)")
    if plan.indicators:
        lines.append("  Indicator columns <column>_imputed / <column>_outlier mark every changed or flagged value.")

    if findings is not None and findings.warnings:
        lines += ["", "Warnings"]
        for warning in findings.warnings:
            wrapped = textwrap.wrap(warning, width=96, initial_indent="  - ", subsequent_indent="    ")
            lines.extend(wrapped)

    lines += [
        "",
        "Next: adjust with plan.configure(column, ...) and plan.add_rule(...), then result = plan.apply(df).",
    ]
    return "\n".join(lines)
