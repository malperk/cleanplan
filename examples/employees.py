"""End-to-end example: cleaning a messy employee table.

The data has exact duplicates, missing values in three columns and three
salaries that are far too high. Run with ``python examples/employees.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import cleanplan as cp


def make_employees(n: int = 150, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(22, 65, n)
    df = pd.DataFrame(
        {
            "employee_id": np.arange(1001, 1001 + n),
            "department": rng.choice(["Sales", "Engineering", "Marketing", "HR", "Finance"], n),
            "age": age.astype(float),
            "salary": rng.integers(35_000, 150_000, n).astype(float),
            # Experience never exceeds age - 19 in the raw data.
            "years_experience": [rng.integers(0, max(1, a - 18)) for a in age],
            "performance_score": rng.uniform(1.0, 5.0, n).round(2),
        }
    )
    df.loc[rng.choice(n, 15, replace=False), "age"] = np.nan
    df.loc[rng.choice(n, 8, replace=False), "salary"] = np.nan
    df.loc[rng.choice(n, 3, replace=False), "department"] = "N/A"
    df.loc[rng.choice(n, 3, replace=False), "salary"] = [350_000, 425_000, 380_000]
    df = pd.concat([df, df.sample(5, random_state=seed)], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    df = make_employees()

    # 1. Inspect: nothing is changed yet. Rules are the domain knowledge the
    #    data cannot reveal on its own.
    plan = cp.inspect(
        df,
        rules=[
            cp.Range("age", min=18, max=70),
            cp.Range("age", min="years_experience + 18"),
            cp.Range("salary", min=0),
        ],
    )
    print(plan)

    # 2. Decide. Salaries above the bounds are treated as entry errors: blank
    #    them and impute from the employee's department instead of dropping
    #    the whole row.
    plan.configure("salary", impute="median", by="department", outliers="set_missing")

    # 3. Apply and look at what happened.
    result = plan.apply(df)
    print("\n" + repr(result))
    print("\nChanges by step:")
    print(result.log.summary().to_string(index=False))
    print("\nCleaned data:")
    print(result.data.head().to_string())

    result.validation.raise_if_failed()


if __name__ == "__main__":
    main()
