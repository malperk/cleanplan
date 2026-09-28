import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def employees() -> pd.DataFrame:
    """Small table with one of each problem.

    - row 2: age missing, 30 years of experience (median age 35 would break age >= exp + 18)
    - row 3: salary missing (Sales)
    - row 10: department missing
    - row 11: salary 400k, far outside the HR range
    """
    return pd.DataFrame(
        {
            "employee_id": list(range(1, 13)),
            "department": ["Sales"] * 5 + ["HR"] * 5 + [None, "HR"],
            "age": [25, 30, np.nan, 40, 45, 28, 33, 38, 43, 50, 35, 29],
            "salary": [50e3, 52e3, 54e3, np.nan, 58e3, 70e3, 72e3, 74e3, 76e3, 78e3, 60e3, 400e3],
            "years_experience": [2, 5, 30, 10, 20, 4, 8, 12, 16, 25, 9, 6],
        }
    )
