import numpy as np
import pandas as pd
import pytest

from cleanplan import Check, Range, Rule, Unique


@pytest.fixture
def people() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [17, 30, np.nan, 40, 25],
            "experience": [0, 20, 5, np.nan, 10],
            "name": ["a", "b", "c", "", "e"],
        }
    )


def test_range_with_numeric_bounds(people):
    rule = Range("age", min=18, max=35)
    assert rule.violations(people).tolist() == [True, False, False, True, False]
    assert rule.name == "age >= 18 and age <= 35"


def test_range_with_expression_bound_ignores_missing_inputs(people):
    rule = Range("age", min="experience + 18")
    # row 2: age missing; row 3: experience missing
    assert rule.violations(people).tolist() == [True, True, False, False, True]


def test_range_needs_a_bound():
    with pytest.raises(ValueError, match="at least one"):
        Range("age")


def test_range_rejects_non_numeric_column(people):
    with pytest.raises(TypeError, match="numeric"):
        Range("name", min=0).violations(people)


def test_range_reports_missing_column(people):
    with pytest.raises(KeyError, match="salary"):
        Range("salary", min=0).violations(people)


def test_range_reports_bad_expression(people):
    with pytest.raises(ValueError, match="Could not evaluate"):
        Range("age", min="nope + 1").violations(people)


def test_check_skips_rows_with_missing_inputs(people):
    rule = Check("age >= experience + 18")
    assert rule.violations(people).tolist() == [True, True, False, False, True]
    assert rule.column is None


def test_check_on_strings(people):
    assert Check("name != ''", column="name").violations(people).tolist() == [False, False, False, True, False]


def test_check_must_return_rows(people):
    with pytest.raises(TypeError, match="one boolean per row"):
        Check("1 > 0").violations(people)


def test_unique_ignores_incomplete_keys():
    df = pd.DataFrame({"id": [1, 1, 2, np.nan, np.nan]})
    assert Unique("id").violations(df).tolist() == [True, True, False, False, False]


def test_unique_over_several_columns():
    df = pd.DataFrame({"a": [1, 1, 1], "b": ["x", "y", "x"]})
    assert Unique("a", "b").violations(df).tolist() == [True, False, True]


@pytest.mark.parametrize(
    "rule",
    [
        Range("age", min=18, max="experience + 60"),
        Check("age > 0", column="age", name="positive age"),
        Unique("id", "name"),
    ],
)
def test_rules_round_trip(rule):
    assert Rule.from_dict(rule.to_dict()) == rule


def test_unknown_rule_kind():
    with pytest.raises(ValueError, match="Unknown rule kind"):
        Rule.from_dict({"kind": "magic"})
