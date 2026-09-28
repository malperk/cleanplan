import json

import numpy as np
import pandas as pd
import pytest

import cleanplan as cp
from cleanplan import CleaningPlan, ColumnPolicy, Range

AGE_RULES = [Range("age", min=18, max=70), Range("age", min="years_experience + 18")]


# --------------------------------------------------------------------- inspect


def test_inspect_changes_nothing(employees):
    before = employees.copy()
    cp.inspect(employees)
    pd.testing.assert_frame_equal(employees, before)


def test_inspect_findings(employees):
    plan = cp.inspect(employees, rules=AGE_RULES)
    f = plan.findings
    assert f.key == ("employee_id",)
    assert f.missing == {"department": 1, "age": 1, "salary": 1}
    assert f.exact_duplicates == 0
    assert list(f.outliers) == ["salary"]
    assert f.outliers["salary"].count == 1
    # Rules are checked on the raw data: nothing breaks them before imputation.
    assert set(f.rule_violations.values()) == {0}


def test_inspect_suggests_conservative_policies(employees):
    plan = cp.inspect(employees)
    assert plan.policies["employee_id"] == ColumnPolicy()
    assert plan.policies["department"].impute == "constant"
    assert plan.policies["department"].fill_value == "Unknown"
    assert plan.policies["age"].impute == "median"
    assert plan.policies["salary"].outliers == "flag"


def test_inspect_does_not_impute_mostly_empty_columns():
    df = pd.DataFrame({"x": [1.0, np.nan, np.nan, np.nan], "y": list("abcd")})
    plan = cp.inspect(df, key=None)
    assert plan.policies["x"].impute == "none"
    assert any("75% missing" in w for w in plan.findings.warnings)


def test_inspect_warns_without_rules_or_key():
    plan = cp.inspect(pd.DataFrame({"x": [1, 2, 3]}), key=None)
    text = " ".join(plan.findings.warnings)
    assert "No rules declared" in text
    assert "No key column" in text


def test_report_mentions_findings_and_plan(employees):
    text = cp.inspect(employees, rules=AGE_RULES).report()
    assert "Findings" in text and "Plan" in text
    assert "salary: 1 outside" in text
    assert "impute median (35)" in text
    assert repr(cp.inspect(employees)).startswith("CleaningPlan for 12 rows")


# ---------------------------------------------------------------- the pipeline


def test_imputed_values_respect_rules(employees):
    """The bug this library exists for: median age 35 for someone with 30 years of experience."""
    result = cp.inspect(employees, rules=AGE_RULES).apply(employees)
    assert result.data.loc[2, "age"] == 48
    assert result.validation.ok
    adjusted = result.log.to_frame().query("step == 'adjust_imputed'")
    assert adjusted[["row", "column", "old", "new"]].values.tolist() == [[2, "age", 35.0, 48.0]]


def test_without_rules_the_inconsistency_is_not_caught(employees):
    result = cp.inspect(employees).apply(employees)
    assert result.data.loc[2, "age"] == 35


def test_input_is_not_modified(employees):
    before = employees.copy()
    cp.inspect(employees, rules=AGE_RULES).apply(employees)
    pd.testing.assert_frame_equal(employees, before)


def test_integers_are_restored_after_imputation(employees):
    df = employees.assign(score=np.linspace(1, 12, 12))  # float, whole numbers, never missing
    data = cp.inspect(df).apply(df).data
    assert data["age"].dtype == "int64"  # float only because of the gap
    assert data["salary"].dtype == "int64"
    assert data["score"].dtype == "float64"  # genuinely float: left alone


def test_dtypes_do_not_depend_on_the_transformed_frame(employees):
    plan = cp.inspect(employees).fit(employees)
    complete = employees.dropna()
    assert plan.transform(complete).data.dtypes.to_dict() == plan.transform(employees).data.dtypes.to_dict()


def test_indicator_columns(employees):
    result = cp.inspect(employees).apply(employees)
    data = result.data
    assert data["age_imputed"].tolist() == [i == 2 for i in range(12)]
    assert data["salary_outlier"].sum() == 1 and data.loc[11, "salary_outlier"]
    assert data.loc[11, "salary"] == 400e3  # flag does not change the value
    assert any(w.check == "outliers" for w in result.validation.warnings)


def test_indicators_can_be_disabled(employees):
    data = cp.inspect(employees, indicators=False).apply(employees).data
    assert list(data.columns) == list(employees.columns)


def test_index_labels_are_preserved():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0, 1.0]}, index=["a", "b", "b", "c"])
    result = cp.inspect(df, key=None).apply(df)
    assert list(result.data.index) == ["a", "b", "b"]
    assert result.log.to_frame().set_index("step").loc["drop_duplicate", "row"] == "c"


# ----------------------------------------------------------------- duplicates


def test_exact_duplicates_are_dropped_and_logged(employees):
    df = pd.concat([employees, employees.iloc[[0, 5]]], ignore_index=True)
    result = cp.inspect(df).apply(df)
    assert len(result.data) == 12
    assert result.log.to_frame().query("step == 'drop_duplicate'")["row"].tolist() == [12, 13]


def test_key_conflicts_are_flagged_by_default(employees):
    df = employees.copy()
    df.loc[1, "employee_id"] = 1  # same id as row 0, different content
    plan = cp.inspect(df)
    assert plan.findings.key_conflicts == 2
    result = plan.apply(df)
    assert len(result.data) == 12
    assert not result.validation.ok
    with pytest.raises(cp.ValidationError, match="employee_id"):
        result.validation.raise_if_failed()


@pytest.mark.parametrize(("action", "rows"), [("keep_first", 11), ("keep_last", 11), ("drop", 10)])
def test_key_conflicts_can_be_resolved(employees, action, rows):
    df = employees.copy()
    df.loc[1, "employee_id"] = 1
    result = cp.inspect(df, key_conflicts=action).apply(df)
    assert len(result.data) == rows
    assert result.validation.ok


# ---------------------------------------------------------- placeholders, rules


def test_placeholders_become_missing_before_imputation():
    df = pd.DataFrame({"city": ["Paris", "N/A", "Rome", "?"], "score": [1.0, -999, 3.0, 4.0]})
    plan = cp.inspect(df, key=None)
    assert plan.findings.sentinels == {"city": {"N/A": 1, "?": 1}, "score": {-999.0: 1}}
    data = plan.apply(df).data
    assert data["city"].tolist() == ["Paris", "Unknown", "Rome", "Unknown"]
    assert data["score"].tolist() == [1, 3, 3, 4]


def test_invalid_values_are_blanked_and_reimputed(employees):
    df = employees.copy()
    df.loc[0, "age"] = 150
    result = cp.inspect(df, rules=AGE_RULES).apply(df)
    # 150 is blanked before the median is learned: median of the 10 valid ages is 36.5
    assert result.data.loc[0, "age"] == 36
    assert result.data.loc[0, "age_imputed"]
    assert result.validation.ok


def test_invalid_values_can_drop_rows(employees):
    df = employees.copy()
    df.loc[0, "age"] = 150
    plan = cp.inspect(df, rules=AGE_RULES).configure("age", invalid="drop_row")
    result = plan.apply(df)
    assert 0 not in result.data.index
    assert result.log.to_frame()["step"].tolist().count("drop_invalid_row") == 1


def test_flagged_invalid_values_fail_validation(employees):
    df = employees.copy()
    df.loc[0, "age"] = 150
    result = cp.inspect(df, rules=AGE_RULES).configure("age", invalid="flag").apply(df)
    assert result.data.loc[0, "age"] == 150
    assert [f.check for f in result.validation.failures] == ["rule 'age >= 18 and age <= 70'"]


def test_rules_without_a_column_are_validated_only(employees):
    plan = cp.inspect(employees, rules=[cp.Check("salary < 100000")])
    result = plan.apply(employees)
    assert result.data.loc[11, "salary"] == 400e3
    assert result.validation.failures[0].rows.tolist() == [11]


# ------------------------------------------------------------------- outliers


def test_outliers_set_missing_then_group_imputed(employees):
    plan = cp.inspect(employees).configure("salary", outliers="set_missing", impute="median", by="department")
    result = plan.apply(employees)
    # HR without the outlier: 70k..78k, median 74k (5 values = min_group_size).
    assert result.data.loc[11, "salary"] == 74e3
    # Sales has only 4 observed salaries, below min_group_size, so the overall median is used.
    overall = employees["salary"].drop(index=11).median()
    assert result.data.loc[3, "salary"] == overall


def test_small_groups_can_be_allowed(employees):
    plan = cp.inspect(employees).configure("salary", impute="median", by="department", min_group_size=4)
    assert plan.apply(employees).data.loc[3, "salary"] == 53e3


def test_missing_group_uses_overall_statistic():
    df = pd.DataFrame({"g": ["a"] * 5 + [None], "x": [1.0, 2, 3, 4, 5, np.nan]})
    plan = cp.inspect(df, key=None).configure("x", impute="mean", by="g", min_group_size=1)
    plan.policies["g"] = ColumnPolicy()
    assert plan.apply(df).data.loc[5, "x"] == 3


def test_outliers_clip(employees):
    result = cp.inspect(employees).configure("salary", outliers="clip").apply(employees)
    upper = result.plan.outlier_bounds_["salary"][1]
    assert result.data.loc[11, "salary"] == pytest.approx(upper)
    assert result.data.loc[11, "salary_outlier"]


def test_outliers_drop_row(employees):
    result = cp.inspect(employees).configure("salary", outliers="drop_row").apply(employees)
    assert 11 not in result.data.index
    assert "salary_outlier" not in result.data


def test_mad_method(employees):
    plan = cp.inspect(employees, outlier_method="mad")
    assert plan.findings.outliers["salary"].method == "mad"


# ------------------------------------------------------------------ fit / transform


def test_fit_on_train_transform_test(employees):
    train, test = employees.iloc[:10], employees.iloc[10:].copy()
    test.loc[11, "age"] = np.nan
    plan = cp.inspect(train, rules=AGE_RULES).fit(train)
    result = plan.transform(test)
    assert result.data.loc[11, "age"] == train["age"].median()
    # Same schema for every transformed frame.
    assert list(result.data.columns) == list(plan.transform(train).data.columns)


def test_transform_requires_fit():
    with pytest.raises(RuntimeError, match="not fitted"):
        CleaningPlan({"x": ColumnPolicy()}).transform(pd.DataFrame({"x": [1]}))


def test_configure_unfits_the_plan(employees):
    plan = cp.inspect(employees)
    assert plan.is_fitted
    plan.configure("age", impute="mean")
    assert not plan.is_fitted


def test_transform_checks_columns(employees):
    plan = cp.inspect(employees)
    with pytest.raises(KeyError, match="salary"):
        plan.transform(employees.drop(columns="salary"))


# ------------------------------------------------------------------ edge cases


def test_categorical_columns_get_new_category():
    df = pd.DataFrame({"i": [1, 2, 3, 4], "c": pd.Categorical(["a", "b", None, "a"])})
    data = cp.inspect(df, key=None).apply(df).data
    assert data["c"].tolist() == ["a", "b", "Unknown", "a"]
    assert isinstance(data["c"].dtype, pd.CategoricalDtype)


def test_mode_for_strings():
    df = pd.DataFrame({"i": [1, 2, 3, 4], "c": ["x", "y", "x", None]})
    plan = cp.inspect(df, key=None).configure("c", impute="mode")
    assert plan.apply(df).data["c"].tolist() == ["x", "y", "x", "x"]


def test_nullable_integer_columns():
    df = pd.DataFrame({"n": pd.array([1, 2, None, 4, 5], dtype="Int64")})
    data = cp.inspect(df, key=None).apply(df).data
    assert data["n"].tolist() == [1, 2, 3, 4, 5]


def test_all_missing_column_fails_validation():
    df = pd.DataFrame({"x": [np.nan, np.nan], "y": [1, 2]})
    plan = CleaningPlan({"x": ColumnPolicy(impute="median")})
    result = plan.apply(df)
    assert result.validation.failures[0].check == "missing values"


def test_mean_of_text_is_rejected():
    df = pd.DataFrame({"c": ["a", None]})
    with pytest.raises(TypeError, match="not numeric"):
        CleaningPlan({"c": ColumnPolicy(impute="mean")}).fit(df)


def test_text_constant_in_numeric_column_is_rejected():
    df = pd.DataFrame({"x": [1.0, np.nan]})
    with pytest.raises(TypeError, match="non-numeric"):
        CleaningPlan({"x": ColumnPolicy(impute="constant", fill_value="?")}).apply(df)


def test_indicator_name_clash():
    df = pd.DataFrame({"x": [1.0, np.nan], "x_imputed": [0, 1]})
    with pytest.raises(ValueError, match="already exist"):
        CleaningPlan({"x": ColumnPolicy(impute="median")}).fit(df)


# ---------------------------------------------------------------- configuration


def test_policy_validation():
    with pytest.raises(ValueError, match="impute must be one of"):
        ColumnPolicy(impute="magic")
    with pytest.raises(ValueError, match="fill_value"):
        ColumnPolicy(impute="constant")
    with pytest.raises(ValueError, match="'by'"):
        ColumnPolicy(impute="constant", fill_value=0, by="g")


def test_configure_rejects_typos(employees):
    plan = cp.inspect(employees)
    with pytest.raises(KeyError, match="salry"):
        plan.configure("salry", impute="mean")
    with pytest.raises(TypeError, match="imputer"):
        plan.configure("salary", imputer="mean")


def test_add_rule_type_check(employees):
    with pytest.raises(TypeError):
        cp.inspect(employees).add_rule("age > 0")


def test_plan_round_trips_through_json(employees):
    plan = cp.inspect(employees, rules=[*AGE_RULES, cp.Unique("employee_id")], key_conflicts="keep_first")
    plan.configure("salary", impute="median", by="department", outliers="clip")
    restored = CleaningPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
    assert restored.to_dict() == plan.to_dict()
    pd.testing.assert_frame_equal(restored.apply(employees).data, plan.apply(employees).data)


def test_change_log(employees):
    log = cp.inspect(employees).apply(employees).log
    frame = log.to_frame()
    assert list(frame.columns) == ["step", "row", "column", "old", "new", "reason"]
    assert len(frame) == len(log)
    assert set(log.summary()["step"]) == {"impute"}


def test_tuple_column_labels():
    df = pd.DataFrame({("a", "x"): [1.0, np.nan, 3.0], ("a", "y"): ["p", "q", "r"]})
    result = cp.inspect(df, key=None, indicators=False).apply(df)
    assert result.log.to_frame()["column"].tolist() == [("a", "x")]
