import numpy as np
import pandas as pd
import pytest

from cleanplan import detect


def test_string_sentinels_ignore_case_and_whitespace():
    s = pd.Series(["ok", " N/A ", "null", "", "None", np.nan, "value"])
    mask = detect.sentinel_mask(s, detect.DEFAULT_STRING_SENTINELS, detect.DEFAULT_NUMERIC_SENTINELS)
    assert mask.tolist() == [False, True, True, True, True, False, False]


def test_numeric_sentinels():
    s = pd.Series([1.0, -999.0, np.nan, 5.0])
    mask = detect.sentinel_mask(s, detect.DEFAULT_STRING_SENTINELS, detect.DEFAULT_NUMERIC_SENTINELS)
    assert mask.tolist() == [False, True, False, False]


def test_sentinels_on_other_types_are_never_matched():
    s = pd.Series(pd.to_datetime(["2024-01-01", None]))
    assert not detect.sentinel_mask(s, {""}, {-999}).any()


def test_iqr_bounds():
    values = pd.Series(np.arange(1, 13, dtype=float))  # Q1 = 3.75, Q3 = 9.25
    lower, upper = detect.outlier_bounds(values, "iqr", 1.5)
    assert lower == pytest.approx(3.75 - 1.5 * 5.5)
    assert upper == pytest.approx(9.25 + 1.5 * 5.5)


def test_mad_bounds_ignore_the_outlier_itself():
    values = pd.Series([10.0, 11, 12, 13, 14, 15, 16, 17, 1000])
    lower, upper = detect.outlier_bounds(values, "mad", 3.5)
    assert lower < 10 and 17 < upper < 1000


def test_bounds_need_spread_and_enough_values():
    assert detect.outlier_bounds(pd.Series([5.0] * 20), "iqr", 1.5) is None
    assert detect.outlier_bounds(pd.Series([5.0] * 20), "mad", 3.5) is None
    assert detect.outlier_bounds(pd.Series([1.0, 2, 3]), "iqr", 1.5) is None


def test_unknown_outlier_method():
    with pytest.raises(ValueError, match="Unknown outlier method"):
        detect.outlier_bounds(pd.Series(np.arange(20.0)), "zscore", 3)


@pytest.mark.parametrize(
    ("values", "expected"),
    [([1.0, 2.0, np.nan], True), ([1.5, 2.0], False), ([1, 2], True), ([np.nan], False), (["a"], False)],
)
def test_is_integral(values, expected):
    assert detect.is_integral(pd.Series(values)) is expected


def test_guess_key():
    df = pd.DataFrame({"name": list("abcd"), "customer_id": [1, 2, 3, 4], "group_id": [1, 1, 2, 2]})
    assert detect.guess_key(df) == "customer_id"
    assert detect.guess_key(df[["name"]]) is None
