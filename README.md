# cleanplan

**Reviewable, rule-aware data cleaning for pandas.**

cleanplan finds data quality problems automatically, but it never fixes them
silently. It proposes a plan, you review and adjust it, and every change it
makes is logged and re-validated against your rules.

```python
import cleanplan as cp

plan = cp.inspect(df, rules=[cp.Range("age", min="years_experience + 18")])
print(plan)  # what was found, what would be done

plan.configure("salary", impute="median", by="department", outliers="set_missing")
result = plan.apply(df)

result.data  # the cleaned DataFrame
result.log.to_frame()  # every change: row, column, old, new, reason
result.validation  # your rules, re-checked on the cleaned data
```

## Why not just `dropna()` and `fillna(median)`?

Typical cleaning scripts get three things wrong, and they are easy to miss:

1. **Imputation creates impossible rows.** If you fill a missing age with the
   median (44) for someone with 32 years of experience, that person started
   working at 12. Nobody checks this, because the check you ran *before*
   imputing passed.
2. **Outliers get deleted with their whole row.** A salary of 425,000 in HR
   may be a typo, but the employee's age, department and performance score
   are still valid. Deleting the row throws them away and biases the data.
3. **Nobody knows afterwards what was changed.** Once a value is filled, it
   looks exactly like a real one.

cleanplan addresses each one:

| Problem | What cleanplan does |
|---|---|
| Impossible imputed values | Rules declare valid ranges, including ones that depend on other columns. Imputed values are clipped into range, and every rule is re-checked on the result. |
| Outliers | Detected with rank-based methods (IQR or MAD). The default is to **flag** them. You choose to blank and re-impute them, cap them, or drop them. |
| Invisible changes | A change log records every cell and row that changed, with the reason. `<column>_imputed` and `<column>_outlier` columns mark affected values in the data. |
| Wrong order of steps | Steps always run in the same order, so errors never feed the statistics used to fill the gaps. |
| Train/test leakage | `fit` on training data, `transform` new data, the same way scikit-learn works. |

## Installation

```bash
pip install cleanplan
```

Requires Python 3.10+ and pandas 2.0+ (pandas 3 is supported).

## Walkthrough

The data in [`examples/employees.py`](examples/employees.py) has exact
duplicates, gaps in three columns, `"N/A"` placeholders and three salaries
that are far too high.

### 1. Inspect

```python
plan = cp.inspect(
    df,
    rules=[
        cp.Range("age", min=18, max=70),
        cp.Range("age", min="years_experience + 18"),
        cp.Range("salary", min=0),
    ],
)
print(plan)
```

```text
CleaningPlan for 155 rows x 6 columns
  key: employee_id

Findings
  Exact duplicates   5 rows
  Key conflicts      none
  Missing values     age: 15 (9.7%)
                     salary: 8 (5.2%)
  Placeholders       department: 'N/A' x3
  Rule violations    none
  Outliers           salary: 3 outside [-39,837.4, 221,220] (IQR 1.5)

Plan
  Rows               drop exact duplicates (keep first)
                     key conflicts: flag (reported, not resolved)
  department         impute 'Unknown'
  age                impute median (44) | outliers: flag (IQR 1.5) | rule violations: set_missing
  salary             impute median (88,450) | outliers: flag (IQR 1.5) | rule violations: set_missing
  years_experience   impute median (11) | outliers: flag (IQR 1.5)
  performance_score  impute median (3.235) | outliers: flag (IQR 1.5)
  Indicator columns <column>_imputed / <column>_outlier mark every changed or flagged value.
```

`inspect` changes nothing. The key column was guessed from its name. You can
pass `key=` yourself, or `key=None` to turn key checks off.

### 2. Decide

The suggestions are deliberately conservative. Outliers are flagged but not
changed, and a missing category becomes `"Unknown"`, not the most common
value. Change what your domain knowledge says is wrong:

```python
# These salaries are entry errors: blank them and impute from the department.
plan.configure("salary", impute="median", by="department", outliers="set_missing")
```

### 3. Apply and review

```python
result = plan.apply(df)
print(result.log.summary())
```

```text
                 step     column                                                           reason  count
normalize_placeholder department                                  placeholder for a missing value      3
       drop_duplicate      (row)                                exact duplicate of an earlier row      5
              outlier     salary                           outside [-39,837.4, 221,220] (IQR 1.5)      3
               impute department                                    missing, filled with constant      3
               impute        age                                      missing, filled with median     15
               impute     salary                        missing, filled with median by department     11
       adjust_imputed        age imputed value adjusted to satisfy 'age >= years_experience + 18'      1
```

The last line is the point of the library. One employee's imputed age of 44
would have broken the experience rule, so it was raised to the lowest valid
value.

```python
result.validation.raise_if_failed()  # use this in pipelines
```

## Concepts

### Order of steps

1. Normalize placeholders (`""`, `"N/A"`, `"null"`, `-999`, ...) to missing.
2. Drop exact duplicates. Resolve rows that share a key but differ, if you
   asked for that.
3. Handle values that break a rule (default: blank them so they get imputed).
4. Handle outliers (default: flag them).
5. Impute missing values, then clip the imputed values into the ranges set by
   the rules.
6. Validate: re-check every rule, key uniqueness and remaining gaps.

Statistics such as medians and outlier bounds are learned after steps 1–4. A
value of 425,000 you have marked as an error therefore never shifts the
median used to replace it.

### Rules

```python
cp.Range("age", min=18, max=70)  # numeric bounds
cp.Range("years_experience", max="age - 18")  # bounds from other columns (pandas eval)
cp.Check("hire_date <= termination_date")  # any row-wise condition: reported only
cp.Check("discount <= price", column="discount")  # ...or blame a column to treat its values as invalid
cp.Unique("customer_id", "order_id")  # composite keys
```

A rule never reports a row whose inputs are missing. Missing values are
handled separately.

### Column policies

| Option | Values | Default suggestion |
|---|---|---|
| `impute` | `median`, `mean`, `mode`, `constant`, `none` | numeric: `median`, text/category: `constant "Unknown"`, bool: `mode` |
| `by` / `min_group_size` | group column / minimum observed values per group | none / 5 |
| `outliers` | `flag`, `set_missing`, `clip`, `drop_row`, `ignore` | `flag` for numeric columns with at least 10 distinct values |
| `outlier_method` / `outlier_threshold` | `iqr` (1.5) or `mad` (3.5) | `iqr` |
| `invalid` | `set_missing`, `flag`, `drop_row` | `set_missing` |

Columns that are more than 50% missing are not imputed by default, because
filling a mostly empty column invents data. Change this with
`inspect(max_missing=...)`.

A classic mean/standard-deviation z-score is intentionally not offered. The
outliers you are looking for inflate the mean and standard deviation, which
lets them hide.

### Plan-level options

```python
cp.inspect(
    df,
    key="customer_id",  # or a list, "auto" (default), or None
    key_conflicts="flag",  # "keep_first", "keep_last", "drop"
    drop_duplicates=True,
    indicators=True,  # add <column>_imputed / <column>_outlier
    restore_integers=True,  # int columns that became float only because of gaps go back to int
    string_sentinels={"", "n/a", "unknown"},
    numeric_sentinels={-1},
)
```

### Train / test

```python
plan = cp.inspect(train, rules=rules).fit(train)
clean_train = plan.transform(train).data
clean_test = plan.transform(test).data  # uses medians and bounds learned from train
```

Indicator columns are chosen from the fitted policies, so every transformed
frame has the same columns and dtypes. They also mark issues that first appear
in test data.

### Saving decisions

```python
import json

json.dump(plan.to_dict(), open("cleaning_plan.json", "w"), indent=2)
plan = cp.CleaningPlan.from_dict(json.load(open("cleaning_plan.json")))
```

Keep the JSON under version control next to your pipeline. It records how the
data was cleaned.

## Compared with other tools

Most of these tools do a different job, and several work well alongside
cleanplan. The table shows which tool fits which job.

| | cleanplan | pandera | Great Expectations | ydata-profiling | scikit-learn imputers | AutoClean | klib |
|---|---|---|---|---|---|---|---|
| **Main job** | Reviewable cleaning | Schema and data validation | Data validation | Profiling reports | Imputation in ML pipelines | Automated cleaning | Cleaning helpers and plots |
| **Finds problems before you write any rules** | ✅ | ❌ You define checks (`infer_schema` drafts a schema) | ❌ You define Expectations | ✅ | ❌ | ✅ | ✅ |
| **Changes the data** | ✅ | Partly: dtype coercion, custom parsers, dropping invalid rows | ❌ | ❌ | Missing values only | ✅ | Partly: drops duplicates and empty rows/columns, converts dtypes |
| **Imputes missing values** | Median, mean, mode, constant, optionally per group | Only via custom parsers | ❌ | ❌ | Mean, median, most frequent, constant, KNN, iterative | Mean, median, most frequent, regression, KNN | ❌ Drops them |
| **Handles outliers** | Flag, blank and re-impute, clip or drop (IQR, MAD) | Only via custom parsers | ❌ | ❌ | ❌ | Winsorize or drop (IQR) | ❌ |
| **Shows what will change before changing it** | ✅ | ❌ | — | — | Partly: learned values in `statistics_` | ❌ | ❌ Prints a summary afterwards |
| **Repairs respect your rules** (e.g. an imputed age stays consistent with experience) | ✅ | ❌ Parsed data is validated, not adjusted | — | — | ❌ | ❌ | ❌ |
| **Record of changes** | Every changed cell and removed row, with the reason | ❌ Dropped rows are not reported | — | — | Missing-value indicator columns | Step-level log file | Summary of changes |
| **Validates the result** | ✅ Re-checks rules, keys and gaps | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Learn on train data, apply to test data** | ✅ `fit` / `transform` | — | — | — | ✅ | ❌ | ❌ |
| **Data it works with** | pandas | pandas, Polars, PySpark, Dask, Modin, Ibis and more | pandas, Spark, SQL databases, files | pandas, Spark | NumPy, pandas, Polars | pandas | pandas |

— means the row does not apply, because the tool does not change data.

### When to use something else

- **Validating data in production**, against a schema contract, or on Polars
  or Spark: use pandera or Great Expectations. cleanplan only validates the
  rules you give it, and only on pandas.
- **Exploring a new dataset** with charts and a full report: use
  ydata-profiling or klib.
- **Model-based imputation** (KNN, iterative) inside a model pipeline: use
  scikit-learn. cleanplan only offers simple statistics for now.
- **Data that does not fit in memory**: cleanplan works on in-memory pandas
  DataFrames only.

A common setup: explore with ydata-profiling, clean with cleanplan, then
guard the pipeline with pandera.

<sub>Checked in September 2026 against pandera 0.33, Great Expectations
(GX Core) 1.23, ydata-profiling 4.18, scikit-learn 1.9, py-AutoClean 1.1.3
and klib 1.4.1. If something here is out of date, please
[open an issue](https://github.com/malperk/cleanplan/issues).</sub>

## Roadmap

- Command line: `cleanplan inspect data.csv`
- HTML report for notebooks
- Model-based imputation (KNN, iterative) that respects rules
- Datetime rules and imputation
- Polars support

Ideas and bug reports are welcome in the
[issue tracker](https://github.com/malperk/cleanplan/issues).

## Development

```bash
git clone https://github.com/malperk/cleanplan
cd cleanplan
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
