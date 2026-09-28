import runpy
from pathlib import Path

EXAMPLES = Path(__file__).parent.parent / "examples"


def test_employees_example_runs(capsys):
    runpy.run_path(str(EXAMPLES / "employees.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "Validation passed." in out
    assert "adjust_imputed" in out
