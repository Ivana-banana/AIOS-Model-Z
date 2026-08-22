"""Критические числа заявки не расходятся между сдаточными документами."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_declared_npv_is_the_same_in_all_current_documents():
    for name in ("docs/submission.md", "docs/report.md", "docs/defence-final.md"):
        assert "13 271,8" in _read(name), f"{name} потерял заявляемый ЧДД"


def test_submission_pairs_declared_npv_with_same_run_baseline_and_uplift():
    submission = _read("docs/submission.md")

    assert "13 271,8 млн руб." in submission
    assert "11 856,2 млн руб." in submission
    assert "+1 415,6 млн руб. (+11,9 %)" in submission


def test_defence_does_not_present_aggressive_uplift_as_the_declared_result():
    defence = _read("docs/defence-final.md")

    assert "Прирост заявляемого сценария | +1 415,6 млн руб. (+11,9 %)" in defence
    assert "Прирост самого доходного сценария | +1 789,5 млн руб. (+15,1 %)" in defence
