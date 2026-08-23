"""Доля закачки, идущей на пределе давления: чем меньше, тем больше числа наше."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "pressure_share", Path(__file__).resolve().parent.parent / "scripts" / "pressure_share.py")
pressure_share = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pressure_share)

LIMITS = {"1": 300.0, "2": 300.0}


def _rows(records: list[tuple[str, str, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        records, columns=["DATA", "well", "BHP", "WWIR", "WWIT_Diff"])


def test_volume_injected_at_the_limit_is_counted_apart_from_the_rest():
    rows = _rows([
        ("2007-01-01", "1", 300.0, 50.0, 1500.0),   # на пределе
        ("2007-02-01", "1", 210.0, 50.0, 1500.0),   # свободный режим
    ])

    result = pressure_share.share(rows, LIMITS, "2007-01-01", tolerance=1.0)

    assert result["volume_share"] == pytest.approx(50.0)
    assert result["month_share"] == pytest.approx(50.0)
    assert result["median_bhp"] == pytest.approx(255.0)


def test_the_tolerance_decides_what_counts_as_at_the_limit():
    """299,5 атм — «на пределе» при допуске 1 атм и «свободный режим» при 0,1."""
    rows = _rows([("2007-01-01", "1", 299.5, 50.0, 1500.0)])

    wide = pressure_share.share(rows, LIMITS, "2007-01-01", tolerance=1.0)
    strict = pressure_share.share(rows, LIMITS, "2007-01-01", tolerance=0.1)

    assert wide["volume_share"] == pytest.approx(100.0)
    assert strict["volume_share"] == pytest.approx(0.0)


def test_months_outside_the_window_and_idle_months_do_not_count():
    rows = _rows([
        ("2006-12-01", "1", 300.0, 50.0, 1500.0),   # до окна
        ("2007-01-01", "1", 300.0, 0.0, 0.0),       # скважина стоит
        ("2007-02-01", "1", 100.0, 50.0, 1500.0),
    ])

    result = pressure_share.share(rows, LIMITS, "2007-01-01", tolerance=1.0)

    assert result["months"] == 1
    assert result["volume_share"] == pytest.approx(0.0)


def test_a_well_without_a_limit_in_the_deck_is_left_out_rather_than_assumed():
    rows = _rows([
        ("2007-01-01", "1", 300.0, 50.0, 1000.0),
        ("2007-01-01", "9", 300.0, 50.0, 1000.0),   # предела в деке нет
    ])

    result = pressure_share.share(rows, LIMITS, "2007-01-01", tolerance=1.0)

    assert result["wells"] == 1
    assert result["volume"] == pytest.approx(1000.0)


def test_the_producer_limit_is_the_lower_one():
    """У добывающей предел снизу: 50 атм — это «на пределе», 120 — свободный режим."""
    rows = pd.DataFrame(
        [("2007-01-01", "1", 50.0, 0.0, 0.0, 30.0, 900.0),
         ("2007-02-01", "1", 120.0, 0.0, 0.0, 30.0, 900.0)],
        columns=["DATA", "well", "BHP", "WWIR", "WWIT_Diff", "WLPR", "WLPT_Diff"])

    result = pressure_share.share(rows, {"1": 50.0}, "2007-01-01",
                                  tolerance=1.0, kind="producer")

    assert result["volume_share"] == pytest.approx(50.0)


def test_the_same_row_is_not_at_the_limit_from_both_sides():
    """Верхний предел нагнетательной и нижний добывающей не путаются местами."""
    rows = pd.DataFrame(
        [("2007-01-01", "1", 120.0, 40.0, 1200.0, 30.0, 900.0)],
        columns=["DATA", "well", "BHP", "WWIR", "WWIT_Diff", "WLPR", "WLPT_Diff"])

    as_injector = pressure_share.share(rows, {"1": 300.0}, "2007-01-01", 1.0, "injector")
    as_producer = pressure_share.share(rows, {"1": 50.0}, "2007-01-01", 1.0, "producer")

    assert as_injector["volume_share"] == pytest.approx(0.0)
    assert as_producer["volume_share"] == pytest.approx(0.0)
