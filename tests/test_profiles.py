"""Профили по скважинам: числа отчёта должны воспроизводиться командой.

Разбор профилей 18.08 выполнил принцип 1 (посмотреть на данные), но не принцип 5
(повторяемость одной командой): числа ушли в отчёт, скрипта не осталось. Здесь
проверяется тот, что появился, и главное в нём — **база обводнённости названа**.
"""

from __future__ import annotations

import pytest
from conftest import make_row

from aios.summary import Fluids
from scripts.profiles import buckets, by_well, mass_watercut, volume_watercut

FLUIDS = Fluids()


def produced(data: str, well: str, oil_m3: float, water_m3: float, rate: float = 30.0):
    oil_t = oil_m3 * FLUIDS.oil_density / 1000
    liquid_t = oil_t + water_m3 * FLUIDS.water_density / 1000
    return make_row(data, well, wlpr=rate, womr=oil_t / 30, liquid=liquid_t, oil=oil_t)


def test_the_two_bases_differ_and_the_gap_is_worth_naming():
    """Объёмные 0,80 — это 0,84 по массе: порог сдвигается на проценты.

    Вода плотнее нефти (1172 против 913), поэтому её доля по массе всегда выше,
    чем по объёму. Пока база не названа, «выше 0,9» означает разные множества
    скважин.
    """
    row = produced("2014-01-01", "1", oil_m3=200.0, water_m3=800.0)
    oil_t, liquid_t = row["WOMT_Diff"], row["WLPT_Diff"]

    assert volume_watercut(oil_t, liquid_t, FLUIDS) == pytest.approx(0.8)
    assert mass_watercut(oil_t, liquid_t) == pytest.approx(0.837, abs=5e-4)


def test_watercut_of_a_dry_well_is_zero_not_undefined():
    assert mass_watercut(0.0, 0.0) == 0.0
    assert volume_watercut(0.0, 0.0, FLUIDS) == 0.0


def test_the_tail_is_measured_separately_from_the_whole_horizon():
    """Средний по горизонту показатель скрывает картину на конец — это и была находка."""
    rows = (
        [produced(f"2013-{month:02d}-01", "1", oil_m3=900.0, water_m3=100.0)
         for month in range(1, 13)]
        + [produced(f"2014-{month:02d}-01", "1", oil_m3=50.0, water_m3=950.0)
           for month in range(1, 13)]
    )

    state = by_well(rows, months=12, fluids=FLUIDS)["1"]

    assert state["wc_mass_all"] < 0.6, "за весь горизонт скважина выглядит сухой"
    assert state["wc_mass_tail"] > 0.9, "а на конец горизонта — обводнённой"


def test_negative_rows_are_skipped_as_the_methodology_requires():
    """Строку с отрицательным приростом методика исключает целиком."""
    good = produced("2014-01-01", "1", oil_m3=500.0, water_m3=500.0)
    bad = make_row("2014-02-01", "1", wlpr=30.0, womr=-1.0, liquid=-10.0, oil=-1.0)

    state = by_well([good, bad], months=0, fluids=FLUIDS)["1"]

    assert state["oil_all"] == pytest.approx(good["WOMT_Diff"])
    assert state["liquid_all"] == pytest.approx(good["WLPT_Diff"])


def test_buckets_cover_the_whole_range_without_gaps():
    counts = dict(buckets([0.05, 0.5, 0.75, 0.85, 0.95, 1.0]))
    assert sum(counts.values()) == 6, "ни одно значение не должно потеряться"
