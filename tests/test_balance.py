"""Баланс воды: закачивать нечего, кроме добытой воды.

Ограничение из разбора организаторов 12.08 (`docs/method/qa-organisers.md`):
дополнительной воды на объекте нет.
"""

from __future__ import annotations

import pytest
from conftest import make_row

from aios import balance
from aios.summary import Fluids

FLUIDS = Fluids()


def produced(data: str, well: str, oil_m3: float, water_m3: float):
    """Строка выгрузки по заданным объёмам: массы считаются теми же плотностями."""
    oil_t = oil_m3 * FLUIDS.oil_density / 1000
    liquid_t = oil_t + water_m3 * FLUIDS.water_density / 1000
    return make_row(data, well, wlpr=(oil_m3 + water_m3) / 30, womr=oil_t / 30,
                    liquid=liquid_t, oil=oil_t)


def injected(data: str, well: str, volume_m3: float):
    return make_row(data, well, wwir=volume_m3 / 30, injection=volume_m3)


def test_volumes_survive_the_round_trip_through_mass():
    """Выгрузка хранит массы; баланс переводит их обратно в объёмы тем же ключом."""
    rows = [produced("2014-01-01", "1", oil_m3=100.0, water_m3=900.0)]
    year = balance.by_year(rows, FLUIDS)[0]

    assert year.oil_m3 == pytest.approx(100.0)
    assert year.water_m3 == pytest.approx(900.0)
    assert year.watercut == pytest.approx(0.9)
    assert year.water_oil_ratio == pytest.approx(9.0)


def test_years_are_aggregated_separately():
    rows = [
        produced("2014-05-01", "1", 100.0, 400.0),
        produced("2015-05-01", "1", 80.0, 600.0),
    ]
    years = balance.by_year(rows, FLUIDS)

    assert [year.year for year in years] == [2014, 2015]
    assert years[1].water_oil_ratio > years[0].water_oil_ratio


def test_compensation_compares_injection_with_liquid():
    rows = [produced("2014-01-01", "1", 100.0, 900.0), injected("2014-01-01", "2", 500.0)]
    year = balance.by_year(rows, FLUIDS)[0]

    assert year.compensation == pytest.approx(0.5)
    assert year.water_reuse == pytest.approx(500.0 / 900.0)


def test_injection_within_produced_water_is_fine():
    rows = [produced("2014-01-01", "1", 100.0, 900.0), injected("2014-01-01", "2", 800.0)]
    assert balance.problems(rows, FLUIDS) == []


def test_injection_beyond_produced_water_is_flagged():
    rows = [produced("2014-01-01", "1", 100.0, 500.0), injected("2014-01-01", "2", 900.0)]
    found = balance.problems(rows, FLUIDS)

    assert len(found) == 1
    assert "дополнительной воды на объекте нет" in found[0]


def test_the_check_is_cumulative_not_monthly():
    """Воду можно копить и закачивать неравномерно — значение имеет накопленный итог."""
    rows = [
        produced("2014-01-01", "1", 100.0, 1000.0),
        injected("2014-01-01", "2", 100.0),
        produced("2015-01-01", "1", 100.0, 100.0),
        injected("2015-01-01", "2", 900.0),   # за год перебор, накопительно — нет
    ]
    assert balance.problems(rows, FLUIDS) == []


def test_injection_without_any_production_has_no_source():
    rows = [injected("2014-01-01", "2", 500.0)]
    found = balance.problems(rows, FLUIDS)

    assert len(found) == 1
    assert "источника закачки нет" in found[0]


def test_negative_rows_are_ignored_as_the_methodology_does():
    rows = [
        produced("2014-01-01", "1", 100.0, 900.0),
        make_row("2014-02-01", "1", liquid=-5.0, oil=1.0),
    ]
    year = balance.by_year(rows, FLUIDS)[0]
    assert year.water_m3 == pytest.approx(900.0)


def test_water_oil_ratio_ranks_the_worst_well():
    rows = [
        produced("2014-01-01", "clean", 100.0, 100.0),
        produced("2014-01-01", "watered", 10.0, 990.0),
    ]
    ratios = balance.well_water_oil_ratio(rows, FLUIDS)

    assert ratios["clean"] == pytest.approx(1.0)
    assert ratios["watered"] == pytest.approx(99.0)
    assert max(ratios, key=ratios.get) == "watered"


def test_a_well_that_produces_no_oil_has_infinite_ratio():
    rows = [produced("2014-01-01", "dead", 0.0, 500.0)]
    assert balance.well_water_oil_ratio(rows, FLUIDS)["dead"] == float("inf")
