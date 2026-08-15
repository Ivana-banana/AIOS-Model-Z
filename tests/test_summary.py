"""Выгрузка симулятора → 14 колонок расчётчика ЧДД.

Настоящего прогона на этой машине нет, поэтому проверяем на синтетическом
`.UNSMRY`, записанном той же библиотекой, которой читаем.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from aios import economics, summary


def _write_case(tmp_path, monthly_oil_m3, monthly_water_m3, monthly_injection_m3=None,
                well="1", start=datetime(2014, 1, 1)):
    """Синтетический расчёт: 1 скважина, месячные отчётные шаги."""
    from resdata.summary import Summary

    writer = Summary.writer(str(tmp_path / "CASE"), start, 10, 10, 3)
    writer.add_variable("WOPT", wgname=well, unit="SM3")
    writer.add_variable("WWPT", wgname=well, unit="SM3")
    writer.add_variable("WWIT", wgname=well, unit="SM3")
    writer.add_variable("WBHP", wgname=well, unit="BARSA")

    oil = water = injection = 0.0
    days = 0.0
    writer.add_t_step(0, sim_days=0.0)  # старт расчёта: всё по нулям

    injections = monthly_injection_m3 or [0.0] * len(monthly_oil_m3)
    for step, (delta_oil, delta_water, delta_injection) in enumerate(
        zip(monthly_oil_m3, monthly_water_m3, injections, strict=True), start=1
    ):
        oil += delta_oil
        water += delta_water
        injection += delta_injection
        days += 31.0
        t_step = writer.add_t_step(step, sim_days=days)
        t_step[f"WOPT:{well}"] = oil
        t_step[f"WWPT:{well}"] = water
        t_step[f"WWIT:{well}"] = injection
        t_step[f"WBHP:{well}"] = 80.0
    writer.fwrite()
    return tmp_path / "CASE"


def test_volumes_are_converted_to_mass_by_deck_densities(tmp_path):
    case = _write_case(tmp_path, monthly_oil_m3=[100.0], monthly_water_m3=[900.0])
    fluids = summary.Fluids(oil_density=913.0765883224138, water_density=1172.0)

    rows = summary.monthly_rows(case, fluids=fluids)

    assert len(rows) == 1
    row = rows[0]
    assert row["WOMT_Diff"] == pytest.approx(100.0 * 913.0765883224138 / 1000)
    assert row["WLPT_Diff"] == pytest.approx(
        100.0 * 913.0765883224138 / 1000 + 900.0 * 1172.0 / 1000
    )
    # дебит — объёмный и средний за месяц: методика подбирает ЭЦН под режим месяца
    assert row["WLPR"] == pytest.approx(1000.0 / 31.0)


def test_month_is_labelled_by_its_start(tmp_path):
    """Приращение между 01 JAN и 01 FEB — это январь."""
    case = _write_case(tmp_path, monthly_oil_m3=[10.0, 20.0], monthly_water_m3=[0.0, 0.0])
    rows = summary.monthly_rows(case)
    assert [row["DATA"] for row in rows] == ["2014-01-01", "2014-02-01"]
    assert rows[0]["WOMT_Diff"] < rows[1]["WOMT_Diff"]


def test_rows_match_the_calculator_contract(tmp_path):
    case = _write_case(tmp_path, monthly_oil_m3=[100.0, 120.0], monthly_water_m3=[50.0, 80.0])
    rows = summary.monthly_rows(case)
    assert list(rows[0].keys()) == economics.REQUIRED_COLUMNS


def test_rows_feed_the_reference_calculator_end_to_end(tmp_path):
    """Сквозной контур без симулятора: выгрузка → ЧДД считается и он положительный."""
    case = _write_case(
        tmp_path,
        monthly_oil_m3=[300.0] * 12,
        monthly_water_m3=[200.0] * 12,
    )
    rows = summary.monthly_rows(case)
    assert summary.sanity_problems(rows) == []

    result = economics.compute_npv(
        rows, headers=economics.REQUIRED_COLUMNS, start_date="2014-01-01"
    )
    assert result.chdd_m > 0
    assert result.raw["annual"][0]["activeWellMonths"] == pytest.approx(12.0)


def test_negative_increments_are_not_silently_clamped(tmp_path):
    """Отрицательное приращение доходит до расчётчика — он исключит строку целиком.

    Обнулить его здесь означало бы завысить свой ЧДД против пересчёта организаторов.
    """
    case = _write_case(tmp_path, monthly_oil_m3=[100.0, -30.0], monthly_water_m3=[0.0, 0.0])
    rows = summary.monthly_rows(case)

    assert rows[1]["WOMT_Diff"] < 0
    assert any("отрицательное" in problem for problem in summary.sanity_problems(rows))

    result = economics.compute_npv(
        rows, headers=economics.REQUIRED_COLUMNS, start_date="2014-01-01"
    )
    assert result.excluded_negative_rows == 1


def test_sanity_catches_rate_above_the_methodology_limit(tmp_path):
    case = _write_case(tmp_path, monthly_oil_m3=[20000.0], monthly_water_m3=[0.0])
    rows = summary.monthly_rows(case)
    assert any("500" in problem for problem in summary.sanity_problems(rows))


def test_injection_only_well_is_reported(tmp_path):
    case = _write_case(
        tmp_path,
        monthly_oil_m3=[0.0, 0.0],
        monthly_water_m3=[0.0, 0.0],
        monthly_injection_m3=[3100.0, 3100.0],
    )
    rows = summary.monthly_rows(case)
    assert rows[0]["WWIT_Diff"] == pytest.approx(3100.0)
    assert rows[0]["WWIR"] == pytest.approx(100.0)
    assert rows[1]["WWIT"] == pytest.approx(6200.0)
    assert summary.sanity_problems(rows) == []


def test_summary_section_lists_every_well_explicitly():
    section = summary.render_summary_section(["1", "2"])
    assert "WOPT\n '1'\n '2'\n/" in section
    assert "WWIT\n '1'\n '2'\n/" in section
    assert "FOPT" in section
    # звёздочку не используем: молча недовыгруженный вектор — это потерянный прогон
    assert "*" not in section
