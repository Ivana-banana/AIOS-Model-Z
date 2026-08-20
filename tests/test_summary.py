"""Выгрузка симулятора → 14 колонок расчётчика ЧДД.

Настоящего прогона на этой машине нет, поэтому проверяем на синтетическом
`.UNSMRY`, записанном той же библиотекой, которой читаем. Эталон раскладки —
экспортёр организаторов `vendor/Model_Z_OPM/USER/Model_Z/script_1.py`.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from aios import economics, summary


def _write_case(tmp_path, monthly_oil_m3, monthly_water_m3, monthly_injection_m3=None,
                oil_rate=None, liquid_rate=None, injection_rate=None,
                well="1", start=datetime(2014, 1, 1)):
    """Синтетический расчёт: одна скважина, месячные отчётные шаги.

    Накопленные задаются приращениями, мгновенные — значениями на шаге.
    Дебиты по умолчанию согласованы с приращениями (месяц 31 сутки).
    """
    from resdata.summary import Summary

    count = len(monthly_oil_m3)
    injections = monthly_injection_m3 or [0.0] * count
    oil_rate = oil_rate or [value / 31.0 for value in monthly_oil_m3]
    liquid_rate = liquid_rate or [
        (oil + water) / 31.0 for oil, water in zip(monthly_oil_m3, monthly_water_m3, strict=True)
    ]
    injection_rate = injection_rate or [value / 31.0 for value in injections]

    writer = Summary.writer(str(tmp_path / "CASE"), start, 10, 10, 3)
    for keyword, unit in (
        ("WOPT", "SM3"), ("WWPT", "SM3"), ("WWIT", "SM3"),
        ("WOPR", "SM3/DAY"), ("WLPR", "SM3/DAY"), ("WWIR", "SM3/DAY"),
        ("WBHP", "BARSA"), ("WBP9", "BARSA"),
    ):
        writer.add_variable(keyword, wgname=well, unit=unit)

    oil = water = injection = 0.0
    days = 0.0
    writer.add_t_step(0, sim_days=0.0)  # старт расчёта: всё по нулям

    for step in range(count):
        oil += monthly_oil_m3[step]
        water += monthly_water_m3[step]
        injection += injections[step]
        days += 31.0
        t_step = writer.add_t_step(step + 1, sim_days=days)
        t_step[f"WOPT:{well}"] = oil
        t_step[f"WWPT:{well}"] = water
        t_step[f"WWIT:{well}"] = injection
        t_step[f"WOPR:{well}"] = oil_rate[step]
        t_step[f"WLPR:{well}"] = liquid_rate[step]
        t_step[f"WWIR:{well}"] = injection_rate[step]
        t_step[f"WBHP:{well}"] = 80.0
        t_step[f"WBP9:{well}"] = 120.0
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
    # накопленные — тоже в массе и на конец месяца
    assert row["WLPT"] == pytest.approx(row["WLPT_Diff"])
    assert row["WOMT"] == pytest.approx(row["WOMT_Diff"])


def test_a_wells_own_pvt_region_overrides_the_field_density(tmp_path):
    """Плотность берётся по PVT-региону скважины, а не одна на весь фонд.

    В Модели Z два региона — 913,08 и 928,16 кг/м³, разница 1,6 % идёт прямо в
    выручку. Какая скважина в каком регионе, определяет `aios/pvt.py`.
    """
    case = _write_case(tmp_path, monthly_oil_m3=[100.0], monthly_water_m3=[900.0], well="1")
    field = summary.Fluids(oil_density=913.0765883224138, water_density=1172.0)
    own = summary.Fluids(oil_density=928.1613457381528, water_density=1173.0)

    rows = summary.monthly_rows(case, fluids=field, well_fluids={"1": own})

    assert rows[0]["WOMT_Diff"] == pytest.approx(100.0 * 928.1613457381528 / 1000)
    assert rows[0]["WLPT_Diff"] == pytest.approx(
        100.0 * 928.1613457381528 / 1000 + 900.0 * 1173.0 / 1000
    )


def test_a_well_without_a_region_keeps_the_field_density(tmp_path):
    """Скважина не нашлась в разборе дека — считаем по общей, а не по нулю."""
    case = _write_case(tmp_path, monthly_oil_m3=[100.0], monthly_water_m3=[900.0], well="1")
    field = summary.Fluids(oil_density=913.0765883224138, water_density=1172.0)

    rows = summary.monthly_rows(case, fluids=field, well_fluids={"другая": summary.Fluids(1.0, 1.0)})

    assert rows[0]["WOMT_Diff"] == pytest.approx(100.0 * 913.0765883224138 / 1000)


def test_rates_are_instantaneous_at_the_end_of_the_month(tmp_path):
    """Организаторы делают shift(-1): дебит в строке — значение на конец месяца.

    Средний за месяц дал бы здесь 1000/31 ≈ 32,3, а не 7.
    """
    case = _write_case(
        tmp_path,
        monthly_oil_m3=[100.0, 100.0],
        monthly_water_m3=[900.0, 900.0],
        liquid_rate=[7.0, 55.0],
        oil_rate=[1.0, 3.0],
    )
    rows = summary.monthly_rows(case)

    assert rows[0]["WLPR"] == pytest.approx(7.0)
    assert rows[1]["WLPR"] == pytest.approx(55.0)
    assert rows[0]["WOMR"] == pytest.approx(1.0 * 913.0765883224138 / 1000)


def test_pressures_follow_the_organisers_mapping(tmp_path):
    """THP — это пластовое давление (WBP9), BHP — забойное (WBHP)."""
    case = _write_case(tmp_path, monthly_oil_m3=[100.0], monthly_water_m3=[0.0])
    row = summary.monthly_rows(case)[0]
    assert row["THP"] == pytest.approx(120.0)
    assert row["BHP"] == pytest.approx(80.0)


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
    case = _write_case(
        tmp_path, monthly_oil_m3=[20000.0], monthly_water_m3=[0.0], liquid_rate=[645.0]
    )
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


def _sanity_row(data, *, well="1", wlpt=0.0, womt=0.0, wwit=0.0,
                wlpt_diff=0.0, womt_diff=0.0, wwit_diff=0.0,
                wlpr=0.0, wwir=0.0):
    """Строка контракта под проверки напрямую, без симулятора.

    Так собирает строки и суррогат: накопленные величины и приращения приходят
    независимо друг от друга, а не одной разностью, — поэтому рассогласовать их
    в тесте можно, и это не искусственный случай.
    """
    return {
        "DATA": data, "well": well,
        "WLPT": wlpt, "WLPR": wlpr, "WOMT": womt, "WOMR": 0.0,
        "WWIR": wwir, "WWIT": wwit, "THP": 120.0, "BHP": 80.0, "WEFF": 1.0,
        "WLPT_Diff": wlpt_diff, "WOMT_Diff": womt_diff, "WWIT_Diff": wwit_diff,
    }


def test_negative_increment_and_the_drop_it_explains_are_one_problem():
    """Отрицательный прирост и вызванное им уменьшение накопленного — один факт.

    Замер на опорном расписании Модели Z (20.08, машина В) давал их парами:
    2 400 строк «отрицательное WLPT_Diff» против 2 378 «накопленное WLPT
    уменьшилось» у тех же 28 скважин, 264 против 264 по WWIT у пяти. Разница в
    22 строки — первые месяцы, у которых нет предыдущего для сравнения.
    """
    rows = [
        _sanity_row("2014-01-01", wwit=100.0, wwit_diff=100.0),
        _sanity_row("2014-02-01", wwit=95.0, wwit_diff=-5.0),
    ]
    assert summary.sanity_problems(rows) == [
        "2014-02-01 скв. 1: отрицательное WWIT_Diff = -5"
    ]


def test_cumulative_drop_without_a_negative_increment_is_still_caught():
    """Рассогласованный ряд: накопленное поехало при неотрицательном приросте.

    На выгрузке симулятора так не бывает — там `*_Diff` и есть эта разность, —
    но прогноз суррогата собирается иначе. Свёртка дубликата не должна снимать
    саму проверку.
    """
    rows = [
        _sanity_row("2014-01-01", wlpt=100.0, wlpt_diff=100.0),
        _sanity_row("2014-02-01", wlpt=90.0, wlpt_diff=0.0),
    ]
    problems = summary.sanity_problems(rows)
    assert len(problems) == 1
    assert "накопленное WLPT уменьшилось" in problems[0]


def test_drop_deeper_than_the_increment_explains_is_a_separate_problem():
    """Прирост отрицательный, но уменьшения не объясняет — это две разные новости."""
    rows = [
        _sanity_row("2014-01-01", wwit=100.0, wwit_diff=100.0),
        _sanity_row("2014-02-01", wwit=90.0, wwit_diff=-1.0),
    ]
    problems = summary.sanity_problems(rows)
    assert len(problems) == 2
    assert any("отрицательное WWIT_Diff" in problem for problem in problems)
    assert any("накопленное WWIT уменьшилось" in problem for problem in problems)


def test_summary_section_lists_every_well_explicitly():
    section = summary.render_summary_section(["1", "2"])
    assert "WOPT\n '1'\n '2'\n/" in section
    assert "WWIT\n '1'\n '2'\n/" in section
    assert "WBP9\n '1'\n '2'\n/" in section
    assert "FOPT" in section
    # звёздочку не используем: молча недовыгруженный вектор — это потерянный прогон
    assert "*" not in section


def test_summary_section_covers_everything_the_organisers_export():
    """Их скрипт берёт wlpt, wlpr, womt, womr, wwir, wwit, wbp9, wbhp, weff.

    Массовых векторов (womt/womr) у OPM нет — они получаются из объёмных,
    поэтому в секции обязаны быть WOPT и WOPR.
    """
    section = summary.render_summary_section(["1"])
    for vector in ("WOPT", "WOPR", "WWPT", "WLPR", "WWIR", "WWIT", "WBHP", "WBP9"):
        assert f"\n{vector}\n" in f"\n{section}"
