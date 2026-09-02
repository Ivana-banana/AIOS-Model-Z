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
                well="1", start=datetime(2014, 1, 1), vectors=None):
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
    all_vectors = (
        ("WOPT", "SM3"), ("WWPT", "SM3"), ("WWIT", "SM3"),
        ("WOPR", "SM3/DAY"), ("WLPR", "SM3/DAY"), ("WWIR", "SM3/DAY"),
        ("WBHP", "BARSA"), ("WBP9", "BARSA"),
    )
    requested = {keyword for keyword, _ in all_vectors} if vectors is None else set(vectors)
    for keyword, unit in all_vectors:
        if keyword in requested:
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
        values = {
            "WOPT": oil, "WWPT": water, "WWIT": injection,
            "WOPR": oil_rate[step], "WLPR": liquid_rate[step],
            "WWIR": injection_rate[step], "WBHP": 80.0, "WBP9": 120.0,
        }
        for keyword, value in values.items():
            if keyword in requested:
                t_step[f"{keyword}:{well}"] = value
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
                wlpr=0.0, wwir=0.0, thp=120.0):
    """Строка контракта под проверки напрямую, без симулятора.

    Так собирает строки и суррогат: накопленные величины и приращения приходят
    независимо друг от друга, а не одной разностью, — поэтому рассогласовать их
    в тесте можно, и это не искусственный случай.
    """
    return {
        "DATA": data, "well": well,
        "WLPT": wlpt, "WLPR": wlpr, "WOMT": womt, "WOMR": 0.0,
        "WWIR": wwir, "WWIT": wwit, "THP": thp, "BHP": 80.0, "WEFF": 1.0,
        "WLPT_Diff": wlpt_diff, "WOMT_Diff": womt_diff, "WWIT_Diff": wwit_diff,
    }


def test_reservoir_pressure_is_not_checked_without_limits():
    """Умолчание — «предел неизвестен»: выдуманная граница уронила бы прогон зря."""
    rows = [_sanity_row("2014-01-01", wlpr=10.0, thp=5.0),
            _sanity_row("2014-01-01", well="9", wwir=10.0, thp=900.0)]
    assert summary.sanity_problems(rows) == []


def test_reservoir_pressure_below_the_floor_is_reported_for_producers():
    rows = [_sanity_row("2014-01-01", wlpr=10.0, thp=45.0)]
    found = summary.sanity_problems(rows, min_reservoir_pressure=50.0)
    assert found == ["2014-01-01 скв. 1: пластовое давление 45 < 50 атм у добывающей"]


def test_reservoir_pressure_above_the_ceiling_is_reported_for_injectors():
    rows = [_sanity_row("2014-01-01", well="9", wwir=10.0, thp=310.0)]
    found = summary.sanity_problems(rows, max_reservoir_pressure=300.0)
    assert found == ["2014-01-01 скв. 9: пластовое давление 310 > 300 атм у нагнетательной"]


def test_each_bound_applies_to_its_own_role_only():
    """Нагнетательная не обязана держать нижнюю границу добывающей, и наоборот.

    Роль берётся так же, как действующий фонд, — по `WLPR` и `WWIR`: другого
    признака в контракте эталонного расчётчика нет.
    """
    rows = [_sanity_row("2014-01-01", wlpr=10.0, thp=310.0),
            _sanity_row("2014-01-01", well="9", wwir=10.0, thp=45.0)]
    assert summary.sanity_problems(rows, min_reservoir_pressure=50.0,
                                   max_reservoir_pressure=300.0) == []


def test_the_declared_scenario_keeps_reservoir_pressure_under_300():
    """Замер 01.09 на заявке: 291,8 атм максимум у нагнетательных, 300 не превышено.

    База даёт 282,2 — то есть заявка съела 9,6 атм запаса, и 17 строк подошли
    выше 290. Если кейс задаст верхнюю границу 300, заявка проходит; если ниже —
    нет. Замер закреплён здесь, чтобы это было числом, а не памятью.
    """
    rows = [_sanity_row("2014-01-01", well="9", wwir=10.0, thp=291.8)]
    assert summary.sanity_problems(rows, max_reservoir_pressure=300.0) == []
    assert summary.sanity_problems(rows, max_reservoir_pressure=290.0)


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


def test_an_export_without_our_summary_section_is_refused_not_read_as_zeros(tmp_path):
    """Выгрузка без скважинных векторов — это не «скважина ничего не дала».

    Файл SUMMARY в поставке организаторов приходит **пустым**: секцию
    пишем мы сами. Расчёт, к которому её не подключили, читается без единой
    ошибки — просто в нём нет ни одного вектора по скважинам. До правки это
    давало полную таблицу нулей, ноль замечаний санити-чека и ЧДД 0,000 млн руб.,
    поданный как обычный результат (замер 20.08). Ноль здесь опаснее отрицательного
    числа: скважина с нулевым дебитом не входит в действующий фонд, поэтому и OPEX
    ей не начисляется, и результат выглядит безобидно.

    Тот же отказ закрывает вторую дорогу к нулям: имена скважин, не совпавшие с
    именами в выгрузке.
    """
    from resdata.summary import Summary

    writer = Summary.writer(str(tmp_path / "CASE"), datetime(2014, 1, 1), 10, 10, 3)
    writer.add_variable("FOPT", unit="SM3")
    writer.add_t_step(0, sim_days=0.0)
    for step in range(3):
        writer.add_t_step(step + 1, sim_days=31.0 * (step + 1))["FOPT"] = 100.0 * (step + 1)
    writer.fwrite()

    with pytest.raises(ValueError, match="SUMMARY"):
        summary.monthly_rows(tmp_path / "CASE", wells=["1"])


def test_watercut_limit_lives_apart_from_the_physical_sanity_checks():
    """Обводнённость — условие кейса, а не физика, и считается отдельной функцией.

    Пункт 28 чек-листа приёмки. Смешение стоило бы дорого: `run_cycle --confirm`
    отказывается подтверждать ЧДД по непустым санити-чекам («выгрузка
    нефизична»), и предел обводнённости закрывал бы путь получения заявляемого
    числа ровно тогда, когда кейс его назовёт (ревизия 02.09).
    """
    from conftest import make_row

    rows = [
        make_row("2007-01-01", "сухая", wlpr=30.0, womr=20.0, liquid=1000.0, oil=400.0),
        make_row("2007-02-01", "мокрая", wlpr=30.0, womr=1.0, liquid=1000.0, oil=50.0),
    ]

    assert summary.sanity_problems(rows) == [], "физика этих строк в порядке"
    assert summary.case_problems(rows) == [], "без ключа условия нет"

    problems = summary.case_problems(rows, max_watercut=0.9)

    assert len(problems) == 1
    assert "мокрая" in problems[0] and "обводнённость" in problems[0]
    assert summary.sanity_problems(rows) == [], (
        "условие кейса не имеет права попасть в физический страж"
    )
def test_watercut_is_counted_by_mass_not_by_volume():
    """Обводнённость считается из тех же двух колонок, что и везде: по массе.

    Мутация «делить объёмы» проходит незаметно на близких плотностях, поэтому
    проверяется точное значение: 1 − нефть/жидкость в тоннах.
    """
    from conftest import make_row

    rows = [make_row("2007-01-01", "W", wlpr=30.0, womr=5.0, liquid=1000.0, oil=250.0)]

    assert summary.case_problems(rows, max_watercut=0.751) == []
    problems = summary.case_problems(rows, max_watercut=0.749)

    assert len(problems) == 1 and "0.75" in problems[0]


def test_an_export_with_only_part_of_the_well_vectors_is_refused(tmp_path):
    """Частичная SUMMARY-секция — не нулевая физика недостающих колонок.

    Проверка «нет ни одного вектора» защищала только полностью пустую секцию.
    Если в кейс попал один `WOPT`, остальные величины подставлялись нулями, и
    неполная выгрузка выглядела обычным, просто очень странным расчётом.
    """
    case = _write_case(
        tmp_path, monthly_oil_m3=[100.0], monthly_water_m3=[900.0], vectors=("WOPT",)
    )

    with pytest.raises(ValueError, match="WWPT"):
        summary.monthly_rows(case)
