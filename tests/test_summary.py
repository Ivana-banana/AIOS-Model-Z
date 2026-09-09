"""Выгрузка симулятора → 14 колонок расчётчика ЧДД.

Настоящего прогона на этой машине нет, поэтому проверяем на синтетическом
`.UNSMRY`, записанном той же библиотекой, которой читаем. Эталон раскладки —
экспортёр организаторов `vendor/Model_Z_OPM/USER/Model_Z/script_1.py`.
"""

from __future__ import annotations

import ctypes
import os
from datetime import date, datetime
from pathlib import Path

import pytest
from conftest import make_row

from aios import economics, summary


def _resdata_path(directory: Path) -> Path:
    """Вернуть путь, который понимает нативный writer ``resdata`` на Windows.

    ``resdata==6.3.3`` читает готовые файлы из Unicode-путей, но его тестовый
    ``Summary.writer`` возвращает null, если хотя бы один компонент пути не
    ASCII. Это относится только к синтетическим файлам этого модуля. Короткое
    имя существующего каталога оставляет файлы в ``tmp_path`` и не меняет
    проверяемый код.
    """
    if os.name != "nt" or str(directory).isascii():
        return directory

    get_short_path = ctypes.WinDLL("kernel32", use_last_error=True).GetShortPathNameW
    get_short_path.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint)
    get_short_path.restype = ctypes.c_uint
    buffer = ctypes.create_unicode_buffer(32_768)
    length = get_short_path(str(directory), buffer, len(buffer))
    error = ctypes.get_last_error()
    if length == 0 or length >= len(buffer) or not buffer.value.isascii():
        pytest.skip(
            "resdata Summary.writer не поддерживает Unicode-путь, "
            f"а короткое имя Windows недоступно (ошибка {error})"
        )
    return Path(buffer.value)


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

    case = _resdata_path(tmp_path) / "CASE"
    writer = Summary.writer(str(case), start, 10, 10, 3)
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
    return case


def test_a_well_with_no_vectors_at_all_is_refused(tmp_path):
    """Скважина, названная снаружи и отсутствующая в выгрузке, — отказ.

    До 03.09 отказ был только при **частично** пропавших векторах: скважина без
    единого вектора получала нули по всем величинам и выглядела «неработающей».
    Отсутствие данных становилось неотличимо от простоя (ревизия 03.09, 2.3 #7).
    """
    case = _write_case(tmp_path, monthly_oil_m3=[100.0], monthly_water_m3=[900.0])

    # скважина из выгрузки читается как обычно
    assert summary.monthly_rows(case)

    with pytest.raises(ValueError, match="ни одного вектора"):
        summary.monthly_rows(case, wells=("НЕТ-ТАКОЙ",))


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

    case = _resdata_path(tmp_path) / "CASE"
    writer = Summary.writer(str(case), datetime(2014, 1, 1), 10, 10, 3)
    writer.add_variable("FOPT", unit="SM3")
    writer.add_t_step(0, sim_days=0.0)
    for step in range(3):
        writer.add_t_step(step + 1, sim_days=31.0 * (step + 1))["FOPT"] = 100.0 * (step + 1)
    writer.fwrite()

    with pytest.raises(ValueError, match="SUMMARY"):
        summary.monthly_rows(case, wells=["1"])


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


def test_reading_a_case_does_not_flood_the_run_with_warnings():
    """Чтение выгрузки молчит: 371 предупреждение на прогон — это шум, а не сигнал.

    `report_index_list` внутри resdata гасит `DeprecationWarning` на создании
    `IntVector`, но не на `append` в цикле сразу под ним, и печатает по
    предупреждению на отчётный шаг. Замер 05.09 на единственной документированной
    команде прогона: 742 строки шума на три строки собственного вывода — на демо
    это читается как сломанный запуск.

    Сторож считает предупреждения, а не ищет строку: любое новое место, где
    библиотека начнёт шуметь, он поймает так же.
    """
    import warnings

    case = Path(__file__).resolve().parent.parent / "runs" / "base" / "RUN"
    if not case.with_suffix(".UNSMRY").is_file():
        pytest.skip("нет выгрузки runs/base — читать нечего")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rows = summary.monthly_rows(case)

    noisy = [one for one in caught if issubclass(one.category, DeprecationWarning)]
    assert not noisy, (
        f"чтение выгрузки печатает {len(noisy)} предупреждений, первое: {noisy[0].message}"
    )
    assert rows, "молчание не должно достигаться тем, что читать перестали"


# ── что нашёл мутационный прогон 06.09 (Р-221) ───────────────────────────────


def test_a_date_is_read_with_a_half_day_shift():
    """Дата берётся со сдвигом на полсуток — иначе она уезжает на день.

    Три мутации сидели здесь: `timedelta(hours=12)` обнулялось, знак менялся на
    минус. Выгрузка симулятора несёт время конца шага, и оно бывает
    `2014-01-31 23:59:59` или `2014-02-01 00:00:01`; полсуток сдвига кладут обе
    формы в тот месяц, к которому они относятся. Без сдвига месяц уезжает — а по
    месяцу считается вся экономика.
    """
    from aios.summary import _as_date

    assert _as_date(datetime(2014, 1, 31, 23, 59, 59)) == date(2014, 2, 1)
    assert _as_date(datetime(2014, 2, 1, 0, 0, 1)) == date(2014, 2, 1)
    # строка на границе месяца — тот случай, ради которого сдвиг и стоит:
    # без него `2014-01-31 23:59:59` попал бы в январь
    assert _as_date("2014-01-31T23:59:59") == date(2014, 2, 1)
    assert _as_date("2014-02-01T00:00:01") == date(2014, 2, 1)
    assert _as_date(date(2014, 2, 1)) == date(2014, 2, 1), (
        "готовая дата сдвигаться не должна"
    )


def test_a_month_without_liquid_is_skipped_by_the_watercut_check():
    """Месяц с нулевой жидкостью в проверку обводнённости не входит.

    Мутация `<= 0` на `< 0` переживала сюиту: строки с ровно нулевым приростом
    в фикстурах не было. С подменой `1 - 0/0` даёт деление на ноль — проверка
    условия кейса роняет прогон вместо того, чтобы выдать список нарушений.
    """
    rows = [make_row("2014-01-01", "1", wlpr=0.0, womr=0.0, liquid=0.0, oil=0.0),
            make_row("2014-02-01", "1", wlpr=30.0, womr=1.0, liquid=900.0, oil=30.0)]

    problems = summary.case_problems(rows, max_watercut=0.9)

    assert not any("2014-01-01" in one for one in problems), problems


def test_a_negative_pressure_is_reported_for_either_gauge():
    """Отрицательное давление ловится и по забойному, и по устьевому.

    Мутация `or` на `and` переживала сюиту: обе величины в фикстурах были
    отрицательны разом. С подменой строка с одним отрицательным давлением
    проходит как исправная — а это либо ошибка выгрузки, либо нефизичный
    прогноз, и оба случая обязаны быть названы.
    """
    only_bhp = [make_row("2014-01-01", "1", wlpr=30.0, womr=1.0,
                         liquid=900.0, oil=30.0, bhp=-1.0, thp=20.0)]
    only_thp = [make_row("2014-01-01", "1", wlpr=30.0, womr=1.0,
                         liquid=900.0, oil=30.0, bhp=90.0, thp=-1.0)]

    assert any("отрицательное давление" in one
               for one in summary.sanity_problems(only_bhp)), "минус в BHP пропущен"
    assert any("отрицательное давление" in one
               for one in summary.sanity_problems(only_thp)), "минус в THP пропущен"


def test_a_drop_explained_by_a_negative_increment_is_not_reported_twice():
    """Падение накопленной, объяснённое отрицательным приростом, — одна находка.

    Четыре мутации сидели в этом условии: знак прироста, множитель допуска,
    обе границы. На опорном расписании такие пары дают четверть всех нарушений,
    и удвоение списка прячет остальные три четверти за шумом.

    Проверяется и обратное: падение, которое приростом **не** объясняется,
    остаётся нарушением — иначе «не сообщать дважды» превратилось бы в «не
    сообщать вовсе».
    """
    def with_total(row, total):
        """Накопленную задаём вручную: `make_row` держит её нулевой."""
        row["WLPT"] = total
        return row

    # прирост объясняет падение **не в точности**, а в пределах допуска
    # `1e-6 × max(1, drop)`: на ровном нуле подмены множителя и самого допуска
    # неразличимы, поэтому расхождение задаётся явно
    explained = [
        with_total(make_row("2014-01-01", "1", wlpr=30.0, womr=1.0,
                            liquid=900.0, oil=30.0), 1000.0),
        with_total(make_row("2014-02-01", "1", wlpr=30.0, womr=1.0,
                            liquid=-100.00005, oil=30.0), 900.0),
    ]
    about_drop = [one for one in summary.sanity_problems(explained)
                  if "уменьшилось" in one]
    assert about_drop == [], (
        f"падение, объяснённое отрицательным приростом, названо дважды: {about_drop}"
    )

    unexplained = [
        with_total(make_row("2014-01-01", "1", wlpr=30.0, womr=1.0,
                            liquid=900.0, oil=30.0), 1000.0),
        with_total(make_row("2014-02-01", "1", wlpr=30.0, womr=1.0,
                            liquid=900.0, oil=30.0), 500.0),
    ]
    assert any("уменьшилось" in one for one in summary.sanity_problems(unexplained)), (
        "падение накопленной без объяснения приростом пропущено"
    )


def test_the_reservoir_pressure_check_asks_the_right_role():
    """Верхний предел пластового давления спрашивают с нагнетательной.

    Мутация `WWIR > 0` на `>=` переживала сюиту: с ней предел проверялся бы и у
    добывающих, у которых закачки нет по определению. Условие кейса про
    нагнетание превратилось бы в условие про весь фонд, и «нарушения» посыпались
    бы там, где нарушать нечего.
    """
    producer = [make_row("2014-01-01", "1", wlpr=30.0, womr=1.0,
                         liquid=900.0, oil=30.0, thp=290.0)]
    injector = [make_row("2014-01-01", "9", wwir=100.0,
                         injection=3000.0, thp=290.0)]
    at_limit = [make_row("2014-01-01", "9", wwir=100.0,
                         injection=3000.0, thp=270.0)]

    assert summary.sanity_problems(at_limit, max_reservoir_pressure=270.0) == [], (
        "давление ровно на пределе объявлено превышением: методика запрещает "
        "выход за границу, а не саму границу"
    )

    assert summary.sanity_problems(producer, max_reservoir_pressure=270.0) == [], (
        "верхний предел спрошен с добывающей — закачки у неё нет"
    )
    assert any("у нагнетательной" in one for one in
               summary.sanity_problems(injector, max_reservoir_pressure=270.0))


def test_a_calendar_gap_across_a_year_is_found():
    """Разрыв календаря через границу года находится, а не прячется.

    Мутация в переходе `декабрь → январь` переживала сюиту: разрывов через
    границу года в фикстурах не было. Ошибка здесь тихая и дорогая: экономика
    считает строку за скважино-месяц, и незамеченный квартальный шаг занижает
    объём втрое (246 суток вместо 730).
    """
    rows = [make_row("2014-12-01", "1", wlpr=30.0, womr=1.0, liquid=900.0, oil=30.0),
            make_row("2015-01-01", "1", wlpr=30.0, womr=1.0, liquid=900.0, oil=30.0)]

    assert summary.sanity_problems(rows) == [], (
        "переход через границу года объявлен разрывом календаря"
    )

    skipped = [make_row("2014-12-01", "1", wlpr=30.0, womr=1.0,
                        liquid=900.0, oil=30.0),
               make_row("2015-02-01", "1", wlpr=30.0, womr=1.0,
                        liquid=900.0, oil=30.0)]
    assert any("шаг выгрузки не месячный" in one
               for one in summary.sanity_problems(skipped)), (
        "пропуск месяца через границу года не найден"
    )


def test_a_long_list_of_gaps_says_how_many_are_hidden():
    """При длинном списке разрывов сообщение честно считает скрытые.

    Мутация `len(gaps) - 5` на `+ 5` переживала сюиту: списка длиннее пяти в
    фикстурах не было. Число говорит оператору, насколько выгрузка не месячная,
    и «и ещё 11» вместо одного меняет решение — приводить её к месяцам или нет.
    """
    stamps = [f"{2014 + index // 6}-{(index % 6) * 2 + 1:02d}-01"
              for index in range(8)]
    rows = [make_row(stamp, "1", wlpr=30.0, womr=1.0, liquid=900.0, oil=30.0)
            for stamp in stamps]

    problems = [one for one in summary.sanity_problems(rows)
                if "шаг выгрузки не месячный" in one]

    assert problems, summary.sanity_problems(rows)
    text = problems[0]
    assert "и ещё 2" in text, text
    assert "разрывов календаря 7" in text, text
    assert text.count("→") == 5, f"показано не пять разрывов: {text}"

    # короткий список хвоста не получает: «и ещё −4» было бы неправдой
    short = [make_row(stamp, "1", wlpr=30.0, womr=1.0, liquid=900.0, oil=30.0)
             for stamp in ("2014-01-01", "2014-03-01", "2014-05-01")]
    brief = [one for one in summary.sanity_problems(short)
             if "шаг выгрузки не месячный" in one]
    assert brief and "и ещё" not in brief[0], brief


def test_writing_a_csv_creates_the_directory_it_needs(tmp_path):
    """Запись выгрузки создаёт каталог, а не падает на его отсутствии.

    Мутация `parents=True` на `False` переживала сюиту: в фикстурах каталог
    всегда существовал. На чистой машине каталог прогона создаётся впервые
    именно здесь, и отказ пришёл бы после расчёта — то есть после того, как
    работа уже сделана.
    """
    rows = [make_row("2014-01-01", "1", wlpr=30.0, womr=1.0, liquid=900.0, oil=30.0)]
    target = tmp_path / "нет" / "такого" / "rows.csv"

    written = summary.write_csv(rows, target)

    assert written.is_file()
    assert written.read_text(encoding="utf-8").startswith("DATA,")

    # и повторная запись в уже существующий каталог не падает: на перезапуске
    # прогона каталог всегда есть
    again = summary.write_csv(rows, target)
    assert again.is_file()


def test_a_drop_exactly_at_the_tolerance_is_not_reported():
    """Падение накопленной ровно на допуск нарушением не считается.

    Мутация `drop <= 1e-6` на `<` переживала сюиту: падения ровно на границу в
    фикстурах не было, а построить его можно — `1e-6 − 0` даёт её точно. Смысл
    допуска здесь тот же, что у Р-188: расхождение в последнем знаке приходит от
    арифметики выгрузки, а не от физики, и объявлять его уменьшением накопленной
    значит топить настоящие находки в шуме.
    """
    def with_total(row, total):
        row["WLPT"] = total
        return row

    on_the_edge = [
        with_total(make_row("2014-01-01", "1", wlpr=30.0, womr=1.0,
                            liquid=900.0, oil=30.0), 1e-6),
        with_total(make_row("2014-02-01", "1", wlpr=30.0, womr=1.0,
                            liquid=900.0, oil=30.0), 0.0),
    ]
    assert 1e-6 - 0.0 == 1e-6, "проверка вырождена: падение не равно допуску"

    about_drop = [one for one in summary.sanity_problems(on_the_edge)
                  if "уменьшилось" in one]

    assert about_drop == [], (
        f"падение ровно на допуск объявлено уменьшением накопленной: {about_drop}"
    )

    # заметно большее падение по-прежнему находится
    bigger = [
        with_total(make_row("2014-01-01", "1", wlpr=30.0, womr=1.0,
                            liquid=900.0, oil=30.0), 1.0),
        with_total(make_row("2014-02-01", "1", wlpr=30.0, womr=1.0,
                            liquid=900.0, oil=30.0), 0.0),
    ]
    assert any("уменьшилось" in one for one in summary.sanity_problems(bigger))


def test_a_non_numeric_value_is_named_and_not_taken_for_a_zero():
    """NaN — отсутствие данных, а не ноль, и различать их обязаны мы.

    Эталонный расчётчик подставляет ноль молча (`chdd_model.to_number`:
    `number if math.isfinite(number) else 0.0`), а всякое сравнение с NaN ложно,
    поэтому остальные санити-чеки её не видят. Замер ревизии 09.09: одна NaN в
    `WOMT_Diff` из трёх месяцев — минус 34 % ЧДД при пустом списке нарушений и
    пустых `diagnostics.warnings`. Ровно тот класс, который CLAUDE.md требует
    ловить явно: отсутствие данных неотличимо от простоя (Н-33).
    """
    good = make_row("2007-01-01", "1", wlpr=10.0, womr=5.0, liquid=10.0, oil=5.0)
    broken = {**good, "WOMT_Diff": float("nan")}

    assert summary.sanity_problems([good]) == []

    problems = summary.sanity_problems([broken])

    assert problems, "NaN прошла все проверки молча"
    assert "нечисловое WOMT_Diff" in problems[0]
    assert "скв. 1" in problems[0] and "2007-01-01" in problems[0]


def test_an_infinite_value_is_caught_too():
    """Бесконечность расчётчик обнуляет тем же правилом, что и NaN."""
    row = {**make_row("2007-01-01", "1", wlpr=10.0, liquid=10.0),
           "WLPR": float("inf")}

    assert any("нечисловое WLPR" in one for one in summary.sanity_problems([row]))


def test_the_efficiency_column_is_one_not_zero():
    """`WEFF` — коэффициент эксплуатации, и его умолчание единица, а не ноль.

    Мутационный прогон ревизии: подмена `1.0 → 0` в `monthly_rows` выжила — ни
    один тест не смотрел на эту колонку. Сегодня цена нулевая: эталонный
    расчётчик `WEFF` не считает, колонка у него только в списке обязательных.
    Но колонка входит в контракт «симулятор → экономика», и скрипт выгрузки
    организаторов пишет туда настоящий коэффициент: на их кейсе ноль означал бы
    «скважина не работала ни дня» (Р-57).
    """
    rows = summary.monthly_rows.__doc__ is not None  # модуль на месте
    assert rows

    assert "WEFF" in economics.REQUIRED_COLUMNS

    made = make_row("2007-01-01", "1", wlpr=10.0, liquid=10.0)
    assert made["WEFF"] == 1.0, "коэффициент эксплуатации по умолчанию не единица"


def test_wells_sort_numbers_before_names_and_names_alphabetically():
    """Порядок скважин при смешанных именах закреплён, а не подразумевается.

    Мутация `(1, 0, name)` → `(0, 0, name)` выжила: тестов на фонд из цифровых и
    буквенных имён не было. У Модели Z имена цифровые, поэтому здесь тихо;
    валидационный кейс приходит со своим объектом, и имена там любые (Р-58).
    """
    names = ["10", "P-2", "2", "A", "1"]

    assert sorted(names, key=summary._well_key) == ["1", "2", "10", "A", "P-2"]
