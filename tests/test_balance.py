"""Баланс воды: закачивать нечего, кроме добытой воды.

Ограничение из разбора организаторов 12.08 (`docs/method/qa-organisers.md`):
дополнительной воды на объекте нет.
"""

from __future__ import annotations

from calendar import monthrange

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


def test_a_cut_exactly_to_the_ceiling_is_not_a_violation():
    """Срезка ровно до потолка — не нарушение, а его соблюдение (Р-188).

    Планировщик масштабирует закачку так, чтобы накопленное отношение легло
    **точно** на предел, и без допуска его собственная срезка объявлялась
    нарушением в последнем знаке. С включённой проверкой fail-closed (Р-165) это
    означало бы, что ни один приведённый к коридору кандидат не пригоден.
    Найдено прогоном цикла 03.09, допуск оказался непокрытым тестом (критик).
    """
    from aios import balance, summary

    fluids = summary.Fluids()
    # отбор: 100 м³ нефти и 900 м³ воды за месяц
    oil_t = 100.0 * fluids.oil_density / 1000
    water_t = 900.0 * fluids.water_density / 1000
    withdrawn = 100.0 + 900.0
    rows = [make_row("2014-01-01", "1", wlpr=30.0, womr=oil_t / 30,
                     liquid=oil_t + water_t, oil=oil_t)]

    exact = rows + [make_row("2014-01-01", "9", wwir=1.0, injection=withdrawn * 1.3)]
    assert balance.compensation_problems(exact, fluids, 1.3) == [], (
        "ровно на потолке — соблюдено"
    )

    # превышение на 0,001 % допуск не прячет: он в последнем знаке double
    over = rows + [make_row("2014-01-01", "9", wwir=1.0,
                            injection=withdrawn * 1.3 * 1.00001)]
    assert balance.compensation_problems(over, fluids, 1.3), (
        "реальное превышение обязано остаться нарушением"
    )


def test_a_fill_exactly_to_the_floor_is_not_a_violation():
    """То же с нижней границей коридора: добор ровно до неё — соблюдение."""
    from aios import balance, summary

    fluids = summary.Fluids()
    oil_t = 100.0 * fluids.oil_density / 1000
    water_t = 900.0 * fluids.water_density / 1000
    withdrawn = 100.0 + 900.0
    rows = [make_row("2014-01-01", "1", wlpr=30.0, womr=oil_t / 30,
                     liquid=oil_t + water_t, oil=oil_t)]

    exact = rows + [make_row("2014-01-01", "9", wwir=1.0, injection=withdrawn * 0.85)]
    assert balance.compensation_problems(exact, fluids, 1.15, floor=0.85) == []

    under = rows + [make_row("2014-01-01", "9", wwir=1.0,
                             injection=withdrawn * 0.85 * 0.99999)]
    assert balance.compensation_problems(under, fluids, 1.15, floor=0.85)


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


# --- компенсация отбора: вторая, физическая формулировка --------------------

def _rows(oil_t: float, water_t: float, injected_m3: float, year: int = 2019):
    """Строки одного года: массы в тоннах, закачка в кубах — как в выгрузке."""
    return [{
        "DATA": f"{year}-01-01", "well": "1",
        "WOMT_Diff": oil_t, "WLPT_Diff": oil_t + water_t, "WWIT_Diff": injected_m3,
    }]


def test_history_violates_the_produced_water_check_but_passes_compensation():
    """Главная находка: буквальное прочтение отвергает саму историю месторождения.

    Числа — исторический сценарий Модели Z (`docs/experiments.md`): нефть 3656,8
    тыс. т, жидкость 9561,6 тыс. т, закачка 11 671 тыс. м³.
    """
    rows = _rows(oil_t=3_656_800, water_t=9_561_600 - 3_656_800, injected_m3=11_671_000)

    assert balance.problems(rows), "проверка по добытой воде обязана видеть превышение"
    assert not balance.compensation_problems(rows), "по компенсации история нормальна"


def test_compensation_of_the_historical_scenario_is_about_130_percent():
    rows = _rows(oil_t=3_656_800, water_t=9_561_600 - 3_656_800, injected_m3=11_671_000)
    year = balance.by_year(rows)[0]
    assert 1.25 < year.compensation < 1.35


def test_injection_far_above_the_withdrawn_volume_is_reported():
    rows = _rows(oil_t=100_000, water_t=100_000, injected_m3=10_000_000)
    problems = balance.compensation_problems(rows)
    assert problems and "выше предела" in problems[0]


def test_injection_without_any_withdrawal_is_reported():
    rows = _rows(oil_t=0.0, water_t=0.0, injected_m3=1000.0)
    assert "возмещать нечего" in balance.compensation_problems(rows)[0]


def test_the_ceiling_is_a_parameter_not_a_constant():
    rows = _rows(oil_t=100_000, water_t=100_000, injected_m3=300_000)
    assert not balance.compensation_problems(rows, ceiling=2.0)
    assert balance.compensation_problems(rows, ceiling=1.1)


def test_without_a_floor_undercompensation_is_not_reported():
    """Умолчание обязано остаться односторонним: иначе заявляемое число поменялось бы."""
    rows = _rows(oil_t=100_000, water_t=100_000, injected_m3=10_000)
    assert not balance.compensation_problems(rows)


def test_injection_below_the_floor_is_reported():
    """Коридор кейса двусторонний — недокомпенсация такое же нарушение, как перекачка."""
    rows = _rows(oil_t=100_000, water_t=100_000, injected_m3=10_000)
    found = balance.compensation_problems(rows, floor=0.85)
    assert found and "ниже границы" in found[0]


def test_injection_inside_the_corridor_passes_both_bounds():
    rows = _rows(oil_t=100_000, water_t=100_000, injected_m3=220_000)
    assert not balance.compensation_problems(rows, ceiling=1.15, floor=0.85)


def test_the_historical_scenario_sits_above_a_085_115_corridor():
    """Историческая компенсация Модели Z — 1,291, то есть вся история выше 1.15.

    Это не абстракция: если кейс придёт с таким коридором, базовый сценарий
    объекта в него не попадает, и знать это нужно до кейса, а не после.
    """
    rows = _rows(oil_t=3_656_800, water_t=9_561_600 - 3_656_800, injected_m3=11_671_000)
    found = balance.compensation_problems(rows, ceiling=1.15, floor=0.85)
    assert found and "выше предела" in found[0]


def test_produced_water_rule_has_the_same_floor():
    rows = _rows(oil_t=100_000, water_t=1_000_000, injected_m3=100_000)
    assert not balance.problems(rows)
    found = balance.problems(rows, floor=0.85)
    assert found and "ниже границы" in found[0]


def test_volume_factors_change_the_withdrawn_volume():
    """На другом объекте Bн заметно больше единицы, и это меняет вывод."""
    rows = _rows(oil_t=1_000_000, water_t=0.0, injected_m3=1_400_000)
    assert balance.compensation_problems(rows, ceiling=1.2, oil_fvf=1.0)
    assert not balance.compensation_problems(rows, ceiling=1.2, oil_fvf=1.3)


def test_a_row_with_negative_injection_is_dropped_whole_not_clamped():
    """Правило методики одно на все три приращения, а проверялись два.

    `by_year` пропускал строку при отрицательных `WOMT_Diff` или `WLPT_Diff` —
    и тут же **клампил** `WWIT_Diff` через `max(..., 0.0)`. Методика исключает
    строку целиком, если отрицательно **любое** из трёх; комментарий в коде
    заявлял именно это, а проверка закрывала две трети.

    Цена перекоса конкретная и односторонняя: добытая вода такой строки идёт в
    баланс, а закачка обнуляется — то есть ограничение «закачиваем только то,
    что подняли» выглядит выполненным щедрее, чем есть. Отрицательная закачка
    в прогнозах суррогата встречается: она в списке оставшихся санити-пометок
    на 20.08.
    """
    good = produced("2014-01-01", "1", oil_m3=100.0, water_m3=900.0)
    good["WWIT_Diff"] = 500.0
    rolled_back = produced("2014-02-01", "1", oil_m3=100.0, water_m3=900.0)
    rolled_back["WWIT_Diff"] = -500.0

    only_good = balance.by_year([good], FLUIDS)[0]
    with_rolled_back = balance.by_year([good, rolled_back], FLUIDS)[0]

    assert with_rolled_back.water_m3 == pytest.approx(only_good.water_m3), (
        "вода из строки с отрицательной закачкой попала в баланс"
    )
    assert with_rolled_back.injected_m3 == pytest.approx(only_good.injected_m3)


def test_days_are_counted_by_actual_months_not_by_full_year():
    """Неполный год не считаем полным: иначе м³/сут занижается втрое и больше."""
    rows = [produced("2014-01-01", "1", 10.0, 90.0), injected("2014-01-01", "2", 310.0)]
    year = balance.by_year(rows, FLUIDS)[0]

    assert year.days == pytest.approx(31.0)
    assert year.external_water_m3 == pytest.approx(310.0 - 90.0)
    assert year.external_water_m3_per_day == pytest.approx(220.0 / 31.0)


def test_external_water_is_negative_when_produced_water_exceeds_injection():
    """Своей воды больше, чем закачали, — источник не нужен, и это не нарушение."""
    rows = [produced("2014-01-01", "1", 10.0, 900.0), injected("2014-01-01", "2", 100.0)]
    year = balance.by_year(rows, FLUIDS)[0]

    assert year.external_water_m3 < 0
    assert balance.external_water_problems(rows, FLUIDS, limit_m3_per_day=0.0) == []


def test_unknown_source_capacity_is_reported_but_not_treated_as_zero():
    """None — «мощность неизвестна». Это не ноль: потребность печатается, режим не подтверждён."""
    rows = [produced("2014-01-01", "1", 10.0, 90.0), injected("2014-01-01", "2", 310.0)]

    found = balance.external_water_problems(rows, FLUIDS, limit_m3_per_day=None)
    assert len(found) == 1
    assert "мощность источника неизвестна" in found[0]


def test_external_water_over_the_limit_is_a_problem():
    rows = [produced("2014-01-01", "1", 10.0, 90.0), injected("2014-01-01", "2", 310.0)]

    assert balance.external_water_problems(rows, FLUIDS, limit_m3_per_day=10.0) == []
    over = balance.external_water_problems(rows, FLUIDS, limit_m3_per_day=5.0)
    assert len(over) == 1
    assert "воды со стороны" in over[0]


def test_historic_limit_is_the_maximum_the_source_actually_delivered():
    """Предел по умолчанию выводится из истории, а не назначается."""
    rows = [
        produced("2014-01-01", "1", 10.0, 90.0), injected("2014-01-01", "2", 310.0),
        produced("2015-01-01", "1", 10.0, 90.0), injected("2015-01-01", "2", 190.0),
    ]

    assert balance.historic_external_water(rows, FLUIDS) == pytest.approx(220.0 / 31.0)


def test_history_of_model_z_fails_produced_water_but_passes_external_water():
    """Ровно тот случай, из-за которого правило и появилось.

    Объект закачивает больше, чем поднимает воды, — по `problems` это нарушение,
    хотя так работает реальная история Модели Z. С названным источником вопроса
    нет, и разница между двумя правилами видна на одних и тех же строках.
    """
    rows = [produced("2014-01-01", "1", 10.0, 90.0), injected("2014-01-01", "2", 310.0)]

    assert balance.problems(rows, FLUIDS), "правило добытой воды обязано отвергнуть"
    assert balance.external_water_problems(
        rows, FLUIDS, limit_m3_per_day=220.0 / 31.0
    ) == [], "с названной мощностью источника тот же режим обеспечен"


# --- мощность водозабора: помесячный расход, а не годовое среднее -------------
# Проверка названа трекером на КТ-2 («помесячный расход не превышает мощности»)
# и закрывает дыру, которой годовой баланс не видит по построению.


def test_monthly_need_sums_to_the_annual_one():
    """Помесячный счёт не расходится с годовым — иначе завелись бы два ответа.

    Формула потребности повторена в `external_water_by_month` отдельным циклом,
    и без этой проверки два счёта разошлись бы при первой же правке одного из них.
    """
    rows = [
        produced("2014-01-01", "1", 10.0, 500.0), injected("2014-01-01", "2", 900.0),
        produced("2014-07-01", "1", 10.0, 400.0), injected("2014-07-01", "2", 1500.0),
    ]

    year = balance.by_year(rows, FLUIDS)[0]
    months = balance.external_water_by_month(rows, FLUIDS)

    assert [stamp for stamp, _ in months] == ["2014-01", "2014-07"]
    volume = sum(need * monthrange(2014, int(stamp[5:7]))[1] for stamp, need in months)
    assert volume == pytest.approx(year.external_water_m3)


def test_a_peak_month_hides_inside_the_annual_average():
    """Год в норме, июль втрое выше мощности — годовая проверка молчит, месячная нет.

    Это и есть цена усреднения: насосная станция не отдаёт за июль больше, чем
    может за июль, сколько бы ни простаивала в январе.
    """
    rows = [
        produced("2014-01-01", "1", 10.0, 500.0), injected("2014-01-01", "2", 500.0),
        produced("2014-07-01", "1", 10.0, 500.0), injected("2014-07-01", "2", 3_000.0),
    ]
    limit = 50.0

    assert balance.external_water_problems(rows, FLUIDS, limit) == []

    peaks = balance.external_water_capacity_problems(rows, FLUIDS, limit)
    assert len(peaks) == 1
    assert "2014-07" in peaks[0]
    # 2 500 м³ за 31 сутки — 80,6 м³/сут против мощности 50
    assert "81 м³/сут" in peaks[0]
    assert "превышение 61%" in peaks[0]


def test_a_month_inside_the_capacity_is_not_a_problem():
    rows = [produced("2014-07-01", "1", 10.0, 500.0), injected("2014-07-01", "2", 800.0)]

    assert balance.external_water_capacity_problems(rows, FLUIDS, 50.0) == []


def test_unknown_capacity_names_the_peak_month_instead_of_staying_silent():
    """Мощность неизвестна — не нарушение, но и не молчание.

    Тот же выбор, что в `external_water_problems`: подставить правдоподобное
    число вместо неизвестного нельзя, а скрыть потребность — тем более.
    """
    rows = [
        produced("2014-01-01", "1", 10.0, 500.0), injected("2014-01-01", "2", 600.0),
        produced("2014-07-01", "1", 10.0, 500.0), injected("2014-07-01", "2", 3_000.0),
    ]

    found = balance.external_water_capacity_problems(rows, FLUIDS, None)

    assert len(found) == 1, "неизвестная мощность не должна печатать все месяцы подряд"
    assert "2014-07" in found[0]
    assert "мощность источника неизвестна" in found[0]


def test_schedule_that_needs_no_outside_water_says_nothing():
    """Закачка ниже попутной воды — потребности нет, и печатать нечего."""
    rows = [produced("2014-07-01", "1", 10.0, 900.0), injected("2014-07-01", "2", 100.0)]

    assert balance.external_water_capacity_problems(rows, FLUIDS, None) == []
    assert balance.external_water_by_month(rows, FLUIDS)[0][1] < 0


def test_zero_capacity_does_not_divide_by_it():
    """Мощность ноль — «источника нет вовсе», а не деление на ноль.

    Из конфигурации ноль сюда не приходит (`_non_negative(...) or None`), но
    функция публичная, и падать `ZeroDivisionError` она не должна.
    """
    rows = [produced("2014-07-01", "1", 10.0, 100.0), injected("2014-07-01", "2", 900.0)]

    found = balance.external_water_capacity_problems(rows, FLUIDS, 0.0)

    assert len(found) == 1
    assert "стороннего источника нет вовсе" in found[0]


# ── что нашёл мутационный прогон 05.09 (Р-221) ───────────────────────────────
#
# `python scripts/mutate.py --module aios/balance.py --limit 25`: из 25 подмен
# смысла 12 пережили даже широкий набор тестов. Ниже закрыто каждое место.


def test_produced_water_exactly_at_the_ceiling_is_not_a_violation():
    """Допуск Р-188 действует и в правиле `produced_water`, а не только в компенсации.

    Мутации «убрать 1e-9» и «`+` вместо `-`» в строке 179 переживали всю сюиту:
    граница проверялась только для `compensation_problems`. Между тем срезка
    планировщика кладёт отношение **точно** на предел, и без допуска собственная
    срезка объявлялась бы нарушением в последнем знаке — при fail-closed это
    отбраковка любого приведённого к коридору кандидата.
    """
    rows = [produced("2014-01-01", "1", oil_m3=100.0, water_m3=900.0)]
    exactly = rows + [injected("2014-01-01", "9", volume_m3=900.0)]

    assert balance.problems(exactly, FLUIDS, tolerance=1.0) == []

    over = rows + [injected("2014-01-01", "9", volume_m3=900.0 * (1 + 1e-6))]
    assert balance.problems(over, FLUIDS, tolerance=1.0), "превышение сверх допуска молчит"


def test_produced_water_exactly_at_the_floor_is_not_a_violation():
    """Та же граница снизу: коридор задаёт обе, и нижняя тоже кладётся точно."""
    rows = [produced("2014-01-01", "1", oil_m3=100.0, water_m3=900.0)]
    exactly = rows + [injected("2014-01-01", "9", volume_m3=900.0 * 0.85)]

    assert balance.problems(exactly, FLUIDS, tolerance=1.3, floor=0.85) == []

    under = rows + [injected("2014-01-01", "9", volume_m3=900.0 * 0.85 * (1 - 1e-6))]
    assert balance.problems(under, FLUIDS, tolerance=1.3, floor=0.85), (
        "недобор ниже границы молчит"
    )


def test_the_violation_message_states_the_right_percentage():
    """Процент в сообщении — часть решения, а не украшение.

    Пять мутаций сидели прямо в текстах: множитель процента, деление вместо
    умножения. Тесты смотрели только на факт нарушения, поэтому сообщение могло
    называть любое число. Слой интерпретируемости обязателен по регламенту, и
    человек, читающий «превышает на 3%» вместо 30%, примет другое решение.
    """
    rows = [produced("2014-01-01", "1", oil_m3=100.0, water_m3=1000.0)]

    # закачали 1 300 при добытой воде 1 000 — превышение ровно на 30 %
    over = balance.problems(rows + [injected("2014-01-01", "9", volume_m3=1300.0)],
                            FLUIDS, tolerance=1.0)
    assert len(over) == 1, over
    assert "на 30%" in over[0], over[0]

    # закачали 600 при границе 0,85 — это 60 % добытой воды
    under = balance.problems(rows + [injected("2014-01-01", "9", volume_m3=600.0)],
                             FLUIDS, tolerance=1.3, floor=0.85)
    assert len(under) == 1, under
    assert "60% добытой воды" in under[0], under[0]
    assert "ниже границы 85%" in under[0], under[0]


def test_the_compensation_message_states_the_right_percentage():
    """То же для компенсации: обе границы называют свой процент."""
    rows = [produced("2014-01-01", "1", oil_m3=100.0, water_m3=900.0)]

    over = balance.compensation_problems(
        rows + [injected("2014-01-01", "9", volume_m3=1500.0)], FLUIDS, ceiling=1.0)
    assert len(over) == 1, over
    assert "на 150%" in over[0], over[0]

    under = balance.compensation_problems(
        rows + [injected("2014-01-01", "9", volume_m3=500.0)],
        FLUIDS, ceiling=1.3, floor=0.85)
    assert len(under) == 1, under
    assert "на 50%" in under[0], under[0]
    assert "ниже границы 85%" in under[0], under[0]


def test_a_year_without_withdrawal_is_skipped_not_divided_by():
    """Год с нулевым отбором пропускается — иначе деление на ноль.

    Мутация «`<= 0` на `< 0`» в строке 350 переживала сюиту: года без отбора в
    фикстурах не было. На кейсе он появится сам — скважина под закачкой стоит,
    добыча ещё не начата, и `injected / withdrawn` уронит подтверждение.
    """
    rows = [injected("2014-01-01", "9", volume_m3=500.0),
            produced("2015-01-01", "1", oil_m3=100.0, water_m3=900.0),
            injected("2015-01-01", "9", volume_m3=1000.0)]

    worst = balance.worst_cumulative_compensation(rows, FLUIDS, "compensation")

    assert worst is not None
    assert worst[0] == 2015, "худшим обязан стать год, в котором отбор был"


def test_explicit_densities_reach_the_monthly_breakdowns():
    """Переданные плотности доходят до расчёта, а не подменяются умолчанием.

    Мутация `fluids or Fluids()` на `and` в строках 377 и 451 переживала сюиту:
    обе функции всегда звали с умолчанием. Между тем во втором PVT-регионе
    Модели Z нефть 928,16 против 913,08 — разница 1,6 %, и она идёт прямо в
    выручку.
    """
    from aios.summary import Fluids as Densities

    rows = [produced("2014-01-01", "1", oil_m3=100.0, water_m3=900.0),
            injected("2014-01-01", "9", volume_m3=2000.0)]

    # объём воды получается из её массы, поэтому сторонней воды касается
    # плотность **воды**; плотность нефти сюда не входит, и менять надо ту,
    # которая в расчёте участвует (первая редакция теста меняла не ту и падала)
    salty = Densities(oil_density=913.0765883224138, water_density=1050.0)
    assert balance.external_water_by_month(rows, FLUIDS)         != balance.external_water_by_month(rows, salty), (
        "плотность воды не дошла до расчёта сторонней воды — переданное значение потеряно"
    )

    # отношение вода/нефть переводит нефть в объём, поэтому здесь работает
    # плотность нефти: 928,16 второго PVT-региона против 913,08 первого
    heavy = Densities(oil_density=928.16, water_density=1172.0)
    assert balance.well_water_oil_ratio(rows, FLUIDS)         != balance.well_water_oil_ratio(rows, heavy), (
        "плотность нефти не дошла до отношения вода/нефть"
    )


def test_a_year_balance_defaults_to_a_full_year():
    """Умолчание `days` — 365, а не ноль: иначе перевод в м³/сут делит на ноль."""
    year = balance.YearBalance(year=2014, oil_m3=100.0, water_m3=900.0,
                               injected_m3=800.0)

    assert year.days == 365.0


def test_the_capacity_message_states_the_right_excess():
    """Процент превышения мощности водозабора назван верно.

    Последняя из мутаций текста: деление вместо умножения в строке 323. Строка
    попадает оператору как «превышение N%», и по ней принимают решение — цифра
    обязана быть настоящей.
    """
    rows = [produced("2014-01-01", "1", oil_m3=100.0, water_m3=900.0),
            injected("2014-01-01", "9", volume_m3=2000.0)]

    months = balance.external_water_by_month(rows, FLUIDS)
    assert len(months) == 1, months
    need = months[0][1]

    # предел вдвое ниже потребности — превышение ровно 100 %
    over = balance.external_water_capacity_problems(rows, FLUIDS, need / 2)

    assert len(over) == 1, over
    assert "превышение 100%" in over[0], over[0]

    # та же цифра в годовой формулировке: соседняя функция считает превышение
    # своей строкой, и мутация показала, что проверялась только одна из двух
    yearly = balance.by_year(rows, FLUIDS)
    assert len(yearly) == 1, yearly
    annual_need = yearly[0].external_water_m3_per_day
    annual = balance.external_water_problems(rows, FLUIDS, annual_need / 4)

    assert len(annual) == 1, annual
    assert "превышение 300%" in annual[0], annual[0]


def test_the_floor_tolerance_forgives_the_last_digit_and_nothing_more():
    """Допуск нижней границы прощает последний знак — и ровно его.

    Две мутации строки 185 («убрать 1e-9», «`<` на `<=`») пережили даже
    закрытую проверку границы: тест ставил закачку **ровно** на границу, а там
    обе редакции ведут себя одинаково. Отличает их только попадание внутрь
    самого допуска, и оно строится явным вычислением, а не круглым числом.
    """
    water_m3 = 900.0
    floor = 0.85
    rows = [produced("2014-01-01", "1", oil_m3=100.0, water_m3=water_m3)]

    # внутри допуска: ниже границы, но в последнем знаке — не нарушение
    inside = water_m3 * floor * (1 - 5e-10)
    assert balance.problems(rows + [injected("2014-01-01", "9", inside)],
                            FLUIDS, tolerance=1.3, floor=floor) == [], (
        "допуск не прощает расхождения в последнем знаке — срезка планировщика "
        "объявляется собственным нарушением (Р-188)"
    )

    # ровно на границе допуска: строгое `<` обязано промолчать
    edge = water_m3 * floor * (1 - 1e-9)
    assert balance.problems(rows + [injected("2014-01-01", "9", edge)],
                            FLUIDS, tolerance=1.3, floor=floor) == [], (
        "сравнение стало нестрогим: значение ровно на границе допуска "
        "объявлено нарушением"
    )

    # заметно ниже — нарушение обязано быть
    assert balance.problems(rows + [injected("2014-01-01", "9", water_m3 * floor * 0.999)],
                            FLUIDS, tolerance=1.3, floor=floor), (
        "допуск стал шире последнего знака и прячет настоящий недобор"
    )

