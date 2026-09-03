"""Пограничные случаи скорректированной методики ЧДД — закреплены навсегда.

Каждый тест здесь соответствует пункту из `docs/method/diff.md`. Если один из них
покраснел, значит либо мы сломали контракт данных, либо организаторы поменяли
методику — и тогда сначала читаем их обновление, а потом правим числа.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import make_row, months

from aios.economics import REQUIRED_COLUMNS, compute_npv

START = "2014-01-01"


def npv(rows):
    return compute_npv(rows, headers=REQUIRED_COLUMNS, start_date=START)


def working(data: str, well: str = "A"):
    return make_row(data, well, wlpr=30, womr=5, liquid=900, oil=150)


def test_negative_diff_row_is_dropped_whole_not_clamped():
    """Отрицательная строка исключается целиком: ЧДД такой же, как если бы её не было."""
    base = [working(data) for data in months(2014, 3)]
    with_negative = base + [
        make_row("2014-04-01", "A", wlpr=30, womr=5, liquid=-10, oil=150)
    ]

    clean = npv(base)
    dirty = npv(with_negative)

    assert dirty.chdd_m == pytest.approx(clean.chdd_m, abs=1e-12)
    assert dirty.excluded_negative_rows == 1
    # именно исключение, а не обнуление: обнулённая строка добавила бы месяц фонду
    assert dirty.raw["annual"][0]["activeWellMonths"] == clean.raw["annual"][0]["activeWellMonths"]


def test_stop_and_start_are_two_paid_events():
    """Остановка и пуск — два отдельных события по 1,0 млн руб."""
    rows = [
        working(months(2014, 3)[0]),
        make_row(months(2014, 3)[1], "A"),  # простой: WLPR = 0 и WWIR = 0
        working(months(2014, 3)[2]),
    ]
    result = npv(rows)

    assert result.start_stop_count == 2
    assert result.raw["annual"][0]["startStopCostM"] == pytest.approx(2.0)
    assert [t["type"] for t in result.raw["activityTransitions"]] == ["stop", "start"]


def test_first_activation_is_not_a_paid_start():
    """Скважина, работающая с первого шага, не платит за «пуск»."""
    result = npv([working(data) for data in months(2014, 3)])
    assert result.start_stop_count == 0
    assert result.raw["activityTransitions"] == []


def test_stopped_well_is_out_of_the_active_fund():
    """За месяц простоя не начисляется содержание фонда (1,0 млн руб./скв.-год)."""
    always_on = npv([working(data) for data in months(2014, 3)])
    with_gap = npv([
        working(months(2014, 3)[0]),
        make_row(months(2014, 3)[1], "A"),
        working(months(2014, 3)[2]),
    ])

    assert always_on.raw["annual"][0]["fundOpexM"] == pytest.approx(3 / 12)
    assert with_gap.raw["annual"][0]["fundOpexM"] == pytest.approx(2 / 12)


def test_conversion_to_injection_costs_five_million_without_pump():
    """ВНС по скорректированной методике — 5,0 млн руб. в OPEX и без стоимости ЭЦН."""
    rows = [
        working(months(2014, 3)[0]),
        make_row(months(2014, 3)[1], "A", wwir=50, injection=1500),
        make_row(months(2014, 3)[2], "A", wwir=50, injection=1500),
    ]
    result = npv(rows)

    assert result.conversion_count == 1
    assert result.raw["annual"][0]["conversionOpexM"] == pytest.approx(5.0)
    assert result.capex_m == pytest.approx(0.0)
    # перевод не считается остановкой: скважина остаётся действующей
    assert result.start_stop_count == 0


def test_pump_upsize_charges_operation_to_opex_and_pump_to_capex():
    """Выход за верхнюю границу интервала → смена ЭЦН: 1,8 млн OPEX + цена насоса CAPEX."""
    rows = [
        working(months(2014, 3)[0]),  # 30 м3/сут → насос 30 (интервал 20–50)
        make_row(months(2014, 3)[1], "A", wlpr=300, womr=50, liquid=9000, oil=1500),
        make_row(months(2014, 3)[2], "A", wlpr=300, womr=50, liquid=9000, oil=1500),
    ]
    result = npv(rows)

    assert result.pump_changes == 1
    assert result.raw["annual"][0]["pumpOperationM"] == pytest.approx(1.8)
    assert result.capex_m == pytest.approx(4.75)  # типоразмер 250, интервал 210–320
    assert result.raw["annual"][0]["pumpInstallCounts"] == {"250": 1}


def test_initial_pump_is_not_capitalised_by_default():
    """Первичное оснащение фонда отдельно не начисляется (chargeInitialPump = False)."""
    result = npv([working(data) for data in months(2014, 3)])
    assert result.capex_m == pytest.approx(0.0)
    assert result.pump_changes == 0


def test_downsize_needs_a_drop_of_more_than_100_below_the_lower_bound():
    """Переход на меньший типоразмер — только при падении WLPR более чем на 100 м³/сут."""
    high = make_row(months(2014, 3)[0], "A", wlpr=300, womr=50, liquid=9000, oil=1500)

    # 200 м3/сут: ниже нижней границы установленного насоса (210) всего на 10 → замены нет
    inside = npv([
        high,
        make_row(months(2014, 3)[1], "A", wlpr=200, womr=30, liquid=6000, oil=900),
        make_row(months(2014, 3)[2], "A", wlpr=200, womr=30, liquid=6000, oil=900),
    ])
    assert inside.pump_changes == 0
    assert inside.capex_m == pytest.approx(0.0)

    # 50 м3/сут: ниже 210 на 160 → замена вниз, ставится типоразмер 80 (интервал 50–100)
    below = npv([
        high,
        make_row(months(2014, 3)[1], "A", wlpr=50, womr=8, liquid=1500, oil=240),
        make_row(months(2014, 3)[2], "A", wlpr=50, womr=8, liquid=1500, oil=240),
    ])
    assert below.pump_changes == 1
    assert below.capex_m == pytest.approx(1.85)
    assert below.raw["annual"][0]["pumpInstallCounts"] == {"80": 1}


def test_liquid_rate_above_500_is_rejected_by_the_reference_calculator():
    """Дебит жидкости свыше 500 м³/сут методика не допускает — расчёт падает."""
    rows = [make_row(data, "A", wlpr=550, womr=90, liquid=16500, oil=2700) for data in months(2014, 3)]
    with pytest.raises(ValueError, match="500"):
        npv(rows)


def test_the_liquid_limit_is_declared_once_and_comes_from_the_norms():
    """500 м³/сут — не литерал методики, а верхняя граница старшего ЭЦН.

    Источник числа — `Нормативы_ЧДД.xlsx` организаторов. На валидационном кейсе
    (ТЗ п. 4.3) нормативы придут их, и если старший насос там другой, наши
    проверки обязаны разойтись с эталоном громко. Сейчас число объявлено в
    нескольких местах, и разъехаться они могут молча: валидатор пропустит режим,
    который эталонный расчётчик потом отвергнет, — а это уже отказ на том
    единственном прогоне, который оценивают.
    """
    import inspect

    from aios import economics, summary
    from aios.schedule import MAX_LIQUID_RATE_M3_PER_DAY

    _assumptions, pumps = economics.load_norms()
    assert pumps[-1]["max"] == MAX_LIQUID_RATE_M3_PER_DAY

    # второе место — умолчание санити-чека на границе «симулятор → экономика»
    default = inspect.signature(summary.sanity_problems).parameters["max_liquid_rate"].default
    assert default == MAX_LIQUID_RATE_M3_PER_DAY

    # третьего быть не должно: в модуле экономики константа не используется вовсе
    assert not hasattr(economics, "MAX_LIQUID_RATE_M3_PER_DAY")


def test_the_vendor_ceiling_is_checked_against_the_vendor_itself():
    """Второе число 500 в модуле — потолок эталона, и он сверяется с эталоном.

    `VENDOR_LIQUID_CEILING_M3_PER_DAY` объявлено рядом с
    `MAX_LIQUID_RATE_M3_PER_DAY` и численно ему равно, то есть выглядит как то
    самое «третье место с тем же литералом», от которого защищает тест выше
    (найдено критиком 03.09). Разница в источнике: у первого источник —
    таблица ЭЦН поставки, у второго — строка в коде расчётчика организаторов.
    Одинаковыми они быть не обязаны: обновят архив, и потолок уедет. Поэтому
    константа проверяется не глазами, а самим вендором.
    """
    import re

    from aios.schedule import VENDOR_LIQUID_CEILING_M3_PER_DAY

    source = (Path(__file__).resolve().parent.parent
              / "vendor" / "CHDD_PYTHON" / "chdd_model.py").read_text(encoding="utf-8")
    # ноль отсеивается намеренно: `WLPR > 0` в расчётчике — признак действующего
    # фонда, а не предел режима. Нас интересует единственное сравнение с потолком
    found = {float(one) for one
             in re.findall(r'row\["WLPR"\]\s*>\s*(\d+(?:\.\d+)?)', source)} - {0.0}

    assert found, "в расчётчике больше нет проверки WLPR — константу нужно пересмотреть"
    assert found == {VENDOR_LIQUID_CEILING_M3_PER_DAY}


def test_injection_only_well_counts_as_active_fund():
    """Скважина в действующем фонде, если WLPR > 0 ИЛИ WWIR > 0."""
    result = npv([
        make_row(data, "A", wwir=50, injection=1500) for data in months(2014, 3)
    ])
    assert result.raw["annual"][0]["activeWellMonths"] == pytest.approx(3.0)
    assert result.raw["annual"][0]["fundOpexM"] == pytest.approx(3 / 12)


def test_discounting_base_follows_the_start_year():
    """Дисконт первого года равен 1, дальше 1/(1+WACC)^(y−y0)."""
    rows = [make_row(f"{year}-06-01", "A", wlpr=30, womr=5, liquid=900, oil=150)
            for year in (2014, 2015, 2016)]
    result = compute_npv(rows, headers=REQUIRED_COLUMNS, start_date="2014-01-01")
    factors = [year["discountFactor"] for year in result.raw["annual"]]
    assert factors[0] == pytest.approx(1.0)
    assert factors[1] == pytest.approx(1 / 1.1)
    assert factors[2] == pytest.approx(1 / 1.1**2)
