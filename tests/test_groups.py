"""Группы фонда: разбиение обязано быть воспроизводимым и объяснимым."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aios.groups import (
    Group,
    build_groups,
    link_matrix,
    problems,
    stability,
    watercut,
)

SINCE = "2007-01-01"


def _rows(records: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    frame["DATA"] = pd.to_datetime(frame["DATA"])
    for column in ("WLPR", "WWIR", "WLPT_Diff", "WOMT_Diff", "WWIT_Diff"):
        if column not in frame:
            frame[column] = 0.0
    return frame


def _history(months: int, wells: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    dates = pd.date_range(SINCE, periods=months, freq="MS")
    records = []
    for well, series in wells.items():
        for index, date in enumerate(dates):
            liquid = float(series.get("WLPR", np.zeros(months))[index])
            injection = float(series.get("WWIR", np.zeros(months))[index])
            oil = float(series.get("WOMR", np.zeros(months))[index])
            records.append({
                "DATA": date, "well": well,
                "WLPR": liquid, "WWIR": injection,
                "WLPT_Diff": liquid * 30.0, "WOMT_Diff": oil * 30.0,
                "WWIT_Diff": injection * 30.0,
            })
    return _rows(records)


def test_a_planted_lagged_link_is_found_and_an_unrelated_well_is_not():
    months = 120
    noise = np.random.default_rng(0).normal(0.0, 1.0, months)
    injection = 100.0 + np.cumsum(noise)
    lag = 3
    response = np.concatenate([np.full(lag, 50.0), 50.0 + 0.4 * injection[:-lag]])
    unrelated = 50.0 + np.cumsum(np.random.default_rng(1).normal(0.0, 1.0, months))

    rows = _history(months, {
        "1": {"WWIR": injection},
        "2": {"WLPR": response, "WOMR": response * 0.1},
        "3": {"WLPR": unrelated, "WOMR": unrelated * 0.1},
    })

    links = link_matrix(rows, SINCE)
    assert links.loc["1", "2"] > 0.9
    assert links.loc["1", "3"] == 0.0


def test_producers_and_injectors_never_share_a_group():
    months = 60
    rate = np.full(months, 100.0)
    rows = _history(months, {
        str(well): ({"WLPR": rate, "WOMR": rate * (0.5 if well < 5 else 0.05)}
                    if well < 9 else {"WWIR": rate})
        for well in range(1, 17)
    })

    groups = build_groups(rows, SINCE)
    producers = {w for g in groups if g.kind == "producer" for w in g.wells}
    injectors = {w for g in groups if g.kind == "injector" for w in g.wells}
    assert not producers & injectors
    assert injectors == {str(w) for w in range(9, 17)}


def test_dry_and_watered_producers_land_in_different_groups():
    months = 60
    rate = np.full(months, 100.0)
    rows = _history(months, {
        "1": {"WLPR": rate, "WOMR": rate * 0.6},
        "2": {"WLPR": rate, "WOMR": rate * 0.5},
        "3": {"WLPR": rate, "WOMR": rate * 0.02},
        "4": {"WLPR": rate, "WOMR": rate * 0.01},
    })

    cut = watercut(rows, SINCE)
    assert cut["1"] < cut["4"]

    dry = next(g for g in build_groups(rows, SINCE) if g.name == "добывающие сухие")
    assert set(dry.wells) == {"1", "2"}


def test_weights_are_shares_of_historical_rate_and_sum_to_one():
    months = 60
    rows = _history(months, {
        "1": {"WLPR": np.full(months, 300.0), "WOMR": np.full(months, 30.0)},
        "2": {"WLPR": np.full(months, 100.0), "WOMR": np.full(months, 10.0)},
    })

    group = next(g for g in build_groups(rows, SINCE) if g.wells)
    assert sum(group.weights.values()) == pytest.approx(1.0)
    assert group.weights["1"] == pytest.approx(0.75)
    assert group.weights["2"] == pytest.approx(0.25)


def test_a_group_of_two_wells_is_reported_as_too_small():
    months = 60
    rate = np.full(months, 100.0)
    rows = _history(months, {
        "1": {"WLPR": rate, "WOMR": rate * 0.5},
        "2": {"WLPR": rate, "WOMR": rate * 0.02},
    })

    found = problems(build_groups(rows, SINCE), rows, SINCE)
    assert any("меньше четырёх" in message for message in found)


def test_a_group_holding_a_sliver_of_the_volume_is_reported():
    months = 60
    rows = _history(months, {
        **{str(w): {"WLPR": np.full(months, 1000.0), "WOMR": np.full(months, 500.0)}
           for w in range(1, 6)},
        **{str(w): {"WLPR": np.full(months, 1.0), "WOMR": np.full(months, 0.01)}
           for w in range(6, 11)},
    })

    found = problems(build_groups(rows, SINCE), rows, SINCE)
    assert any("базового объёма" in message for message in found)


def test_an_empty_window_yields_empty_groups_without_dividing_by_zero():
    rows = _history(12, {"1": {"WLPR": np.zeros(12)}})
    groups = build_groups(rows, SINCE)
    assert all(isinstance(group, Group) and group.weights == {} for group in groups)


def test_stability_is_high_when_the_split_does_not_depend_on_the_window():
    months = 120
    rate = np.full(months, 100.0)
    rows = _history(months, {
        str(well): {"WLPR": rate, "WOMR": rate * (0.5 if well < 5 else 0.02)}
        for well in range(1, 9)
    })

    assert stability(rows, SINCE, draws=5, seed=3) == pytest.approx(1.0)


def test_stability_does_not_pass_when_bootstrap_loses_comparable_wells():
    """ARI пустой выборки равен 1 формально, но устойчивость группы не доказывает."""
    rows = _history(5, {
        "1": {
            "WLPR": np.array([100.0, 0.0, 0.0, 0.0, 0.0]),
            "WOMR": np.array([50.0, 0.0, 0.0, 0.0, 0.0]),
        },
    })

    assert stability(rows, SINCE, draws=10, seed=0) == 0.0


def test_stability_requires_at_least_one_bootstrap_draw():
    rows = _history(3, {"1": {"WLPR": np.full(3, 100.0)}})

    with pytest.raises(ValueError, match="число выборок"):
        stability(rows, SINCE, draws=0, seed=0)


# ── что нашёл мутационный прогон 06.09 (Р-221) ───────────────────────────────


def test_every_window_includes_the_date_it_names():
    """Окно истории включает названную дату — во всех четырёх местах модуля.

    Четыре мутации подменяли `>= since` на `>`, каждая в своей функции. Потеря
    первого месяца сдвигает и обводнённость, и веса скважин внутри группы, и
    доли групп в отборе — то есть само разбиение фонда, по которому дальше
    двигаются множители.
    """
    # январь чистый по нефти, февраль обводнённый: если первый месяц выпадет,
    # доля изменится — на этом и держится проверка
    rows = _rows([
        {"DATA": "2007-01-01", "well": "1", "WLPR": 30.0,
         "WLPT_Diff": 900.0, "WOMT_Diff": 900.0},
        {"DATA": "2007-02-01", "well": "1", "WLPR": 30.0,
         "WLPT_Diff": 900.0, "WOMT_Diff": 0.0},
    ])

    from_january = watercut(rows, "2007-01-01")
    from_february = watercut(rows, "2007-02-01")

    assert from_january["1"] == pytest.approx(0.5), (
        f"январь выпал из окна: доля {from_january['1']} вместо (1800−900)/1800"
    )
    assert from_february["1"] == pytest.approx(1.0)


def test_a_month_without_liquid_gets_no_watercut():
    """Обводнённость месяца с нулевой жидкостью не считается.

    Мутация `liquid > 0` на `>=` переживала сюиту: строки простоя в фикстурах не
    было. С подменой `0/0` даёт `nan`, и он расходится по разбиению — скважина
    попадает не в ту группу молча.
    """
    rows = _rows([
        {"DATA": "2007-01-01", "well": "1", "WLPT_Diff": 0.0, "WOMT_Diff": 0.0},
        {"DATA": "2007-01-01", "well": "2", "WLPT_Diff": 900.0, "WOMT_Diff": 90.0},
    ])

    shares = watercut(rows, SINCE)

    assert not np.isnan(shares.get("2", np.nan)), "рабочая скважина потеряла долю"
    assert shares.get("2") == pytest.approx(0.9)
    idle = shares.get("1")
    assert idle is None or np.isnan(idle), (
        f"скважина без жидкости получила обводнённость {idle}"
    )


def test_a_correlation_needs_at_least_three_months():
    """Связь между скважинами считается не меньше чем по трём месяцам.

    Мутация минимума переживала сюиту. По двум точкам корреляция равна ±1 всегда
    — то есть разбиение фонда строилось бы на числе, которое ничего не выражает.
    """
    from aios.groups import _correlate

    two = np.array([[1.0], [2.0]])
    assert np.allclose(_correlate(two, two), 0.0), (
        "по двум месяцам посчитана связь: корреляция двух точек равна ±1 всегда"
    )

    three = np.array([[1.0], [2.0], [3.0]])
    assert np.allclose(_correlate(three, three), 1.0)


def _month(index: int) -> str:
    return f"20{7 + index // 12:02d}-{index % 12 + 1:02d}-01"


def _field(months: int = 36) -> pd.DataFrame:
    """Фонд из восьми скважин: четыре добывающих и четыре нагнетательных.

    Первый месяц намеренно выделен — у добывающих в нём вся жидкость чистая
    нефть. Так граница окна (`>= since` против `>`) меняет и обводнённость, и
    разбиение, и доли групп: без этого месяца скважины меняются местами.
    """
    records = []
    for index in range(months):
        clean = index == 0
        for number, watercut_share in enumerate(("0.1", "0.3", "0.7", "0.9"), start=1):
            liquid = 900.0
            oil = liquid if clean else liquid * (1 - float(watercut_share))
            records.append({"DATA": _month(index), "well": str(number),
                            "WLPR": 30.0 + number, "WLPT_Diff": liquid,
                            "WOMT_Diff": oil})
        for number in range(5, 9):
            records.append({"DATA": _month(index), "well": str(number),
                            "WWIR": 40.0 + number * (index % 3),
                            "WWIT_Diff": 1200.0 + number * 100})
    return _rows(records)


def test_the_grouping_window_includes_the_date_it_names():
    """Разбиение считается от названной даты включительно.

    Мутации `>= since` на `>` сидят в четырёх разных функциях модуля, и каждая
    молча выбрасывает первый месяц. Здесь он выделен: у добывающих в нём вся
    жидкость — нефть, поэтому его потеря меняет обводнённость, а с ней и то, в
    какую группу попадёт скважина.
    """
    rows = _field()

    from_start = watercut(rows, _month(0))
    without_first = watercut(rows, _month(1))

    assert from_start["1"] != pytest.approx(without_first["1"]), (
        "первый месяц выпал из окна: обводнённость не изменилась"
    )
    assert from_start["1"] < without_first["1"], (
        "чистый по нефти месяц не снизил обводнённость — окно считается не оттуда"
    )


def test_the_groups_split_the_fund_by_the_median():
    """Порог деления — медиана по фонду, и она считается по тому же окну.

    Мутация `<= median` на `<` переживала сюиту: скважины, стоящей ровно на
    медиане, в фикстурах не было. Медиана делит фонд пополам — на чётном числе
    скважин она проходит между ними, и включающая граница отправляет
    «серединную» скважину в сухую группу, а не в обводнённую.
    """
    rows = _field()

    groups = build_groups(rows, _month(0))
    by_name = {group.name: group for group in groups}

    dry = by_name["добывающие сухие"].wells
    wet = by_name["добывающие обводнённые"].wells

    assert set(dry) | set(wet) == {"1", "2", "3", "4"}, (dry, wet)
    assert not set(dry) & set(wet), "скважина попала в обе группы"
    assert "1" in dry and "4" in wet, (dry, wet)


def test_a_thin_group_is_reported_by_share_and_by_count():
    """Группа, которая почти ничего не двигает, названа — по обеим причинам.

    Две мутации сидели в порогах: четыре скважины и десятая часть объёма.
    Смысл один — групповой множитель обязан двигать заметную величину, иначе
    поиск тратит вызов на переменную, которая ни на что не влияет.
    """
    rows = _field()
    groups = build_groups(rows, _month(0))

    found = problems(groups, rows, _month(0))

    # в фикстуре по две скважины на группу — обе меньше четырёх
    assert sum("меньше четырёх" in one for one in found) >= 2, found

    # а группа с ничтожным объёмом называется отдельно: у «крошки» сотая доля
    # объёма фонда, у остальных — всё прочее
    scaled = _rows([
        {"DATA": _month(index), "well": well, "WLPR": 30.0,
         "WLPT_Diff": 9.0 if well == "1" else 900.0, "WOMT_Diff": 1.0}
        for index in range(12) for well in ("1", "2", "3", "4")
    ])
    tiny = Group(name="крошка", kind="producer", weights={"1": 1.0})
    big = Group(name="остальные", kind="producer",
                weights={"2": 0.5, "3": 0.3, "4": 0.2})

    about = problems([tiny, big], scaled, _month(0))

    assert any("базового объёма" in one and "крошка" in one for one in about), about
    assert not any("базового объёма" in one and "остальные" in one for one in about), (
        "группа с почти всем объёмом объявлена ничтожной"
    )


def test_group_weights_are_shares_of_the_group_total():
    """Веса внутри группы — доли, и они считаются по тому же окну.

    Мутация границы окна в `_group` переживала сюиту: веса берутся медианой
    режима, и без первого месяца медиана сдвигается. Доли обязаны давать
    единицу — иначе групповой множитель означает не то, что написано.
    """
    rows = _field()

    group = build_groups(rows, _month(0))[0]

    assert group.weights, "группа пуста"
    assert sum(group.weights.values()) == pytest.approx(1.0), group.weights
    assert all(value >= 0.0 for value in group.weights.values())


def test_links_are_taken_at_their_strongest_lag():
    """Связь берётся по сильнейшему запаздыванию, а не по последнему.

    Мутация `np.maximum` на вычитание переживала сюиту. Отклик добычи на закачку
    приходит с задержкой в месяцы, и брать связь без выбора максимума значит
    объявлять несвязанными как раз те пары, ради которых запаздывание и
    перебирается.
    """
    rows = _field()

    matrix = link_matrix(rows, _month(0), min_link=0.0)

    assert not matrix.empty
    assert (matrix.to_numpy() >= 0.0).all(), (
        "в матрице связей есть отрицательные значения: максимум по модулю не взят"
    )
    assert (matrix.to_numpy() <= 1.0 + 1e-9).all(), "связь больше единицы"

