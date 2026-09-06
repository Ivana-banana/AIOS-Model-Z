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

    assert shares.get("2") == pytest.approx(0.9), "рабочая скважина потеряла долю"
    # скважины без жидкости в ряду нет вовсе — не `nan`, а отсутствие. С
    # подменой границы она появляется с `0/0`, и `nan` расходится по разбиению
    assert "1" not in shares, (
        f"скважина без жидкости попала в ряд обводнённостей: {dict(shares)}"
    )
    assert not shares.isna().any(), f"в ряду есть nan: {dict(shares)}"


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


#: Обводнённость четырёх добывающих за историю — намеренно близкая. Медиана
#: проходит между «2» и «3», и потеря **одного** месяца их переставляет: это и
#: отличает окно `>= since` от `>`. Разброс пошире (0,1 против 0,9) такой
#: подмены не замечает — порядок скважин от одного месяца там не меняется, и
#: первая версия фикстуры мутации не убивала (замер 06.09).
_WATERCUT = {"1": 0.50, "2": 0.52, "3": 0.51, "4": 0.53}


def _field(months: int = 36) -> pd.DataFrame:
    """Фонд из восьми скважин: четыре добывающих и четыре нагнетательных.

    Месяцев 36, а не 18: история короче `MAX_LAG_MONTHS` обрывает срез по
    запаздыванию, и матрица связей не считается вовсе.

    Первый месяц выделен у скважины «2» — в нём вся её жидкость чистая нефть.
    С этим месяцем её доля 0,5056 и она попадает в сухие; без него — 0,52, и
    сухой становится «3». То есть граница окна меняет **само разбиение фонда**,
    а не только четвёртый знак доли.
    """
    records = []
    for index in range(months):
        for well, share in _WATERCUT.items():
            liquid = 900.0
            oil = liquid if (index == 0 and well == "2") else liquid * (1 - share)
            records.append({"DATA": _month(index), "well": well, "WLPR": 30.0,
                            "WLPT_Diff": liquid, "WOMT_Diff": oil})
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

    assert from_start["2"] == pytest.approx(0.5056, abs=1e-4), from_start["2"]
    assert without_first["2"] == pytest.approx(0.52, abs=1e-9), without_first["2"]

    # и это меняет разбиение, а не только четвёртый знак доли
    dry = {one.wells for one in build_groups(rows, _month(0))
           if one.name == "добывающие сухие"}
    assert dry == {("1", "2")}, dry


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
    assert set(dry) == {"1", "2"} and set(wet) == {"3", "4"}, (dry, wet)


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


def _two_months() -> pd.DataFrame:
    """Два месяца, где первый решает: у «1» в нём режим и объём вдесятеро выше.

    Длинная история для этих проверок не годится: один месяц из тридцати шести
    не двигает ни медиану режима, ни долю объёма, и подмена границы окна
    переживает любой такой тест (замер 06.09). На двух месяцах первый — половина
    данных, и граница становится различима.
    """
    return _rows([
        {"DATA": _month(0), "well": "1", "WLPR": 100.0,
          "WLPT_Diff": 10000.0, "WOMT_Diff": 1000.0},
        {"DATA": _month(0), "well": "2", "WLPR": 10.0,
          "WLPT_Diff": 300.0, "WOMT_Diff": 30.0},
        {"DATA": _month(1), "well": "1", "WLPR": 10.0,
          "WLPT_Diff": 300.0, "WOMT_Diff": 30.0},
        {"DATA": _month(1), "well": "2", "WLPR": 10.0,
          "WLPT_Diff": 300.0, "WOMT_Diff": 30.0},
    ])


def test_group_weights_are_taken_from_the_named_month_on():
    """Веса внутри группы считаются от названной даты включительно.

    Мутация `>= since` на `>` в `_group` переживала прежние тесты: на длинной
    истории медиана режима от одного месяца не сдвигается. Здесь сдвигается —
    и вместе с ней доли, которыми групповой множитель распределяется по
    скважинам.
    """
    from aios.groups import _group

    rows = _two_months()

    with_first = _group("тест", "producer", ["1", "2"], rows, _month(0), "WLPR")
    without_first = _group("тест", "producer", ["1", "2"], rows, _month(1), "WLPR")

    assert with_first.weights["1"] == pytest.approx(0.8462, abs=1e-4), (
        f"первый месяц выпал из весов: {with_first.weights}"
    )
    assert without_first.weights["1"] == pytest.approx(0.5)
    assert sum(with_first.weights.values()) == pytest.approx(1.0)


def test_the_share_of_volume_is_measured_from_the_named_month_on():
    """Доля объёма группы считается от названной даты включительно.

    Мутация границы окна в `problems` переживала прежние тесты по той же
    причине. Здесь первый месяц решает: с ним у «мелкой» группы 5,5 % объёма —
    меньше десятой части, и об этом обязано быть сказано; без него у обеих групп
    поровну, и говорить не о чем.
    """
    rows = _two_months()
    big = Group(name="крупная", kind="producer", weights={"1": 1.0})
    small = Group(name="мелкая", kind="producer", weights={"2": 1.0})

    with_first = [one for one in problems([big, small], rows, _month(0))
                  if "объёма" in one]
    without_first = [one for one in problems([big, small], rows, _month(1))
                     if "объёма" in one]

    assert len(with_first) == 1 and "мелкая" in with_first[0], with_first
    assert "5.5%" in with_first[0], with_first[0]
    assert without_first == [], (
        f"при равных объёмах группа объявлена ничтожной: {without_first}"
    )


def test_a_well_exactly_on_the_median_goes_to_the_dry_group():
    """Скважина ровно на медиане попадает в сухую группу.

    Мутация `<= median` на `<` переживала сюиту: на чётном фонде медиана
    проходит **между** скважинами, и ровно на ней никто не стоит. На нечётном
    стоит средняя — и граница становится различима. Смысл включающей границы:
    медиана делит фонд пополам, и «ровно посередине» это ещё не «обводнённая».
    """
    rows = _rows([
        {"DATA": _month(index), "well": well, "WLPR": 30.0,
          "WLPT_Diff": 900.0, "WOMT_Diff": 900.0 * (1 - share)}
        for index in range(3)
        for well, share in (("1", 0.2), ("2", 0.5), ("3", 0.8))
    ])

    groups = {one.name: set(one.wells) for one in build_groups(rows, _month(0))}

    assert groups["добывающие сухие"] == {"1", "2"}, (
        f"скважина ровно на медиане ушла в обводнённые: {groups}"
    )
    assert groups["добывающие обводнённые"] == {"3"}


def test_a_share_exactly_at_a_tenth_is_not_reported():
    """Доля ровно в десятую часть объёма нарушением не считается.

    Мутация `share < 0.10` на `<=` переживала сюиту: доли ровно на пороге в
    фикстурах не было. Порог объявлен как «меньше десятой части», и группа,
    двигающая ровно десятую, ему отвечает — жаловаться не на что.
    """
    rows = _rows([
        {"DATA": _month(0), "well": "1", "WLPT_Diff": 900.0, "WOMT_Diff": 90.0},
        {"DATA": _month(0), "well": "2", "WLPT_Diff": 100.0, "WOMT_Diff": 10.0},
    ])
    big = Group(name="крупная", kind="producer", weights={"1": 1.0})
    edge = Group(name="ровно десятая", kind="producer", weights={"2": 1.0})

    about = [one for one in problems([big, edge], rows, _month(0)) if "объёма" in one]

    assert about == [], (
        f"доля ровно 100/1000 объявлена меньше десятой части: {about}"
    )

    # а чуть меньшая доля называется
    smaller = _rows([
        {"DATA": _month(0), "well": "1", "WLPT_Diff": 900.0, "WOMT_Diff": 90.0},
        {"DATA": _month(0), "well": "2", "WLPT_Diff": 50.0, "WOMT_Diff": 5.0},
    ])
    assert [one for one in problems([big, edge], smaller, _month(0))
            if "объёма" in one], "доля 5 % не названа"
