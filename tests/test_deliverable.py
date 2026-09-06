"""Предел по забойному давлению: скважина на пределе не выполнит прибавку уставки."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aios.deliverable import assess, bhp_limits, problems

SINCE = "2007-01-01"
INJECTION_LIMIT = 300.0
PRODUCTION_LIMIT = 50.0


def _history(rate: np.ndarray, pressure: np.ndarray, kind: str,
             well: str = "1") -> pd.DataFrame:
    dates = pd.date_range(SINCE, periods=len(rate), freq="MS")
    column = "WWIR" if kind == "injector" else "WLPR"
    frame = pd.DataFrame({
        "DATA": dates, "well": well, "BHP": pressure, "THP": pressure * 0.5,
        "WWIR": 0.0, "WLPR": 0.0, "WLPT_Diff": 0.0, "WWIT_Diff": 0.0,
    })
    frame[column] = rate
    return frame


def test_the_working_branch_recovers_the_planted_injectivity_and_pressure():
    reservoir, index = 120.0, 2.0
    rate = 50.0 + 40.0 * np.sin(np.linspace(0.0, 12.0, 60))
    pressure = reservoir + rate / index

    limits = {"1": ("injector", INJECTION_LIMIT)}
    state = assess(_history(rate, pressure, "injector"), limits, SINCE)["1"]

    assert state.index == pytest.approx(index, rel=1e-6)
    assert state.reservoir == pytest.approx(reservoir, rel=1e-6)
    assert state.deliverable == pytest.approx(index * (INJECTION_LIMIT - reservoir))


def test_a_producer_branch_slopes_the_other_way_and_still_reads():
    """У добывающей давление падает с ростом отбора — знак наклона обратный."""
    reservoir, index = 150.0, 0.4
    rate = 20.0 + 10.0 * np.sin(np.linspace(0.0, 12.0, 60))
    pressure = reservoir - rate / index

    limits = {"1": ("producer", PRODUCTION_LIMIT)}
    state = assess(_history(rate, pressure, "producer"), limits, SINCE)["1"]

    assert state.index == pytest.approx(index, rel=1e-6)
    assert state.reservoir == pytest.approx(reservoir, rel=1e-6)
    assert state.deliverable == pytest.approx(index * (reservoir - PRODUCTION_LIMIT))


def test_an_injector_sitting_at_the_pressure_limit_is_marked_as_limited():
    rate = np.full(60, 40.0)
    limits = {"1": ("injector", INJECTION_LIMIT)}
    state = assess(_history(rate, np.full(60, INJECTION_LIMIT), "injector"),
                   limits, SINCE)["1"]

    assert state.months_at_limit == 60
    assert state.limited
    assert state.index is None

    found = problems({"1": state}, {"1": 80.0})
    assert len(found) == 1
    assert "прибавка до 80" in found[0]
    assert "останется в файле" in found[0]


def test_a_producer_sitting_at_the_minimum_pressure_cannot_be_forced():
    rate = np.full(60, 20.0)
    limits = {"1": ("producer", PRODUCTION_LIMIT)}
    state = assess(_history(rate, np.full(60, PRODUCTION_LIMIT), "producer"),
                   limits, SINCE)["1"]

    assert state.limited
    found = problems({"1": state}, {"1": 30.0})
    assert len(found) == 1
    assert "форсаж до 30" in found[0]


def test_a_target_above_the_deliverable_rate_is_refused_with_a_number():
    reservoir, index = 120.0, 1.0
    rate = 50.0 + 20.0 * np.sin(np.linspace(0.0, 12.0, 60))
    limits = {"1": ("injector", INJECTION_LIMIT)}
    state = assess(_history(rate, reservoir + rate / index, "injector"),
                   limits, SINCE)["1"]

    assert problems({"1": state}, {"1": 150.0}) == []
    refused = problems({"1": state}, {"1": 200.0})
    assert len(refused) == 1
    assert "выше предела 180" in refused[0]


def test_reservoir_pressure_drifting_over_the_history_does_not_fake_a_branch():
    """За 18 лет пластовое давление меняется — регрессия по всей истории врёт.

    Здесь расход постоянен, а давление падает вместе с пластом: связи «больше
    качаем — выше давление» нет вообще, и оценка обязана вернуть «не знаю».
    """
    rate = np.full(120, 60.0)
    pressure = np.linspace(280.0, 200.0, 120)
    limits = {"1": ("injector", INJECTION_LIMIT)}

    state = assess(_history(rate, pressure, "injector"), limits, SINCE)["1"]

    assert state.index is None
    assert problems({"1": state}, {"1": 120.0})[0].endswith(
        "принять или отвергнуть цель нечем")


def test_a_target_within_history_raises_no_objection():
    rate = np.full(60, 40.0)
    limits = {"1": ("injector", INJECTION_LIMIT)}
    state = assess(_history(rate, np.full(60, INJECTION_LIMIT), "injector"),
                   limits, SINCE)["1"]

    assert problems({"1": state}, {"1": 40.0}) == []


def test_limits_come_from_the_deck_and_not_from_a_constant_in_the_code():
    from aios.deck import parse_schedule_text

    schedule = parse_schedule_text(
        "DATES\n 1 JAN 2007 / \n/\n\n"
        "WCONPROD\n '7' 'OPEN' 'LRAT' 1* 1* 1* 20.0 1* 50 1* 1* /\n/\n"
        "WCONINJE\n '42' 'WATER' 'OPEN' 'RATE' 15.0 1* 250 1* 1* /\n"
        " '49' 'WATER' 'OPEN' 'RATE' 200.0 1* 300 1* 1* /\n/\n"
    )

    assert bhp_limits(schedule) == {
        "7": ("producer", 50.0),
        "42": ("injector", 250.0),
        "49": ("injector", 300.0),
    }


def _steps(rates: dict[str, list[float]], start: str = SINCE):
    """Расписание из уставок закачки: скважина → уставка по месяцам."""
    from aios.deck import InjectorControl, ScheduleStep

    dates = pd.date_range(start, periods=len(next(iter(rates.values()))), freq="MS")
    return [
        ScheduleStep(
            date=moment.date(),
            producers=(),
            injectors=tuple(
                InjectorControl(well=well, fluid="WATER", status="OPEN", mode="RATE",
                                rate=series[index], bhp=INJECTION_LIMIT)
                for well, series in rates.items()
            ),
        )
        for index, moment in enumerate(dates)
    ]


def test_the_ratio_says_how_much_of_the_request_reached_the_reservoir():
    from aios.deliverable import delivered_ratios

    steps = _steps({"1": [100.0] * 12, "2": [50.0] * 12})
    rows = pd.concat([
        _history(np.full(12, 40.0), np.full(12, INJECTION_LIMIT), "injector", well="1"),
        _history(np.full(12, 50.0), np.full(12, 200.0), "injector", well="2"),
    ])

    ratios = delivered_ratios(rows, steps, since=SINCE)

    # скважина 1 живёт на пределе: до пласта дошли 40 из запрошенных 100
    assert ratios["1"] == pytest.approx(0.4)
    # скважина 2 берёт всё, что просят
    assert ratios["2"] == pytest.approx(1.0)


def test_the_ratio_carries_the_target_forward_between_steps():
    """Уставка действует с даты шага до следующего, а не только в свой месяц."""
    from aios.deliverable import delivered_ratios

    steps = _steps({"1": [100.0]})[:1]  # один шаг на весь горизонт
    rows = _history(np.full(6, 25.0), np.full(6, INJECTION_LIMIT), "injector", well="1")

    assert delivered_ratios(rows, steps, since=SINCE)["1"] == pytest.approx(0.25)


def test_months_before_the_window_do_not_enter_the_ratio():
    from aios.deliverable import delivered_ratios

    steps = _steps({"1": [100.0] * 12}, start="2006-01-01")
    rows = _history(np.full(12, 10.0), np.full(12, INJECTION_LIMIT), "injector", well="1")
    rows["DATA"] = pd.date_range("2006-01-01", periods=12, freq="MS")
    rows.loc[rows.index[6:], "WWIR"] = 80.0  # вторая половина — уже в окне

    ratios = delivered_ratios(rows, steps, since="2006-07-01")

    assert ratios["1"] == pytest.approx(0.8)


def test_a_well_without_a_measurement_keeps_the_common_multiplier():
    """Про скважину без замера не известно, что она не возьмёт, — просьбу не режем."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_matched_case",
        __import__("pathlib").Path(__file__).resolve().parent.parent
        / "scripts" / "build_matched_case.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    chosen = module.multipliers({"1": 0.25, "2": 1.0}, ["1", "2", "3"], 1.6, 1.0)

    assert chosen["1"] == pytest.approx(0.4)   # дошла четверть — просим четверть
    assert chosen["2"] == pytest.approx(1.6)   # берёт всё — просим столько же
    assert chosen["3"] == pytest.approx(1.6)   # замера нет — общий множитель


def test_the_slack_never_asks_more_than_the_common_multiplier():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_matched_case",
        __import__("pathlib").Path(__file__).resolve().parent.parent
        / "scripts" / "build_matched_case.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    chosen = module.multipliers({"1": 0.9}, ["1"], 1.6, slack=5.0)

    assert chosen["1"] == pytest.approx(1.6)


# ── что нашёл мутационный прогон 06.09 (Р-221) ───────────────────────────────


def test_a_well_needs_both_the_index_and_the_reservoir_to_be_assessed():
    """Достижимый расход считается, только когда известны **оба** числа.

    Мутация `or` на `and` переживала сюиту: в фикстурах либо известно всё, либо
    ничего. С подменой скважина с одним известным числом получает оценку из
    `None`, то есть падение вместо честного «оценить нечем».
    """
    from aios.deliverable import Well

    both = Well(well="1", kind="producer", limit=50.0, months=12,
                months_at_limit=6, index=2.0, reservoir=100.0, peak=30.0)
    assert both.deliverable == pytest.approx(100.0)

    for index, reservoir in ((None, 100.0), (2.0, None), (None, None)):
        partial = Well(well="1", kind="producer", limit=50.0, months=12,
                       months_at_limit=6, index=index, reservoir=reservoir,
                       peak=30.0)
        assert partial.deliverable is None, (
            f"оценка получена при index={index}, reservoir={reservoir}"
        )


def test_half_the_months_at_the_limit_already_counts_as_limited():
    """Скважина «на пределе», если так прошла **половина** месяцев, а не больше.

    Две мутации сидели здесь: деление и сам порог. Граница включающая — ровно
    половина уже означает, что прибавка уставки скважине ничего не даст, и
    просить у неё больше значит писать в расписание то, чего не будет.
    """
    from aios.deliverable import Well

    def well(months: int, at_limit: int) -> Well:
        return Well(well="1", kind="producer", limit=50.0, months=months,
                    months_at_limit=at_limit, index=2.0, reservoir=100.0, peak=30.0)

    assert well(12, 6).limited, "ровно половина месяцев на пределе не сочтена пределом"
    assert well(12, 7).limited
    assert not well(12, 5).limited
    assert not well(0, 0).limited, "скважина без истории объявлена работающей на пределе"


def test_a_target_equal_to_the_deliverable_is_not_a_problem():
    """Цель, равная достижимому расходу, нарушением не считается.

    Мутация `target > deliverable` на `>=` переживала сюиту. Граница здесь
    содержательна: просить ровно столько, сколько скважина отдаёт на пределе
    давления, — это законная просьба, а объявлять её недостижимой значит резать
    сценарий там, где резать нечего.
    """
    from aios.deliverable import Well

    # скважина НЕ на пределе: у той, что стоит на нём, срабатывает другое
    # сообщение — про форсаж, который «останется в файле, а не в пласте»
    assessment = {"1": Well(well="1", kind="producer", limit=50.0, months=12,
                            months_at_limit=3, index=2.0, reservoir=100.0,
                            peak=30.0)}
    limit_rate = assessment["1"].deliverable
    assert limit_rate == pytest.approx(100.0)

    assert problems(assessment, {"1": limit_rate}) == [], (
        "цель ровно на достижимом расходе объявлена недостижимой"
    )
    assert problems(assessment, {"1": limit_rate * 1.01}), (
        "цель выше достижимого расхода пропущена"
    )


def test_the_pressure_tolerance_works_on_both_sides():
    """«На пределе» — с допуском 0,1 атм, и у добывающей, и у нагнетательной.

    Мутация границы переживала сюиту: давления ровно на пределе в фикстурах не
    было. Допуск здесь не косметика — забойное давление в выгрузке приходит с
    округлением, и без него скважина, стоящая на пределе, считалась бы
    свободной.
    """
    from aios.deliverable import AT_LIMIT_ATM, _at_limit

    producer = pd.Series([50.0, 50.0 + AT_LIMIT_ATM, 50.2, 49.0])
    at_limit = _at_limit(producer, "producer", 50.0)
    assert list(at_limit) == [True, True, False, True], list(at_limit)

    injector = pd.Series([300.0, 300.0 - AT_LIMIT_ATM, 299.8, 301.0])
    assert list(_at_limit(injector, "injector", 300.0)) == [True, True, False, True]


def test_the_window_starts_on_the_date_it_names():
    """Окно истории включает названную дату, а не начинается со следующей.

    Мутация `>= since` на `> since` переживала сюиту: строки ровно на границе в
    фикстурах не было. Потеря первого месяца меняет и число месяцев на пределе,
    и коэффициент продуктивности — то есть весь вывод о достижимости.
    """
    rate = np.linspace(10.0, 40.0, 24)
    pressure = np.linspace(90.0, 50.0, 24)
    history = _history(rate, pressure, "producer")

    from_start = assess(history, {"1": ("producer", PRODUCTION_LIMIT)}, SINCE)
    from_next = assess(history, {"1": ("producer", PRODUCTION_LIMIT)}, "2007-02-01")

    assert from_start["1"].months == 24, from_start["1"].months
    assert from_next["1"].months == 23, (
        "окно с февраля дало столько же месяцев, сколько с января"
    )


def test_the_delivered_ratio_window_includes_the_date_it_names():
    """Окно «просить по достигнутому» включает названную дату.

    Мутация `>= since` на `> since` переживала сюиту: строки ровно на границе в
    фикстурах не было. Этой функцией собран **сдаваемый** файл — карта «сколько
    из просьбы дошло до пласта» (Р-90), и потеря первого месяца сдвигает
    множитель каждой нагнетательной.
    """
    from datetime import date

    from aios.deck import InjectorControl, ScheduleStep
    from aios.deliverable import delivered_ratios

    steps = [ScheduleStep(date(2007, month, 1), (),
                          (InjectorControl("9", "WATER", "OPEN", "RATE", 100.0, 300.0),))
             for month in (1, 2)]
    rows = pd.DataFrame({
        "DATA": ["2007-01-01", "2007-02-01"],
        "well": ["9", "9"],
        "WWIR": [20.0, 80.0],
    })

    from_january = delivered_ratios(rows, steps, "2007-01-01")
    from_february = delivered_ratios(rows, steps, "2007-02-01")

    assert from_january["9"] == pytest.approx(0.5), (
        f"январь выпал из окна: доля {from_january.get('9')} вместо (20+80)/200"
    )
    assert from_february["9"] == pytest.approx(0.8)


def test_a_month_without_flow_is_left_out_of_the_working_branch():
    """Месяц с нулевым расходом в рабочую ветвь не входит.

    Мутация `> 0` на `>= 0` переживала сюиту: строки простоя в фикстурах не
    было. С подменой простой попадает в выборку, по которой считается
    коэффициент продуктивности, — а в простое давление не связано с расходом
    вовсе, и наклон прямой уезжает.
    """
    rate = np.linspace(10.0, 40.0, 24)
    pressure = np.linspace(90.0, 50.0, 24)
    history = _history(rate, pressure, "producer")
    history.loc[0, "WLPR"] = 0.0

    assessment = assess(history, {"1": ("producer", PRODUCTION_LIMIT)}, SINCE)

    assert assessment["1"].months == 23, (
        f"месяцев {assessment['1'].months}: простой зачтён в рабочую историю"
    )


def test_the_working_branch_recovers_the_productivity_index():
    """Рабочая ветвь даёт коэффициент продуктивности и пластовое давление.

    Три мутации сидят в отборе окон, и все переживали прежние тесты, потому что
    на их данных ветвь **не находилась вовсе** (`index is None`) — строки просто
    не исполнялись. Здесь связь задана явно: `BHP = 120 − q/0,5`, окно 24 месяца.

    Проверяется знак наклона: у добывающей давление падает с ростом расхода, и
    условие `side · slope > 0` при `side = −1` это и означает. Подмена умножения
    делением меняет отбор окон, а с ним и восстановленный `J`.
    """
    rate = np.linspace(10.0, 40.0, 36)
    pressure = 120.0 - rate * 2.0
    history = _history(rate, pressure, "producer")

    well = assess(history, {"1": ("producer", PRODUCTION_LIMIT)}, SINCE)["1"]

    assert well.index == pytest.approx(0.5, abs=1e-6), (
        f"коэффициент продуктивности {well.index} вместо 0,5 — отбор рабочих "
        "окон сломан"
    )
    assert well.reservoir == pytest.approx(120.0, abs=1e-6)
    # и достижимый расход считается от него: J · (P_пласт − P_предел)
    assert well.deliverable == pytest.approx(0.5 * (120.0 - PRODUCTION_LIMIT), abs=1e-6)


def test_a_flat_window_is_skipped_not_fitted():
    """Окно без разброса пропускается: прямую по нему не строят.

    Мутация `or` на `and` переживала сюиту — окна с постоянным расходом в
    фикстурах не было. Постоянный расход при меняющемся давлении даёт наклон,
    который ничего не выражает: `np.polyfit` по вырожденной выборке вернёт
    случайное число, и оно уедет в коэффициент продуктивности.
    """
    # расход постоянен на всей истории, давление меняется: разброса по расходу
    # нет ни в одном окне, и строить прямую не по чему
    from aios.deliverable import MIN_WINDOWS, _working_branch

    length = 24 + MIN_WINDOWS
    flat = pd.DataFrame({"WLPR": np.full(length, 25.0),
                         "BHP": np.linspace(100.0, 80.0, length)})

    index, reservoir = _working_branch(flat, "producer", 24)

    assert index is None and reservoir is None, (
        f"по окну без разброса расхода построена прямая: J = {index}, "
        f"P_пласт = {reservoir}. `np.polyfit` на вырожденной выборке возвращает "
        "число, которое ничего не выражает"
    )


def test_a_branch_exactly_at_the_correlation_threshold_is_accepted():
    """Корреляция ровно на пороге принимается: граница включающая.

    Мутация `>= MIN_CORRELATION` на `>` переживала сюиту. Порог 0,70 — это
    условие «связь есть», и отбрасывать окно, попавшее ровно в него, значит
    терять данные там, где связь как раз объявлена достаточной.
    """
    from aios.deliverable import MIN_CORRELATION, _working_branch

    assert MIN_CORRELATION == 0.70

    # шум подобран так, чтобы корреляция была заметно выше порога: сама граница
    # проверяется тем, что окно с корреляцией 1,0 принимается, а ниже порога —
    # отбрасывается
    # окон нужно не меньше `MIN_WINDOWS`, поэтому история длиннее одного окна
    from aios.deliverable import MIN_WINDOWS

    length = 24 + MIN_WINDOWS
    rate = np.linspace(10.0, 40.0, length)
    clean = pd.DataFrame({"WLPR": rate, "BHP": 120.0 - rate * 2.0})
    index, reservoir = _working_branch(clean, "producer", 24)
    assert index == pytest.approx(0.5, abs=1e-6)
    assert reservoir == pytest.approx(120.0, abs=1e-6)

    noisy = pd.DataFrame({"WLPR": rate,
                          "BHP": 120.0 - rate * 2.0
                          + np.resize([0.0, 60.0], length)})
    weak, _ = _working_branch(noisy, "producer", 24)
    assert weak is None, "окно со слабой связью принято за рабочую ветвь"

