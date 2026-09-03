"""Сквозной цикл управления на тривиальном суррогате.

Ценность этих тестов в том, что контур целиком работает без единого расчёта
ГГДМ: пока товарищи поднимают симулятор, агенты и UI строятся не вслепую.
"""

from __future__ import annotations

from datetime import date

import pytest
from conftest import make_row

from aios import economics, loop, optimise, summary, trust
from aios.deck import ProducerControl, ScheduleStep
from aios.margin import Norms
from aios.surrogate import DeclineSurrogate

FLUIDS = summary.Fluids()
CONTROL_START = date(2014, 1, 1)


@pytest.fixture
def norms():
    assumptions, _ = economics.load_norms()
    return Norms.from_assumptions(assumptions)


def well_history(well: str, watercut_start: float, watercut_step: float, rate: float = 30.0):
    rows = []
    for index in range(12):
        month = date(2013, index + 1, 1)
        days = 30
        liquid_m3 = rate * days
        watercut = min(watercut_start + watercut_step * index, 0.999)
        oil_m3 = liquid_m3 * (1 - watercut)
        oil_t = oil_m3 * FLUIDS.oil_density / 1000
        liquid_t = oil_t + (liquid_m3 - oil_m3) * FLUIDS.water_density / 1000
        rows.append(make_row(month.isoformat(), well, wlpr=rate, womr=oil_t / days,
                             liquid=liquid_t, oil=oil_t))
    return rows


@pytest.fixture
def surrogate():
    """Два фонда: «хороший» с обводнённостью 0,5 и «плохой» на грани 0,99."""
    history = well_history("good", 0.50, 0.005) + well_history("bad", 0.985, 0.0005)
    return DeclineSurrogate.fit(history, FLUIDS)


def plan(months: int = 18):
    steps = []
    for index in range(months):
        year = 2014 + index // 12
        month = index % 12 + 1
        steps.append(
            ScheduleStep(
                date(year, month, 1),
                (
                    ProducerControl("good", "OPEN", "LRAT", 30.0, 50.0),
                    ProducerControl("bad", "OPEN", "LRAT", 30.0, 50.0),
                ),
                (),
            )
        )
    return steps


class _Responsive:
    """Суррогат, чей прогноз следует за уставками расписания.

    Нужен ровно для одного свойства: после того как планировщик срежет отбор,
    следующий прогноз обязан показать **меньший** отбор. На тривиальном
    бейзлайне этого не видно — он предсказывает по истории скважины, а не по
    поданной уставке, — а именно это свойство и делает коридор баланса воды
    неподвижной точкой, а не одним проходом (Р-189).
    """

    def predict(self, steps):
        rows = []
        for step in steps:
            for control in step.producers:
                rate = (control.liquid_rate or 0.0) if control.status == "OPEN" else 0.0
                oil_t = rate * 0.1 * FLUIDS.oil_density / 1000
                water_t = rate * 0.9 * FLUIDS.water_density / 1000
                rows.append(make_row(step.date.isoformat(), control.well, wlpr=rate,
                                     womr=oil_t / 30, liquid=oil_t + water_t, oil=oil_t))
            for control in step.injectors:
                rate = (control.rate or 0.0) if control.status == "OPEN" else 0.0
                rows.append(make_row(step.date.isoformat(), control.well,
                                     wwir=rate, injection=rate * 30.0))
        return rows

    def in_training_range(self, steps):
        return []


def test_an_empty_search_keeps_the_starting_scenario(norms, surrogate, monkeypatch):
    """Пустой ответ поиска — «ничего не менять», а не «вернуться к деку».

    `apply_choices(base_steps, [])` отдаёт опорное расписание и молча теряет
    карту старта из `--base-schedule`: прирост считался бы от одного сценария, а
    расписание собиралось из другого. Поиск возвращает пустой список законно —
    когда стартовая точка вне области доверия (Р-174, Р-190).
    """
    monkeypatch.setattr(loop.optimise, "choose_rates", lambda *a, **k: [])

    start = {("production", "good"): 1.5}
    result = loop.run_cycle(
        surrogate, plan(), norms,
        control_start=CONTROL_START, start_date=CONTROL_START,
        start_multipliers=start,
    )

    rates = {control.well: control.liquid_rate
             for step in result.steps for control in step.producers
             if control.status == "OPEN"}
    assert rates.get("good") == pytest.approx(45.0), (
        f"стартовый сценарий обязан сохраниться, получено {rates}"
    )
    # Прирост при этом нулевым быть не обязан: цикл, кроме подбора режимов,
    # ещё останавливает убыточные скважины — и это отдельное решение, которое
    # пустой ответ поиска не отменяет.
    assert result.base_chdd_m == pytest.approx(
        loop.run_cycle(surrogate, plan(), norms, control_start=CONTROL_START,
                       start_date=CONTROL_START, start_multipliers=start,
                       tune_rates=False).base_chdd_m
    ), "точка отсчёта остаётся стартовым сценарием"


def test_the_corridor_holds_after_the_cuts_that_follow_it(norms):
    """Срезки отбора поднимают компенсацию — цикл обязан довести её до коридора.

    Компенсация есть отношение закачки к отобранному объёму, и каждая срезка
    отбора её поднимает. Пока коридор подбирался первым и одним проходом,
    расписание выходило из планировщика нарушающим тот самый коридор: замер
    ревизии — 16 нарушений и 133–141 % на полном цикле, то есть при включённой
    проверке fail-closed (Р-165) ни один кандидат не проходил сертификацию.
    Лечится порядком **и** пересчётом прогноза (Р-189).
    """
    from aios.deck import InjectorControl, ProducerControl, ScheduleStep
    from aios.planner import Limits

    steps = [ScheduleStep(date(2014, month, 1),
                          (ProducerControl("P1", "OPEN", "LRAT", 100.0, 50.0),),
                          (InjectorControl("I1", "WATER", "OPEN", "RATE", 100.0, 300.0),))
             for month in range(1, 13)]

    result = loop.run_cycle(
        _Responsive(), steps, norms,
        control_start=CONTROL_START, start_date=CONTROL_START,
        tune_rates=False,
        # предел по жидкости срежет отбор вдвое уже ПОСЛЕ того, как коридор
        # подобран по исходному отбору
        limits=Limits(max_field_liquid_m3_per_day=50.0,
                      enforce_water_balance=True,
                      water_balance_rule="compensation",
                      water_tolerance=1.3),
        fluids=FLUIDS,
    )

    assert result.case_problems == (), (
        "расписание, только что приведённое к ограничениям, их нарушает: "
        + "; ".join(result.case_problems)
    )
    assert result.allocations, "срезки обязаны быть названы, а не потеряны"


def test_cycle_stops_the_watered_out_well_and_keeps_the_good_one(surrogate, norms):
    """Скважину останавливает причинное решение, а не множитель перебора.

    До 04.09 остановку давал одиночный перебор режимов: множитель 0,0 выигрывал
    по марже **одной** скважины на прогнозе без отклика соседей. Замер (Б3)
    показал цену: 33 из 72 добывающих на историческом расписании, −63…−68 % ЧДД
    по полному расчёту. Ноль убран из умолчания перебора, и остановка пришла
    оттуда, где ей место, — из `policy.plan_shutdowns`, который смотрит на
    накопленный убыток и **объясняет** решение.
    """
    result = loop.run_cycle(surrogate, plan(), norms,
                            control_start=CONTROL_START, start_date=CONTROL_START)

    chosen = {choice.well: choice.multiplier for choice in result.rate_choices}
    assert chosen["bad"] > 0.0, "остановка вернулась в перебор режимов"
    assert chosen["bad"] < chosen["good"], "обводнённая скважина даже не заглушена"

    shut = [action for action in result.actions
            if action.well == "bad" and action.kind == "shut"]
    assert shut, f"скважину никто не остановил: {[(a.well, a.kind) for a in result.actions]}"
    assert "убыток" in shut[0].reason, "остановка без причины в тексте решения"

    assert result.gain_m > 0
    assert result.is_improvement


def test_the_causal_policy_catches_what_rate_tuning_missed(surrogate, norms):
    """Подбор смотрит на горизонт целиком, политика — только назад.

    Скважина, которая становится убыточной по ходу горизонта уже на выбранном
    режиме, ловится именно политикой, и её решение датировано.
    """
    result = loop.run_cycle(surrogate, plan(), norms, control_start=CONTROL_START,
                            start_date=CONTROL_START, tune_rates=False)

    assert [action.well for action in result.actions] == ["bad"]
    assert result.actions[0].date > CONTROL_START
    assert result.rate_choices == ()


def test_cycle_does_not_force_wells_beyond_their_history(surrogate, norms):
    """Форсаж бейзлайн подтвердить не может, поэтому вверх режимы не идут."""
    result = loop.run_cycle(surrogate, plan(), norms,
                            control_start=CONTROL_START, start_date=CONTROL_START)

    assert all(choice.multiplier <= 1.0 for choice in result.rate_choices)
    assert result.warnings == ()


def test_cycle_leaves_a_profitable_fund_alone(norms):
    history = well_history("good", 0.50, 0.005) + well_history("also-good", 0.55, 0.005)
    model = DeclineSurrogate.fit(history, FLUIDS)
    steps = [
        ScheduleStep(
            date(2014, month, 1),
            (
                ProducerControl("good", "OPEN", "LRAT", 30.0, 50.0),
                ProducerControl("also-good", "OPEN", "LRAT", 30.0, 50.0),
            ),
            (),
        )
        for month in range(1, 13)
    ]

    result = loop.run_cycle(model, steps, norms,
                            control_start=CONTROL_START, start_date=CONTROL_START)

    assert result.actions == ()
    assert all(choice.multiplier == 1.0 for choice in result.rate_choices)
    assert result.gain_m == pytest.approx(0.0)


def test_resulting_schedule_breaks_no_constraints(surrogate, norms):
    result = loop.run_cycle(surrogate, plan(), norms,
                            control_start=CONTROL_START, start_date=CONTROL_START)
    assert result.problems == ()


def test_forecast_is_never_reported_as_confirmed(surrogate, norms):
    """Прирост на суррогате — прогноз. Пока не подтверждён расчётом, так и пишем."""
    result = loop.run_cycle(surrogate, plan(), norms,
                            control_start=CONTROL_START, start_date=CONTROL_START)

    assert result.confirmed is False
    assert "ПРОГНОЗ, не подтверждён" in result.summary()


def test_uncalibrated_margin_is_visible_in_the_cycle_report(surrogate, norms):
    margin = trust.OptimismMargin.fit([(100.0, 90.0)])
    result = loop.run_cycle(surrogate, plan(), norms, control_start=CONTROL_START,
                            start_date=CONTROL_START, margin=margin)

    assert "НЕ КАЛИБРОВАН" in result.summary()


def test_cycle_report_shows_the_raw_and_reserved_forecast(surrogate, norms):
    margin = trust.OptimismMargin.fit([(100.0 + gap, 100.0) for gap in range(1, 10)])
    result = loop.run_cycle(surrogate, plan(), norms, control_start=CONTROL_START,
                            start_date=CONTROL_START, margin=margin)

    assert margin.calibrated
    assert "Прогноз после запаса" in result.summary()
    assert "сырой" in result.summary()


def test_confirmation_replaces_the_forecast_with_the_real_calculation(surrogate, norms):
    result = loop.run_cycle(surrogate, plan(), norms,
                            control_start=CONTROL_START, start_date=CONTROL_START)

    # «выгрузка ГГДМ»: тот же фонд, но нефти вдвое меньше прогноза
    measured = [dict(row, WOMT_Diff=row["WOMT_Diff"] / 2) for row in result.rows]
    confirmed = loop.confirm(result, measured, CONTROL_START, base_rows=result.rows)

    assert confirmed.confirmed is True
    assert confirmed.chdd_m < result.chdd_m
    assert confirmed.actions == result.actions
    assert "подтверждено полным расчётом" in confirmed.summary()


def test_extrapolation_warnings_reach_the_result(surrogate, norms):
    """Форсаж выше исторического обязан дойти до решения, а не осесть в логе."""
    forced = [
        ScheduleStep(
            step.date,
            tuple(
                ProducerControl(c.well, c.status, c.mode, 200.0, c.bhp) for c in step.producers
            ),
            step.injectors,
        )
        for step in plan(6)
    ]

    result = loop.run_cycle(surrogate, forced, norms,
                            control_start=CONTROL_START, start_date=CONTROL_START)

    assert result.warnings
    assert any("исторического максимума" in warning for warning in result.warnings)
    assert "Предупреждения" in result.summary()


def test_summary_names_both_numbers(surrogate, norms):
    result = loop.run_cycle(surrogate, plan(), norms,
                            control_start=CONTROL_START, start_date=CONTROL_START)
    text = result.summary()

    assert "ЧДД базового сценария" in text
    assert "ЧДД после решений" in text
    assert "Прирост" in text
    assert "скв. bad" in text


def test_planner_has_the_last_word_in_the_cycle(surrogate, norms):
    """Ограничения объекта применяются после скважинных решений, а не вместо них."""
    from aios.planner import Limits

    result = loop.run_cycle(
        surrogate, plan(), norms,
        control_start=CONTROL_START, start_date=CONTROL_START,
        limits=Limits(max_field_liquid_m3_per_day=20.0, enforce_water_balance=False),
    )

    assert result.allocations
    assert {item.resource for item in result.allocations} == {"жидкость"}
    for step in result.steps:
        working = sum(c.liquid_rate or 0.0 for c in step.producers if c.status == "OPEN")
        assert working <= 20.0 + 1e-9
    assert "Согласование" in result.summary() or "жидкость:" in result.summary()


def test_without_limits_the_planner_stays_out(surrogate, norms):
    result = loop.run_cycle(surrogate, plan(), norms,
                            control_start=CONTROL_START, start_date=CONTROL_START)
    assert result.allocations == ()


def test_joint_search_sees_the_same_limits_the_planner_enforces(surrogate, norms, monkeypatch):
    """Поиск режимов обязан узнавать об ограничении раньше планировщика (Р-54, `docs/decisions.md`).

    Иначе он выбирает режимы по ЧДД, которого не будет: планировщик режет их
    после. `choose_joint_rates` умеет учитывать `limits`/`norms`/`fluids` с
    21.08 (`aios/optimise.py`), но `run_cycle` их не передавал.
    """
    from aios.planner import Limits

    limits = Limits(max_field_liquid_m3_per_day=20.0, enforce_water_balance=False)
    seen = {}
    original = optimise.choose_joint_rates

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(loop.optimise, "choose_joint_rates", spy)

    loop.run_cycle(
        surrogate, plan(), norms,
        control_start=CONTROL_START, start_date=CONTROL_START,
        joint_rate_search=True, limits=limits, fluids=FLUIDS,
    )

    assert seen.get("limits") is limits
    assert seen.get("norms") is norms
    assert seen.get("fluids") is FLUIDS


def test_forecast_reports_rows_the_calculator_threw_away():
    """Прогноз считается по строкам суррогата, а негодные расчётчик выбрасывает.

    Методика исключает строку с отрицательным приростом целиком, поэтому
    прогнозный ЧДД падает почти пропорционально доле таких строк: 5 % негодных
    дают −5,3 %, 20 % — −20,0 % (измерено 20.08). Само по себе это правильно, но
    молча: цикл берёт из расчёта только число денег и теряет счётчик.

    На странице настоящего прогона счётчик показывается, а в прогнозе — нет, то
    есть честный указатель стоит ровно там, где он наименее нужен. Замер А4 на
    сценарии с ВНС дал 22 563 нарушения физических проверок, так что случай не
    гипотетический.
    """
    from datetime import date

    from conftest import make_row

    from aios import loop as loop_module

    rows = []
    for month in range(6):
        broken = month % 3 == 0
        rows.append(make_row(f"2014-{month + 1:02d}-01", "1", wlpr=30.0,
                             womr=-1.0 if broken else 5.0,
                             liquid=-30.0 if broken else 900.0,
                             oil=-30.0 if broken else 150.0))

    result = loop_module._npv(rows, date(2014, 1, 1))
    assert result.excluded_negative_rows == 2, "строки с отрицательным приростом исключены"


def test_summary_names_the_discarded_rows_when_there_are_any():
    """Прогноз, посчитанный на обрезанном наборе строк, обязан это сказать."""
    from aios.loop import CycleResult

    clean = CycleResult(base_chdd_m=100.0, chdd_m=110.0, actions=(), rate_choices=(),
                        allocations=(), steps=(), warnings=(), problems=())
    assert "исключено" not in clean.summary().lower()

    spoiled = CycleResult(base_chdd_m=100.0, chdd_m=110.0, actions=(), rate_choices=(),
                          allocations=(), steps=(), warnings=(), problems=(),
                          excluded_rows=4231)
    text = spoiled.summary()
    assert "4231" in text
    assert "исключ" in text.lower()


def test_confirmed_gain_is_a_difference_of_two_full_calculations(surrogate, norms):
    """Подтверждённый прирост нельзя считать от прогнозного базового.

    `confirm` заменял только `chdd_m`, а `base_chdd_m` оставлял прогнозным — и
    `gain_m` становился разностью двух разных источников, подписанной
    «подтверждено полным расчётом». Замер 21.08 на синтетике: прогноз обещал
    прирост −100,0, после подтверждения печаталось −999,2, потому что базовое
    число осталось от суррогата.

    `scripts/run_cycle.py` эту ловушку обходил вручную — считал базовый ЧДД сам и
    печатал разность мимо `gain_m`. То есть защита держалась на том, что
    единственный вызывающий про неё знает.
    """
    result = loop.run_cycle(surrogate, plan(), norms,
                            control_start=CONTROL_START, start_date=CONTROL_START)
    measured = [dict(row, WOMT_Diff=row["WOMT_Diff"] / 2) for row in result.rows]
    base_measured = [dict(row, WOMT_Diff=row["WOMT_Diff"] * 0.9) for row in result.rows]

    confirmed = loop.confirm(result, measured, CONTROL_START, base_rows=base_measured)

    from aios import economics
    expected_base = economics.compute_npv(
        base_measured, headers=economics.REQUIRED_COLUMNS,
        start_date=CONTROL_START.isoformat(),
    ).chdd_m
    assert confirmed.base_chdd_m == pytest.approx(expected_base), (
        "базовое число осталось прогнозным"
    )
    assert confirmed.gain_m == pytest.approx(confirmed.chdd_m - expected_base)
