"""Сквозной цикл управления на тривиальном суррогате.

Ценность этих тестов в том, что контур целиком работает без единого расчёта
ГГДМ: пока товарищи поднимают симулятор, агенты и UI строятся не вслепую.
"""

from __future__ import annotations

from datetime import date

import pytest
from conftest import make_row

from aios import economics, loop, summary
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


def test_cycle_stops_the_watered_out_well_and_keeps_the_good_one(surrogate, norms):
    result = loop.run_cycle(surrogate, plan(), norms,
                            control_start=CONTROL_START, start_date=CONTROL_START)

    chosen = {choice.well: choice.multiplier for choice in result.rate_choices}
    assert chosen["bad"] == 0.0
    assert chosen["good"] > 0.0
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


def test_confirmation_replaces_the_forecast_with_the_real_calculation(surrogate, norms):
    result = loop.run_cycle(surrogate, plan(), norms,
                            control_start=CONTROL_START, start_date=CONTROL_START)

    # «выгрузка ГГДМ»: тот же фонд, но нефти вдвое меньше прогноза
    measured = [dict(row, WOMT_Diff=row["WOMT_Diff"] / 2) for row in result.rows]
    confirmed = loop.confirm(result, measured, CONTROL_START)

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
