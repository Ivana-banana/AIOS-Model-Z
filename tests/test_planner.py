"""Планировщик: согласование скважинных решений через общие ограничения."""

from __future__ import annotations

from datetime import date

import pytest
from conftest import make_row

from aios import balance, economics, planner
from aios.deck import InjectorControl, ProducerControl, ScheduleStep
from aios.margin import Norms, month_margin
from aios.schedule import validate
from aios.summary import Fluids

FLUIDS = Fluids()


@pytest.fixture
def norms():
    assumptions, _ = economics.load_norms()
    return Norms.from_assumptions(assumptions)


def produced(data: str, well: str, oil_m3: float, water_m3: float, rate: float = 30.0):
    oil_t = oil_m3 * FLUIDS.oil_density / 1000
    liquid_t = oil_t + water_m3 * FLUIDS.water_density / 1000
    return make_row(data, well, wlpr=rate, womr=oil_t / 30, liquid=liquid_t, oil=oil_t)


def injected(data: str, well: str, volume_m3: float):
    return make_row(data, well, wwir=volume_m3 / 30, injection=volume_m3)


def step(producers=(), injectors=(), month: int = 1):
    return ScheduleStep(date(2007, month, 1), tuple(producers), tuple(injectors))


def prod(well: str, rate: float):
    return ProducerControl(well, "OPEN", "LRAT", rate, 50.0)


def inj(well: str, rate: float):
    return InjectorControl(well, "WATER", "OPEN", "RATE", rate, 300.0)


# ── баланс воды ──────────────────────────────────────────────────────────────

def test_injection_within_produced_water_is_left_alone(norms):
    forecast = [produced("2007-01-01", "1", 100.0, 900.0), injected("2007-01-01", "9", 800.0)]
    steps = [step(producers=[prod("1", 30.0)], injectors=[inj("9", 26.0)])]

    result, decisions = planner.enforce(steps, forecast, norms, planner.Limits(), FLUIDS)

    assert decisions == []
    assert result == steps


def test_injection_beyond_produced_water_is_cut_to_fit(norms):
    """Дополнительной воды на объекте нет — закачивать больше добытой нечего."""
    forecast = [produced("2007-01-01", "1", 100.0, 500.0), injected("2007-01-01", "9", 1000.0)]
    steps = [step(producers=[prod("1", 30.0)], injectors=[inj("9", 33.0)])]

    result, decisions = planner.enforce(steps, forecast, norms, planner.Limits(), FLUIDS)

    assert len(decisions) == 1
    assert decisions[0].resource == "закачка"
    assert result[0].injectors[0].rate == pytest.approx(33.0 * 500.0 / 1000.0)
    assert "источника сверх неё нет" in decisions[0].reason


def test_the_cut_is_proportional_across_injectors(norms):
    """Чья закачка полезнее — бейзлайн не знает, и приоритет не выдумывает."""
    forecast = [produced("2007-01-01", "1", 100.0, 400.0),
                injected("2007-01-01", "8", 600.0), injected("2007-01-01", "9", 200.0)]
    steps = [step(producers=[prod("1", 30.0)], injectors=[inj("8", 60.0), inj("9", 20.0)])]

    result, _ = planner.enforce(steps, forecast, norms, planner.Limits(), FLUIDS)
    rates = {control.well: control.rate for control in result[0].injectors}

    assert rates["8"] / rates["9"] == pytest.approx(3.0)
    assert rates["8"] + rates["9"] == pytest.approx(80.0 * 400.0 / 800.0)


def test_the_corrected_schedule_satisfies_the_balance_check(norms):
    """Итог обязан проходить ту же проверку, которой мы ловим нарушения по факту."""
    forecast = [produced("2007-01-01", "1", 100.0, 500.0), injected("2007-01-01", "9", 1000.0)]
    steps = [step(producers=[prod("1", 30.0)], injectors=[inj("9", 33.0)])]

    result, _ = planner.enforce(steps, forecast, norms, planner.Limits(), FLUIDS)

    factor = result[0].injectors[0].rate / 33.0
    corrected = [row for row in forecast if row["WWIT_Diff"] == 0] + [
        injected("2007-01-01", "9", 1000.0 * factor)
    ]
    assert balance.problems(corrected, FLUIDS) == []


def test_balance_enforcement_can_be_switched_off(norms):
    forecast = [produced("2007-01-01", "1", 100.0, 100.0), injected("2007-01-01", "9", 5000.0)]
    steps = [step(producers=[prod("1", 30.0)], injectors=[inj("9", 160.0)])]

    _, decisions = planner.enforce(steps, forecast, norms,
                                   planner.Limits(enforce_water_balance=False), FLUIDS)
    assert decisions == []


def test_shut_injectors_are_not_touched(norms):
    forecast = [produced("2007-01-01", "1", 10.0, 10.0), injected("2007-01-01", "9", 5000.0)]
    steps = [step(producers=[prod("1", 30.0)],
                  injectors=[InjectorControl("9", "WATER", "SHUT", "RATE", None, 300.0)])]

    result, _ = planner.enforce(steps, forecast, norms, planner.Limits(), FLUIDS)
    assert result[0].injectors[0].status == "SHUT"


def test_surface_injection_capacity_is_respected_on_each_step(norms):
    steps = [
        step(injectors=[inj("8", 80.0), inj("9", 40.0)], month=1),
        step(injectors=[inj("8", 30.0), inj("9", 20.0)], month=2),
    ]

    result, decisions = planner.enforce(
        steps,
        forecast=[],
        norms=norms,
        limits=planner.Limits(
            max_field_injection_m3_per_day=60.0, enforce_water_balance=False
        ),
        fluids=FLUIDS,
    )

    assert sum(control.rate or 0.0 for control in result[0].injectors) == pytest.approx(60.0)
    assert sum(control.rate or 0.0 for control in result[1].injectors) == pytest.approx(50.0)
    assert {decision.resource for decision in decisions} == {"мощность ППД"}
    assert validate(result) == []


# ── мощность по жидкости ─────────────────────────────────────────────────────

def test_liquid_within_capacity_is_left_alone(norms):
    forecast = [produced("2007-01-01", "1", 50.0, 50.0), produced("2007-01-01", "2", 50.0, 50.0)]
    steps = [step(producers=[prod("1", 30.0), prod("2", 30.0)])]

    result, decisions = planner.enforce(
        steps, forecast, norms,
        planner.Limits(max_field_liquid_m3_per_day=100.0, enforce_water_balance=False), FLUIDS
    )
    assert decisions == []
    assert result == steps


def test_the_worst_earner_per_cubic_metre_is_cut_first(norms):
    """Порядок реза экономический, и его можно защитить: деньги на кубометр."""
    forecast = [
        produced("2007-01-01", "rich", 90.0, 10.0),   # почти чистая нефть
        produced("2007-01-01", "poor", 1.0, 99.0),    # почти чистая вода
    ]
    steps = [step(producers=[prod("rich", 40.0), prod("poor", 40.0)])]

    result, decisions = planner.enforce(
        steps, forecast, norms,
        planner.Limits(max_field_liquid_m3_per_day=50.0, enforce_water_balance=False), FLUIDS
    )
    rates = {c.well: (c.liquid_rate or 0.0) for c in result[0].producers}

    assert rates["rich"] == pytest.approx(40.0)   # прибыльную не тронули
    assert rates["poor"] == pytest.approx(10.0)   # обводнённую урезали
    assert [d.well for d in decisions] == ["poor"]
    assert "наименьшая в фонде" in decisions[0].reason


def test_a_fully_cut_well_is_shut_not_left_open_at_zero(norms):
    forecast = [produced("2007-01-01", "rich", 90.0, 10.0),
                produced("2007-01-01", "poor", 1.0, 99.0)]
    steps = [step(producers=[prod("rich", 40.0), prod("poor", 40.0)])]

    result, _ = planner.enforce(
        steps, forecast, norms,
        planner.Limits(max_field_liquid_m3_per_day=40.0, enforce_water_balance=False), FLUIDS
    )
    poor = next(c for c in result[0].producers if c.well == "poor")

    assert poor.status == "SHUT"
    assert poor.liquid_rate is None
    assert validate(result) == []


def test_capacity_is_respected_on_every_step(norms):
    forecast = [produced("2007-01-01", "1", 50.0, 50.0), produced("2007-01-01", "2", 10.0, 90.0)]
    steps = [
        step(producers=[prod("1", 60.0), prod("2", 60.0)], month=1),
        step(producers=[prod("1", 30.0), prod("2", 30.0)], month=2),
    ]

    result, _ = planner.enforce(
        steps, forecast, norms,
        planner.Limits(max_field_liquid_m3_per_day=70.0, enforce_water_balance=False), FLUIDS
    )
    for corrected in result:
        total = sum(c.liquid_rate or 0.0 for c in corrected.producers if c.status == "OPEN")
        assert total <= 70.0 + 1e-9


def test_both_limits_apply_together(norms):
    forecast = [produced("2007-01-01", "1", 50.0, 50.0), injected("2007-01-01", "9", 500.0)]
    steps = [step(producers=[prod("1", 100.0)], injectors=[inj("9", 16.0)])]

    _, decisions = planner.enforce(
        steps, forecast, norms,
        planner.Limits(max_field_liquid_m3_per_day=60.0), FLUIDS
    )
    assert {d.resource for d in decisions} == {"закачка", "жидкость"}


# ── объяснимость ─────────────────────────────────────────────────────────────

def test_explain_says_when_nothing_was_needed():
    assert "не вмешивался" in planner.explain([])


def test_explain_groups_by_resource():
    decisions = [
        planner.Allocation("1", "закачка", 100.0, 50.0, "воды не хватает"),
        planner.Allocation("2", "жидкость", 80.0, 40.0, "предел системы сбора"),
    ]
    text = planner.explain(decisions)

    assert "закачка:" in text and "жидкость:" in text
    assert "скв. 1" in text and "скв. 2" in text


# ── единицы ранжирования ─────────────────────────────────────────────────────

def test_margin_is_ranked_per_cubic_metre_not_per_tonne(norms):
    """Дефицитный ресурс — м³/сут, значит и деньги считаются на кубометр.

    `WLPT_Diff` в контракте расчётчика — **тонны**, а предел системы сбора,
    уставка `WCONPROD` и сам срез заданы в **м³/сут**. Делить маржу на тонны и
    подписывать результат «млн руб./м³» значит ранжировать фонд по чужой единице:
    плотность жидкости меняется с обводнённостью от 913 до 1172 кг/м³.

    На порядок среза это влияет слабо — маржу определяет нефть, и обе метрики
    упорядочивают фонд почти одинаково. Но число уходит в объяснение решения
    человеку, а объяснение с чужой единицей — это то же самое, что неверное число.
    """
    rows = [produced("2007-01-01", "1", oil_m3=300.0, water_m3=600.0)]

    value = planner._margin_per_liquid(rows, norms, FLUIDS)

    assert value["1"] == pytest.approx(month_margin(rows[0], norms) / 900.0)
    # а по тоннам получилось бы заметно другое число
    assert value["1"] != pytest.approx(month_margin(rows[0], norms) / rows[0]["WLPT_Diff"])


def test_the_cut_reason_reports_the_same_unit_it_ranks_by(norms):
    forecast = [produced("2007-01-01", "1", oil_m3=300.0, water_m3=600.0),
                produced("2007-01-01", "2", oil_m3=10.0, water_m3=890.0)]
    steps = [step(producers=(prod("1", 40.0), prod("2", 40.0)))]

    _, decisions = planner.enforce(
        steps, forecast, norms,
        planner.Limits(max_field_liquid_m3_per_day=50.0, enforce_water_balance=False), FLUIDS
    )

    liquid = [d for d in decisions if d.resource == "жидкость"]
    assert liquid, "предел по жидкости не сработал"
    assert "млн руб./м³" in liquid[0].reason
