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


def test_compensation_rule_measures_injection_against_withdrawn_liquid(norms):
    """Компенсация считается от отобранной жидкости, а не от добытой воды.

    Закачка 1 150 м³ при добытой воде 900 м³ буквальным прочтением отвергается,
    хотя отобрано 1 000 м³ пластового объёма и компенсация равна 1,15 — обычная
    величина для заводнения с поддержанием давления. Историческая компенсация
    Модели Z на расчётном окне — 1,291 (`docs/manus/vrr-water-balance.md`,
    Р-61), и правило `compensation` не должно резать то, что её не превышает.
    """
    forecast = [produced("2007-01-01", "1", 100.0, 900.0), injected("2007-01-01", "9", 1150.0)]
    steps = [step(producers=[prod("1", 30.0)], injectors=[inj("9", 38.0)])]

    result, decisions = planner.enforce(
        steps, forecast, norms,
        planner.Limits(water_balance_rule="compensation", water_tolerance=1.3), FLUIDS,
    )

    assert decisions == []
    assert result == steps


def test_compensation_above_the_ceiling_is_cut_to_the_ceiling(norms):
    forecast = [produced("2007-01-01", "1", 100.0, 900.0), injected("2007-01-01", "9", 1500.0)]
    steps = [step(producers=[prod("1", 30.0)], injectors=[inj("9", 50.0)])]

    result, decisions = planner.enforce(
        steps, forecast, norms,
        planner.Limits(water_balance_rule="compensation", water_tolerance=1.3), FLUIDS,
    )

    assert [decision.resource for decision in decisions] == ["закачка"]
    assert result[0].injectors[0].rate == pytest.approx(50.0 * 1300.0 / 1500.0)
    assert "компенсац" in decisions[0].reason


def test_without_a_floor_the_corridor_stays_one_sided(norms):
    """Умолчание не должно менять поведение: заявляемое число получено без нижней границы."""
    forecast = [produced("2007-01-01", "1", 100.0, 900.0), injected("2007-01-01", "9", 100.0)]
    steps = [step(producers=[prod("1", 30.0)], injectors=[inj("9", 3.0)])]

    result, decisions = planner.enforce(
        steps, forecast, norms,
        planner.Limits(water_balance_rule="compensation", water_tolerance=1.3), FLUIDS,
    )

    assert decisions == []
    assert result == steps


def test_compensation_below_the_floor_is_raised_to_the_floor(norms):
    """Отобрано 1 000 м³, закачано 100 — при коридоре снизу 0.85 нужно 850."""
    forecast = [produced("2007-01-01", "1", 100.0, 900.0), injected("2007-01-01", "9", 100.0)]
    steps = [step(producers=[prod("1", 30.0)], injectors=[inj("9", 3.0)])]

    result, decisions = planner.enforce(
        steps, forecast, norms,
        planner.Limits(water_balance_rule="compensation", water_tolerance=1.15,
                       water_tolerance_min=0.85), FLUIDS,
    )

    assert [decision.resource for decision in decisions] == ["закачка"]
    assert result[0].injectors[0].rate == pytest.approx(3.0 * 850.0 / 100.0)
    assert "ниже границы коридора" in decisions[0].reason


def test_raising_to_the_floor_is_proportional_across_injectors(norms):
    """Добор распределяется тем же правилом, что и срезка: приоритета мы не знаем."""
    forecast = [produced("2007-01-01", "1", 100.0, 900.0),
                injected("2007-01-01", "8", 75.0), injected("2007-01-01", "9", 25.0)]
    steps = [step(producers=[prod("1", 30.0)], injectors=[inj("8", 3.0), inj("9", 1.0)])]

    result, _ = planner.enforce(
        steps, forecast, norms,
        planner.Limits(water_balance_rule="compensation", water_tolerance=1.15,
                       water_tolerance_min=0.85), FLUIDS,
    )
    rates = {control.well: control.rate for control in result[0].injectors}

    assert rates["8"] / rates["9"] == pytest.approx(3.0)
    assert rates["8"] + rates["9"] == pytest.approx(4.0 * 850.0 / 100.0)


def test_injection_inside_the_corridor_is_left_alone(norms):
    forecast = [produced("2007-01-01", "1", 100.0, 900.0), injected("2007-01-01", "9", 1000.0)]
    steps = [step(producers=[prod("1", 30.0)], injectors=[inj("9", 33.0)])]

    result, decisions = planner.enforce(
        steps, forecast, norms,
        planner.Limits(water_balance_rule="compensation", water_tolerance=1.15,
                       water_tolerance_min=0.85), FLUIDS,
    )

    assert decisions == []
    assert result == steps


def test_nothing_to_raise_when_there_are_no_injectors(norms):
    """Расписанием такое не лечится — планировщик молчит, ловит аудит по факту."""
    forecast = [produced("2007-01-01", "1", 100.0, 900.0)]
    steps = [step(producers=[prod("1", 30.0)])]

    result, decisions = planner.enforce(
        steps, forecast, norms,
        planner.Limits(water_balance_rule="compensation", water_tolerance=1.15,
                       water_tolerance_min=0.85), FLUIDS,
    )

    assert decisions == []
    assert result == steps
    assert balance.compensation_problems(forecast, FLUIDS, ceiling=1.15, floor=0.85)


def test_an_empty_corridor_is_refused_by_name(norms):
    forecast = [produced("2007-01-01", "1", 100.0, 900.0), injected("2007-01-01", "9", 100.0)]
    steps = [step(producers=[prod("1", 30.0)], injectors=[inj("9", 3.0)])]

    with pytest.raises(ValueError, match="пустой коридор"):
        planner.enforce(
            steps, forecast, norms,
            planner.Limits(water_balance_rule="compensation", water_tolerance=0.9,
                           water_tolerance_min=1.1), FLUIDS,
        )


def test_a_floor_makes_no_sense_for_the_external_water_rule(norms):
    """Правило про источник, а не про отношение объёмов: молча игнорировать нельзя."""
    forecast = [produced("2007-01-01", "1", 100.0, 900.0), injected("2007-01-01", "9", 100.0)]
    steps = [step(producers=[prod("1", 30.0)], injectors=[inj("9", 3.0)])]

    with pytest.raises(ValueError, match="external_water"):
        planner.enforce(
            steps, forecast, norms,
            planner.Limits(water_balance_rule="external_water",
                           max_external_water_m3_per_day=10.0,
                           water_tolerance_min=0.85), FLUIDS,
        )


def test_the_raised_schedule_satisfies_the_same_audit(norms):
    """Итог добора обязан проходить ту же проверку, которой ловим нарушение по факту."""
    forecast = [produced("2007-01-01", "1", 100.0, 900.0), injected("2007-01-01", "9", 100.0)]
    steps = [step(producers=[prod("1", 30.0)], injectors=[inj("9", 3.0)])]

    result, _ = planner.enforce(
        steps, forecast, norms,
        planner.Limits(water_balance_rule="compensation", water_tolerance=1.15,
                       water_tolerance_min=0.85), FLUIDS,
    )

    factor = result[0].injectors[0].rate / 3.0
    corrected = [row for row in forecast if row["WWIT_Diff"] == 0] + [
        injected("2007-01-01", "9", 100.0 * factor)
    ]
    assert balance.compensation_problems(corrected, FLUIDS, ceiling=1.15, floor=0.85) == []


def test_the_source_limit_applies_under_the_compensation_rule_too(norms):
    """Докачка N — физика источника, коридор — норма. Кейс задаёт оба сразу.

    Отобрано 1 000 м³ пластового объёма, попутной воды 900, закачано 1 100 —
    компенсация 1,10, внутри коридора 0.85–1.15, правило её не трогает. Но
    сторонней воды нужно 200 м³ за январь (31 сутки), то есть 6,45 м³/сут, а
    источник даёт 3. Значит закачка обязана ужаться до 900 + 3 × 31 = 993 м³.
    """
    forecast = [produced("2007-01-01", "1", 100.0, 900.0), injected("2007-01-01", "9", 1100.0)]
    steps = [step(producers=[prod("1", 30.0)], injectors=[inj("9", 36.0)])]

    result, decisions = planner.enforce(
        steps, forecast, norms,
        planner.Limits(water_balance_rule="compensation", water_tolerance=1.15,
                       water_tolerance_min=0.85,
                       max_external_water_m3_per_day=3.0), FLUIDS,
    )

    assert [decision.resource for decision in decisions] == ["закачка"]
    assert result[0].injectors[0].rate == pytest.approx(36.0 * 993.0 / 1100.0)
    assert "воды со стороны" in decisions[0].reason


def test_an_unknown_source_capacity_cuts_nothing_under_compensation(norms):
    """None — «мощность неизвестна». Выдуманный предел изменил бы заявляемое ЧДД."""
    forecast = [produced("2007-01-01", "1", 100.0, 900.0), injected("2007-01-01", "9", 1100.0)]
    steps = [step(producers=[prod("1", 30.0)], injectors=[inj("9", 36.0)])]

    result, decisions = planner.enforce(
        steps, forecast, norms,
        planner.Limits(water_balance_rule="compensation", water_tolerance=1.15), FLUIDS,
    )

    assert decisions == []
    assert result == steps


def test_the_source_limit_wins_over_the_corridor_floor(norms):
    """Воды, которой нет, никакая норма не создаёт — и это видно в решениях.

    Нижняя граница просит добрать до 850 м³, но своей воды 100, а источник даёт
    3 м³/сут — это 93 м³ за январь. Итог 193 м³ ниже коридора, и так и должно
    быть: приоритет у физики, а несоответствие называет аудит.
    """
    forecast = [produced("2007-01-01", "1", 900.0, 100.0), injected("2007-01-01", "9", 100.0)]
    steps = [step(producers=[prod("1", 30.0)], injectors=[inj("9", 3.0)])]

    result, decisions = planner.enforce(
        steps, forecast, norms,
        planner.Limits(water_balance_rule="compensation", water_tolerance=1.15,
                       water_tolerance_min=0.85,
                       max_external_water_m3_per_day=3.0), FLUIDS,
    )

    # Решение ровно одно: коридор и источник считаются в одном шаге, иначе два
    # последовательных множителя дали бы «двойную экономию», которой нет в пласте.
    assert [decision.resource for decision in decisions] == ["закачка"]
    assert result[0].injectors[0].rate == pytest.approx(3.0 * 193.0 / 100.0)
    assert balance.compensation_problems(
        [row for row in forecast if row["WWIT_Diff"] == 0]
        + [injected("2007-01-01", "9", 193.0)],
        FLUIDS, ceiling=1.15, floor=0.85,
    ), "аудит обязан назвать то, что расписанием не лечится"


def test_external_water_rule_allows_injection_the_source_can_cover(norms):
    """Январь 2007: 31 сутки, своей воды 900 м³, источник даёт 10 м³/сут — это 310 м³."""
    forecast = [produced("2007-01-01", "1", 100.0, 900.0), injected("2007-01-01", "9", 1200.0)]
    steps = [step(producers=[prod("1", 30.0)], injectors=[inj("9", 40.0)])]

    result, decisions = planner.enforce(
        steps, forecast, norms,
        planner.Limits(water_balance_rule="external_water",
                       max_external_water_m3_per_day=10.0), FLUIDS,
    )

    assert decisions == []
    assert result == steps


def test_external_water_rule_cuts_what_the_source_cannot_cover(norms):
    forecast = [produced("2007-01-01", "1", 100.0, 900.0), injected("2007-01-01", "9", 1200.0)]
    steps = [step(producers=[prod("1", 30.0)], injectors=[inj("9", 40.0)])]

    result, decisions = planner.enforce(
        steps, forecast, norms,
        planner.Limits(water_balance_rule="external_water",
                       max_external_water_m3_per_day=1.0), FLUIDS,
    )

    assert [decision.resource for decision in decisions] == ["закачка"]
    assert result[0].injectors[0].rate == pytest.approx(40.0 * 931.0 / 1200.0)
    assert "воды со стороны" in decisions[0].reason


def test_unknown_source_capacity_does_not_invent_a_limit(norms):
    """Мощность источника неизвестна — не режем.

    Выдуманный предел изменил бы заявляемый ЧДД на число, которого никто не
    измерял. Потребность в сторонней воде при этом видна через
    `aios.balance.external_water_problems` и идёт в отчёт оговоркой.
    """
    forecast = [produced("2007-01-01", "1", 100.0, 100.0), injected("2007-01-01", "9", 9000.0)]
    steps = [step(producers=[prod("1", 30.0)], injectors=[inj("9", 300.0)])]

    result, decisions = planner.enforce(
        steps, forecast, norms,
        planner.Limits(water_balance_rule="external_water"), FLUIDS,
    )

    assert decisions == []
    assert result == steps


def test_an_unknown_balance_rule_is_refused_not_silently_ignored(norms):
    with pytest.raises(ValueError, match="water_balance_rule"):
        planner.enforce(
            [step(injectors=[inj("9", 10.0)])], [], norms,
            planner.Limits(water_balance_rule="voidage"), FLUIDS,
        )


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


def test_a_cut_well_is_told_its_place_in_the_order_not_called_the_worst(norms):
    """Обоснование обязано быть верным для каждой урезанной скважины, а не для первой.

    Планировщик режет по возрастанию удельной маржи, пока отбор не уложится в
    предел, — то есть под нож идёт несколько скважин. В обосновании у каждой
    стояло «маржа … — наименьшая в фонде», хотя наименьшая она только у первой.
    Замер 20.08: при пределе 300 из 900 м³/сут урезаны две скважины, обе названы
    наименьшими, причём обе с одинаково округлённым числом `-0.0001`.

    Это тот же класс, что и остановка, которой не происходит: слой
    интерпретируемости обязан объяснять принятое решение, а не звучать правдоподобно.
    Ложное обоснование хуже отсутствующего — его нельзя проверить на глаз.
    """
    rows = [make_row("2014-01-01", well, wlpr=300.0, womr=oil / 30,
                     liquid=9000.0, oil=oil)
            for well, oil in (("A", 1.0), ("B", 50.0), ("C", 200.0))]
    steps = [ScheduleStep(date(2014, 1, 1),
                          tuple(ProducerControl(well, "OPEN", "LRAT", 300.0, 50.0)
                                for well in ("A", "B", "C")), ())]

    _, decisions = planner.enforce(
        steps, rows, norms, planner.Limits(max_field_liquid_m3_per_day=300.0)
    )

    assert len(decisions) == 2, [str(d) for d in decisions]
    worst, second = decisions[0], decisions[1]
    assert "наименьшая" in worst.reason, worst.reason
    assert "наименьшая" not in second.reason, (
        "вторая по счёту скважина названа наименьшей в фонде — это неправда"
    )
    assert "2-я" in second.reason, second.reason
    # число, которым обосновано решение, обязано различать скважины
    assert worst.reason.split("маржа")[1] != second.reason.split("маржа")[1], (
        "обоснования двух разных решений неразличимы по числу"
    )


def test_ranking_ignores_rows_the_methodology_drops(norms):
    """Скважину нельзя ставить под нож из-за строк, которых нет в экономике.

    `_margin_per_liquid` обрезал отрицательные приращения в **знаменателе**
    (`max(..., 0.0)`), а в числителе считал строку как есть: месяц с
    `WOMT_Diff = −150` давал марже −1,427 млн руб. (замер 20.08). Скважина с
    откатом накопленного выглядела худшей в фонде и уходила под нож раньше
    настоящей убыточной — то есть решение принималось по деньгам, которых
    методика не признаёт.

    После правки её строки в ранжирование не идут вовсе, и она получает
    значение по умолчанию — «сведений нет». Это ставит её между убыточными и
    прибыльными, и такой порядок защитим: про убыточную мы знаем, что она
    теряет, про эту — не знаем ничего.
    """
    rows = [make_row("2014-01-01", "убыточная", wlpr=40.0, womr=0.1,
                     liquid=900.0, oil=3.0),
            make_row("2014-01-01", "с откатом", wlpr=40.0, womr=5.0,
                     liquid=900.0, oil=150.0)]
    rows[1]["WOMT_Diff"] = -150.0    # строка, которую методика выбрасывает
    steps = [ScheduleStep(date(2014, 1, 1),
                          (ProducerControl("убыточная", "OPEN", "LRAT", 40.0, 50.0),
                           ProducerControl("с откатом", "OPEN", "LRAT", 40.0, 50.0)), ())]

    _, decisions = planner.enforce(
        steps, rows, norms,
        planner.Limits(max_field_liquid_m3_per_day=40.0, enforce_water_balance=False),
        FLUIDS,
    )

    liquid = [d for d in decisions if d.resource == "жидкость"]
    assert liquid, "предел по жидкости не сработал"
    assert liquid[0].well == "убыточная", (
        f"первой урезана «{liquid[0].well}» — решение принято по строке, "
        "исключённой методикой"
    )


def test_external_water_refuses_a_forecast_without_dates_instead_of_guessing(norms):
    """Лимит источника суточный: без календаря объём не посчитать.

    Тихая подстановка 365 суток на девятнадцатилетнем горизонте занизила бы
    допустимую закачку в двадцать раз и срезала бы её без причины — то есть
    изменила бы заявляемый ЧДД молча.
    """
    forecast = [
        {k: v for k, v in produced("2007-01-01", "1", 100.0, 900.0).items() if k != "DATA"},
        {k: v for k, v in injected("2007-01-01", "9", 1200.0).items() if k != "DATA"},
    ]
    steps = [step(producers=[prod("1", 30.0)], injectors=[inj("9", 40.0)])]

    with pytest.raises(ValueError, match="нужна колонка DATA"):
        planner.enforce(
            steps, forecast, norms,
            planner.Limits(water_balance_rule="external_water",
                           max_external_water_m3_per_day=1.0), FLUIDS,
        )
# ── формы ограничений валидационного кейса, заведённые 01.09 ─────────────────
#
# Каждая проверка построена так, как требует протокол приёмки кейса
# (`docs/validation-case-checklist.md`): решение нарушает ровно одно условие, и
# видно, что срезано именно оно.


def month_step(year: int, month: int, producers=(), injectors=()):
    return ScheduleStep(date(year, month, 1), tuple(producers), tuple(injectors))


def test_shut_window_rejects_reversed_dates():
    """Окно с концом раньше начала — опечатка кейса, а не «пустое окно»."""
    with pytest.raises(ValueError, match="раньше начала"):
        planner.ShutWindow(well="1", start=date(2027, 6, 1), end=date(2027, 1, 1))


def test_forced_shut_window_closes_the_well_only_inside_it(norms):
    """До 01.09 выбор был из двух неверных: остановить навсегда или не остановить.

    Кейс говорит «скважину W остановить с 01.2027 по 06.2027»; вне окна режим
    остаётся тот, который назначил цикл.
    """
    steps = [
        month_step(2026, 12, producers=[prod("1", 30.0)]),
        month_step(2027, 3, producers=[prod("1", 30.0)]),
        month_step(2027, 9, producers=[prod("1", 30.0)]),
    ]
    limits = planner.Limits(
        enforce_water_balance=False,
        forced_shut_windows=(planner.ShutWindow("1", date(2027, 1, 1), date(2027, 6, 30)),),
    )

    result, decisions = planner.enforce(steps, [], norms, limits, FLUIDS)

    assert [control.status for step_ in result for control in step_.producers] == \
        ["OPEN", "SHUT", "OPEN"]
    assert len(decisions) == 1
    assert "окно обязательной остановки 2027-01-01 … 2027-06-30" in decisions[0].reason


def test_forced_shut_window_closes_injectors_too(norms):
    """«Не управлять этой скважиной в эти месяцы» относится и к нагнетательной."""
    steps = [month_step(2027, 3, injectors=[inj("9", 40.0)])]
    limits = planner.Limits(
        enforce_water_balance=False,
        forced_shut_windows=(planner.ShutWindow("9", date(2027, 1, 1), date(2027, 6, 30)),),
    )

    result, decisions = planner.enforce(steps, [], norms, limits, FLUIDS)

    assert result[0].injectors[0].status == "SHUT"
    assert result[0].injectors[0].rate is None
    assert decisions[0].before == 40.0


def test_two_windows_of_one_well_are_explained_separately(norms):
    """Два ремонта в разные годы — два решения, а не одно повторённое."""
    steps = [
        month_step(2027, 3, producers=[prod("1", 30.0)]),
        month_step(2028, 3, producers=[prod("1", 30.0)]),
    ]
    limits = planner.Limits(
        enforce_water_balance=False,
        forced_shut_windows=(
            planner.ShutWindow("1", date(2027, 1, 1), date(2027, 6, 30)),
            planner.ShutWindow("1", date(2028, 1, 1), date(2028, 6, 30)),
        ),
    )

    _, decisions = planner.enforce(steps, [], norms, limits, FLUIDS)

    assert len(decisions) == 2
    assert {decision.reason for decision in decisions} == {
        "окно обязательной остановки 2027-01-01 … 2027-06-30",
        "окно обязательной остановки 2028-01-01 … 2028-06-30",
    }


def test_annual_liquid_volume_cuts_only_the_year_over_the_norm(norms):
    """Годовой аккумулятор был только у закачки: «за год не более V» по отбору."""
    steps = [
        month_step(2007, month, producers=[prod("1", 100.0)]) for month in (1, 2)
    ] + [month_step(2008, 1, producers=[prod("1", 10.0)])]
    # январь + февраль 2007 при 100 м³/сут это 5 900 м³; 2008 — 310
    limits = planner.Limits(enforce_water_balance=False,
                            max_field_liquid_m3_per_year=3_000.0)

    result, decisions = planner.enforce(steps, [], norms, limits, FLUIDS)

    rates = [control.liquid_rate for step_ in result for control in step_.producers]
    assert rates[2] == 10.0, "год в пределах нормы трогать нельзя"
    assert sum(rate * days for rate, days in zip(rates[:2], (31, 28), strict=True)) == \
        pytest.approx(3_000.0)
    assert len(decisions) == 1
    assert "годовой объём отбора жидкости, 2007" in decisions[0].resource


def test_daily_field_limit_can_be_read_as_annual_average(norms):
    """Пик, прочитанный как среднее, пропускает нарушение — поэтому это ключ.

    Один месяц выше суточного предела, год в среднем — ниже. При обычном чтении
    (пик каждого месяца) срезка есть, при чтении «среднее за год» — нет.
    """
    steps = [
        month_step(2007, 1, producers=[prod("1", 150.0)]),
        month_step(2007, 2, producers=[prod("1", 10.0)]),
    ]
    peak = planner.Limits(enforce_water_balance=False, max_field_liquid_m3_per_day=100.0)
    average = planner.Limits(enforce_water_balance=False, max_field_liquid_m3_per_day=100.0,
                             field_limits_are_annual_average=True)
    forecast = [produced("2007-01-01", "1", 100.0, 900.0)]

    cut, cut_decisions = planner.enforce(steps, forecast, norms, peak, FLUIDS)
    kept, kept_decisions = planner.enforce(steps, forecast, norms, average, FLUIDS)

    assert cut[0].producers[0].liquid_rate == 100.0 and cut_decisions
    assert kept == steps and kept_decisions == []


def test_annual_average_still_catches_the_year_that_breaks_it(norms):
    """Обратная сторона того же ключа: год выше среднего режется целиком."""
    steps = [month_step(2007, month, producers=[prod("1", 150.0)]) for month in (1, 2)]
    limits = planner.Limits(enforce_water_balance=False, max_field_liquid_m3_per_day=100.0,
                            field_limits_are_annual_average=True)

    result, decisions = planner.enforce(steps, [], norms, limits, FLUIDS)

    assert result[0].producers[0].liquid_rate == pytest.approx(100.0)
    assert "среднесуточный отбор за год" in decisions[0].resource


def test_annual_average_counts_only_the_months_present():
    """Неполный год делится на свои дни, а не на 365 — иначе норма завышена вчетверо."""
    steps = [month_step(2007, month) for month in (11, 12)]

    budget = planner._annual_budget(steps, 100.0)

    assert budget == {2007: 100.0 * (30 + 31)}


def test_annual_average_applies_to_injection_too(norms):
    """Тот же ключ читает и предел по закачке: формы приходят парой."""
    steps = [
        month_step(2007, 1, injectors=[inj("9", 150.0)]),
        month_step(2007, 2, injectors=[inj("9", 10.0)]),
    ]
    limits = planner.Limits(enforce_water_balance=False,
                            max_field_injection_m3_per_day=100.0,
                            field_limits_are_annual_average=True)

    result, decisions = planner.enforce(steps, [], norms, limits, FLUIDS)

    assert result == steps and decisions == []
def test_field_oil_limit_cuts_liquid_in_the_same_share(norms):
    """Мощность УПН выражается по нефти, и объектным пределом по жидкости не заменяется."""
    steps = [month_step(2007, 1, producers=[prod("1", 100.0)])]
    # 100 м³ нефти за месяц ≈ 91,3 т, то есть 2,94 т/сут при 31 дне
    forecast = [produced("2007-01-01", "1", 100.0, 900.0, rate=100.0)]
    limits = planner.Limits(enforce_water_balance=False, max_field_oil_t_per_day=1.47)

    result, decisions = planner.enforce(steps, forecast, norms, limits, FLUIDS)

    assert result[0].producers[0].liquid_rate == pytest.approx(50.0, rel=1e-2)
    assert "нефть" in decisions[0].resource
    assert "от достигнутого" in decisions[0].reason


def test_field_oil_limit_cuts_from_the_achieved_rate_not_from_the_setpoint(norms):
    """Уставка выше факта: пропорциональный срез уставки отклика не меняет вовсе.

    Найдено репетицией приёмки 01.09 — на Модели Z уставки выше достигнутых
    дебитов в 1,47 раза, и после «успешной» срезки предел по нефти оставался
    нарушенным. Резать надо до доли **достигнутого**: уставка 100 при факте 30 и
    половинном пределе даёт 15, а не 50.
    """
    steps = [month_step(2007, 1, producers=[prod("1", 100.0)])]
    forecast = [produced("2007-01-01", "1", 100.0, 900.0, rate=30.0)]
    limits = planner.Limits(enforce_water_balance=False, max_field_oil_t_per_day=1.47)

    result, _ = planner.enforce(steps, forecast, norms, limits, FLUIDS)

    assert result[0].producers[0].liquid_rate == pytest.approx(15.0, rel=1e-2)


def test_a_well_without_a_response_is_left_alone_by_the_phase_limit(norms):
    """Скважина, которой в прогнозе нет, фазы не даёт — резать её не за что."""
    steps = [month_step(2007, 1, producers=[prod("1", 100.0), prod("новая", 80.0)])]
    forecast = [produced("2007-01-01", "1", 100.0, 900.0, rate=100.0)]
    limits = planner.Limits(enforce_water_balance=False, max_field_oil_t_per_day=1.47)

    result, _ = planner.enforce(steps, forecast, norms, limits, FLUIDS)
    rates = {control.well: control.liquid_rate for control in result[0].producers}

    assert rates["новая"] == 80.0


def test_field_water_limit_uses_liquid_minus_oil(norms):
    """Отдельной колонки воды в контракте расчётчика нет — вода это жидкость минус нефть."""
    steps = [month_step(2007, 1, producers=[prod("1", 100.0)])]
    forecast = [produced("2007-01-01", "1", 100.0, 900.0, rate=100.0)]
    water_t = 900.0 * FLUIDS.water_density / 1000
    limits = planner.Limits(enforce_water_balance=False,
                            max_field_water_t_per_day=water_t / 31 / 2)

    result, decisions = planner.enforce(steps, forecast, norms, limits, FLUIDS)

    assert result[0].producers[0].liquid_rate == pytest.approx(50.0, rel=1e-6)
    assert "попутная вода" in decisions[0].resource


def test_phase_limit_without_a_forecast_is_refused(norms):
    """Объявленное ограничение, которое нечем посчитать, хуже отсутствующего."""
    steps = [month_step(2007, 1, producers=[prod("1", 100.0)])]
    limits = planner.Limits(enforce_water_balance=False, max_field_oil_t_per_day=1.0)

    with pytest.raises(ValueError, match="прогноза нет"):
        planner.enforce(steps, [], norms, limits, FLUIDS)


def test_group_liquid_cap_cuts_the_worst_by_margin_inside_the_group(norms):
    """Между объектом и скважиной не было ничего, а «по кусту не более» — обычная форма."""
    steps = [month_step(2007, 1, producers=[prod("1", 60.0), prod("2", 60.0), prod("3", 60.0)])]
    forecast = [
        produced("2007-01-01", "1", 200.0, 800.0),   # богатая нефтью
        produced("2007-01-01", "2", 5.0, 995.0),     # почти вода
        produced("2007-01-01", "3", 200.0, 800.0),   # вне группы
    ]
    limits = planner.Limits(
        enforce_water_balance=False,
        well_groups={"куст-1": frozenset({"1", "2"})},
        group_liquid_caps_m3_per_day={"куст-1": 80.0},
    )

    result, decisions = planner.enforce(steps, forecast, norms, limits, FLUIDS)
    rates = {control.well: control.liquid_rate for control in result[0].producers}

    assert rates["3"] == 60.0, "скважина вне группы не трогается"
    assert rates["1"] + rates["2"] == pytest.approx(80.0)
    assert rates["2"] < rates["1"], "первой режется худшая по деньгам на кубометр"
    assert decisions and "куст-1" in decisions[0].resource


def test_group_injection_cap_is_shared_proportionally(norms):
    """У нагнетательных приоритета нет — по той же причине, что и в балансе воды."""
    steps = [month_step(2007, 1, injectors=[inj("9", 60.0), inj("8", 40.0)])]
    limits = planner.Limits(
        enforce_water_balance=False,
        well_groups={"кнс-1": frozenset({"9", "8"})},
        group_injection_caps_m3_per_day={"кнс-1": 50.0},
    )

    result, _ = planner.enforce(steps, [], norms, limits, FLUIDS)
    rates = {control.well: control.rate for control in result[0].injectors}

    assert rates["9"] == pytest.approx(30.0) and rates["8"] == pytest.approx(20.0)


def test_a_well_can_belong_to_two_groups(norms):
    """Куст и блок заводнения — разные разрезы одного фонда, пересечение законно.

    Действует наименьший потолок, и объясняет срезку та группа, которая её
    фактически сделала: группы применяются по очереди, и после срезки до 20 м³/сут
    потолок 50 уже не нарушен — писать про него решение значило бы отчитаться о
    срезке, которой не было.
    """
    steps = [month_step(2007, 1, injectors=[inj("9", 60.0)])]
    limits = planner.Limits(
        enforce_water_balance=False,
        well_groups={"куст-1": frozenset({"9"}), "блок-А": frozenset({"9"})},
        group_injection_caps_m3_per_day={"куст-1": 50.0, "блок-А": 20.0},
    )

    result, decisions = planner.enforce(steps, [], norms, limits, FLUIDS)

    assert result[0].injectors[0].rate == pytest.approx(20.0), "действует наименьший потолок"
    assert [decision.resource for decision in decisions] == ["закачка группы «блок-А»"]


def test_group_order_does_not_change_the_result(norms):
    """Порядок групп в конфигурации не должен менять числа: он алфавитный и стабильный."""
    steps = [month_step(2007, 1, injectors=[inj("9", 60.0)])]
    tight = planner.Limits(
        enforce_water_balance=False,
        well_groups={"куст-1": frozenset({"9"}), "блок-А": frozenset({"9"})},
        group_injection_caps_m3_per_day={"блок-А": 20.0, "куст-1": 50.0},
    )
    loose = planner.Limits(
        enforce_water_balance=False,
        well_groups={"блок-А": frozenset({"9"}), "куст-1": frozenset({"9"})},
        group_injection_caps_m3_per_day={"куст-1": 50.0, "блок-А": 20.0},
    )

    first, _ = planner.enforce(steps, [], norms, tight, FLUIDS)
    second, _ = planner.enforce(steps, [], norms, loose, FLUIDS)

    assert first == second


def test_group_cap_without_the_group_is_refused(norms):
    """Потолок группы, которой нет в составе, — опечатка кейса, а не пустое условие."""
    limits = planner.Limits(enforce_water_balance=False,
                            group_liquid_caps_m3_per_day={"куст-7": 10.0})

    with pytest.raises(ValueError, match="куст-7"):
        planner.enforce([month_step(2007, 1)], [], norms, limits, FLUIDS)
def test_ranking_survives_a_forecast_read_from_csv(norms):
    """Имя скважины из pandas — число, в расписании — строка: совпадений ноль.

    Найдено репетицией приёмки 01.09. Без нормализации маржа не находилась ни
    одной скважине, `value.get(well, 0.0)` возвращал ноль всем, и «режем худшую по
    деньгам» превращалось в «режем в порядке перечисления» — молча, с правильным
    на вид объяснением и соблюдённым пределом.
    """
    steps = [step(producers=[prod("1", 60.0), prod("2", 60.0)])]
    rich = produced("2007-01-01", "1", 200.0, 800.0)
    poor = produced("2007-01-01", "2", 5.0, 995.0)
    as_csv = [row | {"well": int(row["well"])} for row in (rich, poor)]
    limits = planner.Limits(enforce_water_balance=False, max_field_liquid_m3_per_day=80.0)

    result, decisions = planner.enforce(steps, as_csv, norms, limits, FLUIDS)
    rates = {control.well: control.liquid_rate for control in result[0].producers}

    assert rates["1"] + rates["2"] == pytest.approx(80.0)
    assert rates["2"] < rates["1"], "первой обязана резаться скважина с худшей маржой"
    assert decisions[0].well == "2"
# ── квота мероприятий против жёстких ограничений (найдено критиком 01.09) ────

def test_quota_does_not_cancel_a_shutdown_it_did_not_ask_for(norms):
    """Квота отменяла остановку, поставленную ограничением, — и нарушала его.

    `_fit_events_to_annual_quota` идёт последним и при исчерпанной квоте
    возвращает скважину в состояние предыдущего шага. Остановка, которую только
    что поставило окно `forced_shut_windows`, для квоты выглядит обычным
    мероприятием — и отменялась. Скважина снова открыта внутри окна.
    """
    steps = [
        ScheduleStep(date(2027, 1, 1), (prod("A", 50.0), prod("B", 100.0)), ()),
        # своя остановка A в феврале съедает квоту
        ScheduleStep(date(2027, 2, 1),
                     (ProducerControl("A", "SHUT", "LRAT", None, 50.0), prod("B", 100.0)), ()),
        ScheduleStep(date(2027, 3, 1),
                     (ProducerControl("A", "SHUT", "LRAT", None, 50.0), prod("B", 100.0)), ()),
    ]
    limits = planner.Limits(
        enforce_water_balance=False, max_events_per_year=1,
        forced_shut_windows=(planner.ShutWindow("B", date(2027, 3, 1), date(2027, 12, 31)),),
    )

    result, decisions = planner.enforce(steps, [], norms, limits, FLUIDS)
    march = result[2]

    assert [c.status for c in march.producers if c.well == "B"] == ["SHUT"]
    assert any("оставлен вопреки квоте" in decision.reason for decision in decisions), (
        "превышение квоты обязательным мероприятием обязано быть названо вслух"
    )


def test_quota_does_not_restore_a_rate_the_field_limit_cut_to_zero(norms):
    """Тот же механизм бил по деньгам: срезка в ноль это «остановка» для квоты.

    Худшая по марже скважина срезается объектным пределом до нуля, то есть до
    `SHUT`. Квота видит остановку, отменяет её и возвращает скважине режим
    прошлого шага — объектный предел снова нарушен, и это уже деньги, а не форма.
    """
    steps = [
        ScheduleStep(date(2027, 1, 1), (prod("A", 50.0), prod("B", 50.0)), ()),
        ScheduleStep(date(2027, 2, 1), (prod("A", 150.0), prod("B", 100.0)), ()),
    ]
    forecast = [
        produced("2027-01-01", "A", 200.0, 800.0, rate=50.0),   # богатая нефтью
        produced("2027-01-01", "B", 2.0, 998.0, rate=50.0),     # почти вода — режется первой
    ]
    limits = planner.Limits(enforce_water_balance=False,
                            max_field_liquid_m3_per_day=150.0, max_events_per_year=0)

    result, _ = planner.enforce(steps, forecast, norms, limits, FLUIDS)
    february = sum(c.liquid_rate or 0.0 for c in result[1].producers if c.status == "OPEN")

    assert february <= 150.0, "квота вернула отбор, который срезал объектный предел"


def test_quota_still_cancels_what_the_cycle_asked_for(norms):
    """Защита не должна отменять саму квоту: предложения цикла режутся как прежде."""
    steps = [
        ScheduleStep(date(2027, 1, 1), (prod("A", 50.0), prod("B", 50.0)), ()),
        ScheduleStep(date(2027, 2, 1),
                     (ProducerControl("A", "SHUT", "LRAT", None, 50.0),
                      ProducerControl("B", "SHUT", "LRAT", None, 50.0)), ()),
    ]
    limits = planner.Limits(enforce_water_balance=False, max_events_per_year=1)

    result, decisions = planner.enforce(steps, [], norms, limits, FLUIDS)
    statuses = sorted(c.status for c in result[1].producers)

    assert statuses == ["OPEN", "SHUT"], "одна остановка проходит по квоте, вторая отменяется"
    assert any("отменён: квота мероприятий" in decision.reason for decision in decisions)
def test_a_group_cut_to_zero_shuts_the_well_instead_of_leaving_it_open(norms):
    """Открытая запись с нулевым дебитом делает расписание непригодным целиком.

    Найдено критиком 01.09: объектный предел в этом положении ставил SHUT, а
    групповой оставлял OPEN с нулём — `schedule.validate` такое расписание
    отвергает, а проверка условий при этом говорила «соблюдено».
    """
    from aios.schedule import validate

    steps = [month_step(2027, 1, producers=[prod("W1", 60.0), prod("W2", 60.0)])]
    forecast = [produced("2027-01-01", "W1", 2.0, 998.0, rate=60.0),
                produced("2027-01-01", "W2", 200.0, 800.0, rate=60.0)]
    limits = planner.Limits(
        enforce_water_balance=False,
        well_groups={"куст-1": frozenset({"W1", "W2"})},
        group_liquid_caps_m3_per_day={"куст-1": 10.0},
    )

    result, _ = planner.enforce(steps, forecast, norms, limits, FLUIDS)
    shut = [control for control in result[0].producers if control.status == "SHUT"]

    assert shut and shut[0].liquid_rate is None
    assert not [problem for problem in validate(result)
                if "нулевой дебит" in problem.reason]


def test_a_zero_well_cap_shuts_the_well(norms):
    """Адресный потолок ноль означает «не давать работать», а не «открыта с нулём»."""
    from aios.schedule import validate

    steps = [month_step(2027, 1, producers=[prod("W1", 60.0), prod("W2", 60.0)])]
    limits = planner.Limits(enforce_water_balance=False,
                            well_caps_m3_per_day={"W1": 0.0})

    result, _ = planner.enforce(steps, [], norms, limits, FLUIDS)
    rates = {control.well: (control.status, control.liquid_rate)
             for control in result[0].producers}

    assert rates["W1"] == ("SHUT", None)
    assert rates["W2"] == ("OPEN", 60.0)
    assert not [problem for problem in validate(result)
                if "нулевой дебит" in problem.reason]
def test_phase_limit_does_not_cut_a_month_another_limit_already_fixed(norms):
    """Фаза считается по текущим уставкам, а не по сырому прогнозу.

    Найдено критиком 01.09: на репетиции предел по воде (1400 т/сут) был уже
    соблюдён после объектной срезки — фактические 1203, — а планировщик всё равно
    сделал 33 срезки и снял 471 939 м³ жидкости за горизонт.
    """
    steps = [month_step(2007, 1, producers=[prod("1", 100.0)])]
    forecast = [produced("2007-01-01", "1", 100.0, 900.0, rate=100.0)]
    oil_t = 100.0 * FLUIDS.oil_density / 1000
    # предел ровно вдвое ниже прогноза, но объектный предел уже урезал уставку
    # вдвое — значит фаза в него укладывается и резать нечего
    limits = planner.Limits(enforce_water_balance=False,
                            max_field_liquid_m3_per_day=50.0,
                            max_field_oil_t_per_day=oil_t / 31 / 2)

    result, decisions = planner.enforce(steps, forecast, norms, limits, FLUIDS)

    assert result[0].producers[0].liquid_rate == pytest.approx(50.0)
    assert not [one for one in decisions if "нефть" in one.resource], (
        "месяц, приведённый чужой срезкой, фазовый предел резать не должен"
    )
