"""Планирование эксперимента: выборка сценариев для обучения суррогата."""

from __future__ import annotations

from datetime import date

import pytest

from aios import doe
from aios.deck import InjectorControl, ProducerControl, ScheduleStep
from aios.schedule import MAX_LIQUID_RATE_M3_PER_DAY, validate

SEED = 20260817


def plan(wells, months: int = 6, rate: float = 40.0):
    return [
        ScheduleStep(
            date(2007, month, 1),
            tuple(ProducerControl(well, "OPEN", "LRAT", rate, 50.0) for well in wells),
            (InjectorControl("9", "WATER", "OPEN", "RATE", 30.0, 300.0),),
        )
        for month in range(1, months + 1)
    ]


def test_latin_hypercube_puts_one_point_in_every_stratum():
    """Свойство, ради которого он и берётся: по каждой оси нет ни дыр, ни сгущений."""
    samples, dimensions = 20, 5
    points = doe.latin_hypercube(samples, dimensions, SEED)

    assert len(points) == samples
    for axis in range(dimensions):
        strata = sorted(int(row[axis] * samples) for row in points)
        assert strata == list(range(samples))


def test_latin_hypercube_is_reproducible_from_the_seed():
    """Регламент запрещает неконтролируемую случайность."""
    assert doe.latin_hypercube(8, 3, SEED) == doe.latin_hypercube(8, 3, SEED)
    assert doe.latin_hypercube(8, 3, SEED) != doe.latin_hypercube(8, 3, SEED + 1)


def test_latin_hypercube_rejects_empty_designs():
    with pytest.raises(ValueError):
        doe.latin_hypercube(0, 3, SEED)


def test_design_is_reproducible():
    base = plan(["1", "2", "3"])
    first = doe.design(base, samples=5, seed=SEED)
    again = doe.design(base, samples=5, seed=SEED)

    assert [s.multipliers for s in first] == [s.multipliers for s in again]
    assert [s.name for s in first] == ["doe-000", "doe-001", "doe-002", "doe-003", "doe-004"]


def test_multipliers_stay_inside_the_bounds():
    scenarios = doe.design(plan(["1", "2"]), samples=12, seed=SEED, bounds=(0.2, 1.1))
    values = [value for s in scenarios for value in s.multipliers.values()]

    assert min(values) >= 0.2
    assert max(values) <= 1.1


def test_near_zero_multipliers_snap_to_a_real_shutdown():
    """Полпроцента дебита физически неотличимы от остановки, экономически — нет.

    Остановленная скважина выпадает из действующего фонда и не платит за
    содержание; открытая с околонулевым дебитом платит. Датасет не должен
    размазывать эту границу.
    """
    scenarios = doe.design(plan(["1"]), samples=30, seed=SEED, bounds=(0.0, 1.0))
    values = {value for s in scenarios for value in s.multipliers.values()}

    assert 0.0 in values
    assert not any(0.0 < value < 0.05 for value in values)


def test_shut_wells_are_really_shut():
    scenarios = doe.design(plan(["1", "2", "3"]), samples=20, seed=SEED)
    scenario = next(s for s in scenarios if s.shut_wells)

    shut = set(scenario.shut_wells)
    for step in scenario.steps:
        for control in step.producers:
            if control.well in shut:
                assert control.status == "SHUT"
                assert control.liquid_rate is None


def test_every_scenario_passes_the_schedule_validator():
    """Невалидный сценарий — это потерянные часы расчёта, ловим до симулятора."""
    for scenario in doe.design(plan(["1", "2", "3"], rate=450.0), samples=10, seed=SEED):
        assert validate(list(scenario.steps)) == []
        rates = [
            control.liquid_rate
            for step in scenario.steps
            for control in step.producers
            if control.liquid_rate is not None
        ]
        assert max(rates) <= MAX_LIQUID_RATE_M3_PER_DAY


def test_injectors_are_not_touched_by_the_design():
    scenarios = doe.design(plan(["1", "2"]), samples=6, seed=SEED)
    for scenario in scenarios:
        assert scenario.steps[0].injectors[0].rate == 30.0


def test_injector_design_varies_injection_and_keeps_production_fixed():
    scenarios = doe.design(
        plan(["1", "2"]), samples=8, seed=SEED, bounds=(0.5, 1.5),
        control_kind="injectors",
    )

    rates = {scenario.steps[0].injectors[0].rate for scenario in scenarios}

    assert len(rates) == 8
    assert all(scenario.steps[0].producers[0].liquid_rate == 40.0 for scenario in scenarios)
    assert all(scenario.multipliers == {} for scenario in scenarios)
    assert all(set(scenario.injector_multipliers) == {"9"} for scenario in scenarios)
    assert all(validate(list(scenario.steps)) == [] for scenario in scenarios)


def test_all_control_design_keeps_roles_separate_for_a_converted_well():
    steps = [
        ScheduleStep(
            date(2007, 1, 1),
            (ProducerControl("9", "OPEN", "LRAT", 40.0, 50.0),),
            (InjectorControl("9", "WATER", "SHUT", "RATE", None, 300.0),),
        ),
        ScheduleStep(
            date(2007, 2, 1),
            (ProducerControl("9", "SHUT", "LRAT", None, 50.0),),
            (InjectorControl("9", "WATER", "OPEN", "RATE", 30.0, 300.0),),
        ),
    ]

    scenario = doe.design(
        steps, samples=2, seed=SEED, bounds=(0.5, 1.5), control_kind="all"
    )[0]

    assert set(scenario.multipliers) == {"9"}
    assert set(scenario.injector_multipliers) == {"9"}
    assert validate(list(scenario.steps)) == []


def test_scenarios_actually_differ():
    scenarios = doe.design(plan(["1", "2", "3"]), samples=8, seed=SEED)
    signatures = {tuple(sorted(s.multipliers.items())) for s in scenarios}
    assert len(signatures) == 8


def test_design_needs_producers():
    steps = [ScheduleStep(date(2007, 1, 1), (), ())]
    with pytest.raises(ValueError, match="добывающих скважин"):
        doe.design(steps, samples=3, seed=SEED)


def test_design_rejects_a_broken_range():
    with pytest.raises(ValueError, match="диапазон"):
        doe.design(plan(["1"]), samples=3, seed=SEED, bounds=(1.0, 0.5))


def test_coverage_reports_the_span_per_well():
    scenarios = doe.design(plan(["1", "2", "3"]), samples=25, seed=SEED)
    report = doe.coverage(scenarios)

    assert report["scenarios"] == 25
    assert report["wells"] == 3
    # латинский гиперкуб обязан покрыть почти весь диапазон по каждой скважине
    assert report["min_span"] > 0.9 * (doe.DEFAULT_RANGE[1] - doe.DEFAULT_RANGE[0])


def test_coverage_of_nothing_is_not_a_crash():
    assert doe.coverage([]) == {"scenarios": 0}


def test_scenario_summary_is_readable():
    scenario = doe.design(plan(["1", "2"]), samples=4, seed=SEED)[0]
    text = scenario.summary()

    assert "doe-000" in text
    assert "скважин 2" in text


def test_levels_spread_scenarios_apart_not_just_wells():
    """Без общего уровня все сценарии садятся на один уровень отбора.

    Латинский гиперкуб разыгрывает каждую скважину независимо, поэтому средний
    множитель по сценарию — это среднее десятков независимых величин, и по закону
    больших чисел оно почти одинаково у всех точек выборки. Разнообразие
    получается внутри сценария, но не между сценариями, а суррогату нужно именно
    второе: он должен предсказывать отклик объекта на изменение уровня отбора.

    Поймано на реальных расчётах 16.08: шесть сценариев дали разброс ЧДД 5,9 %
    при множителях по скважинам от 0 до 1.25 (`docs/experiments.md`).
    """
    steps = [
        ScheduleStep(
            date=date(2007, 1, 1),
            producers=tuple(
                ProducerControl(well=str(number), status="OPEN", mode="LRAT",
                                     liquid_rate=100.0, bhp=50.0)
                for number in range(1, 41)
            ),
            injectors=(),
        ),
    ]

    def spread(scenarios):
        averages = [
            sum(s.multipliers.values()) / len(s.multipliers) for s in scenarios
        ]
        return max(averages) - min(averages)

    flat = doe.design(steps, samples=12, seed=1, bounds=(0.0, 1.25))
    layered = doe.design(steps, samples=12, seed=1, bounds=(0.8, 1.2), levels=(0.5, 1.5))

    # сравнение относительное, а не по абсолютному порогу: разброс средних падает
    # с ростом числа скважин (на 72 скважинах Модели Z он вдвое у́же, чем здесь),
    # и абсолютный порог означал бы разные требования на разных фондах
    assert spread(layered) > 2 * spread(flat), (
        f"уровни не развели сценарии: без уровней {spread(flat):.3f}, "
        f"с уровнями {spread(layered):.3f}"
    )


# ── третий вид управления: перевод под закачку ───────────────────────────────
#
# Контракт между машинами: сценарий с ВНС обязан записать по переведённой
# скважине **обе** величины — добычу в ноль и закачку больше нуля. Только тогда
# у неё появится колонка `inj:` в датасете, и вход суррогата перестанет быть
# немым. Без этого предложение перевести скважину не искажается, а исчезает.

def test_conversion_scenarios_record_both_multipliers():
    scenarios = doe.design(plan(["1", "2", "3", "4"]), samples=4, seed=7,
                           control_kind="conversions")

    converted = [
        (s, well) for s in scenarios for well in s.converted_wells
    ]
    assert converted, "хотя бы один сценарий обязан кого-то перевести"
    for scenario, well in converted:
        assert scenario.multipliers[well] == 0.0, "добыча переведённой — ноль"
        assert scenario.injector_multipliers[well] > 0.0, "а закачка больше нуля"


def test_conversion_actually_changes_the_schedule():
    scenarios = doe.design(plan(["1", "2", "3", "4"]), samples=4, seed=7,
                           control_kind="conversions")
    scenario = next(s for s in scenarios if s.converted_wells)
    well = scenario.converted_wells[0]

    injectors = {c.well for step in scenario.steps for c in step.injectors}
    assert well in injectors, "переведённая обязана появиться среди нагнетательных"
    assert all(c.status == "SHUT" for step in scenario.steps
               for c in step.producers if c.well == well)


def test_conversion_scenarios_stay_valid():
    for scenario in doe.design(plan(["1", "2", "3"]), samples=3, seed=1,
                               control_kind="conversions"):
        assert validate(list(scenario.steps)) == []


def test_conversion_share_is_reproducible_from_the_seed():
    first = doe.design(plan(["1", "2", "3", "4"]), samples=4, seed=11,
                       control_kind="conversions")
    second = doe.design(plan(["1", "2", "3", "4"]), samples=4, seed=11,
                        control_kind="conversions")
    assert [s.converted_wells for s in first] == [s.converted_wells for s in second]


def test_coverage_reports_conversions():
    scenarios = doe.design(plan(["1", "2", "3", "4"]), samples=4, seed=7,
                           control_kind="conversions")
    report = doe.coverage(scenarios)
    assert report["converted_total"] >= 1


# ── пустые переводы: скважина, уже работающая под закачкой ────────────────────
#
# Измерено на настоящем деке 19.08 (`docs/experiments.md`): в кандидаты на ВНС
# попадает скважина, которая в горизонте управления не добывает ни месяца, — в
# Модели Z 30 скважин переведены под закачку за историю. Расписание при этом
# корректно, переводить нечего; портится манифест: сборщик датасета запишет такой
# скважине `production_multiplier = 0` и `injection_multiplier > 0`, и в обучающей
# выборке появится строка «скважину перевели», отклик у которой в точности
# опорный. Сеть выучит, что перевод под закачку не даёт ничего — ровно на том
# классе решений, ради которого пакет и затевается.

def already_injecting(months: int = 6):
    """Расписание, где скважина 102 к началу управления уже под закачкой.

    Списано со скважины 102 Модели Z: одна закрывающая запись `WCONPROD` на
    01.2007 (`SHUT`, дебит 0) и закачка 90 м³/сут во всех месяцах после неё.
    Скважина 1 рядом — настоящий кандидат на перевод.
    """
    steps = []
    for month in range(1, months + 1):
        producers = [ProducerControl("1", "OPEN", "LRAT", 40.0, 50.0)]
        if month == 1:
            producers.append(ProducerControl("102", "SHUT", "LRAT", 0.0, 50.0))
        steps.append(ScheduleStep(
            date(2007, month, 1),
            tuple(producers),
            (InjectorControl("102", "WATER", "OPEN", "RATE", 90.0, 300.0),),
        ))
    return steps


def records_of(steps, well: str):
    """Все записи расписания по одной скважине — с чем сравнивать «изменилось»."""
    return [
        (step.date, kind, control)
        for step in steps
        for kind, controls in (("prod", step.producers), ("inj", step.injectors))
        for control in controls
        if control.well == well
    ]


def test_a_well_already_under_injection_is_not_offered_for_conversion():
    """Переводить под закачку ту, что и так качает, нечего."""
    for scenario in doe.design(already_injecting(), samples=6, seed=SEED,
                               control_kind="conversions"):
        assert "102" not in scenario.converted_wells
        assert "102" not in scenario.multipliers
        assert "102" not in scenario.injector_multipliers


def test_converted_wells_lists_only_wells_whose_schedule_changed():
    """Контракт манифеста: названа переведённой — значит, расписание другое."""
    base = already_injecting()
    for scenario in doe.design(base, samples=4, seed=SEED, control_kind="conversions"):
        assert scenario.converted_wells, "сценарий обязан кого-то перевести"
        for well in scenario.converted_wells:
            assert records_of(scenario.steps, well) != records_of(base, well)


def test_coverage_counts_only_real_conversions():
    """Отчёт о покрытии — число настоящих переводов, а не заявленных."""
    scenarios = doe.design(already_injecting(), samples=5, seed=SEED,
                           control_kind="conversions")
    report = doe.coverage(scenarios)

    assert report["converted_total"] == sum(len(s.converted_wells) for s in scenarios)
    assert report["converted_total"] == len(scenarios), "кандидат здесь ровно один"


def test_conversions_need_a_well_that_actually_produces():
    """Фонд, целиком работающий под закачкой, — отказ, а не пустой пакет."""
    steps = [
        ScheduleStep(
            date(2007, 1, 1),
            (ProducerControl("102", "SHUT", "LRAT", 0.0, 50.0),),
            (InjectorControl("102", "WATER", "OPEN", "RATE", 90.0, 300.0),),
        ),
    ]
    with pytest.raises(ValueError, match="добывающих скважин"):
        doe.design(steps, samples=2, seed=SEED, control_kind="conversions")
