"""Проверка расписания на соблюдение условий кейса — по результату, а не по процессу.

Смысл модуля в одном: планировщик может ничего не сделать, и проверка процесса
это пройдёт. Здесь считается факт против предела — так 30.08 нашлась мёртвая
ручка `max_events_per_year` (Р-113), и так же будет найдена следующая.
"""

from __future__ import annotations

from datetime import date

import pytest
from conftest import make_row

from aios import casecheck, planner
from aios.deck import InjectorControl, ProducerControl, ScheduleStep
from aios.summary import Fluids

FLUIDS = Fluids()


def step(year: int, month: int, producers=(), injectors=()):
    return ScheduleStep(date(year, month, 1), tuple(producers), tuple(injectors))


def prod(well: str, rate: float | None, status: str = "OPEN"):
    return ProducerControl(well, status, "LRAT", rate, 50.0)


def inj(well: str, rate: float | None, status: str = "OPEN"):
    return InjectorControl(well, "WATER", status, "RATE", rate, 300.0)


def test_no_limits_declared_means_nothing_to_check():
    """Пустые ограничения дают пустой список, а не выдуманные проверки."""
    assert casecheck.check([step(2007, 1, producers=[prod("1", 100.0)])], [],
                           planner.Limits()) == []


def test_field_rate_names_the_month_where_the_peak_is():
    """«Фактическое значение, предел и период» — требование протокола приёмки."""
    steps = [step(2007, 1, producers=[prod("1", 90.0)]),
             step(2007, 2, producers=[prod("1", 130.0)])]

    checks = casecheck.check(steps, [], planner.Limits(max_field_liquid_m3_per_day=100.0))

    assert len(checks) == 1
    assert checks[0].value == 130.0 and checks[0].where == "2007-02"
    assert not checks[0].satisfied
    assert "НАРУШЕНО" in str(checks[0])


def test_average_reading_checks_the_year_not_the_month():
    """Тот же предел под ключом «среднее за год» проверяется иначе, и это видно."""
    steps = [step(2007, 1, producers=[prod("1", 150.0)]),
             step(2007, 2, producers=[prod("1", 10.0)])]

    peak = casecheck.check(steps, [], planner.Limits(max_field_liquid_m3_per_day=100.0))
    average = casecheck.check(steps, [], planner.Limits(
        max_field_liquid_m3_per_day=100.0, field_limits_are_annual_average=True))

    assert not peak[0].satisfied
    assert average[0].satisfied and "среднее за год" in average[0].where


def test_annual_volume_is_checked_by_calendar_year():
    steps = [step(2007, 1, injectors=[inj("9", 100.0)]),
             step(2008, 1, injectors=[inj("9", 10.0)])]

    checks = casecheck.check(steps, [], planner.Limits(
        max_field_injection_m3_per_year=1_000.0))

    assert checks[0].value == pytest.approx(3_100.0) and checks[0].where == "2007"
    assert not checks[0].satisfied


def test_group_cap_is_checked_only_over_its_members():
    steps = [step(2007, 1, producers=[prod("1", 60.0), prod("2", 60.0), prod("3", 60.0)])]
    limits = planner.Limits(
        well_groups={"куст-1": frozenset({"1", "2"})},
        group_liquid_caps_m3_per_day={"куст-1": 100.0},
    )

    checks = casecheck.check(steps, [], limits)

    assert checks[0].value == 120.0, "третья скважина в группу не входит"
    assert "куст-1" in checks[0].key


def test_forced_shut_window_violation_names_the_well_and_the_month():
    steps = [step(2027, 3, producers=[prod("1", 30.0)]),
             step(2027, 9, producers=[prod("1", 30.0)])]
    limits = planner.Limits(forced_shut_windows=(
        planner.ShutWindow("1", date(2027, 1, 1), date(2027, 6, 30)),))

    checks = casecheck.check(steps, [], limits)

    assert checks[0].value == 1.0, "нарушение только внутри окна"
    assert checks[0].where == "скв. 1, 2027-03"
    assert not checks[0].satisfied


def test_forced_shut_window_is_satisfied_when_the_well_is_closed():
    steps = [step(2027, 3, producers=[prod("1", None, status="SHUT")])]
    limits = planner.Limits(forced_shut_windows=(
        planner.ShutWindow("1", date(2027, 1, 1), date(2027, 6, 30)),))

    assert casecheck.check(steps, [], limits)[0].satisfied


def test_phase_is_scaled_by_the_change_of_the_liquid_rate():
    """Прогноз считался для прежнего расписания — фаза масштабируется уставкой.

    Скважина в прогнозе работала на 100 м³/сут и дала 91,3 т нефти за январь; в
    расписании ей оставили 50. Значит нефти ожидается вдвое меньше — и проверка
    обязана это учитывать, иначе срезанное расписание всё равно «нарушает».
    """
    forecast = [make_row("2007-01-01", "1", wlpr=100.0, womr=3.0,
                         liquid=1000.0, oil=91.3)]
    full = casecheck.check([step(2007, 1, producers=[prod("1", 100.0)])], forecast,
                           planner.Limits(max_field_oil_t_per_day=2.0))
    halved = casecheck.check([step(2007, 1, producers=[prod("1", 50.0)])], forecast,
                             planner.Limits(max_field_oil_t_per_day=2.0))

    assert full[0].value == pytest.approx(91.3 / 31, rel=1e-6)
    assert halved[0].value == pytest.approx(full[0].value / 2, rel=1e-6)
    assert not full[0].satisfied and halved[0].satisfied


def test_a_raised_setpoint_does_not_promise_more_oil():
    """Уставка выше достигнутого — надежда, а не отклик; отношение ограничено единицей.

    Найдено репетицией 01.09: исторические уставки Модели Z выше достигнутых
    дебитов в 1,47 раза, и без ограничения проверка объявляла нарушенной фазу,
    которую планировщик уже уложил в предел.
    """
    forecast = [make_row("2007-01-01", "1", wlpr=100.0, womr=3.0,
                         liquid=1000.0, oil=91.3)]
    limits = planner.Limits(max_field_oil_t_per_day=10.0)

    kept = casecheck.check([step(2007, 1, producers=[prod("1", 100.0)])], forecast, limits)
    raised = casecheck.check([step(2007, 1, producers=[prod("1", 300.0)])], forecast, limits)

    assert raised[0].value == kept[0].value == pytest.approx(91.3 / 31, rel=1e-6)


def test_water_phase_is_liquid_minus_oil():
    forecast = [make_row("2007-01-01", "1", wlpr=100.0, womr=3.0,
                         liquid=1000.0, oil=91.3)]

    checks = casecheck.check([step(2007, 1, producers=[prod("1", 100.0)])], forecast,
                             planner.Limits(max_field_water_t_per_day=1.0))

    assert checks[0].value == pytest.approx((1000.0 - 91.3) / 31, rel=1e-6)


def test_a_well_absent_from_the_forecast_does_not_invent_a_response():
    """Скважине без прогноза отклик не выдумывается: она в проверку не входит."""
    forecast = [make_row("2007-01-01", "1", wlpr=100.0, womr=3.0,
                         liquid=1000.0, oil=91.3)]
    steps = [step(2007, 1, producers=[prod("1", 100.0), prod("новая", 100.0)])]

    checks = casecheck.check(steps, forecast, planner.Limits(max_field_oil_t_per_day=10.0))

    assert checks[0].value == pytest.approx(91.3 / 31, rel=1e-6)


def test_well_caps_report_the_worst_offender_against_its_own_cap():
    """Адресные потолки: у каждой скважины свой предел, и назван должен быть худший."""
    steps = [step(2007, 1, producers=[prod("1", 300.0), prod("2", 120.0)])]
    limits = planner.Limits(well_caps_m3_per_day={"1": 250.0, "2": 100.0})

    checks = casecheck.check(steps, [], limits)

    assert checks[0].key == "well_caps_m3_per_day"
    assert checks[0].value == 300.0 and checks[0].limit == 250.0
    assert checks[0].where == "скв. 1, 2007-01"
    assert not checks[0].satisfied


def test_well_caps_compare_each_well_with_its_own_cap_not_by_absolute_rate():
    """Скважина 900/1000 не должна закрывать собой скважину 100/10.

    Найдено критиком 01.09: ранжирование шло по абсолютной ставке, и нарушитель
    с маленьким потолком терялся за большим соседом — `violations()` возвращал
    пусто на расписании, которое нарушает адресный потолок вдесятеро.
    """
    steps = [step(2007, 1, producers=[prod("большая", 900.0), prod("малая", 100.0)])]
    limits = planner.Limits(well_caps_m3_per_day={"большая": 1000.0, "малая": 10.0})

    checks = casecheck.check(steps, [], limits)

    assert not checks[0].satisfied
    assert checks[0].where == "скв. малая, 2007-01"
    assert checks[0].value == 100.0 and checks[0].limit == 10.0


def test_well_caps_are_satisfied_when_every_well_is_under_its_own():
    """Соблюдение считается по своему потолку, а не по чужому."""
    steps = [step(2007, 1, producers=[prod("1", 200.0), prod("2", 90.0)])]
    limits = planner.Limits(well_caps_m3_per_day={"1": 250.0, "2": 100.0})

    assert casecheck.check(steps, [], limits)[0].satisfied


def test_violations_lists_only_the_broken_ones():
    steps = [step(2007, 1, producers=[prod("1", 120.0)], injectors=[inj("9", 10.0)])]
    limits = planner.Limits(max_field_liquid_m3_per_day=100.0,
                            max_field_injection_m3_per_day=100.0)

    broken = casecheck.violations(casecheck.check(steps, [], limits))

    assert [one.key for one in broken] == ["max_field_liquid_m3_per_day"]


def test_a_schedule_fixed_by_the_planner_passes_the_check():
    """Сквозная связка: что планировщик привёл, то проверка обязана принять.

    Это главный тест модуля. Планировщик и проверка написаны раздельно и по
    разным формулам; совпадение здесь и означает, что срезка действительно
    выполняет объявленное условие, а не выглядит выполняющей.
    """
    from aios import economics
    from aios.margin import Norms

    assumptions, _ = economics.load_norms()
    norms = Norms.from_assumptions(assumptions)

    steps = [step(2007, month, producers=[prod("1", 150.0), prod("2", 90.0)],
                  injectors=[inj("9", 200.0)])
             for month in (1, 2, 3)]
    forecast = [make_row(f"2007-{month:02d}-01", well, wlpr=150.0, womr=3.0,
                         liquid=1000.0, oil=91.3)
                for month in (1, 2, 3) for well in ("1", "2")]
    limits = planner.Limits(
        enforce_water_balance=False,
        max_field_liquid_m3_per_day=200.0,
        max_field_injection_m3_per_year=1_000_000.0,
        well_groups={"куст-1": frozenset({"1", "2"})},
        group_liquid_caps_m3_per_day={"куст-1": 180.0},
        forced_shut_windows=(planner.ShutWindow("2", date(2007, 3, 1), date(2007, 3, 31)),),
    )

    before = casecheck.check(steps, forecast, limits)
    assert casecheck.violations(before), "тест бессмыслен, если нарушать нечего"

    fixed, decisions = planner.enforce(steps, forecast, norms, limits, FLUIDS)

    assert decisions
    assert casecheck.violations(casecheck.check(fixed, forecast, limits)) == []
def test_events_per_year_is_checked_and_can_be_exceeded_visibly():
    """Ключ, который проверка не знала вовсе, — и который назван причиной модуля.

    После Р-139 квота может быть превышена законно: мероприятие, порождённое
    жёстким ограничением, не отменяется. Значит превышение обязано быть видно.
    """
    steps = [
        step(2027, 1, producers=[prod("A", 50.0), prod("B", 50.0)]),
        step(2027, 2, producers=[prod("A", None, "SHUT"), prod("B", None, "SHUT")]),
    ]

    checks = casecheck.check(steps, [], planner.Limits(max_events_per_year=1))

    assert checks[0].key == "max_events_per_year"
    assert checks[0].value == 2.0 and checks[0].where == "2027"
    assert not checks[0].satisfied


def test_events_per_year_counts_the_same_events_the_planner_does():
    """Два счётчика одной величины разъезжаются молча — поэтому код один."""
    steps = [
        step(2027, 1, producers=[prod("A", 50.0)]),
        step(2027, 2, producers=[prod("A", None, "SHUT")]),
        step(2027, 3, producers=[prod("A", 50.0)]),
    ]

    checks = casecheck.check(steps, [], planner.Limits(max_events_per_year=5))

    assert checks[0].value == float(len(planner.events_of(steps)))
    assert checks[0].value == 2.0, "остановка и пуск — два мероприятия"


def test_annual_average_divides_by_the_days_present_not_by_365():
    """Мутация `/365` вместо `/дни года` не должна выживать: год бывает неполным."""
    steps = [step(2007, 11, producers=[prod("1", 100.0)]),
             step(2007, 12, producers=[prod("1", 100.0)])]
    limits = planner.Limits(max_field_liquid_m3_per_day=100.0,
                            field_limits_are_annual_average=True)

    check = casecheck.check(steps, [], limits)[0]

    assert check.value == pytest.approx(100.0), (
        "два месяца по 100 м³/сут дают среднесуточное 100, а не 100 × 61 / 365"
    )
    assert check.satisfied


def test_annual_volume_sums_the_months_instead_of_taking_the_largest():
    """Мутация `max(...)` вместо `+=` не должна выживать: объём — сумма."""
    steps = [step(2007, month, injectors=[inj("9", 10.0)]) for month in (1, 2, 3)]

    check = casecheck.check(steps, [], planner.Limits(
        max_field_injection_m3_per_year=1_000.0))[0]

    assert check.value == pytest.approx(10.0 * (31 + 28 + 31))


def test_active_wells_and_forced_shut_count_injectors_too():
    """Нагнетательные — часть действующего фонда и часть обязательных остановок."""
    steps = [step(2007, 1, producers=[prod("1", 50.0)],
                  injectors=[inj("9", 40.0), inj("8", 40.0)])]

    active = casecheck.check(steps, [], planner.Limits(max_active_wells=2))[0]
    forced = casecheck.check(steps, [], planner.Limits(
        forced_shut_wells=frozenset({"9"})))[0]

    assert active.value == 3.0 and not active.satisfied
    assert forced.value == 1.0 and forced.where == "скв. 9, 2007-01"
def test_a_zero_cap_is_violated_by_any_work():
    """Потолок ноль абсолютен: «не давать работать». Раньше он единственный не проверялся."""
    steps = [step(2007, 1, producers=[prod("W", 5.0)])]

    check = casecheck.check(steps, [], planner.Limits(well_caps_m3_per_day={"W": 0.0}))[0]

    assert not check.satisfied
    assert check.value == 5.0 and check.limit == 0.0


def test_field_limits_see_wells_that_keep_running_from_an_earlier_step():
    """Дек несёт дельты: скважина без записи на шаге продолжает работать.

    Ревизия 02.09: пределы объекта, группы и фонда суммировали только записи
    шага, поэтому три нагнетателя по 400 м³/сут, открытые в три разных месяца,
    не нарушали предела 800 — и проверка соглашалась.
    """
    steps = [
        step(2027, 1, injectors=[inj("A", 400.0)]),
        step(2027, 2, injectors=[inj("B", 400.0)]),
        step(2027, 3, injectors=[inj("C", 400.0)]),
    ]
    limits = planner.Limits(max_field_injection_m3_per_day=800.0)

    check = casecheck.check(planner.expand_deltas(steps), [], limits)[0]

    assert check.value == pytest.approx(1200.0) and not check.satisfied
