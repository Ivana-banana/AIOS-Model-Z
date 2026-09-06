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
def test_monthly_event_quota_is_checked_separately_from_the_annual_one():
    """Две квоты — две проверки: год из двенадцати и месяц из двенадцати это разное."""
    steps = [
        step(2027, 1, producers=[prod("A", 10.0), prod("B", 10.0)]),
        step(2027, 2, producers=[prod("A", None, "SHUT"), prod("B", None, "SHUT")]),
    ]
    limits = planner.Limits(max_events_per_year=12, max_events_per_month=1)

    checks = {check.key: check for check in casecheck.check(steps, [], limits)}

    assert checks["max_events_per_year"].satisfied
    assert not checks["max_events_per_month"].satisfied
    assert checks["max_events_per_month"].where == "2027-02"
def test_minimum_rate_is_the_only_check_that_looks_downward():
    """У всех проверок нарушение — превышение; у этой наоборот, недобор до порога."""
    steps = [step(2007, 1, producers=[prod("мало", 2.0), prod("много", 40.0)])]

    check = casecheck.check(steps, [], planner.Limits(
        min_producer_rate_m3_per_day=5.0))[0]

    assert not check.satisfied
    assert check.value == pytest.approx(3.0), "недобор 5 − 2"
    assert "скв. мало" in check.where
def test_horizon_volume_is_one_basket_not_yearly_ones():
    steps = [step(2027, 1, injectors=[inj("W", 100.0)]),
             step(2028, 1, injectors=[inj("W", 100.0)])]

    check = casecheck.check(steps, [], planner.Limits(
        max_field_injection_m3_total=1_000.0))[0]

    assert check.value == pytest.approx(6_200.0)
    assert not check.satisfied and "2027-01 … 2028-01" in check.where
def test_minimum_rate_sees_an_open_well_with_zero_rate():
    """Открытая запись с нулём — то состояние, которое производят объёмные срезки.

    Ревизия 02.09: проверка пропускала `rate == 0` и `None`, то есть была слепа
    ровно к тому, что создаёт планировщик.
    """
    steps = [step(2007, 1, producers=[prod("нуль", 0.0), prod("норма", 40.0)])]

    check = casecheck.check(steps, [], planner.Limits(
        min_producer_rate_m3_per_day=5.0))[0]

    assert not check.satisfied
    assert "скв. нуль" in check.where and check.value == pytest.approx(5.0)


# ── что нашёл мутационный прогон 06.09 (Р-221) ───────────────────────────────


def test_each_limit_measures_the_role_it_names():
    """Предел для добывающих меряет добычу, для нагнетательных — закачку.

    Четыре мутации подменяли `producers=True` на `False` и обратно, и все
    переживали сюиту: в фикстурах добыча и закачка редко расходятся так, чтобы
    подмена дала другое число. Между тем это молчаливая подмена смысла — предел
    «на скважину по жидкости» начинал бы мерить приёмистость, и условие кейса
    проверялось бы не то, которое объявлено.

    Здесь роли расходятся намеренно: добывающая на 100, нагнетательная на 400.
    """
    steps = [step(2014, 1, producers=(prod("p", 100.0),), injectors=(inj("i", 400.0),))]

    liquid = casecheck.check(steps, [], planner.Limits(
        enforce_water_balance=False, max_well_liquid_m3_per_day=1000.0))
    injection = casecheck.check(steps, [], planner.Limits(
        enforce_water_balance=False, max_well_injection_m3_per_day=1000.0))

    by_key = {one.key: one.value for one in liquid + injection}
    assert by_key["max_well_liquid_m3_per_day"] == 100.0, (
        f"предел по жидкости померил {by_key['max_well_liquid_m3_per_day']} — "
        "это приёмистость, а не отбор"
    )
    assert by_key["max_well_injection_m3_per_day"] == 400.0, (
        f"предел по закачке померил {by_key['max_well_injection_m3_per_day']}"
    )

    yearly_liquid = casecheck.check(steps, [], planner.Limits(
        enforce_water_balance=False, max_field_liquid_m3_per_year=1e9))
    yearly_injection = casecheck.check(steps, [], planner.Limits(
        enforce_water_balance=False, max_field_injection_m3_per_year=1e9))

    assert yearly_liquid[0].value < yearly_injection[0].value, (
        f"годовые объёмы совпали ({yearly_liquid[0].value}): роли перепутаны"
    )


def test_a_group_cap_measures_its_own_resource():
    """Групповой предел по закачке считает закачку, а не отбор.

    Та же подмена роли, но в групповых пределах — там она стоит отдельным
    аргументом и потому проверялась отдельной мутацией.
    """
    steps = [step(2014, 1, producers=(prod("p", 100.0),), injectors=(inj("i", 400.0),))]
    limits = planner.Limits(
        enforce_water_balance=False,
        well_groups={"куст": ("p", "i")},
        group_liquid_caps_m3_per_day={"куст": 1000.0},
        group_injection_caps_m3_per_day={"куст": 1000.0},
    )

    found = {one.key: one.value for one in casecheck.check(steps, [], limits)}

    assert found["group_liquid_caps_m3_per_day[куст]"] == 100.0
    assert found["group_injection_caps_m3_per_day[куст]"] == 400.0, (
        "групповой предел по закачке померил отбор"
    )


def test_a_well_without_a_reference_rate_does_not_divide_by_zero():
    """Скважина с нулевым `WLPR` в прогнозе доли не даёт — и не роняет проверку.

    Мутация `reference > 0` на `>=` переживала сюиту: строки с нулевым дебитом
    в прогнозе не встречалось. На выгрузке она обычна — остановленная скважина
    даёт ровно её, и деление на ноль уронило бы проверку условий кейса вместо
    списка нарушений.
    """
    steps = [step(2014, 1, producers=(prod("p", 100.0),))]
    forecast = [make_row("2014-01-01", "p", wlpr=0.0, womr=0.0,
                         liquid=0.0, oil=0.0)]

    checks = casecheck.check(steps, forecast, planner.Limits(
        enforce_water_balance=False, max_field_oil_t_per_day=1000.0))

    assert checks, "проверка по фазе не выполнена"
    assert checks[0].value == 0.0, (
        f"скважина без дебита дала вклад {checks[0].value} в добычу нефти"
    )


def test_the_worst_case_names_the_first_month_it_happened():
    """При равных пиках называется **первый** месяц, а не последний.

    Три мутации меняли `>` на `>=` в поиске максимума. Значение от этого не
    меняется — меняется `where`, то есть строка, по которой человек идёт
    разбираться. «Пик в декабре» вместо «пик в январе» отправляет искать причину
    не туда, а на ровной полке пики совпадают часто.
    """
    steps = [step(2014, month, producers=(prod("p", 100.0),)) for month in (1, 2, 3)]

    daily = casecheck.check(steps, [], planner.Limits(
        enforce_water_balance=False, max_field_liquid_m3_per_day=1000.0))[0]

    assert daily.value == 100.0
    assert daily.where.startswith("2014-01"), (
        f"назван месяц {daily.where} — при равных пиках берётся последний, а не первый"
    )


def test_the_shortfall_names_the_first_well_that_hit_it():
    """То же для недобора: при равных провалах называется первый по порядку."""
    steps = [step(2014, 1, producers=(prod("a", 1.0), prod("b", 1.0)))]

    checks = casecheck.check(steps, [], planner.Limits(
        enforce_water_balance=False, min_producer_rate_m3_per_day=5.0))
    shortfall = [one for one in checks
                 if one.key == "min_producer_rate_m3_per_day"][0]

    assert shortfall.value == 4.0
    assert "скв. a" in shortfall.where, (
        f"назван {shortfall.where} — при равном недоборе берётся не первая скважина"
    )


def test_a_zero_cap_forbids_any_work_of_either_role():
    """Нулевой потолок скважины запрещает работу — и добычу, и закачку.

    Три мутации сидели в этой ветке: роль в объединении ставок и оба условия
    абсолютного потолка. Ноль означает «не давать работать», и пропустить такую
    скважину значит не проверить единственный случай, когда потолок абсолютен.
    """
    limits = planner.Limits(enforce_water_balance=False,
                            well_caps_m3_per_day={"i": 0.0})
    working = [step(2014, 1, injectors=(inj("i", 400.0),))]
    idle = [step(2014, 1, injectors=(inj("i", 0.0, status="SHUT"),))]

    breach = [one for one in casecheck.check(working, [], limits)
              if one.key == "well_caps_m3_per_day"]
    assert breach and not breach[0].satisfied, (
        f"работа при нулевом потолке не объявлена нарушением: {breach}"
    )
    assert "скв. i" in breach[0].where

    quiet = [one for one in casecheck.check(idle, [], limits)
             if one.key == "well_caps_m3_per_day"]
    assert not quiet or quiet[0].satisfied, (
        "остановленная скважина объявлена нарушившей нулевой потолок"
    )


def test_the_annual_average_names_the_first_year_of_equal_peaks():
    """Среднегодовой предел при равных годах называет первый.

    Отдельная мутация от суточной: средние за год и пик месяца считаются разными
    ветками, и закрыта была одна. Год в отчёте — то, с чего начинают разбор.
    """
    steps = [step(year, month, producers=(prod("p", 100.0),))
             for year in (2014, 2015) for month in range(1, 13)]

    # среднегодовая ветка включается отдельным ключом кейса: без него тот же
    # предел считается пиком месяца, и проверка ушла бы в другую функцию
    yearly = casecheck.check(steps, [], planner.Limits(
        enforce_water_balance=False, max_field_liquid_m3_per_day=1000.0,
        field_limits_are_annual_average=True))[0]

    assert "среднее за год" in yearly.where, yearly.where
    assert yearly.where.startswith("2014"), (
        f"назван {yearly.where} — при равных годах берётся последний, а не первый"
    )


def test_a_zero_cap_ignores_a_well_that_does_not_work():
    """Открытая запись с нулевым расходом нулевой потолок не нарушает.

    Мутация `and` на `or` переживала прежний тест: остановленная запись в
    ставки вообще не попадает, поэтому до условия дело не доходило. Разделяет
    их **открытая** запись с нулём — состояние, которое создаёт объёмная срезка.
    """
    limits = planner.Limits(enforce_water_balance=False,
                            well_caps_m3_per_day={"i": 0.0})
    open_at_zero = [step(2014, 1, injectors=(inj("i", 0.0),))]

    checks = [one for one in casecheck.check(open_at_zero, [], limits)
              if one.key == "well_caps_m3_per_day"]

    assert not checks or checks[0].value == 0.0, (
        f"открытая запись с нулевым расходом объявлена нарушившей нулевой потолок: "
        f"{[(one.value, one.where) for one in checks]}"
    )


def test_a_zero_cap_reports_the_first_offender():
    """При нескольких нарушителях нулевого потолка называется первый.

    Мутация `worst_ratio < inf` на `<=` переживала сюиту: нарушитель в фикстурах
    был один. Со второй записью подмена перезаписывает `where` — и разбираться
    отправляют к последней скважине вместо первой.
    """
    limits = planner.Limits(enforce_water_balance=False,
                            well_caps_m3_per_day={"a": 0.0, "b": 0.0})
    steps = [step(2014, 1, injectors=(inj("a", 100.0), inj("b", 400.0)))]

    breach = [one for one in casecheck.check(steps, [], limits)
              if one.key == "well_caps_m3_per_day"][0]

    assert not breach.satisfied
    assert "скв. a" in breach.where, (
        f"назван {breach.where} — при равных (бесконечных) отношениях берётся "
        "последний нарушитель, а не первый"
    )
    assert breach.value == 100.0, f"в отчёт попал расход не первой скважины: {breach.value}"

