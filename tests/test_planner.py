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
