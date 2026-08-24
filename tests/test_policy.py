"""Бейзлайн-политика: от диагноза к расписанию."""

from __future__ import annotations

from datetime import date

import pytest
from conftest import make_row, months

from aios import economics, policy
from aios.deck import InjectorControl, ProducerControl, ScheduleStep
from aios.margin import Norms
from aios.schedule import validate

CONTROL_START = date(2014, 1, 1)


@pytest.fixture
def norms():
    assumptions, _ = economics.load_norms()
    return Norms.from_assumptions(assumptions)


def losing(data: str, well: str = "bad"):
    """Скважина с почти чистой водой: убыток около 0,17 млн руб. в месяц."""
    return make_row(data, well, wlpr=30, womr=0.1, liquid=900, oil=3)


def earning(data: str, well: str = "good"):
    return make_row(data, well, wlpr=30, womr=5, liquid=900, oil=150)


def test_no_actions_while_the_loss_is_short(norms):
    rows = [losing(data) for data in months(2014, 4)]
    assert policy.plan_shutdowns(rows, norms, control_start=CONTROL_START) == []


def test_no_actions_for_a_profitable_well(norms):
    rows = [earning(data) for data in months(2014, 24)[:12]]
    assert policy.plan_shutdowns(rows, norms, control_start=CONTROL_START) == []


def test_shutdown_takes_effect_the_month_after_the_decision(norms):
    """Решение принимается по данным до даты, а действует со следующего месяца."""
    rows = [losing(data) for data in months(2014, 12)]
    actions = policy.plan_shutdowns(rows, norms, control_start=CONTROL_START)

    assert len(actions) == 1
    action = actions[0]
    assert action.well == "bad"
    assert action.kind == policy.SHUT
    assert action.cost_m == pytest.approx(1.0)
    # шесть убыточных месяцев (янв–июн) дают убыток > 1 млн только к седьмому
    assert action.date > date(2014, 6, 1)


def test_decision_uses_only_the_past(norms):
    """Обрезав данные после даты решения, получаем то же самое действие.

    Это и есть проверка причинности: правило, подглядывающее в будущее, здесь
    дало бы другой ответ.
    """
    full = [losing(data) for data in months(2014, 12)]
    action = policy.plan_shutdowns(full, norms, control_start=CONTROL_START)[0]

    cutoff = action.date.isoformat()
    truncated = [row for row in full if row["DATA"] <= cutoff]
    again = policy.plan_shutdowns(truncated, norms, control_start=CONTROL_START)

    assert [a.date for a in again] == [action.date]
    assert again[0].reason == action.reason


def test_a_recovered_month_resets_the_streak(norms):
    rows = [losing(data) for data in months(2014, 5)]
    rows.append(earning(months(2014, 6)[5], well="bad"))  # шестой месяц прибыльный
    rows.extend(losing(data) for data in months(2014, 5, first=7))
    assert policy.plan_shutdowns(rows, norms, control_start=CONTROL_START) == []


def test_actions_never_precede_the_control_start(norms):
    rows = [losing(data) for data in months(2006, 12)] + [losing(data) for data in months(2007, 12)]
    actions = policy.plan_shutdowns(rows, norms, control_start=date(2007, 1, 1))
    assert all(action.date >= date(2007, 1, 1) for action in actions)


def test_a_well_is_shut_only_once(norms):
    rows = [losing(data) for data in months(2014, 12)] + [losing(data) for data in months(2015, 12)]
    actions = policy.plan_shutdowns(rows, norms, control_start=CONTROL_START)
    assert [action.well for action in actions] == ["bad"]


def _steps(dates, wells=("1", "2")):
    return [
        ScheduleStep(
            day,
            tuple(ProducerControl(well, "OPEN", "LRAT", 45.0, 50.0) for well in wells),
            (InjectorControl("9", "WATER", "OPEN", "RATE", 30.0, 300.0),),
        )
        for day in dates
    ]


def test_apply_writes_shut_once_and_then_leaves_the_well_alone():
    """Режим в Eclipse действует до следующего изменения — повторять SHUT незачем."""
    steps = _steps([date(2014, month, 1) for month in (1, 2, 3, 4)])
    actions = [policy.Action(date(2014, 2, 1), "1", policy.SHUT, "убыточна", 1.0)]

    result = policy.apply(steps, actions)

    assert [c.well for c in result[0].producers] == ["1", "2"]
    assert result[0].producers[0].status == "OPEN"

    shut = [c for c in result[1].producers if c.well == "1"]
    assert len(shut) == 1 and shut[0].status == "SHUT" and shut[0].liquid_rate is None

    for step in result[2:]:
        assert [c.well for c in step.producers] == ["2"]


def test_apply_keeps_injectors_untouched():
    steps = _steps([date(2014, 1, 1)])
    result = policy.apply(steps, [policy.Action(date(2014, 1, 1), "1", policy.SHUT, "x", 1.0)])
    assert result[0].injectors == steps[0].injectors


def test_result_passes_the_schedule_validator():
    steps = _steps([date(2014, month, 1) for month in (1, 2, 3)])
    result = policy.apply(steps, [policy.Action(date(2014, 2, 1), "1", policy.SHUT, "x", 1.0)])
    assert validate(result) == []


def test_apply_rejects_unknown_actions():
    with pytest.raises(ValueError, match="неизвестное действие"):
        policy.apply(_steps([date(2014, 1, 1)]), [policy.Action(date(2014, 1, 1), "1", "fly", "", 0)])


def test_explain_is_readable(norms):
    rows = [losing(data) for data in months(2014, 12)]
    actions = policy.plan_shutdowns(rows, norms, control_start=CONTROL_START)
    text = policy.explain(actions)

    assert "Действий: 1" in text
    assert "скв. bad" in text
    assert "обводнённость" in text
    assert "1.0 млн руб" in text


def test_explain_says_so_when_there_is_nothing_to_do():
    assert "Действий нет" in policy.explain([])


def injecting(data: str, well: str = "нагн"):
    """Нагнетательная: закачка есть, добычи нет — выручки нет по построению."""
    return make_row(data, well, wlpr=0.0, womr=0.0, liquid=0.0, oil=0.0,
                    wwir=300.0, injection=9000.0)


def test_an_injector_is_not_judged_by_a_rule_written_for_producers(norms):
    """Политика не должна выносить решение, которого расписание не исполняет.

    У нагнетательной скважины выручки нет по построению, поэтому её месячная
    маржа отрицательна всегда: закачка — чистый расход. Правило «убыточна
    `lookback` месяцев подряд» срабатывало на ней автоматически, и политика
    выносила остановку. Расписание её при этом не исполняло: `apply` выдаёт
    только `ProducerControl SHUT` и нагнетательные пропускает. Замер 20.08:
    решение вынесено (`shut`, 1,0 млн руб., «убыток 6 мес. подряд, накоплено
    2,12 млн»), реально остановленных месяцев — ноль.

    Цена не в деньгах, а в доверии к слою интерпретируемости: решение попадает
    в таблицу «Принятые решения», объясняется языковой моделью и стоит миллион
    на бумаге — а его не происходит. В базовом расписании Модели Z 41
    нагнетательная скважина (11 никогда не добывали, 30 переведены под закачку).

    Остановка закачки — отдельное решение со своими последствиями для энергии
    пласта, и принимать его как побочный эффект правила про добывающие нельзя:
    замер того же дня показал, что срез закачки обрушивает нефть вчетверо.
    """
    rows = [injecting(data) for data in months(2014, 12)]

    assert policy.plan_shutdowns(rows, norms, control_start=CONTROL_START) == []


def test_a_well_converted_to_injection_stops_accruing_a_producer_loss(norms):
    """Перевод под закачку обрывает счётчик убытка, а не продолжает его.

    После перевода скважина не добывает: судить её правилом про добывающие
    нечем, и накопленный до перевода убыток продолжать нельзя.
    """
    rows = [losing(data, "внс") for data in months(2014, 3)]
    rows += [injecting(data, "внс") for data in months(2014, 12)[3:]]

    assert policy.plan_shutdowns(rows, norms, control_start=CONTROL_START) == []


def test_a_row_the_methodology_drops_does_not_count_as_a_loss(norms):
    """Строка, исключённая методикой, не должна закрывать прибыльную скважину.

    Методика исключает строку с отрицательным `WLPT_Diff`, `WOMT_Diff` или
    `WWIT_Diff` из экономики **целиком** — не обнуляет и не клампит. Скрининг
    считал её как есть, то есть приписывал скважине убыток, которого в деньгах
    не существует.

    Замер 20.08: месяц с `WOMT_Diff = −150` даёт маржу −1,427 млн руб. против
    +1,081 у того же месяца с нормальным приращением. Прибыльная скважина, у
    которой выгрузка раз за разом откатывает накопленное, останавливалась с
    обоснованием «убыток 6 мес. подряд, накоплено 8,56 млн руб.; обводнённость
    **116,7 %**» — обводнённость выше ста процентов физически невозможна и сама
    выдаёт, что решение принято по несуществующим деньгам.

    Отрицательные строки на этом объекте не гипотетические: 102 штуки за
    2025-09 в выгрузке организаторов (`JOURNAL.md`, принцип 1).
    """
    rows = []
    for month in months(2014, 12):
        row = earning(month, "П1")
        if month >= "2014-04-01":
            row["WOMT_Diff"] = -150.0
        rows.append(row)

    assert policy.plan_shutdowns(rows, norms, control_start=CONTROL_START) == []


def _watery(well: str, months: int, *, watercut: float, liquid: float = 900.0,
            start_year: int = 2014):
    """Скважина, поднимающая почти одну воду и потому убыточная."""
    rows = []
    for index in range(months):
        year = start_year + index // 12
        month = index % 12 + 1
        rows.append(make_row(f"{year}-{month:02d}-01", well, wlpr=30.0,
                             womr=liquid * (1 - watercut) / 30,
                             liquid=liquid, oil=liquid * (1 - watercut)))
    return rows


def test_a_water_producer_is_offered_for_conversion_not_only_for_shutdown(norms):
    """Скважина, поднимающая одну воду, — кандидат на перевод под закачку.

    Остановка убирает и воду, и энергию пласта; перевод оставляет объём в
    системе, но разворачивает его — вода идёт обратно и поддерживает давление у
    соседей. Методика оценивает событие отдельно: 5,0 млн руб. ВНС без стоимости
    ЭЦН (Р-30), то есть планка впятеро выше, чем у остановки.
    """
    from aios.policy import CONVERT, plan_conversions

    # сорок восемь месяцев не для красоты: накопленный убыток обязан перекрыть
    # 5,0 млн руб. ВНС, а это впятеро выше планки остановки — на двух годах
    # такая скважина её не набирает
    rows = _watery("вода", 48, watercut=0.995)

    actions = plan_conversions(rows, norms, control_start=date(2014, 1, 1))

    assert actions, "кандидат на перевод не найден"
    assert actions[0].kind == CONVERT
    assert actions[0].cost_m == norms.conversion_m == 5.0
    assert "обводнённость" in actions[0].reason
    assert "полным расчётом" in actions[0].reason, (
        "предложение обязано говорить, что выигрыш по соседям суррогатом не оценён"
    )


def test_a_well_with_oil_is_not_offered_for_conversion(norms):
    """С заметной нефтью перевод отнимает добычу, а размен оценить нечем."""
    from aios.policy import plan_conversions

    rows = _watery("нефтяная", 24, watercut=0.80)

    assert plan_conversions(rows, norms, control_start=date(2014, 1, 1)) == []


def test_conversion_is_not_offered_without_headroom_under_the_water_ceiling(norms):
    """Без запаса под потолком компенсации перевод — обещание, которое срежут."""
    from aios.policy import plan_conversions

    rows = _watery("вода", 48, watercut=0.995)

    assert plan_conversions(rows, norms, control_start=date(2014, 1, 1),
                            headroom=False) == []


def test_after_conversion_the_well_leaves_the_producing_fund(norms):
    """Роль меняется: добыча закрыта, появилась запись закачки.

    Это третье условие задачи: планировщик и баланс воды обязаны увидеть
    скважину в новой роли, а не в старой.
    """
    from aios.deck import ProducerControl, ScheduleStep
    from aios.policy import CONVERT, Action, apply

    steps = [ScheduleStep(date(2014, month, 1),
                          (ProducerControl("вода", "OPEN", "LRAT", 30.0, 50.0),), ())
             for month in range(1, 7)]
    action = Action(date=date(2014, 3, 1), well="вода", kind=CONVERT,
                    reason="обводнённость 99,5 %", cost_m=norms.conversion_m)

    result = apply(steps, [action])

    before = result[0]
    assert before.producers[0].status == "OPEN" and not before.injectors

    # запись закачки ставится один раз: в Eclipse режим действует до следующего
    # изменения, и повторять её на каждом шаге не нужно
    after = result[2]
    assert [control.status for control in after.producers] == ["SHUT"]
    assert len(after.injectors) == 1, "скважина обязана появиться как нагнетательная"
    assert after.injectors[0].well == "вода"
    assert after.injectors[0].rate == 30.0, "закачивает столько, сколько поднимала"
    # а добыча остаётся закрытой и дальше: дек переоткрывает её на каждом шаге
    assert all(control.status == "SHUT"
               for step in result[2:] for control in step.producers)
