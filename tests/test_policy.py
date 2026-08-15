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
