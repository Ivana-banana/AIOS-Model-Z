"""Генерация и валидация `wells_schedule.inc`.

Файл уходит организаторам и считается ровно один раз, поэтому валидатор обязан
ловить нарушения до отправки, а не после.
"""

from __future__ import annotations

from datetime import date

import pytest
from conftest import MODEL_Z_SCHEDULE

from aios import deck, schedule
from aios.deck import InjectorControl, ProducerControl, ScheduleStep


def producer(well="1", rate=45.0, status="OPEN", mode="LRAT", bhp=50.0):
    return ProducerControl(well=well, status=status, mode=mode, liquid_rate=rate, bhp=bhp)


def injector(well="2", rate=30.0, status="OPEN", mode="RATE", bhp=300.0, fluid="WATER"):
    return InjectorControl(well=well, fluid=fluid, status=status, mode=mode, rate=rate, bhp=bhp)


def step(day=1, month=1, year=2007, producers=(), injectors=()):
    return ScheduleStep(date(year, month, day), tuple(producers), tuple(injectors))


def test_valid_schedule_has_no_problems():
    steps = [
        step(producers=[producer()], injectors=[injector()]),
        step(month=2, producers=[producer(rate=60.0)]),
    ]
    assert schedule.validate(steps) == []


def test_render_is_parsed_back_identically():
    """То, что мы записали, симулятор прочитает так же, как читаем мы."""
    steps = [
        step(producers=[producer(), producer(well="3", rate=12.5)], injectors=[injector()]),
        step(month=2, producers=[producer(status="SHUT", rate=None)]),
    ]
    parsed = deck.parse_schedule_text(schedule.render(steps))

    assert [s.date for s in parsed.steps] == [date(2007, 1, 1), date(2007, 2, 1)]
    assert parsed.steps[0].producers == steps[0].producers
    assert parsed.steps[0].injectors == steps[0].injectors
    assert parsed.steps[1].producers[0].status == "SHUT"


def test_liquid_rate_above_500_is_rejected():
    problems = schedule.validate([step(producers=[producer(rate=501.0)])])
    assert len(problems) == 1
    assert "500" in problems[0].message


def test_open_producer_with_zero_rate_is_rejected():
    """Ноль на открытой скважине — это не остановка, а некорректная запись."""
    problems = schedule.validate([step(producers=[producer(rate=0.0)])])
    assert len(problems) == 1
    assert "нулевой дебит" in problems[0].message


def test_injection_pressure_above_300_atm_is_rejected():
    """Организаторы на разборе: выше 300 атм в жизни раскрываются трещины.

    Модель такой режим воспроизведёт и покажет красивую добычу — но решение
    оценивается ещё и по физической реализуемости.
    """
    problems = schedule.validate([step(injectors=[injector(bhp=350.0)])])
    assert len(problems) == 1
    assert "раскрытие трещин" in problems[0].message


def test_injection_pressure_at_the_limit_is_allowed():
    """300 атм стоит во всей истории дека — это рабочее значение, а не нарушение."""
    assert schedule.validate([step(injectors=[injector(bhp=300.0)])]) == []


def test_shut_producer_may_have_no_rate():
    assert schedule.validate([step(producers=[producer(status="SHUT", rate=None)])]) == []


def test_dates_must_increase():
    steps = [step(month=2), step(month=1)]
    problems = schedule.validate(steps)
    assert any("не возрастает" in problem.message for problem in problems)


def test_well_cannot_be_open_for_production_and_injection_at_once():
    problems = schedule.validate([step(producers=[producer(well="7")], injectors=[injector(well="7")])])
    assert any("одновременно открыта" in problem.message for problem in problems)


def test_conversion_to_injection_is_written_as_shut_plus_open():
    """Так перевод под закачку записан в деке организаторов, и это законно.

    Скважина 102 на 01.01.2007: `WCONPROD` со `SHUT` закрывает добычу,
    `WCONINJE` с `OPEN` открывает закачку — обе записи на одном шаге. Запрет на
    такую пару сделал бы невозможной операцию, без которой трек 2 не решается.
    """
    steps = [
        step(
            producers=[producer(well="102", status="SHUT", rate=None)],
            injectors=[injector(well="102", rate=90.0)],
        )
    ]
    assert schedule.validate(steps) == []


def test_duplicate_records_for_the_same_role_are_still_rejected():
    problems = schedule.validate([step(producers=[producer(well="7"), producer(well="7")])])
    assert any("дублируется" in problem.message for problem in problems)


def test_unknown_well_is_rejected_when_the_fund_is_given():
    problems = schedule.validate([step(producers=[producer(well="999")])], known_wells=frozenset({"1"}))
    assert any("нет в деке" in problem.message for problem in problems)


def test_invalid_schedule_is_never_written(tmp_path):
    target = tmp_path / "wells_schedule.inc"
    with pytest.raises(schedule.ScheduleError):
        schedule.write(target, [step(producers=[producer(rate=900.0)])])
    assert not target.exists()


def test_written_file_is_readable_by_the_parser(tmp_path):
    target = tmp_path / "wells_schedule.inc"
    steps = [step(producers=[producer()], injectors=[injector()])]
    schedule.write(target, steps, known_wells=frozenset({"1", "2"}))

    text = target.read_text(encoding="utf-8")
    assert "DATES" in text and "WCONPROD" in text and "WCONINJE" in text
    assert deck.parse_schedule(target).steps[0].producers == steps[0].producers


@pytest.mark.skipif(not MODEL_Z_SCHEDULE.is_file(), reason="дек Модели Z не распакован")
def test_the_organisers_own_schedule_passes_our_validator():
    """Историческое управление организаторов не нарушает ни одного нашего правила.

    Проверка в обе стороны: если наш валидатор забракует их же дек, значит,
    неправ валидатор, а не дек. Так и вскрылось, что перевод под закачку
    записывается парой SHUT + OPEN на одном шаге.
    """
    steps = [
        step for step in deck.parse_schedule(MODEL_Z_SCHEDULE).steps
        if step.date >= date(2007, 1, 1)
    ]
    assert len(steps) == 225
    assert schedule.validate(steps) == []
