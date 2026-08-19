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


def test_invalid_schedule_stops_at_assert_valid():
    """Проверка отделена от записи намеренно — см. Р-04 в `docs/decisions.md`.

    Записывает сдаваемый файл `aios.run.build_case`: он склеивает историю дека с
    нашими шагами, а не рендерит одни шаги, поэтому единой функции «проверить и
    записать» здесь быть не может. Гарантия живёт в двух местах: `assert_valid`
    отвергает брак, а `build_case` вызывает её до того, как что-либо напишет.
    """
    with pytest.raises(schedule.ScheduleError):
        schedule.assert_valid([step(producers=[producer(rate=900.0)])])


def test_rendered_text_is_readable_by_the_parser(tmp_path):
    """Круговая проверка: что отрисовали, то и разбираем обратно."""
    steps = [step(producers=[producer()], injectors=[injector()])]
    target = tmp_path / "wells_schedule.inc"
    target.write_text(schedule.render(steps), encoding="utf-8")

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


def test_shut_producer_gets_an_explicit_zero_rate():
    """Остановка без числа — файл, который симулятор не читает.

    OPM Flow отвергает `WCONPROD` с режимом LRAT и `1*` вместо дебита:
    «Trying to set CMODE to: LRAT - no value has been specified for this
    control», и весь прогон падает на разборе входа. Дек организаторов пишет
    остановку именно с нулём: `'42' 'SHUT' 'LRAT' 1* 1* 1* 0.0 1* 50 1* 1* /`.

    Поймано первым же прогоном сценария с остановками: 7 077 таких записей.
    """
    text = schedule.render([
        ScheduleStep(
            date=date(2007, 1, 1),
            producers=(ProducerControl(well="3", status="SHUT", mode="LRAT",
                                       liquid_rate=None, bhp=50.0),),
            injectors=(),
        ),
    ])
    record = next(line for line in text.splitlines() if "'3'" in line)
    assert "1* 1* 1* 1*" not in record, f"дебит остановленной скважины не задан: {record}"
    assert " 0 " in record or " 0.0 " in record, record


def test_shut_injector_gets_an_explicit_zero_rate():
    """То же для нагнетательной: `WCONINJE` с RATE и `1*` падает так же."""
    text = schedule.render([
        ScheduleStep(
            date=date(2007, 1, 1),
            producers=(),
            injectors=(InjectorControl(well="6", fluid="WATER", status="SHUT", mode="RATE",
                                       rate=None, bhp=300.0),),
        ),
    ])
    record = next(line for line in text.splitlines() if "'6'" in line)
    rate_field = record.split("'RATE'", 1)[1].split()[0]
    assert rate_field != "1*", f"закачка остановленной скважины не задана: {record}"
    assert float(rate_field) == 0.0, record
