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


def test_case_limits_reads_both_limits_from_the_case():
    """Помощник собирает то, что `validate` ждёт, и ничего не выдумывает."""
    from aios.schedule import case_limits

    assert case_limits() == {}, "без нормативов и без ключа выдумывать нечего"
    assert case_limits([{"max": 320.0}], None) == {"max_liquid_rate": 320.0}
    assert case_limits(None, 250.0) == {"max_injection_bhp": 250.0}

    both = case_limits([{"max": 320.0}], 250.0)
    assert both == {"max_liquid_rate": 320.0, "max_injection_bhp": 250.0}


def test_case_limits_still_caps_the_liquid_rate_at_the_vendor_ceiling():
    """Таблица кейса с насосом на 700 не поднимает предел выше 500.

    У эталонного расчётчика 500 зашито литералом и роняет расчёт целиком, а не
    считает иначе, — поэтому потолок действует при любой таблице.
    """
    from aios.schedule import case_limits

    assert case_limits([{"max": 700.0}])["max_liquid_rate"] == 500.0


def test_the_generated_header_is_pure_ascii():
    """Заголовок сдаваемого файла — только ASCII, и это закреплено (Р-179).

    Комментарий безвреден для парсера, но файл читает чужой симулятор в чужой
    локали: под cp1251 кириллица превращается в мусор в первых строках
    результата — там, где проверяющий смотрит первым делом. И наш собственный
    входной контроль не-ASCII вне комментария считает отказом: держать у себя то,
    что мы помечаем у других, — двойной стандарт.

    Сторожа на это не было: решение записали закрытым, а проверка отсутствовала
    (Б5 выжимки `STOP-BEFORE-SUBMIT.md`).
    """
    from aios.schedule import HEADER

    assert HEADER.isascii(), (
        "в заголовке снова не-ASCII: " + repr([line for line in HEADER.splitlines()
                                               if not line.isascii()])
    )


def test_a_rendered_schedule_carries_no_non_ascii_of_its_own():
    """Не только заголовок: весь наш сгенерированный текст обязан быть ASCII.

    Заголовок — единственное место, где кириллица уже заводилась, но проверять
    надо результат целиком: следующая строка на русском появится не в HEADER.
    """
    from datetime import date as date_type

    from aios.deck import InjectorControl, ProducerControl, ScheduleStep
    from aios.schedule import render

    text = render([ScheduleStep(
        date_type(2014, 1, 1),
        (ProducerControl("1", "OPEN", "LRAT", 100.0, 50.0),),
        (InjectorControl("2", "WATER", "OPEN", "RATE", 40.0, 250.0),),
    )])

    assert text.isascii(), [line for line in text.splitlines() if not line.isascii()]


# ── что нашёл мутационный прогон 05.09 (Р-221) ───────────────────────────────


def test_a_repeated_date_is_a_problem_not_a_step():
    """Два шага с одной датой — нарушение: календарь обязан возрастать строго.

    Мутация `<=` на `<` переживала сюиту: проверялось только убывание. Дубликат
    `DATES` — обычная ошибка сборки дека, и симулятор примет её молча, посчитав
    два месяца за один.
    """
    same = date(2014, 3, 1)
    steps = [
        ScheduleStep(same, (ProducerControl("1", "OPEN", "LRAT", 30.0, 50.0),), ()),
        ScheduleStep(same, (ProducerControl("1", "OPEN", "LRAT", 35.0, 50.0),), ()),
    ]

    problems = schedule.validate(steps)

    assert problems, "повтор даты не замечен — два месяца слились в один"
    assert any("дат" in one.message.lower() or "поряд" in one.message.lower()
               for one in problems), [one.message for one in problems]


def test_exactly_the_liquid_limit_is_allowed():
    """Ровно 500 м3/сут законно: верхний типоразмер ЭЦН закрывает 320-500.

    Мутация `>` на `>=` переживала сюиту. Методика запрещает **превышение**
    500, а не саму границу; отбрасывая её, мы теряли бы верхний интервал ЭЦН
    целиком — и делали бы это молча, при fail-closed отбраковывая кандидата.
    """
    limit = schedule.MAX_LIQUID_RATE_M3_PER_DAY
    at_limit = [ScheduleStep(date(2014, 1, 1),
                             (ProducerControl("1", "OPEN", "LRAT", limit, 50.0),), ())]
    above = [ScheduleStep(date(2014, 1, 1),
                          (ProducerControl("1", "OPEN", "LRAT", limit * 1.0001, 50.0),), ())]

    assert schedule.validate(at_limit) == [], schedule.validate(at_limit)
    assert schedule.validate(above), "превышение предела пропущено"


def test_an_open_injector_at_zero_or_below_is_refused():
    """Открытая нагнетательная не закачивает ноль, и приёмистость не бывает отрицательной.

    Четыре мутации строк 295-304 переживали сюиту: проверялись только
    добывающие. Нагнетательная с нулём — это остановка, записанная как работа;
    отрицательная приёмистость — отбор, записанный как закачка.
    """
    def one(rate, *, status="OPEN", mode="RATE", bhp=300.0):
        return [ScheduleStep(date(2014, 1, 1), (),
                             (InjectorControl("9", "WATER", status, mode, rate, bhp),))]

    assert schedule.validate(one(0.0)), "открытая нагнетательная с нулём принята"
    assert schedule.validate(one(None)), "режим RATE без приёмистости принят"
    assert schedule.validate(one(-5.0, status="SHUT", mode="BHP")), "отрицательная приёмистость принята"
    assert schedule.validate(one(-1.0, bhp=-1.0, status="SHUT", mode="BHP")), "отрицательное давление принято"
    assert schedule.validate(one(400.0)) == [], schedule.validate(one(400.0))


def test_a_shut_producer_without_a_rate_is_written_as_an_explicit_zero():
    """Остановка пишется явным нулём, а открытая скважина без уставки — не пишется.

    Мутация `and` на `or` в `_rate_value` переживала сюиту. С `or` открытая
    скважина без уставки тоже получила бы «0.0», то есть остановку вместо
    отказа: расписание собралось бы, а скважина молча встала.
    """
    from aios.schedule import _rate_value

    assert _rate_value(None, "SHUT") == "0.0"
    assert _rate_value(None, "OPEN") != "0.0", (
        "открытая скважина без уставки записана нулём — это молчаливая остановка"
    )
    assert _rate_value(30.0, "OPEN").startswith("30")


def test_a_long_list_of_problems_says_how_many_were_hidden():
    """При длинном списке нарушений сообщение честно говорит, сколько не показано.

    Две мутации строки 166 переживали сюиту: текст ошибки никто не читал. Он —
    единственное, что видит собирающий расписание, и «... ещё N» отличает
    «пятьдесят нарушений» от «пятисот».
    """
    steps = [
        ScheduleStep(date(2014, month, 1),
                     (ProducerControl("1", "OPEN", "LRAT", 9999.0, 50.0),), ())
        for month in range(1, 13)
    ] + [
        ScheduleStep(date(2015, month, 1),
                     (ProducerControl(str(well), "OPEN", "LRAT", 9999.0, 50.0),), ())
        for month in range(1, 13) for well in range(1, 6)
    ]

    with pytest.raises(schedule.ScheduleError) as raised:
        schedule.assert_valid(steps)

    text = str(raised.value)
    assert "ещё" in text, text[:400]
    hidden = len(schedule.validate(steps)) - 50
    assert f"ещё {hidden}" in text, text[-200:]


def test_only_an_open_rate_injector_is_checked_for_zero():
    """Условие проверки нуля — «открыта И режим RATE», а не «или».

    Мутация `and` на `or` переживала прежний тест: он не подавал остановленную
    скважину с нулём. С `or` остановка, записанная нулём (а дек организаторов
    пишет её именно так), объявлялась бы нарушением — то есть законная запись
    отвергалась бы валидатором.
    """
    shut_at_zero = [ScheduleStep(date(2014, 1, 1), (),
                                 (InjectorControl("9", "WATER", "SHUT", "RATE", 0.0, 300.0),))]
    open_on_pressure = [ScheduleStep(date(2014, 1, 1), (),
                                     (InjectorControl("9", "WATER", "OPEN", "BHP", 0.0, 300.0),))]

    assert schedule.validate(shut_at_zero) == [], (
        "остановка, записанная нулём, объявлена нарушением: так пишет дек организаторов"
    )
    assert schedule.validate(open_on_pressure) == [], (
        "под BHP поле расхода — не уставка, и ноль в нём законен"
    )


def test_a_missing_rate_outside_rate_mode_is_not_compared_to_zero():
    """Пустая приёмистость вне режима RATE не сравнивается с нулём.

    Мутация `and` на `or` в строке 300 переживала сюиту: с ней `None < 0`
    роняет валидатор `TypeError` — то есть проверка расписания падает вместо
    того, чтобы выдать список нарушений.
    """
    steps = [ScheduleStep(date(2014, 1, 1), (),
                          (InjectorControl("9", "WATER", "SHUT", "BHP", None, 300.0),))]

    assert schedule.validate(steps) == [], schedule.validate(steps)


def test_zero_injection_pressure_is_refused():
    """Давление нагнетания ровно ноль — нарушение: граница включающая."""
    def one(bhp):
        return [ScheduleStep(date(2014, 1, 1), (),
                             (InjectorControl("9", "WATER", "OPEN", "RATE", 30.0, bhp),))]

    assert schedule.validate(one(0.0)), "нулевое давление нагнетания принято"
    assert schedule.validate(one(300.0)) == [], schedule.validate(one(300.0))


def test_exactly_fifty_problems_are_listed_without_a_tail():
    """Ровно пятьдесят нарушений показываются целиком, без «... ещё 0».

    Мутация `<=` на `<` в строке 166 переживала сюиту: список длиной ровно в
    порог не проверялся, и хвост печатал бы «ещё 0» — строку, которая говорит
    неправду о том, что что-то скрыто.
    """
    # один шаг с пятьюдесятью нарушающими скважинами
    steps = [ScheduleStep(date(2014, 1, 1),
                          tuple(ProducerControl(str(index), "OPEN", "LRAT", 9999.0, 50.0)
                                for index in range(50)), ())]
    assert len(schedule.validate(steps)) == 50, len(schedule.validate(steps))

    with pytest.raises(schedule.ScheduleError) as raised:
        schedule.assert_valid(steps)

    assert "ещё" not in str(raised.value), (
        "при ровно пятидесяти нарушениях напечатан хвост «... ещё 0»"
    )

