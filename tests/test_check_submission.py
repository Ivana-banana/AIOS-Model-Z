"""Предполётная проверка сдаваемого `wells_schedule.inc`.

Проверяется файл, а не структура в памяти: он мог быть собран прежней версией
кода, скопирован между машинами или обрезан. Расчёт у организаторов один
(`CLAUDE.md`), поэтому цена пропущенного дефекта — весь результат.

Тексты собираются нашим же генератором (`aios.schedule.render`): так проверка
заодно ловит расхождение генератора с парсером — если мы перестанем читать
собственный файл, тНавигатор тем более прочтёт его иначе.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from aios.deck import InjectorControl, ProducerControl, ScheduleStep, parse_schedule_text
from aios.schedule import render

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_submission import check, events  # noqa: E402

CONTROL_START = date(2007, 1, 1)


def producer(well: str, rate: float, status: str = "OPEN") -> ProducerControl:
    return ProducerControl(well, status, "LRAT", rate, 50.0)


def injector(well: str, rate: float, status: str = "OPEN") -> InjectorControl:
    return InjectorControl(well, "WATER", status, "RATE", rate, 250.0)


def steps(*months: tuple[date, tuple, tuple]) -> list[ScheduleStep]:
    return [ScheduleStep(day, producers, injectors) for day, producers, injectors in months]


DECK = render(steps(
    (date(2007, 1, 1), (producer("1", 100.0), producer("2", 50.0)), (injector("3", 200.0),)),
    (date(2007, 2, 1), (producer("1", 90.0),), ()),
    (date(2007, 3, 1), (producer("1", 80.0),), ()),
))


def test_a_schedule_from_our_own_generator_passes(tmp_path):
    """Опорный случай: то, что мы генерируем, обязано проходить проверку целиком."""
    failures, notes = check(DECK, DECK, CONTROL_START)

    assert failures == []
    assert not [note for note in notes if "горизонт короче" in note]


def test_a_rate_above_the_methodology_limit_is_refused():
    text = render(steps((date(2007, 1, 1), (producer("1", 501.0),), ())))

    failures, _ = check(text, DECK, CONTROL_START)

    assert any("запрещено методикой" in failure for failure in failures)


def test_a_well_missing_from_the_deck_is_refused():
    """Скважина, которой в деке нет, — это другой фонд, а не другой режим."""
    text = render(steps((date(2007, 1, 1), (producer("нет-в-деке", 100.0),), ())))

    failures, _ = check(text, DECK, CONTROL_START)

    assert any("скважины нет в деке" in failure for failure in failures)


def test_an_empty_file_is_refused_not_silently_accepted():
    failures, _ = check("-- только комментарий\n", DECK, CONTROL_START)

    assert any("нет ни одного шага" in failure for failure in failures)


def test_history_before_the_control_start_is_not_judged_by_our_rules():
    """История задана организаторами: браковать их дек по нашим правилам нельзя."""
    text = render(steps(
        (date(1995, 6, 1), (producer("1", 900.0),), ()),   # выше предела, но это история
        (date(2007, 1, 1), (producer("1", 100.0),), ()),
    ))

    failures, _ = check(text, DECK, CONTROL_START)

    assert failures == []


def test_a_short_horizon_is_a_warning_with_the_number_of_months():
    """Расписание, кончающееся раньше дека, — это недосчитанные годы выручки."""
    text = render(steps((date(2007, 1, 1), (producer("1", 100.0),), ())))

    _, notes = check(text, DECK, CONTROL_START)

    assert any("горизонт короче дека на 2 мес." in note for note in notes)


def test_wells_that_never_appear_in_the_file_are_named():
    """Молчание про скважину — не остановка: по семантике Eclipse режим сохраняется."""
    text = render(steps((date(2007, 1, 1), (producer("1", 100.0),), ())))

    _, notes = check(text, DECK, CONTROL_START)

    assert any("без единого режима в файле: 2" in note for note in notes)


def test_a_file_without_completions_says_so_instead_of_blaming_every_well():
    """В форме «хвост» вскрытий нет вовсе — это не 109 скважин без вскрытия."""
    _, notes = check(DECK, DECK, CONTROL_START)

    assert any("вскрытий (COMPDAT) в файле нет" in note for note in notes)


def test_controls_we_cannot_parse_are_reported_not_ignored():
    """Неразобранное ключевое слово означает, что часть режима мы не видели."""
    text = DECK + "\nWELTARG\n '1' LRAT 200 /\n/\n"

    _, notes = check(text, DECK, CONTROL_START)

    assert any("WELTARG" in note for note in notes)


# --- события расписания -------------------------------------------------------
#
# Деньги считает эталонный расчётчик по выгрузке, а не расписание. Здесь считается
# другое: насколько подвижное расписание мы отправляем — остановка и пуск это два
# отдельных события методики, и они не должны сливаться в одно.

def test_a_stop_and_a_start_are_counted_separately():
    parsed = parse_schedule_text(render(steps(
        (date(2007, 1, 1), (producer("1", 100.0),), ()),
        (date(2007, 2, 1), (producer("1", 100.0, status="SHUT"),), ()),
        (date(2007, 3, 1), (producer("1", 100.0),), ()),
    )))

    assert events(parsed.steps)["остановок"] == 1
    assert events(parsed.steps)["пусков"] == 1


def test_a_conversion_to_injection_is_counted():
    parsed = parse_schedule_text(render(steps(
        (date(2007, 1, 1), (producer("1", 100.0),), ()),
        (date(2007, 2, 1), (), (injector("1", 200.0),)),
    )))

    assert events(parsed.steps)["переводов под закачку"] == 1


def test_a_well_that_never_changes_produces_no_events():
    parsed = parse_schedule_text(render(steps(
        (date(2007, 1, 1), (producer("1", 100.0),), ()),
        (date(2007, 2, 1), (producer("1", 90.0),), ()),
    )))

    assert events(parsed.steps) == {"остановок": 0, "пусков": 0, "переводов под закачку": 0}


def test_a_conversion_in_one_step_is_not_also_a_stop_and_a_start():
    """Перевод пишется двумя строками одного шага — событие при этом одно.

    По записям режима, а не по состоянию на конец шага, историческое расписание
    Модели Z давало 30 остановок, 30 пусков и 30 переводов — одно и то же
    тридцатикратно (найдено прогоном 20.08).
    """
    parsed = parse_schedule_text(render(steps(
        (date(2007, 1, 1), (producer("1", 100.0),), ()),
        (date(2007, 2, 1), (producer("1", 100.0, status="SHUT"),), (injector("1", 200.0),)),
    )))

    assert events(parsed.steps) == {"остановок": 0, "пусков": 0, "переводов под закачку": 1}


def test_a_conversion_on_the_very_first_step_is_counted():
    """Перевод под закачку на первом же шаге файла — тоже перевод.

    Заявка с 08.09 сдаётся хвостом за период управления, поэтому «состояния до»
    в файле нет вовсе: оно осталось в истории организаторов. Признак перевода
    при этом виден и без истории — одна и та же скважина закрыта на добычу и
    открыта на закачку в одном шаге. В сдаваемом
    `runs/matched-0995/submission/wells_schedule.inc` так переводится скважина
    102 на 2007-01-01: эталонный расчётчик её перевод считает
    (`conversionTransitions`, 5,0 млн руб.), а счётчик расписания терял, потому
    что на первом шаге у него нет предыдущего состояния.
    """
    parsed = parse_schedule_text(render(steps(
        (date(2007, 1, 1), (producer("1", 100.0, status="SHUT"),), (injector("1", 200.0),)),
        (date(2007, 2, 1), (), (injector("1", 200.0),)),
    )))

    assert events(parsed.steps) == {"остановок": 0, "пусков": 0, "переводов под закачку": 1}


def test_a_control_before_the_first_dates_is_checked_too_not_skipped():
    """Режим до первого DATES обязан проверяться наравне с остальными.

    `aios.deck` кладёт такое ключевое слово в преамбулу, а не в шаг: даты у него
    ещё нет. Проверка смотрела только `steps`, поэтому дебит 900 м³/сут,
    записанный до первого DATES, проходил предполётную проверку с нулём отказов —
    и файл объявлялся пригодным к отправке. Найдено прогоном 20.08.

    Случай не выдуманный: сам смысл этой проверки в том, что файл мог быть
    собран прежней версией кода, скопирован между машинами или отредактирован
    руками (докстрока `scripts/check_submission.py`). Наш генератор преамбулу
    режимами не наполняет — а проверяем мы не его, а файл.
    """
    text = (
        "WCONPROD\n 'W1' 'OPEN' 'LRAT' 3* 900 1* 50 /\n/\n\n"
        + render(steps((date(2007, 1, 1), (producer("1", 100.0),), ())))
    )

    failures, _ = check(text, DECK, CONTROL_START)

    assert any("запрещено методикой" in failure for failure in failures), (
        "нарушение до первого DATES прошло проверку"
    )
    assert any("до первого DATES" in failure for failure in failures)


def test_wells_controlled_but_not_declared_in_the_file_are_reported():
    """Скважина получает режим, но `WELSPECS` для неё в файле нет.

    Разбор семантики SCHEDULE (`docs/manus/schedule-semantics.md`, п. 4 чек-листа)
    требует объявления `WELSPECS` до первого `COMPDAT`/`WCONPROD`/`WCONINJE` для
    каждой скважины, на которую файл ссылается. Требование действует ровно в том
    прочтении, где наш файл **подменяет** всю секцию SCHEDULE: наследовать
    объявление там неоткуда. Прочтение не выбрано (В-03), поэтому это
    предупреждение, а не отказ, — но молчать о нём нельзя: parser-ошибка вылезет
    у организаторов на единственном расчёте.
    """
    text = render(steps((date(2007, 1, 1), (producer("1", 100.0),), (injector("3", 200.0),))))

    _, notes = check(text, DECK, CONTROL_START)

    assert any("WELSPECS" in note for note in notes), "необъявленные скважины прошли молча"


def test_a_file_declaring_every_controlled_well_keeps_quiet():
    text = (
        "WELSPECS\n 'W1' 'GROUP' 1 1 1* 'OIL' /\n 'W3' 'GROUP' 2 2 1* 'OIL' /\n/\n\n"
        + render(steps((date(2007, 1, 1), (producer("W1", 100.0),), (injector("W3", 200.0),))))
    )

    _, notes = check(text, DECK, CONTROL_START)

    assert not [note for note in notes if "WELSPECS" in note]


# --- Б5: не-ASCII в сдаваемом файле ------------------------------------------
# Заголовок генератора переведён на ASCII (Р-179), но файл, собранный до этой
# правки, по-прежнему несёт две кириллические строки. Сторожа на это не было:
# решение записали закрытым, а проверки не завели.


def test_non_ascii_bytes_are_reported_as_a_note_with_line_numbers():
    """Кириллица в файле — заметка с номерами строк, а не отказ.

    Отказ объявил бы непригодным файл, которым посчитано заявляемое число.
    Заметка называет, где именно байты и чем это грозит: под чужой локалью они
    превратятся в мусор ровно в тех строках, куда проверяющий смотрит первым
    делом.
    """
    cyrillic = "-- wells_schedule.inc" + chr(10) + "-- Управляющие последовательности"
    text = DECK.replace("-- wells_schedule.inc", cyrillic)

    failures, notes = check(text, DECK, CONTROL_START)

    about = [note for note in notes if "ASCII" in note]
    assert about, notes
    assert "2" in about[0], "заметка не называет строку"
    assert not [failure for failure in failures if "ASCII" in failure], (
        "не-ASCII объявлен отказом — так заявляемый файл станет непригодным"
    )


def test_a_pure_ascii_schedule_gets_no_such_note():
    """Проверка не должна жаловаться на файл, с которым всё в порядке."""
    _, notes = check(DECK, DECK, CONTROL_START)

    assert DECK.isascii(), "фикстура сама не ASCII — тест проверяет не то"
    assert not [note for note in notes if "ASCII" in note]


def test_the_shipped_schedule_is_checked_for_it_too():
    """Сдаваемый файл под трекингом — значит, его состояние проверяется здесь.

    Не «должно быть ASCII»: файл собран до Р-179 и две кириллические строки в нём
    есть. Тест стережёт, что проверка их **видит**, — превращение известного
    дефекта в замолчанный стоило бы дороже самого дефекта.
    """
    import pytest

    shipped = Path(__file__).resolve().parent.parent / "runs" / "matched-0995" / "wells_schedule.inc"
    if not shipped.is_file():
        pytest.skip("сдаваемого расписания нет на этой машине")

    text = shipped.read_text(encoding="utf-8")
    lines = [number for number, line in enumerate(text.splitlines(), 1)
             if not line.isascii()]

    if not lines:
        return  # файл перегенерирован — заметке взяться неоткуда, и это хорошо

    _, notes = check(text, text, CONTROL_START)
    about = [note for note in notes if "ASCII" in note]

    assert about, "в сдаваемом файле есть не-ASCII, а проверка молчит"
    assert str(lines[0]) in about[0]
