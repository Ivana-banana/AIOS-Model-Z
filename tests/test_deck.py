"""Разбор SCHEDULE-секции дека."""

from __future__ import annotations

import pathlib
from datetime import date

import pytest
from conftest import MODEL_Z_SCHEDULE

from aios import deck

ROOT = pathlib.Path(__file__).resolve().parent.parent

SAMPLE = """\
-- комментарий, который не должен попасть в разбор
RPTSCHED
 'WELLS=1' 'SUMMARY=1' /

DATES
 01 NOV 1994 /
/

WCONPROD
 '1' 'OPEN' 'LRAT' 1* 1* 1* 45.0 1* 50 1* 1* /
 '3' 'SHUT' 'LRAT' 1* 1* 1* 0.0 1* 50 1* 1* /
/

WCONINJE
 '2' 'WATER' 'OPEN' 'RATE' 30.0 1* 300 1* 1* /
/

DATES
 01 DEC 1994 /
/

WCONPROD
 '1' 'OPEN' 'LRAT' 1* 1* 1* 60.0 1* 50 1* 1* /
/
"""


def test_an_unquoted_well_name_is_not_taken_for_a_keyword():
    """Имя скважины без кавычек закрывало только что открытый блок — молча.

    Шаблон ключевого слова `[A-Z][A-Z0-9]{1,11}` совпадает и с именем скважины,
    поэтому `WCONPROD` с незакавыченным `P1` разбирался в **пустой** блок: ноль
    добывающих, ни одной ошибки, `unsupported_controls` пуст. На деке
    валидационного кейса, где имена не закавычены, скважина исчезла бы из фонда
    вместе со своей экономикой (ревизия 03.09, 2.3 #1).
    """
    text = (
        "DATES\n 1 JAN 2009 /\n/\n"
        "WELSPECS\n P1 G1 10 10 1* OIL /\n/\n"
        "WCONPROD\n P1 OPEN LRAT 1* 1* 1* 20 1* 50 /\n/\n"
    )

    parsed = deck.parse_schedule_text(text)

    assert [control.well for control in parsed.steps[0].producers] == ["P1"]
    assert parsed.steps[0].producers[0].liquid_rate == pytest.approx(20.0)
    # и то же самое с кавычками разбирается так же
    quoted = deck.parse_schedule_text(
        text.replace(" P1 ", " 'P1' ").replace(" G1 ", " 'G1' "))
    assert [control.well for control in quoted.steps[0].producers] == ["P1"]


def test_a_well_name_in_the_middle_of_a_record_is_not_a_keyword():
    """Запись из нескольких строк не разрывается именем в её середине."""
    text = (
        "DATES\n 1 JAN 2009 /\n/\n"
        "WCONPROD\n"
        " P1 OPEN LRAT 1* 1* 1* 20 1* 50 /\n"
        " P2 OPEN LRAT 1* 1* 1* 30 1* 50 /\n"
        "/\n"
    )

    parsed = deck.parse_schedule_text(text)

    assert [control.well for control in parsed.steps[0].producers] == ["P1", "P2"]


def test_defaults_and_repeats_are_expanded():
    record = deck._expand_defaults(["'A'", "1*", "3*", "2*50", "7"])
    assert record == ("A", None, None, None, None, "50", "50", "7")


def test_comments_are_stripped():
    keywords = deck.parse_keywords("WCONPROD -- это комментарий\n 'A' 'OPEN' /\n/\n")
    assert [keyword.name for keyword in keywords] == ["WCONPROD"]
    assert keywords[0].records == (("A", "OPEN"),)


def test_dates_round_trip():
    parsed = deck.parse_date_record(("01", "NOV", "1994"))
    assert parsed == date(1994, 11, 1)
    assert deck.format_date_record(parsed) == "01 NOV 1994"


def test_schedule_is_split_into_steps():
    schedule = deck.parse_schedule_text(SAMPLE)

    assert [step.date for step in schedule.steps] == [date(1994, 11, 1), date(1994, 12, 1)]
    assert [keyword.name for keyword in schedule.preamble] == ["RPTSCHED"]

    first = schedule.steps[0]
    assert first.producers[0] == deck.ProducerControl("1", "OPEN", "LRAT", 45.0, 50.0)
    assert first.producers[1].status == "SHUT"
    assert first.injectors[0] == deck.InjectorControl("2", "WATER", "OPEN", "RATE", 30.0, 300.0)

    assert schedule.producer_wells == ("1", "3")
    assert schedule.injector_wells == ("2",)
    assert schedule.wells == ("1", "2", "3")


def test_controls_are_read_by_position_not_by_guessing():
    """LRAT — седьмое поле WCONPROD, BHP — девятое; путаница здесь стоит дисквалификации."""
    keywords = deck.parse_keywords(
        "WCONPROD\n '9' 'OPEN' 'LRAT' 1* 1* 1* 123.5 1* 77 1* 1* /\n/\n"
    )
    record = keywords[0].records[0]
    assert record[deck._WCONPROD_LRAT] == "123.5"
    assert record[deck._WCONPROD_BHP] == "77"


@pytest.mark.skipif(not MODEL_Z_SCHEDULE.is_file(), reason="дек Модели Z не распакован")
def test_real_model_z_schedule():
    """Разбор реального дека: 371 месячный шаг, 103 скважины, 11.1994 → 09.2025."""
    schedule = deck.parse_schedule(MODEL_Z_SCHEDULE)

    assert len(schedule.steps) == 371
    assert schedule.steps[0].date == date(1994, 11, 1)
    assert schedule.steps[-1].date == date(2025, 9, 1)
    assert len(schedule.wells) == 103
    assert len(schedule.producer_wells) == 92
    assert len(schedule.injector_wells) == 41

    # даты строго возрастают и идут месячным шагом
    dates = [step.date for step in schedule.steps]
    assert dates == sorted(dates) and len(set(dates)) == len(dates)

    # история не нарушает ограничение методики по дебиту жидкости
    rates = [c.liquid_rate for step in schedule.steps for c in step.producers
             if c.liquid_rate is not None]
    assert max(rates) <= 500.0

    # 30 скважин за историю переведены из добычи под закачку
    converted = set(schedule.producer_wells) & set(schedule.injector_wells)
    assert len(converted) == 30

    # OPM-поставка: вскрытия заданы стандартным COMPDAT, тНавигаторовских
    # WELLTRACK/COMPDATMD в ней нет — на этом держится совместимость с OPM Flow
    text = MODEL_Z_SCHEDULE.read_text(encoding="utf-8", errors="replace")
    assert "COMPDAT" in text
    assert "WELLTRACK" not in text and "COMPDATMD" not in text


# ── управление, которого мы не разбираем ─────────────────────────────────────
#
# Р-03: поддерживаются только нужные ключевые слова, остальное пропускается.
# Для COMPDATMD и WPIMULT это верно — они не режимы. Но WCONHIST, WELTARG и WEFAC
# управлением являются, и в исторических деках WCONHIST встречается сплошь.
# На таком деке разбор дал бы шаги без единого управления добычей, а базовый
# сценарий, DoE и маржа считались бы от пустоты — молча.

WITH_WCONHIST = """\
WELSPECS
 '1' 'GROUP' 23 17 1* 'OIL' /
/

DATES
 01 JAN 2007 /
/

WCONHIST
 '1' 'OPEN' 'ORAT' 30.0 5.0 0.0 /
/
"""

WITH_WPIMULT = """\
DATES
 01 JAN 2007 /
/

WPIMULT
 '1' 1.5 /
/

WCONPROD
 '1' 'OPEN' 'LRAT' 1* 1* 1* 30.0 1* 50 1* 1* /
/
"""


def test_unsupported_control_keywords_are_named_not_dropped():
    """Молчаливый пропуск WCONHIST даёт базовый сценарий без добычи вовсе."""
    schedule = deck.parse_schedule_text(WITH_WCONHIST)

    assert schedule.steps[0].producers == (), "мы его действительно не разбираем"
    assert "WCONHIST" in schedule.unsupported_controls, "и обязаны об этом сказать"


def test_non_control_keywords_stay_silent():
    """WPIMULT и вскрытия режимами не являются — шуметь про них незачем."""
    schedule = deck.parse_schedule_text(WITH_WPIMULT)

    assert schedule.unsupported_controls == ()
    assert len(schedule.steps[0].producers) == 1


@pytest.mark.skipif(not MODEL_Z_SCHEDULE.is_file(), reason="дек Модели Z не распакован")
def test_model_z_schedule_has_nothing_unsupported():
    """Контроль на настоящем деке: сегодня предупреждать не о чем."""
    schedule = deck.parse_schedule(MODEL_Z_SCHEDULE)
    assert schedule.unsupported_controls == ()


WITH_INCLUDE = """\
WELSPECS
 '1' 'G' 1 1 1* 'OIL' /
/

DATES
 1 'JAN' 2007 /
/

WCONPROD
 '1' 'OPEN' 'LRAT' 1* 1* 1* 30.0 1* 50 1* 1* /
/

INCLUDE
 'part2.inc' /
"""


def test_an_include_inside_the_schedule_is_named():
    """Разбиение расписания по файлам — обычное дело в больших деках.

    Наш разбор включения не раскрывает, и до 18.08 делал это молча: шаги и
    скважины из включённого файла просто исчезали. Базовый сценарий покрывал бы
    часть горизонта, DoE масштабировал бы часть фонда, а собранное по таким шагам
    расписание уехало бы организаторам неполным.
    """
    schedule = deck.parse_schedule_text(WITH_INCLUDE)

    assert len(schedule.steps) == 1, "включение мы действительно не раскрываем"
    assert "INCLUDE" in schedule.unsupported_controls, "и обязаны об этом сказать"


def test_control_keyword_sets_of_deck_and_run_do_not_share_a_name():
    """Два разных набора под одним именем — ловушка, а не стиль.

    В `aios/run.py` имя `CONTROL_KEYWORDS` занято под смысл «что наш генератор
    перепишет сам, переносить дословно не надо», и туда входит `DATES`. В
    `aios/deck.py` набор про другое — «что назначает скважине режим», и `DATES`
    там быть не должно. Модули импортируют друг из друга; совпади имена, и
    добавление его в список импорта молча подменило бы локальный набор, а
    `carried_keywords` начал бы переносить блоки дат дословно и продублировал
    шаги расписания.

    Проверяется само свойство, а не текущие имена: наборы обязаны либо
    называться по-разному, либо совпадать по составу.
    """
    from aios import run as run_module

    deck_names = {name for name in dir(deck)
                  if name.endswith("CONTROL_KEYWORDS") and not name.startswith("_")}
    run_names = {name for name in dir(run_module)
                 if name.endswith("CONTROL_KEYWORDS") and not name.startswith("_")}

    for shared in deck_names & run_names:
        assert getattr(deck, shared) == getattr(run_module, shared), (
            f"«{shared}» значит разное в aios/deck.py и aios/run.py: "
            f"{sorted(getattr(deck, shared))} против {sorted(getattr(run_module, shared))}"
        )

    assert "DATES" not in deck.WELL_CONTROL_KEYWORDS, (
        "DATES не назначает режим скважине"
    )
    assert "DATES" in run_module.CONTROL_KEYWORDS, (
        "carried_keywords обязан считать DATES своим и не переносить его дословно"
    )


def test_a_double_dash_inside_quotes_is_not_a_comment():
    """Каталог ловушек Eclipse, ECL-003: `--` в кавычках комментарием не является.

    Дефект был тихим: путь `'model--z/sch.inc'` резался до `'model`, токен
    оставался синтаксически годным, и INCLUDE начинал указывать в никуда без
    единого сообщения.
    """
    found = deck.parse_keywords("INCLUDE\n  'model--z/sch.inc' /\n")

    assert found[0].records == (("model--z/sch.inc",),)


def test_a_comment_outside_quotes_is_still_removed():
    found = deck.parse_keywords("DATES\n  1 'JAN' 2007 /  -- начало горизонта\n/\n")

    assert found[0].records == (("1", "JAN", "2007"),)


# --- ключевое слово — только первый токен строки -----------------------------
# Ревизия 03.09 (п. 2.3 #1): `_KEYWORD_RE` считала ключевым словом любой заглавный
# токен, и запись без кавычек разбиралась как цепочка пустых ключевых слов, а
# скважина исчезала из фонда молча.


def test_data_on_the_keyword_line_is_not_mistaken_for_keywords():
    """`WCONPROD 'P1' OPEN LRAT …` — одна запись, а не четыре ключевых слова.

    До правки `P1`, `OPEN` и `LRAT` становились именами ключевых слов: они
    заглавные и подходят под шаблон. Скважина при этом пропадала — `producers`
    оставался пустым **без единой жалобы**.
    """
    text = """
DATES
 1 JAN 2014 /
/
WCONPROD 'P1' OPEN LRAT 4* 100 50 /
/
"""

    assert [one.name for one in deck.parse_keywords(text)] == ["DATES", "WCONPROD"]

    schedule = deck.parse_schedule_text(text)
    assert [one.well for step in schedule.steps for one in step.producers] == ["P1"]


def test_an_unquoted_well_name_on_the_keyword_line_is_read_too():
    """Кавычки перестали быть условием разбора, когда запись на строке слова."""
    text = """
DATES
 1 JAN 2014 /
/
WCONPROD P1 OPEN LRAT 4* 100 50 /
/
"""

    schedule = deck.parse_schedule_text(text)
    assert [one.well for step in schedule.steps for one in step.producers] == ["P1"]


def test_an_empty_control_block_is_refused_instead_of_losing_the_well():
    """Блок режима без записей — не «нечего назначать», а непрочитанная запись.

    Вторая сетка под `BLOCK_KEYWORDS` (Р-177), а не замена ей. Список закрывает
    известную причину — имя скважины, принятое за ключевое слово. Эта проверка
    ловит **последствие** любой причины, включая ключевое слово чужого дека,
    которого в списке ещё нет: такое разбирается как данные и уходит в записи
    соседа, а блок режима остаётся пустым. Цена молчания здесь — скважина,
    пропавшая из фонда, SUMMARY-секции и ЧДД.
    """
    text = """
DATES
 1 JAN 2014 /
/
WCONPROD
DATES
 1 FEB 2014 /
/
"""

    with pytest.raises(ValueError) as refusal:
        deck.parse_schedule_text(text)

    assert "WCONPROD" in str(refusal.value)
    assert "без записей" in str(refusal.value)


def test_an_unquoted_well_name_no_longer_empties_the_block_at_all():
    """Ради чего заведён `BLOCK_KEYWORDS`: `P1` с новой строки читается как запись.

    До Р-177 первый же заглавный токен закрывал только что открытый блок, и
    `producers` оставался пустым. Теперь имя скважины отличается от ключевого
    слова тем, что ключевое слово **названо** в списке.
    """
    text = """
DATES
 1 JAN 2014 /
/
WCONPROD
 P1 OPEN LRAT 4* 100 50 /
/
"""

    schedule = deck.parse_schedule_text(text)
    assert [one.well for step in schedule.steps for one in step.producers] == ["P1"]


def test_the_quoted_form_of_the_same_record_still_reads():
    """Проверка отказа не должна ловить нормальный дек: тот же режим в кавычках."""
    text = """
DATES
 1 JAN 2014 /
/
WCONPROD
 'P1' OPEN LRAT 4* 100 50 /
/
"""

    schedule = deck.parse_schedule_text(text)
    assert [one.well for step in schedule.steps for one in step.producers] == ["P1"]


def test_the_real_submitted_schedule_parses_the_same_after_the_change():
    """Сдаваемый файл — 371 шаг и 103 скважины — читается как читался.

    Правка трогает распознавание ключевых слов, то есть самый низ разбора. Без
    этой проверки «стало строже» неотличимо от «стало ломать наш же файл».
    """
    path = ROOT / "runs" / "base" / "wells_schedule.inc"
    if not path.is_file():
        pytest.skip("нет выгрузки runs/base на этой машине")

    schedule = deck.parse_schedule(path)
    assert len(schedule.steps) == 371
    assert len(schedule.wells) == 103
    assert len(schedule.producer_wells) == 92
    assert len(schedule.injector_wells) == 41


# --- С10: имя скважины, совпавшее со словом из BLOCK_KEYWORDS ----------------
# Выжимка `STOP-BEFORE-SUBMIT.md`, С10. В список попали `OIL`, `GAS`, `RS`, `RV`,
# `ADD`, `COPY`, `END`, `TIME`; незакавыченное имя, равное такому слову, рвало
# `WCONPROD` посреди записей. На Модели Z имена цифровые — дефект спал до чужого
# дека.


def test_a_well_named_like_a_keyword_does_not_break_the_block():
    """Скважина «OIL» читается как запись, а не закрывает `WCONPROD`.

    До правки терялась не только она: пропадало **всё, что стояло ниже** в блоке,
    и при этом отказа не было — у блока оставалась первая запись, а проверка
    «блок без записей» не срабатывала.
    """
    schedule = deck.parse_schedule_text("""
DATES
 1 JAN 2014 /
/
WCONPROD
 'P1' OPEN LRAT 4* 100 50 /
 OIL OPEN LRAT 4* 90 50 /
/
""")

    assert [one.well for step in schedule.steps for one in step.producers] == ["P1", "OIL"]


def test_the_same_name_in_the_first_record_is_refused_not_lost():
    """Если такое имя стоит первой записью, различить его нечем — тогда отказ.

    Разница с предыдущим тестом не в имени, а в том, что у блока не остаётся ни
    одной прочитанной записи: закрыть его следующим словом — единственное
    поведение, совместимое со словами без записей (`SKIPREST`, `END`). Поэтому
    здесь честный отказ, а не догадка.
    """
    with pytest.raises(ValueError) as refusal:
        deck.parse_schedule_text("""
DATES
 1 JAN 2014 /
/
WCONPROD
 OIL OPEN LRAT 4* 90 50 /
 'P1' OPEN LRAT 4* 100 50 /
/
""")

    assert "WCONPROD" in str(refusal.value)


def test_a_block_without_its_terminator_is_refused_by_the_shape_of_the_record():
    """Незакрытый блок втягивает соседнее слово — и это видно по статусу записи.

    Цена правки С10: блок с прочитанной записью больше не закрывается следующим
    ключевым словом, поэтому `WCONPROD` без одиночного `/` съедает идущий следом
    `DATES`. Молча этого допустить нельзя — потерянный шаг выглядит как «скважина
    так и работала». Признак надёжный: на месте статуса режима у псевдозаписи
    стоит что попало.
    """
    with pytest.raises(ValueError) as refusal:
        deck.parse_schedule_text("""
DATES
 1 JAN 2014 /
/
WCONPROD
 'P1' OPEN LRAT 4* 100 50 /
DATES
 1 FEB 2014 /
/
""")

    assert "статуса" in str(refusal.value)


def test_array_keywords_are_untouched_by_the_record_rule():
    """У массивов GRID одиночный `/` закрывает само слово — их склеивать нельзя.

    Первая редакция правила С10 держала любой блок с записью до одиночного `/` и
    склеивала `PORO` со следующим ключевым словом: десять тестов входного
    контроля дека покраснели разом.
    """
    found = deck.parse_keywords("""
PORO
 0.2 0.2 0.2 /
PERMX
 100 100 100 /
""")

    assert [one.name for one in found] == ["PORO", "PERMX"]
    assert found[0].records == (("0.2", "0.2", "0.2"),)


# ── что нашёл мутационный прогон 06.09 (Р-221) ───────────────────────────────


def test_deck_records_cannot_be_edited_after_parsing():
    """Разобранные записи дека неизменяемы.

    Четыре мутации `frozen=True` на `False` переживали сюиту. Дек — вход всей
    системы: изменяемая запись позволяет поправить режим скважины между
    разбором и записью в файл, и в сдаваемом расписании окажется не то, что
    прочитали.
    """
    import dataclasses

    producer = deck.ProducerControl("1", "OPEN", "LRAT", 30.0, 50.0)
    injector = deck.InjectorControl("2", "WATER", "OPEN", "RATE", 40.0, 300.0)
    keyword = deck.Keyword(name="WCONPROD", records=())

    for one in (producer, injector, keyword):
        with pytest.raises(dataclasses.FrozenInstanceError):
            one.well = "подмена" if hasattr(one, "well") else None
    with pytest.raises(dataclasses.FrozenInstanceError):
        keyword.name = "COMPDAT"


def test_a_block_is_considered_terminated_unless_said_otherwise():
    """Умолчание `terminated` — «блок закрыт», а не «не закрыт».

    Мутация умолчания переживала сюиту: разбор проставляет флаг явно. Ложь по
    умолчанию означала бы, что каждый блок, собранный не разбором (в тесте, в
    генераторе), считается незакрытым — и проверка формата ругалась бы на
    исправный дек.
    """
    assert deck.Keyword(name="WCONPROD", records=()).terminated is True


def test_a_short_date_record_is_refused():
    """Запись `DATES` короче трёх полей или с пропуском — отказ, а не догадка.

    Две мутации сидели здесь: длина `< 3` подменялась на `< 0`, а `or` на `and`.
    С ними запись `01 NOV` разбиралась бы молча — год брался бы неизвестно
    откуда, и календарь расписания уехал бы без единого сообщения.
    """
    assert deck.parse_date_record(("01", "NOV", "1994")) == date(1994, 11, 1)

    for broken in (("01", "NOV"), ("01",), (), ("01", None, "1994"),
                   (None, "NOV", "1994")):
        with pytest.raises(ValueError):
            deck.parse_date_record(broken)


def test_quotes_are_stripped_only_from_a_matching_pair():
    """Кавычки снимаются парой, и только когда они настоящая пара.

    Три мутации сидели в этой функции: длина `>= 2`, срезы `[1:-1]`. Имя
    скважины приходит из дека как есть, и лишний срез превращает `'101'` в `10`
    — то есть в **другую скважину**, молча и на всём горизонте.
    """
    from aios.deck import _unquote

    assert _unquote("'101'") == "101"
    assert _unquote('"101"') == "101"
    assert _unquote("101") == "101", "имя без кавычек не должно резаться"
    assert _unquote("'") == "'", "одиночная кавычка — не пара"
    assert _unquote("") == ""
    assert _unquote("'101") == "'101", "непарная кавычка оставляется как есть"
    # пустая пара — тоже пара: длина ровно два, и граница здесь включающая.
    # Именно её различают `>= 2` и `> 2`, и в деке `''` встречается как пустое
    # поле в кавычках
    assert _unquote("''") == "", "пустая пара кавычек не снята"
    assert _unquote('""') == ''


def test_wells_are_sorted_numerically_then_by_name():
    """Скважины сортируются числом, а не строкой.

    Мутация ключа сортировки переживала сюиту. Строковый порядок ставит «10»
    перед «2», и список фонда в отчёте, в объяснении и в расписании перестаёт
    читаться — а на Модели Z имена как раз цифровые.
    """
    from aios.deck import _sorted_wells

    assert _sorted_wells({"2", "10", "1"}) == ("1", "2", "10")
    # нечисловые имена идут после числовых и сортируются между собой
    assert _sorted_wells({"10", "A1", "2", "B"}) == ("2", "10", "A1", "B")


def test_an_injector_without_a_named_fluid_is_water():
    """Нагнетательная без явной фазы считается водяной.

    Мутация `or "WATER"` на `and` переживала сюиту: в фикстурах фаза всегда
    названа. С подменой она либо всегда «WATER» (даже когда в деке газ), либо
    пустая — и запись `WCONINJE` уходит в файл без фазы.
    """
    text = chr(10).join([
        "SCHEDULE",
        "DATES",
        "  1 JAN 2014 /",
        "/",
        "WCONINJE",
        "  '9' 1* OPEN RATE 40.0 1* 300.0 /",
        "/",
        "WCONINJE",
        "  '8' GAS OPEN RATE 40.0 1* 300.0 /",
        "/",
    ])

    parsed = deck.parse_schedule_text(text)
    fluids = {one.well: one.fluid for one in parsed.steps[0].injectors}

    assert fluids["9"] == "WATER", "пропущенная фаза не подставлена водой"
    assert fluids["8"] == "GAS", "названная фаза подменена водой"


def test_july_is_read_in_both_canonical_spellings():
    """`JLY` — второе каноническое сокращение июля в Eclipse, и деки им пишут.

    Таблица входного контроля (`aios/deckcheck.py`) его знала, а таблица
    разбора — нет: дек с такой датой проходил проверку и валился уже на разборе
    голым `KeyError('JLY')` — без кода отказа и без подсказки. Две копии одной
    таблицы разошлись ровно так, как расходятся все такие копии; теперь она одна.

    Генератор при этом обязан писать `JUL`: наш файл читает их симулятор, и
    вторая форма ему не нужна.
    """
    from aios import deckcheck

    assert deck.parse_date_record(("1", "JLY", "2007")) == date(2007, 7, 1)
    assert deck.parse_date_record(("1", "JUL", "2007")) == date(2007, 7, 1)
    assert deck.parse_schedule_text("DATES\n 1 JLY 2007 /\n/\n").steps[0].date == \
        date(2007, 7, 1)
    assert deck.format_date_record(date(2007, 7, 1)) == "01 JUL 2007"
    assert deckcheck._MONTHS is deck._MONTHS, (
        "у контроля дека снова своя таблица месяцев — они разойдутся молча"
    )
