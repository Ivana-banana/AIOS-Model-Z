"""Разбор SCHEDULE-секции дека."""

from __future__ import annotations

from datetime import date

import pytest
from conftest import MODEL_Z_SCHEDULE

from aios import deck

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
