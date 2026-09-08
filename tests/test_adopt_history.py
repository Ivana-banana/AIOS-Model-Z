"""Приёмка истории разработки организаторов: вторая часть валидационного кейса.

История приходит 11.09 в 17:00, сдача трека 2 — в 21:00. Проверяется не то, что
скрипт «отработал», а то, что материал, который контур прочитает не так, как он
написан, останавливается **до** сборки дека: под таймером обнаружить это на
первом запуске симулятора значит потерять окно.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from aios.deck import parse_schedule_text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from adopt_history import cadence, history_problems, main, next_month  # noqa: E402

MONTHLY = """\
WELSPECS
 '1' 'GROUP' 23 17 1* 'OIL' /
/

DATES
 01 JAN 2007 /
/

WCONPROD
 '1' 'OPEN' 'LRAT' 1* 1* 1* 45.0 1* 50 1* 1* /
/

DATES
 01 FEB 2007 /
/

WCONPROD
 '1' 'OPEN' 'LRAT' 1* 1* 1* 40.0 1* 50 1* 1* /
/
"""


def test_a_clean_monthly_history_has_no_objections():
    assert history_problems(MONTHLY) == []


def test_history_without_a_single_date_is_refused():
    """Дек, ведущий время через TSTEP, мы не разбираем — и говорим это словами."""
    problems = history_problems("WELSPECS\n '1' 'GROUP' 23 17 1* 'OIL' /\n/\n")

    assert len(problems) == 1
    assert "нет ни одного шага DATES" in problems[0]


def test_a_control_before_the_first_date_is_refused():
    """Режим в преамбуле датируется START из RUNSPEC, а мы читаем только SCHEDULE.

    Скважина, чей единственный режим стоит там, выпала бы из фонда и из выгрузки
    молча — и ЧДД разошёлся бы с пересчётом организаторов без единой ошибки.
    """
    text = "WCONPROD\n '1' 'OPEN' 'LRAT' 1* 1* 1* 45.0 1* 50 1* 1* /\n/\n" + MONTHLY
    problems = history_problems(text)

    assert any("до первого DATES" in problem for problem in problems)


def test_historical_controls_are_refused_by_name():
    """`WCONHIST` заменяет то, что генерируем мы: в файл попало бы и то и другое."""
    text = MONTHLY.replace("WCONPROD", "WCONHIST", 1)
    problems = history_problems(text)

    assert any("WCONHIST" in problem for problem in problems)


def test_non_increasing_dates_are_refused():
    text = MONTHLY.replace("01 FEB 2007", "01 JAN 2007")
    problems = history_problems(text)

    assert any("не возрастает" in problem for problem in problems)


def test_quarterly_cadence_is_named_and_not_silently_accepted():
    """Немесячный шаг — это другая экономика, а не другая частота.

    Строка выгрузки у нас и у эталона — скважино-месяц, и OPEX за период
    считается по числу строк: квартальный шаг удешевил бы его втрое молча.
    """
    text = MONTHLY.replace("01 FEB 2007", "01 APR 2007")
    label, odd = cadence(parse_schedule_text(text).steps)

    assert label == "3 мес."
    assert odd == []


def test_a_single_odd_gap_is_listed():
    text = MONTHLY + """
DATES
 01 MAY 2007 /
/
"""
    label, odd = cadence(parse_schedule_text(text).steps)

    assert label == "месячный"
    assert odd == ["2007-02-01 → 2007-05-01 (3 мес.)"]


def test_control_start_is_the_month_after_the_history_ends():
    assert next_month(date(2007, 1, 1)) == date(2007, 2, 1)
    assert next_month(date(2025, 12, 1)) == date(2026, 1, 1)


def test_adopted_deck_is_the_one_our_own_reader_resolves(tmp_path, monkeypatch):
    """Подменённый файл обязан оказаться тем, на который смотрит INCLUDE.

    Проверяется делом: после сборки дек перечитывается нашим же разбором, и его
    SCHEDULE сверяется с текстом истории. Ошибиться здесь значит собрать кейс на
    прежнем расписании и не заметить этого до самого расчёта.
    """
    from aios.deckfiles import layout

    deck = tmp_path / "deck"
    deck.mkdir()
    (deck / "M.data").write_text(
        "RUNSPEC\nMETRIC\n\nSCHEDULE\nINCLUDE\n'M_sch.inc' /\n", encoding="utf-8")
    (deck / "M_sch.inc").write_text("DATES\n 01 JAN 1990 /\n/\n", encoding="utf-8")

    history = tmp_path / "their_history.inc"
    history.write_text(MONTHLY, encoding="utf-8")
    out = tmp_path / "case"

    code = main(["--history", str(history), "--deck", str(deck), "--out", str(out)])

    assert code == 0
    adopted = layout(out).one("SCHEDULE")
    assert adopted.name == "M_sch.inc"
    assert parse_schedule_text(adopted.read_text(encoding="utf-8")).steps[0].date == date(
        2007, 1, 1)


def test_refused_history_leaves_no_half_built_deck(tmp_path):
    """Отказ случается до копирования: полусобранный дек хуже отсутствующего."""
    deck = tmp_path / "deck"
    deck.mkdir()
    (deck / "M.data").write_text(
        "RUNSPEC\nMETRIC\n\nSCHEDULE\nINCLUDE\n'M_sch.inc' /\n", encoding="utf-8")
    (deck / "M_sch.inc").write_text("DATES\n 01 JAN 1990 /\n/\n", encoding="utf-8")

    history = tmp_path / "their_history.inc"
    history.write_text(MONTHLY.replace("WCONPROD", "WCONHIST", 1), encoding="utf-8")
    out = tmp_path / "case"

    code = main(["--history", str(history), "--deck", str(deck), "--out", str(out)])

    assert code == 1
    assert not out.exists()
