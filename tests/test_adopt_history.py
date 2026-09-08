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
from aios.deckfiles import layout

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


MINIMAL_DECK = """\
RUNSPEC
METRIC

SCHEDULE
INCLUDE
'M_sch.inc' /
"""

MINIMAL_SCHEDULE = """\
DATES
 01 JAN 1990 /
/
"""


def test_the_printed_config_lines_actually_parse_as_toml(tmp_path, capsys):
    r"""Напечатанные строки конфигурации вклеиваются в `config.toml` и работают.

    Скрипт печатает три строки, чтобы их скопировали руками, — так задумано: это
    параметры кейса, а не логика. Но путь печатался через `str(path)`, а на
    Windows это обратные слэши, и в TOML обратный слэш начинает escape:
    `model_dir = "C:	mp\case"` даёт табуляцию вместо `	`, а `\c` — ошибку
    разбора. Вклеенная строка ломала бы конфигурацию в вечер 11.09, причём не
    при вклеивании, а на первом чтении конфига, — и чинить было бы уже нельзя.

    Каталог кейса нарочно берётся вне репозитория: путь внутри и так уходил
    через `as_posix()`, дефект жил ровно в ветке «снаружи».
    """
    import tomllib

    deck = tmp_path / "deck"
    deck.mkdir()
    (deck / "M.data").write_text(MINIMAL_DECK, encoding="utf-8")
    (deck / "M_sch.inc").write_text(MINIMAL_SCHEDULE, encoding="utf-8")
    history = tmp_path / "their_history.inc"
    history.write_text(MONTHLY, encoding="utf-8")
    out = tmp_path / "case"

    assert main(["--history", str(history), "--deck", str(deck), "--out", str(out)]) == 0

    printed = capsys.readouterr().out
    lines = [line.strip().split("] ", 1)[1] for line in printed.splitlines()
             if line.strip().startswith(("[case] ", "[economics] "))]
    assert len(lines) == 3, printed

    parsed = tomllib.loads("\n".join(lines))

    assert Path(parsed["model_dir"]) == out.resolve()
    assert parsed["control_start"] == parsed["start_date"] == date(2007, 3, 1)


DECK_WITH_HORIZON = """\
DATES
 01 JAN 2006 /
/

WCONPROD
 '1' 'OPEN' 'LRAT' 1* 1* 1* 30.0 1* 50 1* 1* /
/

DATES
 01 APR 2007 /
/

WCONPROD
 '1' 'OPEN' 'LRAT' 1* 1* 1* 25.0 1* 50 1* 1* /
/

DATES
 01 MAY 2007 /
/

WCONPROD
 '1' 'OPEN' 'LRAT' 1* 1* 1* 20.0 1* 50 1* 1* /
/
"""


def _deck_with(tmp_path, schedule_text):
    """Минимальный дек, у которого SCHEDULE найдётся нашим же разбором."""
    deck = tmp_path / "deck"
    deck.mkdir()
    (deck / "M.data").write_text(MINIMAL_DECK, encoding="utf-8")
    (deck / "M_sch.inc").write_text(schedule_text, encoding="utf-8")
    return deck


def test_the_deck_horizon_past_their_history_is_kept_not_thrown_away(tmp_path, capsys):
    """Их история заменяет историческую часть, а не весь файл.

    Команда затирала SCHEDULE дека целиком. Наши режимы вешаются на существующие
    `DATES` — своих мы не порождаем, — поэтому у кейса, собранного затиранием,
    управляемых шагов **ноль**: сдавать нечего, а обнаружилось бы это на шаге
    цикла, уже под таймером 11.09.

    Проверяется то, ради чего команда написана: после приёмки в деке есть шаги
    за концом их истории, и это шаги поставленного дека.
    """
    deck = _deck_with(tmp_path, DECK_WITH_HORIZON)
    history = tmp_path / "their_history.inc"
    history.write_text(MONTHLY, encoding="utf-8")   # 01.2007 … 02.2007
    out = tmp_path / "case"

    assert main(["--history", str(history), "--deck", str(deck), "--out", str(out)]) == 0

    adopted = parse_schedule_text(
        layout(out).one("SCHEDULE").read_text(encoding="utf-8"))
    dates = [step.date for step in adopted.steps]

    assert date(2007, 1, 1) in dates and date(2007, 2, 1) in dates, dates
    managed = [day for day in dates if day >= date(2007, 3, 1)]
    assert managed == [date(2007, 4, 1), date(2007, 5, 1)], dates
    # шаг дека 01.2006 — часть его истории, и он вытеснен их историей
    assert date(2006, 1, 1) not in dates, dates
    printed = capsys.readouterr().out
    assert "управление:  2007-04-01 — 2007-05-01" in printed
    # управлять есть чем, значит предупреждения быть не должно: прежнее условие
    # `control_start > last` тождественно истинно и печатало его всегда
    assert "ВНИМАНИЕ" not in printed, printed


def test_the_seam_keeps_dates_strictly_increasing(tmp_path):
    """Шов не даёт повторов и убываний: такой `DATES` симулятор считает ошибкой."""
    deck = _deck_with(tmp_path, DECK_WITH_HORIZON)
    history = tmp_path / "their_history.inc"
    history.write_text(MONTHLY, encoding="utf-8")
    out = tmp_path / "case"

    assert main(["--history", str(history), "--deck", str(deck), "--out", str(out)]) == 0

    dates = [step.date for step in parse_schedule_text(
        layout(out).one("SCHEDULE").read_text(encoding="utf-8")).steps]

    assert all(after > before for before, after in zip(dates, dates[1:], strict=False)), dates


def test_a_history_covering_the_whole_horizon_says_there_is_nothing_to_manage(
        tmp_path, capsys):
    """Предупреждение звучит, только когда управлять действительно нечем.

    Условием было `control_start > last`, а это тождественно истинно:
    `control_start` и есть следующий месяц за `last`. Предупреждение печаталось
    всегда — в том числе когда горизонт был, — то есть не отличало опасный
    случай от обычного и потому не значило ничего.
    """
    # дек кончается там же, где их история: управлять нечем
    deck = _deck_with(tmp_path, MONTHLY)
    history = tmp_path / "their_history.inc"
    history.write_text(MONTHLY, encoding="utf-8")
    out = tmp_path / "case"

    assert main(["--history", str(history), "--deck", str(deck), "--out", str(out)]) == 0

    printed = capsys.readouterr().out
    assert "ВНИМАНИЕ:" in printed
    assert "ни в присланной истории, ни в горизонте" in printed
    assert "управление:" not in printed
