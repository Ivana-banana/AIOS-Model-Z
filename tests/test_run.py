"""Сборка расчётного кейса поверх дека организаторов."""

from __future__ import annotations

from datetime import date

import pytest
from conftest import MODEL_Z_SCHEDULE

from aios import run
from aios.deck import (
    InjectorControl,
    ProducerControl,
    ScheduleStep,
    parse_schedule,
    parse_schedule_text,
)

SCHEDULE = """\
RPTSCHED
 'WELLS=1' /

WELSPECS
 '1' 'GROUP' 23 17 1* 'OIL' /
/

DATES
 01 NOV 2006 /
/

COMPDATMD
 '1' 1* 1326.0 1329.3 MD OPEN 2* 0.146 1* 2 1* 1 /
/

WCONPROD
 '1' 'OPEN' 'LRAT' 1* 1* 1* 45.0 1* 50 1* 1* /
/

DATES
 01 JAN 2007 /
/

WCONPROD
 '1' 'OPEN' 'LRAT' 1* 1* 1* 30.0 1* 50 1* 1* /
/
"""

DATA = """\
RUNSPEC
METRIC

GRID
INCLUDE
'Model_Z_grid.inc' /

SUMMARY
INCLUDE
'Model_Z_summary.inc' /

SCHEDULE
INCLUDE
'Model_Z_sch.inc' /

END
"""


@pytest.fixture
def model_dir(tmp_path):
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "Model_Z.data").write_text(DATA, encoding="utf-8")
    (directory / "Model_Z_sch.inc").write_text(SCHEDULE, encoding="utf-8")
    (directory / "Model_Z_summary.inc").write_text("", encoding="utf-8")
    (directory / "Model_Z_grid.inc").write_text("-- сетка\n", encoding="utf-8")
    return directory


def test_history_prefix_keeps_completions_and_cuts_at_the_control_date():
    prefix = run.history_prefix(SCHEDULE, date(2007, 1, 1))

    assert "COMPDATMD" in prefix  # вскрытия сохранены дословно
    assert "01 NOV 2006" in prefix
    assert "01 JAN 2007" not in prefix
    assert "30.0" not in prefix  # управление с даты отсечки не попало


def test_history_prefix_returns_everything_when_the_date_is_beyond_history():
    assert run.history_prefix(SCHEDULE, date(2030, 1, 1)) == SCHEDULE


def test_base_case_keeps_the_original_schedule(model_dir, tmp_path):
    case = run.build_case(tmp_path / "run", model_dir)

    assert case.schedule_file.read_text(encoding="utf-8") == SCHEDULE
    assert case.summary_case == case.data_file.with_suffix("")


def test_case_points_to_our_summary_and_schedule_and_to_their_grid(model_dir, tmp_path):
    case = run.build_case(tmp_path / "run", model_dir)
    text = case.data_file.read_text(encoding="utf-8")

    assert case.schedule_file.resolve().as_posix() in text
    assert case.summary_file.resolve().as_posix() in text
    assert (model_dir / "Model_Z_grid.inc").resolve().as_posix() in text
    # дек организаторов не тронут
    assert (model_dir / "Model_Z_summary.inc").read_text(encoding="utf-8") == ""


def test_summary_section_covers_every_well_of_the_deck(model_dir, tmp_path):
    case = run.build_case(tmp_path / "run", model_dir)
    section = case.summary_file.read_text(encoding="utf-8")
    assert "WOPT\n '1'\n/" in section


def test_our_steps_replace_control_from_the_start_date(model_dir, tmp_path):
    steps = [
        ScheduleStep(
            date(2007, 1, 1),
            (ProducerControl("1", "OPEN", "LRAT", 99.0, 50.0),),
            (InjectorControl("2", "WATER", "OPEN", "RATE", 20.0, 300.0),),
        )
    ]
    case = run.build_case(
        tmp_path / "run", model_dir, steps=steps, control_start=date(2007, 1, 1), wells=("1", "2")
    )

    text = case.schedule_file.read_text(encoding="utf-8")
    assert "COMPDATMD" in text          # история со вскрытиями осталась
    assert "45.0" in text               # управление до отсечки осталось
    assert "30.0" not in text           # управление после отсечки заменено нашим
    assert "99" in text and "WCONINJE" in text

    parsed = parse_schedule_text(text)
    assert parsed.steps[-1].date == date(2007, 1, 1)
    assert parsed.steps[-1].producers[0].liquid_rate == 99.0


def test_own_steps_require_a_control_start(model_dir, tmp_path):
    with pytest.raises(ValueError, match="control_start"):
        run.build_case(tmp_path / "run", model_dir, steps=[])


@pytest.mark.skipif(not MODEL_Z_SCHEDULE.is_file(), reason="дек Модели Z не распакован")
def test_cutting_the_real_deck_at_the_track_2_control_date():
    """01.01.2007 — начало управления фондом по методике для трека 2."""
    text = MODEL_Z_SCHEDULE.read_text(encoding="utf-8", errors="replace")
    prefix = run.history_prefix(text, date(2007, 1, 1))

    kept = parse_schedule_text(prefix)
    assert kept.steps[-1].date == date(2006, 12, 1)
    assert len(kept.steps) == 146
    assert len(kept.wells) > 80  # фонд к 2007 году уже разбурен
    assert "COMPDAT" in prefix  # вскрытия фонда сохранены


def test_carried_keywords_survive_a_rebuilt_schedule():
    """Вскрытия после начала управления обязаны пережить пересборку расписания.

    В деке Модели Z после 2007 года бурится 22 скважины, и их `COMPDAT` лежит
    между `DATES` вместе с режимами. Генератор пишет только режимы, поэтому без
    явного переноса такая скважина получает `WCONPROD`, но ни одного соединения
    с пластом — и даёт ноль весь горизонт. Симулятор об этом только
    предупреждает («has no connections to grid and will remain SHUT»), а прогон
    считается успешным.

    Стоило это 1106 млн руб. мнимой «цены решений» в первом же сравнении
    сценариев (`docs/experiments.md`, 16.08).
    """
    text = """\
DATES
 01 JAN 2007 /
/

COMPDAT
 '12' 18 67 3 6 OPEN 1* 1* 0.146 1* 0 1* Z /
/

WCONPROD
 '12' 'OPEN' 'LRAT' 1* 1* 1* 20.0 1* 50 1* 1* /
/

DATES
 01 FEB 2007 /
/

WPIMULT
 '12' 1.5 /
/
"""
    carried = run.carried_keywords(text, date(2007, 1, 1))

    assert set(carried) == {date(2007, 1, 1), date(2007, 2, 1)}
    assert "COMPDAT" in carried[date(2007, 1, 1)]
    assert "'12' 18 67 3 6 OPEN" in carried[date(2007, 1, 1)]
    assert "WPIMULT" in carried[date(2007, 2, 1)]
    # режимы генерируем сами — они не должны дублироваться переносом
    assert "WCONPROD" not in carried[date(2007, 1, 1)]


def test_history_before_control_start_is_not_carried_twice():
    """До начала управления история и так сохраняется дословно префиксом."""
    text = """\
DATES
 01 JAN 2006 /
/

COMPDAT
 '1' 23 17 1 2 OPEN 1* 1* 0.146 1* 0 1* Z /
/
"""
    assert run.carried_keywords(text, date(2007, 1, 1)) == {}


@pytest.mark.skipif(not MODEL_Z_SCHEDULE.is_file(), reason="дек не распакован")
def test_rebuilt_case_keeps_every_completion_of_the_real_deck(tmp_path):
    """На реальном деке: пересобранное расписание не теряет ни одного COMPDAT."""
    original = MODEL_Z_SCHEDULE.read_text(encoding="utf-8", errors="replace")
    schedule = parse_schedule(MODEL_Z_SCHEDULE)
    control_start = date(2007, 1, 1)
    steps = [step for step in schedule.steps if step.date >= control_start]

    case = run.build_case(tmp_path / "case", MODEL_Z_SCHEDULE.parent,
                          steps=steps, control_start=control_start)
    rebuilt = case.schedule_file.read_text(encoding="utf-8")

    assert rebuilt.count("COMPDAT") == original.count("COMPDAT"), (
        "потеряны вскрытия: скважины останутся без соединений с пластом"
    )
