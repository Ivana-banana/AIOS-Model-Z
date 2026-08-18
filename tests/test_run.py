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


FOREIGN_DATA = """\
-- Дек стороннего месторождения: ни одно имя не совпадает с Моделью Z.
RUNSPEC
METRIC

DIMENS
 10 20 5 /

GRID
INCLUDE
'geometry.grdecl' /

SUMMARY
INCLUDE
'vectors.inc' /

SCHEDULE
INCLUDE
'history.inc' /

END
"""


@pytest.fixture
def foreign_model_dir(tmp_path):
    """Дек, в котором ничто не названо по-нашему: проверка на ТЗ п. 4.3."""
    directory = tmp_path / "FieldX"
    directory.mkdir()
    (directory / "FIELD_X.DATA").write_text(FOREIGN_DATA, encoding="utf-8")
    (directory / "history.inc").write_text(SCHEDULE, encoding="utf-8")
    (directory / "vectors.inc").write_text("", encoding="utf-8")
    (directory / "geometry.grdecl").write_text("-- сетка\n", encoding="utf-8")
    return directory


def test_case_is_built_on_a_deck_with_foreign_file_names(foreign_model_dir, tmp_path):
    """Заявляемый ЧДД считается на кейсе организаторов, а он назван не по-нашему.

    До 18.08 имена `Model_Z.data` и `Model_Z_sch.inc` были зашиты в коде, и на
    таком деке сборка кейса просто не находила файлов.
    """
    case = run.build_case(tmp_path / "run", foreign_model_dir)

    assert case.data_file.is_file()
    assert case.schedule_file.read_text(encoding="utf-8") == SCHEDULE

    data = case.data_file.read_text(encoding="utf-8")
    assert case.schedule_file.resolve().as_posix() in data, "наш SCHEDULE не подставлен"
    assert case.summary_file.resolve().as_posix() in data, "наш SUMMARY не подставлен"
    assert "geometry.grdecl" in data, "сетка организаторов должна остаться на месте"


def test_wells_are_read_from_the_foreign_deck(foreign_model_dir, tmp_path):
    case = run.build_case(tmp_path / "run", foreign_model_dir)
    assert "'1'" in case.summary_file.read_text(encoding="utf-8")


def test_deck_without_a_data_file_fails_loudly(tmp_path):
    """Пустой каталог — это не «соберём как получится», а отказ с причиной."""
    empty = tmp_path / "пусто"
    empty.mkdir()
    with pytest.raises(ValueError, match="нет ни одного .DATA"):
        run.build_case(tmp_path / "run", empty)


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


# ── чужие условия задачи, а не чужой дек (ТЗ п. 4.3) ─────────────────────────
#
# В валидационном кейсе меняются условия, а не объект. Значит контур обязан
# внятно отказывать на том, чего в Модели Z не встречается, — и отказывать
# **до** прогона, а не после часа счёта с неверным диагнозом.

NO_WELLS = """\
RPTSCHED
 'WELLS=1' /

DATES
 01 JAN 2007 /
/
"""

UNCOMPLETED = """\
WELSPECS
 '1' 'GROUP' 23 17 1* 'OIL' /
 '2' 'GROUP' 24 18 1* 'OIL' /
/

DATES
 01 JAN 2007 /
/

COMPDATMD
 '1' 1* 1326.0 1329.3 MD OPEN 2* 0.146 1* 2 1* 1 /
/

WCONPROD
 '1' 'OPEN' 'LRAT' 1* 1* 1* 30.0 1* 50 1* 1* /
 '2' 'OPEN' 'LRAT' 1* 1* 1* 30.0 1* 50 1* 1* /
/
"""


def _deck_with(tmp_path, schedule: str):
    directory = tmp_path / "FieldY"
    directory.mkdir()
    (directory / "FIELD_Y.DATA").write_text(FOREIGN_DATA, encoding="utf-8")
    (directory / "history.inc").write_text(schedule, encoding="utf-8")
    (directory / "vectors.inc").write_text("", encoding="utf-8")
    (directory / "geometry.grdecl").write_text("-- сетка\n", encoding="utf-8")
    return directory


def test_a_deck_without_wells_is_refused_before_the_simulator_runs(tmp_path):
    """Пустой фонд — это отказ, а не SUMMARY без единого вектора.

    Сегодня такой кейс собирается молча, симулятор считает его час и отдаёт
    выгрузку без строк, а `export_rows.py` объявляет причиной «SUMMARY-секция не
    попала в кейс» — то есть не ту. Диагноз известен заранее.
    """
    with pytest.raises(ValueError, match="ни одной скважины"):
        run.build_case(tmp_path / "run", _deck_with(tmp_path, NO_WELLS))


def test_a_well_declared_but_never_completed_is_named(tmp_path):
    """Скважина без вскрытий даёт ноль весь горизонт, и симулятор об этом молчит.

    Ровно так проект уже потерял 22 скважины и все сценарные прогоны 16.08.
    Проверка обязана считать вскрытием и `COMPDAT`, и `COMPDATMD`: в поставке
    тНавигатора есть только второй.
    """
    case = run.build_case(tmp_path / "run", _deck_with(tmp_path, UNCOMPLETED))
    assert case.wells_without_completions == ("2",)


def test_a_fully_completed_deck_reports_nothing(tmp_path, foreign_model_dir):
    case = run.build_case(tmp_path / "run", foreign_model_dir)
    assert case.wells_without_completions == ()


def test_a_control_start_past_the_deck_horizon_is_refused(tmp_path):
    """Дата начала управления за концом дека даёт расписание с повтором даты.

    Сегодня оно собирается молча: история отдаётся целиком, наши шаги
    дописываются следом, и последний шаг дублирует последнюю дату истории. Для
    симулятора шаг, не превосходящий текущего времени, — ошибка, то есть в
    сдаваемый файл уезжает брак.

    Валидатор расписания возрастание дат проверяет, но смотрит на объект `steps`,
    а дубль возникает при склейке истории с ними — в тексте, которого валидатор
    не видит. Поэтому проверка нужна на самой склейке.
    """
    deck = _deck_with(tmp_path, SCHEDULE)
    steps = parse_schedule(deck / "history.inc").steps[-1:]

    with pytest.raises(ValueError, match="не возрастает"):
        run.build_case(tmp_path / "run", deck, steps=steps, control_start=date(2030, 1, 1))


def test_a_normal_control_start_still_assembles(tmp_path):
    """Проверка склейки не должна мешать обычной сборке."""
    deck = _deck_with(tmp_path, SCHEDULE)
    steps = [step for step in parse_schedule(deck / "history.inc").steps
             if step.date >= date(2007, 1, 1)]

    case = run.build_case(tmp_path / "run", deck, steps=steps, control_start=date(2007, 1, 1))

    dates = [item.date for item in parse_schedule(case.schedule_file).steps]
    assert dates == sorted(set(dates)), "даты обязаны строго возрастать"


def test_an_unsupported_control_keyword_stops_the_build(tmp_path):
    """WCONHIST мы не разбираем — значит собирать по такому деку сценарий нельзя.

    Пропустить его молча означает получить шаги без добычи вовсе: базовый
    сценарий, DoE и маржа посчитаются от пустоты, и выглядеть это будет как
    «скважины стояли», а не как «мы не прочли».
    """
    schedule = SCHEDULE + "\nWCONHIST\n '1' 'OPEN' 'ORAT' 30.0 5.0 0.0 /\n/\n"
    deck = _deck_with(tmp_path, schedule)
    steps = [item for item in parse_schedule(deck / "history.inc").steps
             if item.date >= date(2007, 1, 1)]

    with pytest.raises(ValueError, match="WCONHIST"):
        run.build_case(tmp_path / "run", deck, steps=steps, control_start=date(2007, 1, 1))
