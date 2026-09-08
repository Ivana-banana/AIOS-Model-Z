"""Сборка пакета за период управления: файл без истории и его ЧДД.

Письмо организаторов от 08.09.2026 сменило форму сдачи: и `wells_schedule.inc`,
и заявляемое число — только за период, когда месторождением управляет AIOS.
Историю организаторы подставляют сами.

Проверяется не «функция что-то вернула», а два свойства, ценой ошибки в которых
будет единственный расчёт организаторов: разрез ничего не теряет и не удваивает,
и окно ЧДД совпадает с периодом сдаваемого файла.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

from aios.deck import InjectorControl, ProducerControl, ScheduleStep, parse_schedule_text
from aios.schedule import render

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_submission import meaningful_lines, split_off_history  # noqa: E402

CONTROL_START = date(2007, 1, 1)


def test_output_cannot_overwrite_the_full_run_schedule(tmp_path, capsys):
    from build_submission import main

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    source = run_dir / "wells_schedule.inc"
    source.write_text(SCHEDULE, encoding="utf-8")

    assert main(["--run", str(run_dir), "--out", str(run_dir)]) == 1
    assert source.read_text(encoding="utf-8") == SCHEDULE
    assert "исходного прогона" in capsys.readouterr().err


def test_failed_rebuild_invalidates_previous_confirmation(tmp_path, monkeypatch):
    import build_submission
    from check_package import _trace_problem

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "wells_schedule.inc").write_text(SCHEDULE, encoding="utf-8")
    out = run_dir / "submission"
    out.mkdir()
    target = out / "wells_schedule.inc"
    target.write_text(split_off_history(SCHEDULE, CONTROL_START), encoding="utf-8")
    trace = out / "npv.json"
    trace.write_text(json.dumps({
        "agreed": True, "chddM": 123.0, "claimedM": 123.0,
        "scheduleMd5": build_submission.digest(target),
    }), encoding="utf-8")
    monkeypatch.setattr(build_submission, "norms_refusal", lambda _: None)

    assert build_submission.main([
        "--run", str(run_dir), "--model-dir", str(tmp_path / "missing-deck")
    ]) == 1

    record = json.loads(trace.read_text(encoding="utf-8"))
    assert record["agreed"] is False
    assert record["status"] == "unconfirmed"
    assert "сборка пакета не подтверждена" in _trace_problem(target, "123,0 млн руб.")


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

COMPDATMD
 '2' 1* 1400.0 1410.0 MD OPEN 2* 0.146 1* 2 1* 1 /
/

WCONPROD
 '1' 'OPEN' 'LRAT' 1* 1* 1* 30.0 1* 50 1* 1* /
/
"""


def test_history_stays_with_the_organisers():
    """В хвосте нет ни одной строки истории — иначе даты повторятся дважды."""
    tail = split_off_history(SCHEDULE, CONTROL_START)

    assert "01 NOV 2006" not in tail
    assert "45.0" not in tail
    assert tail.startswith("DATES\n 01 JAN 2007 /")


def test_completions_of_wells_drilled_after_the_cut_travel_with_the_tail():
    """Скважина, вскрытая после отсечки, обязана унести своё вскрытие с собой.

    Потерять его — значит получить скважину с режимом и без единого соединения
    с сеткой: симулятор на этом не падает, а скважина даёт ноль весь горизонт.
    """
    tail = split_off_history(SCHEDULE, CONTROL_START)

    assert "COMPDATMD" in tail
    assert "'2'" in tail


def test_the_cut_neither_loses_nor_duplicates_a_single_line():
    """История плюс хвост — это ровно исходное расписание, строка в строку."""
    tail = split_off_history(SCHEDULE, CONTROL_START)
    prefix = SCHEDULE[: SCHEDULE.index("DATES\n 01 JAN 2007")]

    assert meaningful_lines(prefix) + meaningful_lines(tail) == meaningful_lines(SCHEDULE)


def test_nothing_to_hand_over_is_a_refusal_and_not_an_empty_file():
    """За концом горизонта сдавать нечего — и это отказ, а не пустой файл."""
    with pytest.raises(ValueError, match="сдавать за период управления нечего"):
        split_off_history(SCHEDULE, date(2030, 1, 1))


def test_generated_tail_survives_a_round_trip_through_our_own_parser():
    """Собранный хвост читается нашим же разбором: генератор и парсер не разошлись."""
    steps = [
        ScheduleStep(date(2007, 1, 1),
                     (ProducerControl("1", "OPEN", "LRAT", 30.0, 50.0),),
                     (InjectorControl("2", "WATER", "OPEN", "RATE", 120.0, 300.0),)),
        ScheduleStep(date(2007, 2, 1),
                     (ProducerControl("1", "SHUT", "LRAT", None, 50.0),), ()),
    ]
    tail = split_off_history(render(steps), CONTROL_START)
    parsed = parse_schedule_text(tail)

    assert [step.date for step in parsed.steps] == [date(2007, 1, 1), date(2007, 2, 1)]
    assert parsed.steps[0].injectors[0].well == "2"


def test_window_of_the_declared_number_matches_the_period_of_the_declared_file():
    """Окно ЧДД в конфиге равно началу управления — иначе файл и число о разном.

    Ровно этот разъезд и есть способ провалить единственный расчёт организаторов:
    файл сдан за период управления, а число посчитано вместе с историей.
    """
    from aios import config as config_module

    config = config_module.load(Path(__file__).resolve().parent.parent / "config.toml")

    assert config.economics.start_date == config.case.control_start
