"""Связь управляющего манифеста DoE с месячным откликом."""

from datetime import date
from pathlib import Path

from aios.deck import InjectorControl, ProducerControl, ScheduleStep
from scripts.build_dataset import control_roles


def test_reference_schedule_preserves_role_when_scenario_response_is_zero():
    steps = [
        ScheduleStep(
            date(2014, 1, 1),
            (ProducerControl("9", "OPEN", "LRAT", 30.0, 50.0),),
            (InjectorControl("9", "WATER", "SHUT", "RATE", None, 300.0),),
        ),
        ScheduleStep(
            date(2014, 2, 1),
            (ProducerControl("9", "SHUT", "LRAT", None, 50.0),),
            (InjectorControl("9", "WATER", "OPEN", "RATE", 40.0, 300.0),),
        ),
    ]

    roles = control_roles(steps, {"2014-01-01", "2014-02-01", "2014-03-01"})

    assert roles[("2014-01-01", "9")] == "production"
    assert roles[("2014-02-01", "9")] == "injection"
    assert roles[("2014-03-01", "9")] == "injection"


def test_a_case_without_the_done_marker_is_not_collected(tmp_path, capsys):
    """Оборванный прогон оставляет `.UNSMRY`, но это обрезок истории.

    Настоящий случай 18.08: сборщик подхватил сценарий, который в этот момент
    считался заново, и положил в датасет 7107 строк вместо 38110 — молча, без
    единой жалобы. Суррогат, обученный на такой выборке, врёт именно там, где
    данных не хватило.
    """
    import json

    from scripts.build_dataset import main

    root = tmp_path / "doe"
    (root / "doe-000").mkdir(parents=True)
    (root / "doe.json").write_text(
        json.dumps({"scenarios": [{"name": "doe-000", "multipliers": {}}]}),
        encoding="utf-8",
    )
    # результат есть, отметки о завершении нет — значит прогон оборвали
    (root / "doe-000" / "RUN.UNSMRY").write_text("обрывок", encoding="utf-8")

    status = main([
        "--run-root", str(root),
        "--out", str(tmp_path / "dataset.csv"),
        "--config", str(_tiny_project(tmp_path)),
    ])
    output = capsys.readouterr().out

    assert status == 1, "собранных сценариев нет — это отказ, а не успех"
    assert "нет отметки о завершении" in output
    assert "0 собрано" in output


DECK_DATA = """\
RUNSPEC

DIMENS
 2 2 2 /

PROPS
INCLUDE
'p.inc' /

REGIONS
INCLUDE
'r.inc' /

SCHEDULE
INCLUDE
'h.inc' /

END
"""

DECK_SCHEDULE = """\
WELSPECS
 '1' 'G' 1 1 1* 'OIL' /
/

DATES
 01 JAN 2007 /
/

WCONPROD
 '1' 'OPEN' 'LRAT' 1* 1* 1* 30.0 1* 50 1* 1* /
/
"""

TINY_CONFIG = """\
[case]
model_dir = "deck"
run_dir = "runs"
case_name = "RUN"
control_start = 2007-01-01

[economics]
start_date = 2007-01-01
norms_file = "norms.xlsx"

[fluids]
oil_density = 913.0765883224138
water_density = 1172.0

[limits]
max_field_liquid_m3_per_day = 0
max_field_injection_m3_per_day = 0
enforce_water_balance = true
water_tolerance = 1.0

[explain]
enabled = false
model = "claude-opus-5"
max_tokens = 1024
effort = "low"
cache_dir = "artifacts"

[policy]
lookback_months = 6

[run]
seed = 1
processes = 1
"""


def _tiny_project(tmp_path) -> Path:
    """Крошечный дек и конфиг под него.

    Предмет теста — отметка `.doe-done`, к деку отношения не имеющая. Пока тест
    брал дек из корневого `config.toml`, он проходил только там, где распакована
    Модель Z: на чистом клоне падал — а именно так решение запускает проверяющий.
    """
    deck = tmp_path / "deck"
    deck.mkdir()
    (deck / "FIELD.DATA").write_text(DECK_DATA, encoding="utf-8")
    (deck / "p.inc").write_text("-- пусто\n", encoding="utf-8")
    (deck / "r.inc").write_text("-- пусто\n", encoding="utf-8")
    (deck / "h.inc").write_text(DECK_SCHEDULE, encoding="utf-8")

    config = tmp_path / "config.toml"
    config.write_text(TINY_CONFIG, encoding="utf-8")
    return config
