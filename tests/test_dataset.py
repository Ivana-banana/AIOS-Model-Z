"""Связь управляющего манифеста DoE с месячным откликом."""

from datetime import date
from pathlib import Path

import pytest

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

SUMMARY
INCLUDE
's.inc' /

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
max_external_water_m3_per_day = 0
water_balance_rule = "compensation"
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


def test_a_converted_well_carries_both_multipliers(tmp_path, monkeypatch):
    """Перевод под закачку обязан доехать до датасета обеими величинами.

    Ради этого и делался третий вид DoE: колонка `inj:<скважина>` появляется у
    скважины только тогда, когда у неё непустой множитель закачки, а иначе
    предложение перевести её под закачку до модели не доходит вовсе
    (`aios/neural.py`, `_pivot_controls`).

    Разрыв на стыке, найденный 19.08: роль скважины сборщик берёт из **опорного**
    расписания, а там переведённая скважина — добывающая. Множитель закачки
    отбрасывался, и пакет ВНС дал бы датасет, неотличимый от обычного, — молча,
    после двух с половиной часов счёта.
    """
    import csv
    import json

    from conftest import make_row

    from scripts import build_dataset

    root = tmp_path / "doe"
    (root / "doe-000").mkdir(parents=True)
    (root / "doe.json").write_text(
        json.dumps({"scenarios": [{
            "name": "doe-000",
            "multipliers": {"1": 0.0},
            "injector_multipliers": {"1": 1.4},
            "converted_wells": ["1"],
        }]}),
        encoding="utf-8",
    )
    (root / "doe-000" / "RUN.UNSMRY").write_text("", encoding="utf-8")
    (root / "doe-000" / ".doe-done").write_text("2026-08-19", encoding="utf-8")
    rows = [make_row("2007-01-01", "1", wwir=40.0, injection=1200.0)]
    monkeypatch.setattr(build_dataset.summary, "monthly_rows", lambda *a, **k: rows)

    target = tmp_path / "dataset.csv"
    status = build_dataset.main([
        "--run-root", str(root), "--out", str(target),
        "--config", str(_tiny_project(tmp_path)),
    ])

    assert status == 0
    with target.open(encoding="utf-8", newline="") as handle:
        written = list(csv.DictReader(handle))
    assert written[0]["production_multiplier"] == "0.0", written[0]
    assert float(written[0]["injection_multiplier"]) == 1.4, (
        "множитель закачки переведённой скважины потерян — колонка inj: не появится"
    )


def test_the_manifest_names_the_converted_wells(tmp_path):
    """Вторая половина того же стыка: список обязан быть в `doe.json`.

    Сборщик датасета читает манифест, а не объекты в памяти. Пока `doe.json` не
    нёс `converted_wells`, знание о переводе терялось между генератором и
    сборщиком целиком — и обнаружилось бы только метриками суррогата, который
    ничему по этим сценариям не научился.
    """
    import json

    from scripts import build_doe

    root = tmp_path / "pack"
    status = build_doe.main([
        "--control-kind", "conversions", "--samples", "3",
        "--run-root", str(root), "--config", str(_tiny_project(tmp_path)),
    ])

    assert status == 0
    manifest = json.loads((root / "doe.json").read_text(encoding="utf-8"))
    converted = [well for s in manifest["scenarios"] for well in s.get("converted_wells", ())]
    assert converted, "манифест не назвал ни одной переведённой скважины"


def test_control_periods_change_the_multiplier_inside_one_scenario(tmp_path, monkeypatch):
    """Режим по периодам — способ получить больше точек без новых прогонов.

    Измерено 19.08: множитель в DoE постоянен по месяцам, поэтому пакет из 12
    прогонов даёт суррогату 12 независимых точек управления, а не 457 320 строк.
    Периоды внутри сценария умножают число точек на число периодов, а прогон
    остаётся один.

    Схема закреплена здесь **до** появления генератора: прошлый стык (перевод под
    закачку) сломался именно потому, что контракт был описан словами, а проверялся
    только на одной стороне.
    """
    import csv
    import json

    from conftest import make_row

    from scripts import build_dataset

    root = tmp_path / "doe"
    (root / "doe-000").mkdir(parents=True)
    (root / "doe.json").write_text(
        json.dumps({"scenarios": [{
            "name": "doe-000",
            "periods": [
                {"from": "2007-01-01", "multipliers": {"1": 0.8}},
                {"from": "2010-01-01", "multipliers": {"1": 1.3}},
            ],
        }]}),
        encoding="utf-8",
    )
    (root / "doe-000" / "RUN.UNSMRY").write_text("", encoding="utf-8")
    (root / "doe-000" / ".doe-done").write_text("2026-08-19", encoding="utf-8")
    rows = [
        make_row("2007-05-01", "1", wlpr=30.0, womr=20.0, liquid=900.0, oil=600.0),
        make_row("2009-12-01", "1", wlpr=30.0, womr=20.0, liquid=900.0, oil=600.0),
        make_row("2010-01-01", "1", wlpr=30.0, womr=20.0, liquid=900.0, oil=600.0),
        make_row("2014-07-01", "1", wlpr=30.0, womr=20.0, liquid=900.0, oil=600.0),
    ]
    monkeypatch.setattr(build_dataset.summary, "monthly_rows", lambda *a, **k: rows)

    target = tmp_path / "dataset.csv"
    assert build_dataset.main([
        "--run-root", str(root), "--out", str(target),
        "--config", str(_tiny_project(tmp_path)),
    ]) == 0

    with target.open(encoding="utf-8", newline="") as handle:
        written = {row["DATA"]: row["production_multiplier"] for row in csv.DictReader(handle)}
    assert written["2007-05-01"] == "0.8"
    assert written["2009-12-01"] == "0.8", "период закончился раньше своей даты"
    assert written["2010-01-01"] == "1.3", "новый период начинается со своей даты включительно"
    assert written["2014-07-01"] == "1.3"


def test_a_manifest_without_periods_is_still_read(tmp_path, monkeypatch):
    """Уже посчитанные пакеты собираются как были — иначе три пакета надо считать заново."""
    import csv
    import json

    from conftest import make_row

    from scripts import build_dataset

    root = tmp_path / "doe"
    (root / "doe-000").mkdir(parents=True)
    (root / "doe.json").write_text(
        json.dumps({"scenarios": [{"name": "doe-000", "multipliers": {"1": 1.1}}]}),
        encoding="utf-8",
    )
    (root / "doe-000" / "RUN.UNSMRY").write_text("", encoding="utf-8")
    (root / "doe-000" / ".doe-done").write_text("2026-08-19", encoding="utf-8")
    monkeypatch.setattr(
        build_dataset.summary, "monthly_rows",
        lambda *a, **k: [make_row("2007-05-01", "1", wlpr=30.0, womr=20.0,
                                  liquid=900.0, oil=600.0)],
    )

    target = tmp_path / "dataset.csv"
    assert build_dataset.main([
        "--run-root", str(root), "--out", str(target),
        "--config", str(_tiny_project(tmp_path)),
    ]) == 0
    with target.open(encoding="utf-8", newline="") as handle:
        assert next(csv.DictReader(handle))["production_multiplier"] == "1.1"


@pytest.mark.parametrize("manifest_controls", [
    {"periods": [{"from": "2007-01-01", "multipliers": {"1": 0.8}}]},
    {"multipliers": {"1": 0.8}},
])
def test_history_before_control_start_carries_no_scenario_mode(
    tmp_path, monkeypatch, manifest_controls
):
    """История до начала управления не должна нести режим сценария.

    Сценарий DoE меняет расписание только с `control_start`, поэтому история у
    всех сценариев пакета побайтово одна и отклик на ней один и тот же. Если
    приписать этим месяцам множители сценария, суррогат учится на них тому, что
    управление на отклик не влияет — на Модели Z это 14 935 строк из 38 110 на
    сценарий, 39,2 % выборки (`docs/experiments.md`, ревизия 19.08).

    Проверяются обе схемы манифеста: старая красит историю единственным периодом
    «с начала времён», новая — режимом первого периода.
    """
    import csv
    import json

    from conftest import make_row

    from scripts import build_dataset

    root = tmp_path / "doe"
    (root / "doe-000").mkdir(parents=True)
    (root / "doe.json").write_text(
        json.dumps({"control_start": "2007-01-01",
                    "scenarios": [{"name": "doe-000", **manifest_controls}]}),
        encoding="utf-8",
    )
    (root / "doe-000" / "RUN.UNSMRY").write_text("", encoding="utf-8")
    (root / "doe-000" / ".doe-done").write_text("2026-08-19", encoding="utf-8")
    rows = [
        make_row("2005-06-01", "1", wlpr=30.0, womr=20.0, liquid=900.0, oil=600.0),
        make_row("2006-12-01", "1", wlpr=30.0, womr=20.0, liquid=900.0, oil=600.0),
        make_row("2007-01-01", "1", wlpr=30.0, womr=20.0, liquid=900.0, oil=600.0),
        make_row("2007-05-01", "1", wlpr=30.0, womr=20.0, liquid=900.0, oil=600.0),
    ]
    monkeypatch.setattr(build_dataset.summary, "monthly_rows", lambda *a, **k: rows)

    config = _tiny_project(tmp_path)
    # скважина работает с 2005-го: иначе роли на исторических месяцах нет и
    # множитель не пишется по другой причине — тест прошёл бы, не проверив ничего
    (tmp_path / "deck" / "h.inc").write_text(
        DECK_SCHEDULE.replace("01 JAN 2007", "01 JAN 2005"), encoding="utf-8"
    )

    target = tmp_path / "dataset.csv"
    assert build_dataset.main([
        "--run-root", str(root), "--out", str(target), "--config", str(config),
    ]) == 0

    with target.open(encoding="utf-8", newline="") as handle:
        written = {row["DATA"]: row for row in csv.DictReader(handle)}
    for month in ("2005-06-01", "2006-12-01"):
        assert written[month]["production_multiplier"] == "", (
            f"{month}: истории приписан режим сценария")
        assert written[month]["multiplier"] == ""
        assert written[month]["is_shut"] == ""
    assert written["2007-01-01"]["production_multiplier"] == "0.8", (
        "управление начинается со своей даты включительно")
    assert written["2007-05-01"]["production_multiplier"] == "0.8"
