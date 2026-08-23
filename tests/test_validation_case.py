"""Валидационный кейс приходит с чужими условиями — проверка подстановкой.

ТЗ п. 4.3 и 5.1: в рейтинг идёт ЧДД на кейсе организаторов, а не наше число на
Модели Z. Меняются при этом **условия задачи, а не объект**: нормативы, пределы
объекта, окно управления, состав фонда. Значит, ценность проверки не в
рассуждении «у нас всё в конфиге», а в подстановке: меняем величину, запускаем,
смотрим вывод.

Каждый тест здесь закрывает один дефект, найденный такой подстановкой 23.08
(задача Б28). Разбор целиком — `docs/validation-case.md`.
"""

from __future__ import annotations

import ast
import csv
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pytest
from conftest import make_row
from openpyxl import load_workbook

from aios import economics, loop, summary
from aios.margin import Norms, screen

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def alien_norms(tmp_path: Path) -> Path:
    """Нормативы организаторов «как на валидации»: цена нефти 42 000 вместо 28 000."""
    path = tmp_path / "Нормативы_организаторов.xlsx"
    shutil.copy(economics.NORMS_FILE, path)
    book = load_workbook(path)
    for row in book["Нормативы"].iter_rows(min_row=2):
        if row[0].value == "oilPriceRubT":
            row[2].value = 42000.0
    book.save(path)
    return path


@pytest.fixture
def alien_pumps(tmp_path: Path) -> Path:
    """Нормативы, где старший ЭЦН закрывает 250–400, а не 320–500."""
    path = tmp_path / "Нормативы_с_другим_ЭЦН.xlsx"
    shutil.copy(economics.NORMS_FILE, path)
    book = load_workbook(path)
    sheet = book["ЭЦН"]
    for row in sheet.iter_rows(min_row=2):
        if row[0].value in (400.0, 400):
            sheet.delete_rows(row[0].row)
            break
    book.save(path)
    return path


def rows_for(well: str, watercut: float, months: int = 12) -> list[dict[str, object]]:
    fluids = summary.Fluids()
    rows = []
    for index in range(months):
        liquid_m3 = 30.0 * 30
        oil_m3 = liquid_m3 * (1 - watercut)
        oil_t = oil_m3 * fluids.oil_density / 1000
        liquid_t = oil_t + (liquid_m3 - oil_m3) * fluids.water_density / 1000
        rows.append(make_row(date(2014, index + 1, 1).isoformat(), well, wlpr=30.0,
                             womr=oil_t / 30, liquid=liquid_t, oil=oil_t))
    return rows


def test_alien_norms_change_the_declared_number(alien_norms: Path):
    """Другой файл нормативов обязан менять ЧДД, а не только маржу скрининга.

    Замер 23.08 до правки: на странице прогона маржа скважины шла по файлу из
    `[economics].norms_file` (53,4 → 148,1 млн при цене 42 000), а ЧДД считался
    по вендоренному файлу и оставался 11,8 млн. Одна страница, две разные цены
    нефти, и увидеть это можно было только в исходниках.
    """
    rows = rows_for("A", 0.5)
    ours = economics.compute_npv(rows, headers=economics.REQUIRED_COLUMNS,
                                 start_date="2014-01-01")
    theirs = economics.compute_npv(rows, headers=economics.REQUIRED_COLUMNS,
                                   start_date="2014-01-01", norms_file=alien_norms)

    assert theirs.chdd_m > ours.chdd_m * 1.5
    # скрининг и ЧДД слышат один и тот же файл
    assert screen(rows, Norms.from_file(alien_norms))[0].total_margin_m > \
        screen(rows, Norms.from_file())[0].total_margin_m


def test_the_cycle_confirms_by_the_norms_it_was_given(alien_norms: Path):
    """`loop.confirm` — последний шаг перед заявкой числа, и нормативы у него свои."""
    rows = rows_for("A", 0.5)
    result = loop.CycleResult(
        base_chdd_m=0.0, chdd_m=0.0, actions=(), rate_choices=(), allocations=(),
        steps=(), warnings=(), problems=(),
    )

    ours = loop.confirm(result, rows, date(2014, 1, 1), base_rows=rows)
    theirs = loop.confirm(result, rows, date(2014, 1, 1), base_rows=rows,
                          norms_file=alien_norms)

    assert theirs.chdd_m != ours.chdd_m
    assert theirs.base_chdd_m != ours.base_chdd_m


def test_the_web_page_shows_the_number_of_the_configured_norms(tmp_path: Path,
                                                               alien_norms: Path):
    """UI — тоже путь, по которому число попадает человеку в глаза."""
    from fastapi.testclient import TestClient

    from aios.web.app import create_app

    config = (ROOT / "tests" / "test_web.py").read_text(encoding="utf-8")
    config = config.split('CONFIG = """')[1].split('"""')[0]
    project = tmp_path / "project"
    run_dir = project / "runs" / "base"
    run_dir.mkdir(parents=True)
    (project / "config.toml").write_text(
        config.format(norms=alien_norms.resolve().as_posix()), encoding="utf-8"
    )
    (run_dir / "wells_schedule.inc").write_text("DATES\n 01 JAN 2014 /\n/\n", encoding="utf-8")
    rows = rows_for("A", 0.5)
    with (run_dir / "rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=economics.REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    page = TestClient(create_app(project)).get("/run/base").text

    expected = economics.compute_npv(rows, headers=economics.REQUIRED_COLUMNS,
                                     start_date="2014-01-01",
                                     norms_file=alien_norms).chdd_m
    assert f"{expected:.1f}".replace(".", ",") in page or f"{expected:.1f}" in page


def _our_sources() -> list[Path]:
    """Наши модули и скрипты — списком из git, а не маской по каталогу.

    На Windows каталог venv `Scripts/` — для файловой системы тот же путь, что
    наш `scripts/`, поэтому маска затянула бы чужие файлы.
    """
    listed = subprocess.run(
        ["git", "ls-files", "aios", "scripts"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    return [ROOT / name for name in listed if name.endswith(".py")]


def test_every_npv_call_names_its_norms():
    """Ни один вызов расчётчика в нашем коде не берёт нормативы молча.

    Умолчание `compute_npv` — файл действующей поставки, и это верно для тестов
    и для Модели Z. Но в рабочем пути умолчание означает «кейс организаторов
    посчитан по нашим нормативам», а увидеть это нельзя ниоткуда: ошибки не
    будет, будет другое число. Проверка статическая — она не требует ни
    симулятора, ни кейса.
    """
    offenders: list[str] = []
    for path in _our_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name not in {"compute_npv", "compute_npv_from_file"}:
                continue
            passed = {keyword.arg for keyword in node.keywords}
            if not passed & {"norms_file", "assumptions"}:
                offenders.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}")

    assert offenders == [], (
        "вызовы ЧДД без явного источника нормативов:\n  " + "\n  ".join(offenders)
    )


def test_the_liquid_limit_travels_with_the_pump_table(alien_pumps: Path):
    """Предел дебита жидкости — из таблицы ЭЦН кейса, а не литерал 500 от Модели Z.

    Методика запрещает дебит выше верхней границы старшего насоса. Число 500
    принадлежит `Нормативы_ЧДД.xlsx` действующей поставки; на валидационном
    кейсе (ТЗ п. 4.3) таблица приходит от организаторов. Литерал в коде означал
    бы, что мы либо генерируем режим, который их расчётчик отвергнет, либо
    запрещаем законный.
    """
    from aios import schedule as schedule_module
    from aios.deck import ProducerControl, ScheduleStep

    _assumptions, pumps = economics.load_norms(alien_pumps)
    limit = schedule_module.liquid_limit(pumps)
    assert limit == 400.0
    assert schedule_module.liquid_limit(economics.load_norms()[1]) ==         schedule_module.MAX_LIQUID_RATE_M3_PER_DAY

    steps = [ScheduleStep(date(2014, 1, 1),
                          (ProducerControl("A", "OPEN", "LRAT", 450.0, 50.0),), ())]
    assert schedule_module.validate(steps) == []
    problems = schedule_module.validate(steps, max_liquid_rate=limit)
    assert len(problems) == 1 and "450" in str(problems[0])


def test_a_non_monthly_export_is_refused_by_name():
    """Строка = скважино-месяц: квартальный шаг молча удешевляет фонд втрое.

    Замер 23.08: тот же годовой объём на квартальном шаге даёт OPEX фонда 0,333
    вместо 1,0 млн руб. в год и ЧДД на 1,7 % выше — ни ошибки, ни предупреждения
    при этом не возникало.
    """
    monthly = [make_row(f"2014-{month:02d}-01", "A", wlpr=30.0, womr=5.0,
                        liquid=900.0, oil=450.0) for month in range(1, 13)]
    quarterly = [row for row in monthly if int(str(row["DATA"])[5:7]) in (1, 4, 7, 10)]

    assert summary.sanity_problems(monthly) == []
    problems = summary.sanity_problems(quarterly)
    assert len(problems) == 1
    assert "не месячный" in problems[0]

    # и деньги действительно разъезжаются — ради этого проверка и стоит
    money = [economics.compute_npv(rows, headers=economics.REQUIRED_COLUMNS,
                                   start_date="2014-01-01",
                                   norms_file=economics.NORMS_FILE).annual[0]["fundOpexM"]
             for rows in (monthly, quarterly)]
    assert money[0] == pytest.approx(1.0)
    assert money[1] == pytest.approx(1 / 3)


def test_planner_reports_a_fractional_limit_as_it_is():
    """Предел объекта печатается без округления: «предела 0» при пределе 0,5 — ложь."""
    from aios import planner
    from aios.deck import ProducerControl, ScheduleStep

    steps = [ScheduleStep(date(2014, 1, 1),
                          (ProducerControl("A", "OPEN", "LRAT", 30.0, 50.0),), ())]
    forecast = [make_row("2014-01-01", "A", wlpr=30.0, womr=5.0, liquid=900.0, oil=450.0)]
    limits = planner.Limits(max_field_liquid_m3_per_day=0.5, enforce_water_balance=False)

    _fitted, decisions = planner.enforce(steps, forecast, Norms.from_file(), limits)

    assert decisions and "предела 0,5" in decisions[0].reason.replace(".", ",")


# --- Б29: чужие режимы управления -------------------------------------------
#
# `optimise.scale` умножает поле расхода независимо от режима. Под `LRAT` и
# `RATE` это уставка, и множитель осмыслен; под `BHP`, `THP`, `GRUP`, `ORAT`,
# `WRAT`, `RESV` то же поле — верхний предел. На Модели Z все 13 382 записи
# добывающих под `LRAT` и все 8 212 записей нагнетательных под `RATE` (замер
# 23.08), поэтому путь не исполнялся ни разу.

WELLS_HEAD = """WELSPECS
 'P1' 'G' 1 1 1* 'OIL' /
 'I1' 'G' 2 2 1* 'WATER' /
/

DATES
 01 JAN 2007 /
/

"""


def _deck_project(tmp_path: Path, schedule_text: str) -> Path:
    """Крошечный дек с заданной SCHEDULE-секцией и конфиг под него."""
    from test_dataset import DECK_DATA, TINY_CONFIG

    deck = tmp_path / "deck"
    deck.mkdir()
    (deck / "FIELD.DATA").write_text(DECK_DATA, encoding="utf-8")
    for name in ("p.inc", "r.inc"):
        (deck / name).write_text("-- пусто\n", encoding="utf-8")
    (deck / "h.inc").write_text(schedule_text, encoding="utf-8")
    config = tmp_path / "config.toml"
    config.write_text(
        TINY_CONFIG.format(norms=Path(economics.NORMS_FILE).resolve().as_posix()),
        encoding="utf-8",
    )
    return config


@pytest.mark.parametrize("mode, record", [
    ("ORAT", " 'P1' 'OPEN' 'ORAT' 20.0 1* 1* 1* 1* 50 1* 1* /"),
    ("WRAT", " 'P1' 'OPEN' 'WRAT' 1* 25.0 1* 1* 1* 50 1* 1* /"),
    ("RESV", " 'P1' 'OPEN' 'RESV' 1* 1* 1* 1* 40.0 50 1* 1* /"),
])
def test_a_producer_without_a_rate_field_is_not_shut_by_scaling(mode: str, record: str):
    """Работающая скважина без колонки LRAT не должна закрываться множителем.

    До 23.08 `_scaled` считал `(None or 0) × 1.5 = 0` и превращал запись в
    `SHUT`: на чужом деке сценарий «форсаж ×1.5» **останавливал** скважины под
    `ORAT`, `WRAT` и `RESV`, а печатная нота при этом говорила «поле дебита это
    предел, а не уставка» — ровно наоборот произошедшему.
    """
    from aios.deck import parse_schedule_text
    from aios.optimise import scale

    steps = parse_schedule_text(WELLS_HEAD + "WCONPROD\n" + record + "\n/\n").steps
    scaled = scale(steps, 1.5)

    control = scaled[0].producers[0]
    assert control.status == "OPEN", f"{mode}: скважина закрыта множителем"
    assert control.mode == mode
    assert control.liquid_rate is None, "поля дебита не было — не должно и появиться"


def test_an_injector_without_a_rate_field_is_not_shut_by_scaling():
    """То же для нагнетательной под `RESV`: приёмистость стоит в своей колонке."""
    from aios.deck import parse_schedule_text
    from aios.optimise import scale_injection

    text = WELLS_HEAD + "WCONINJE\n 'I1' 'WATER' 'OPEN' 'RESV' 1* 220.0 300 1* /\n/\n"
    scaled = scale_injection(parse_schedule_text(text).steps, 1.6)

    control = scaled[0].injectors[0]
    assert control.status == "OPEN"
    assert control.rate is None


def test_the_scaling_report_classifies_every_mode():
    """Каждый режим попадает ровно в одну корзину, и доля считается по всем.

    Раньше нота смотрела только на добывающих: нагнетательная под `RESV` молча
    останавливалась, а строка сообщала «множитель применён буквально ко всем
    записям, все под LRAT».
    """
    from aios.deck import InjectorControl, ProducerControl, ScheduleStep
    from aios.optimise import scale, scale_injection, scaling_report

    before = [ScheduleStep(date(2007, 1, 1), (
        ProducerControl("уставка", "OPEN", "LRAT", 100.0, 50.0),
        ProducerControl("предел", "OPEN", "LRAT", 400.0, 50.0),
        ProducerControl("давление", "OPEN", "BHP", 100.0, 90.0),
        ProducerControl("устье", "OPEN", "THP", 100.0, 90.0),
        ProducerControl("группа", "OPEN", "GRUP", 100.0, 50.0),
        ProducerControl("нефть", "OPEN", "ORAT", None, 50.0),
        ProducerControl("стоит", "SHUT", "LRAT", None, 50.0),
    ), (
        InjectorControl("расход", "WATER", "OPEN", "RATE", 100.0, 300.0),
        InjectorControl("давление", "WATER", "OPEN", "BHP", 100.0, 300.0),
        InjectorControl("объём", "WATER", "OPEN", "RESV", None, 300.0),
    ))]
    after = scale_injection(scale(before, 1.5), 1.6)
    report = scaling_report(before, after, production=1.5, injection=1.6)

    assert report.records == 9, "остановленная запись не масштабируется и не считается"
    assert report.literal == 2, "буквально применён только к LRAT и RATE с уставкой"
    assert report.capped == 1
    assert report.off_mode == {"BHP": 2, "GRUP": 1, "THP": 1}
    assert report.without_rate == {"ORAT": 1, "RESV": 1}
    assert report.share == pytest.approx(2 / 9)


def test_the_preflight_refuses_a_label_that_does_not_describe_the_file(tmp_path: Path):
    """Доля буквально применённых записей — величина для машины, а не для лога."""
    import json
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from check_package import _scaling_problem

    schedule = tmp_path / "wells_schedule.inc"
    schedule.write_text("-- пусто\n", encoding="utf-8")

    missing, detail = _scaling_problem(schedule)
    assert missing is None and "не измерена" in detail

    (tmp_path / "scaling.json").write_text(
        json.dumps({"records": 10, "literal": 4, "share": 0.4}), encoding="utf-8"
    )
    problem, _detail = _scaling_problem(schedule)
    assert problem is not None and "40.0%" in problem.replace(",", ".")

    (tmp_path / "scaling.json").write_text(
        json.dumps({"records": 10, "literal": 10, "share": 1.0}), encoding="utf-8"
    )
    problem, detail = _scaling_problem(schedule)
    assert problem is None and "всем 10 записям" in detail


def test_a_deck_with_wconhist_is_refused_by_name(tmp_path: Path, capsys):
    """Дек с историческими режимами мы не считаем — и говорим это строкой.

    `WCONHIST` занимает место нашего `WCONPROD`: разобранные шаги остаются без
    режимов, а в собранный файл попали бы оба ключевых слова сразу. Отказ
    законен, трассировка на середине сборки — нет.
    """
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import build_scaled_case

    text = (WELLS_HEAD + "WCONHIST\n 'P1' 'OPEN' 'LRAT' 20.0 12.0 1* 1* 1* 1* /\n/\n"
            + "WCONINJH\n 'I1' 'WATER' 'OPEN' 200.0 1* 300 /\n/\n")
    config = _deck_project(tmp_path, text)

    code = build_scaled_case.main([str(tmp_path / "run"), "--production", "1.5",
                                   "--config", str(config)])

    assert code == 1
    printed = capsys.readouterr()
    assert "кейс не собран" in printed.err
    assert "WCONHIST" in printed.err
