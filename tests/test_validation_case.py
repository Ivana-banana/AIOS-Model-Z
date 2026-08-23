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
