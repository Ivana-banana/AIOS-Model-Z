"""Веб-платформа: состояние фонда, экономика прогона, запуск цикла МАС.

Обязательный по регламенту артефакт — без UI решение к валидации не допускают.
Устроена так, чтобы у проверяющего она поднялась без интернета и без сборки
фронтенда: сервер отдаёт готовый HTML, графики рисуются inline-SVG
(`aios/web/charts.py`), внешних скриптов и шрифтов нет ни одного.

Данные берутся из каталога прогона (`runs/<имя>`): сначала `rows.csv` — выгрузка
в контракте расчётчика, которой можно обменяться без 200-мегабайтных бинарников;
если его нет, читается `.UNSMRY` симулятора.

Запуск:

    uvicorn aios.web.app:app --reload
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from aios import balance as balance_module
from aios import config as config_module
from aios import deckfiles, economics, loop, summary
from aios.deck import ScheduleStep, parse_schedule
from aios.margin import Norms, screen
from aios.model import SurrogateLoadError, for_cycle
from aios.web.charts import Series, bar_chart, line_chart

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

NUMERIC_COLUMNS = tuple(column for column in economics.REQUIRED_COLUMNS if column not in ("DATA", "well"))
QUALITY_TARGETS = ("WLPT_Diff", "WOMT_Diff")


@dataclass(frozen=True)
class Run:
    """Каталог прогона и то, что в нём удалось найти."""

    name: str
    path: Path

    @property
    def rows_csv(self) -> Path:
        return self.path / "rows.csv"

    @property
    def summary_case(self) -> Path | None:
        for candidate in sorted(self.path.glob("*.UNSMRY")):
            return candidate.with_suffix("")
        return None

    @property
    def schedule(self) -> Path:
        return self.path / "wells_schedule.inc"

    @property
    def has_results(self) -> bool:
        return self.rows_csv.is_file() or self.summary_case is not None

    @property
    def status(self) -> str:
        if self.rows_csv.is_file():
            return "выгрузка готова"
        if self.summary_case is not None:
            return "есть расчёт симулятора"
        if self.schedule.is_file():
            return "кейс собран, расчёта нет"
        return "пусто"


@dataclass(frozen=True)
class SurrogateStatus:
    """Краткий и безопасный для UI срез отчёта об обучении."""

    state: str
    train_scenarios: int = 0
    test_scenarios: int = 0
    wells: int = 0
    rows: int = 0
    metrics: tuple[
        tuple[str, float, float, float | None, float | None, float | None, float | None], ...
    ] = ()
    quality_passed: bool = False
    seed: int | None = None
    sklearn_version: str | None = None
    pandas_version: str | None = None
    datasets: tuple[tuple[str, str], ...] = ()


def load_surrogate_status(path: Path) -> SurrogateStatus:
    if not path.is_file():
        return SurrogateStatus("missing")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        metrics = tuple(
            (
                target,
                float(values["neural_mae"]),
                float(values["linear_mae"]),
                _optional_float(values.get("neural_rmse")),
                _optional_float(values.get("linear_rmse")),
                _optional_float(values.get("neural_r2")),
                _optional_float(values.get("linear_r2")),
            )
            for target, values in raw["metrics"].items()
        )
        if not metrics or not all(
            math.isfinite(value)
            for metric in metrics
            for value in metric[1:]
            if value is not None
        ):
            raise ValueError("метрики должны быть конечными числами")
        by_target = {metric[0]: metric[1:3] for metric in metrics}
        quality_passed = all(
            by_target[target][0] < by_target[target][1] for target in QUALITY_TARGETS
        )
        training = raw.get("training", {})
        datasets = tuple(
            (str(dataset["path"]), str(dataset["sha256"]))
            for dataset in training.get("datasets", [])
        )
        return SurrogateStatus(
            state="ready",
            train_scenarios=len(raw["train_scenarios"]),
            test_scenarios=len(raw["test_scenarios"]),
            wells=int(raw["wells"]),
            rows=int(raw["rows"]),
            metrics=metrics,
            quality_passed=quality_passed,
            seed=int(training["seed"]) if "seed" in training else None,
            sklearn_version=(
                str(training["sklearn_version"]) if "sklearn_version" in training else None
            ),
            pandas_version=(
                str(training["pandas_version"]) if "pandas_version" in training else None
            ),
            datasets=datasets,
        )
    except (KeyError, OSError, TypeError, ValueError, UnicodeError, json.JSONDecodeError):
        return SurrogateStatus("invalid")


def create_app(root: Path | None = None) -> FastAPI:
    """Собрать приложение. `root` — корень проекта (для `runs/` и `config.toml`)."""
    project_root = root or Path(__file__).resolve().parent.parent.parent
    application = FastAPI(title="AIOS · трек 2 · Модель Z")

    def runs() -> list[Run]:
        directory = project_root / "runs"
        if not directory.is_dir():
            return []
        return [Run(item.name, item) for item in sorted(directory.iterdir()) if item.is_dir()]

    def settings():
        return config_module.load(project_root / "config.toml")

    @application.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "runs": runs(),
                "settings": settings(),
                "surrogate": load_surrogate_status(
                    project_root / "artifacts" / "surrogate-metrics.json"
                ),
            },
        )

    @application.get("/run/{name}", response_class=HTMLResponse)
    def run_page(request: Request, name: str) -> HTMLResponse:
        run = _find(runs(), name)
        rows = load_rows(run, settings().fluids)
        if not rows:
            raise HTTPException(404, f"В прогоне «{name}» нет результатов расчёта")

        configuration = settings()
        result = economics.compute_npv(
            rows,
            headers=economics.REQUIRED_COLUMNS,
            start_date=configuration.economics.start_date.isoformat(),
        )
        assumptions, _ = economics.load_norms(configuration.economics.norms_file)
        verdicts = screen(rows, Norms.from_assumptions(assumptions))
        years = [str(year["year"]) for year in result.annual]
        water = balance_module.by_year(rows, configuration.fluids)

        return TEMPLATES.TemplateResponse(
            request=request,
            name="run.html",
            context={
                "run": run,
                "result": result,
                "verdicts": verdicts[:25],
                "problems": summary.sanity_problems(rows)[:25],
                "water": water,
                "water_problems": balance_module.problems(rows, configuration.fluids)[:10],
                "chart_wor": line_chart(
                    [str(year.year) for year in water],
                    [
                        Series("ВНФ, м³/м³", [year.water_oil_ratio for year in water]),
                        Series("компенсация", [year.compensation for year in water]),
                    ],
                ),
                "settings": configuration,
                "neural_available": _neural_available(project_root),
                "chart_production": line_chart(
                    years,
                    [
                        Series("нефть, тыс. т", [year["oilKt"] for year in result.annual]),
                        Series("жидкость, тыс. т", [year["liquidKt"] for year in result.annual]),
                        Series("закачка, тыс. м³", [year["injectionKm3"] for year in result.annual]),
                    ],
                ),
                "chart_chdd": bar_chart(
                    years, [year["chddM"] for year in result.annual], unit=""
                ),
                "chart_wells": bar_chart(
                    [verdict.well for verdict in verdicts[:25]],
                    [verdict.total_margin_m for verdict in verdicts[:25]],
                ),
            },
        )

    @application.get("/compare", response_class=HTMLResponse)
    def compare_page(request: Request, a: str, b: str) -> HTMLResponse:
        """Два сценария рядом: где именно разошлись деньги."""
        configuration = settings()
        assumptions, _ = economics.load_norms(configuration.economics.norms_file)
        norms = Norms.from_assumptions(assumptions)

        left, right = _find(runs(), a), _find(runs(), b)
        left_rows = load_rows(left, configuration.fluids)
        right_rows = load_rows(right, configuration.fluids)
        if not left_rows or not right_rows:
            raise HTTPException(404, "У одного из прогонов нет результатов расчёта")

        start = configuration.economics.start_date.isoformat()
        left_npv = economics.compute_npv(left_rows, headers=economics.REQUIRED_COLUMNS,
                                         start_date=start)
        right_npv = economics.compute_npv(right_rows, headers=economics.REQUIRED_COLUMNS,
                                          start_date=start)

        left_annual = {year["year"]: year for year in left_npv.annual}
        right_annual = {year["year"]: year for year in right_npv.annual}
        years = sorted(set(left_annual) | set(right_annual))

        left_wells = screen(left_rows, norms)
        right_wells = {verdict.well: verdict for verdict in screen(right_rows, norms)}
        deltas = sorted(
            (
                (
                    verdict.well,
                    verdict.total_margin_m,
                    right_wells[verdict.well].total_margin_m if verdict.well in right_wells else 0.0,
                )
                for verdict in left_wells
            ),
            key=lambda item: -abs(item[2] - item[1]),
        )[:20]

        return TEMPLATES.TemplateResponse(
            request=request,
            name="compare.html",
            context={
                "left": left, "right": right,
                "left_npv": left_npv, "right_npv": right_npv,
                "deltas": deltas,
                "settings": configuration,
                "chart_chdd": line_chart(
                    [str(year) for year in years],
                    [
                        Series(left.name, [left_annual.get(y, {}).get("chddM", 0.0) for y in years]),
                        Series(right.name, [right_annual.get(y, {}).get("chddM", 0.0) for y in years]),
                    ],
                ),
                "chart_oil": line_chart(
                    [str(year) for year in years],
                    [
                        Series(left.name, [left_annual.get(y, {}).get("oilKt", 0.0) for y in years]),
                        Series(right.name, [right_annual.get(y, {}).get("oilKt", 0.0) for y in years]),
                    ],
                ),
                "chart_delta": bar_chart(
                    [well for well, _, _ in deltas],
                    [after - before for _, before, after in deltas],
                ),
            },
        )

    @application.post("/run/{name}/cycle", response_class=HTMLResponse)
    def cycle_page(request: Request, name: str, model: str = "decline") -> HTMLResponse:
        run = _find(runs(), name)
        configuration = settings()
        rows = load_rows(run, configuration.fluids)
        if not rows:
            raise HTTPException(404, f"В прогоне «{name}» нет результатов расчёта")
        if not run.schedule.is_file():
            raise HTTPException(404, f"В прогоне «{name}» нет wells_schedule.inc")

        assumptions, _ = economics.load_norms(configuration.economics.norms_file)
        base_steps = _steps_from(run.schedule, configuration)
        # SCHEDULE ищется по составу дека, а не по имени файла: на кейсе
        # организаторов он будет называться иначе (ТЗ п. 4.3)
        try:
            reference_schedule = deckfiles.layout(configuration.case.model_dir).one("SCHEDULE")
        except (ValueError, OSError):
            reference_schedule = run.schedule
        reference_steps = _steps_from(
            reference_schedule if reference_schedule.is_file() else run.schedule,
            configuration,
        )
        if model not in {"decline", "neural"}:
            raise HTTPException(400, "неизвестный тип суррогата")
        try:
            selected = for_cycle(
                neural_path=(
                    project_root / "artifacts" / "surrogate.joblib"
                    if model == "neural" else None
                ),
                history=rows,
                reference_steps=reference_steps,
                fluids=configuration.fluids,
            )
        except SurrogateLoadError as error:
            raise HTTPException(409, str(error)) from error

        result = loop.run_cycle(
            selected.model,
            base_steps,
            Norms.from_assumptions(assumptions),
            control_start=configuration.case.control_start,
            start_date=configuration.economics.start_date,
            lookback_months=configuration.policy.lookback_months,
            joint_rate_search=selected.joint_rate_search,
            limits=configuration.limits,
            fluids=configuration.fluids,
        )
        return TEMPLATES.TemplateResponse(
            request=request,
            name="cycle.html",
            context={
                "run": run,
                "result": result,
                "settings": configuration,
                "surrogate_name": selected.name,
            },
        )

    return application


def load_rows(run: Run, fluids: summary.Fluids) -> list[dict[str, Any]]:
    """Строки прогона: сначала `rows.csv`, иначе `.UNSMRY` симулятора."""
    if run.rows_csv.is_file():
        return read_rows_csv(run.rows_csv)
    case = run.summary_case
    if case is not None:
        return summary.monthly_rows(case, fluids=fluids)
    return []


def read_rows_csv(path: Path) -> list[dict[str, Any]]:
    """Прочитать выгрузку, приведя числовые колонки к float.

    Расчётчик и сам разбирает строки, но диагностика и графики считают
    арифметику, и строка «0.0» в сравнении с нулём ведёт себя не как число.
    """
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for column in NUMERIC_COLUMNS:
                row[column] = float(row.get(column) or 0.0)
            rows.append(row)
    return rows


def _steps_from(schedule_file: Path, configuration) -> list[ScheduleStep]:
    """Шаги управления начиная с даты, с которой мы управляем фондом."""
    parsed = parse_schedule(schedule_file)
    return [step for step in parsed.steps if step.date >= configuration.case.control_start]


def _find(items: list[Run], name: str) -> Run:
    for item in items:
        if item.name == name:
            return item
    raise HTTPException(404, f"Прогон «{name}» не найден")


def _neural_available(project_root: Path) -> bool:
    status = load_surrogate_status(project_root / "artifacts" / "surrogate-metrics.json")
    return (
        status.state == "ready"
        and status.quality_passed
        and (project_root / "artifacts" / "surrogate.joblib").is_file()
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


app = create_app()
