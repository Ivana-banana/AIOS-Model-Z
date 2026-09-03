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
from aios import deckfiles, economics, loop, pvt, summary
from aios import explain as explain_module
from aios.deck import ScheduleStep, parse_schedule
from aios.margin import Norms, screen
from aios.model import SurrogateLoadError, for_cycle
from aios.run import build_case
from aios.surrogate import fold_warnings
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
    metrics: tuple[dict[str, Any], ...] = ()
    """По строке на выход. Ключи именованные, а не позиции в кортеже.

    Кортеж пришлось развернуть 22.08: метрик стало одиннадцать, и распаковка
    `for target, neural, linear, ... in` в шаблоне перестала читаться — а это тот
    самый код, где перепутанные местами два столбца никто не заметит.
    """
    quality_passed: bool = False
    seed: int | None = None
    sklearn_version: str | None = None
    pandas_version: str | None = None
    datasets: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class CycleOptions:
    """Чем именно запускали цикл. Всё остальное берётся из `config.toml`.

    Вынесены наружу ровно те три параметра, которые задают режим работы МАС, а
    не экономику: нормативы, лимиты объекта и границы дисконтирования остаются в
    конфигурации, иначе число из интерфейса нельзя было бы повторить командой.
    """

    model: str
    """`decline` — тривиальный суррогат, `neural` — обученная модель."""

    tune_rates: bool
    """Подбор режимов. Выключенный оставляет только остановку убыточных."""

    lookback_months: int
    """Окно, по которому ставится диагноз скважине."""

    @property
    def mode(self) -> str:
        return "full" if self.tune_rates else "stops"

    @property
    def query(self) -> str:
        """Параметры в виде строки запроса: адрес страницы повторяет запуск."""
        return f"model={self.model}&mode={self.mode}&lookback={self.lookback_months}"


@dataclass(frozen=True)
class CycleRun:
    """Результат прохода цикла вместе с тем, что понадобится для сборки кейса."""

    result: loop.CycleResult
    surrogate_name: str
    deck_wells: tuple[str, ...] | None
    """Фонд из дека организаторов. `None` — дек недоступен, кейс не собрать."""


@dataclass(frozen=True)
class TrustStatus:
    """Что интерфейс говорит про меру доверия к суррогату.

    Разбор 4-7 требует этого отдельным пунктом: если хоть один обязательный блок
    артефакта доверия отсутствует — нет калибровочных сценариев, нет значения
    полного расчёта, — интерфейс обязан показать **«мера доверия не калибрована;
    кандидат обязателен к полному расчёту»**, а не нейтральный зелёный статус.
    Молчание здесь читается как «всё в порядке», а это неправда.
    """

    state: str
    """`missing`, `invalid`, `uncalibrated` или `calibrated`."""

    scenarios: int = 0
    threshold: float | None = None
    samples: int = 0
    margin: float | None = None
    reason: str = ""
    scope: str = ""
    audit: dict[str, Any] | None = None

    @property
    def verdict(self) -> str:
        if self.state == "calibrated":
            return "калибрована"
        if self.state == "uncalibrated":
            return "не калибрована; кандидат обязателен к полному расчёту"
        if self.state == "invalid":
            return "артефакт доверия повреждён"
        return "не собрана"


def load_trust_status(path: Path) -> TrustStatus:
    """Прочитать `artifacts/trust-gate.json`. Отсутствие файла — не ошибка, а состояние."""
    if not path.is_file():
        return TrustStatus("missing")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        calibrated = bool(raw.get("calibrated"))
        return TrustStatus(
            state="calibrated" if calibrated else "uncalibrated",
            scenarios=len(raw.get("training_scenario_ids", [])),
            threshold=_optional_float(raw.get("threshold_tau")),
            samples=int(raw.get("conformal_samples") or 0),
            margin=_optional_float(raw.get("optimism_margin")),
            reason=str(raw.get("margin_reason") or ""),
            scope=str(raw.get("guarantee_scope") or ""),
            audit=raw.get("shift_audit"),
        )
    except (ValueError, TypeError):
        return TrustStatus("invalid")


def _passes_response(metric: dict[str, Any]) -> bool:
    """Проходит ли выход порог по **отклику**, а не по цели.

    Интерфейс до 22.08 показывал «лучше/хуже» по одному MAE и тем самым говорил,
    что модель годна, когда порог допуска её не пропускает. Здесь та же логика,
    что в `aios.neural.assert_quality`: положительный `R²` отклика **и**
    положительная медиана `Skill_MAE` по сценариям. `None` — «не измерено», и за
    это отвергать нечего: артефакты до 22.08 этих полей не несут.
    """
    response = metric.get("neural_response_r2")
    median = metric.get("neural_skill_median")
    return (response is None or response > 0) and (median is None or median > 0)


def load_surrogate_status(path: Path) -> SurrogateStatus:
    if not path.is_file():
        return SurrogateStatus("missing")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        metrics = tuple(
            {
                "target": target,
                "neural_mae": float(values["neural_mae"]),
                "linear_mae": float(values["linear_mae"]),
                "neural_r2": _optional_float(values.get("neural_r2")),
                # уровень и отклик порознь: в `neural_r2` 91 % занимает
                # тривиальный уровень скважины, и таблица из одного `R²`
                # показывала отличную модель там, где управление не предсказано
                "level_r2": _optional_float(values.get("level_r2")),
                "neural_response_r2": _optional_float(values.get("neural_response_r2")),
                "slope_response_r2": _optional_float(values.get("slope_response_r2")),
                "neural_skill": _optional_float(values.get("neural_skill")),
                "neural_skill_median": _optional_float(values.get("neural_skill_median")),
                "neural_skill_iqr": _optional_float(values.get("neural_skill_iqr")),
                "linear_skill_median": _optional_float(values.get("linear_skill_median")),
            }
            for target, values in raw["metrics"].items()
        )
        if not metrics or not all(
            math.isfinite(value)
            for metric in metrics
            for key, value in metric.items()
            if key != "target" and value is not None
        ):
            raise ValueError("метрики должны быть конечными числами")
        by_target = {metric["target"]: metric for metric in metrics}
        quality_passed = all(
            by_target[target]["neural_mae"] < by_target[target]["linear_mae"]
            and _passes_response(by_target[target])
            for target in QUALITY_TARGETS
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

    deck_fluids_cache: dict[str, pvt.DeckFluids] = {}

    def deck_fluids() -> pvt.DeckFluids:
        """Плотности по PVT-регионам скважин. Разбор дека кэшируется на процесс.

        Без них весь фонд считался бы по одной плотности из конфигурации — на
        Модели Z это совпадает, на чужом пласте нет (ТЗ п. 4.3).
        """
        key = str(settings().case.model_dir)
        if key not in deck_fluids_cache:
            deck_fluids_cache[key] = pvt.fluids_by_well(settings().case.model_dir)
        return deck_fluids_cache[key]

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
                "trust": load_trust_status(project_root / "artifacts" / "trust-gate.json"),
            },
        )

    @application.get("/run/{name}", response_class=HTMLResponse)
    def run_page(request: Request, name: str) -> HTMLResponse:
        run = _find(runs(), name)
        rows = load_rows(run, settings().fluids, deck_fluids().by_well)
        if not rows:
            raise HTTPException(404, f"В прогоне «{name}» нет результатов расчёта")

        configuration = settings()
        result = economics.compute_npv(
            rows,
            headers=economics.REQUIRED_COLUMNS,
            start_date=configuration.economics.start_date.isoformat(),
            norms_file=configuration.economics.norms_file,
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
                # откуда взяты плотности — рядом с числом ЧДД, а не в логе:
                # общая плотность вместо подеквенной меняет выручку молча
                "deck_fluids": deck_fluids(),
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
        left_rows = load_rows(left, configuration.fluids, deck_fluids().by_well)
        right_rows = load_rows(right, configuration.fluids, deck_fluids().by_well)
        if not left_rows or not right_rows:
            raise HTTPException(404, "У одного из прогонов нет результатов расчёта")

        start = configuration.economics.start_date.isoformat()
        left_npv = economics.compute_npv(left_rows, headers=economics.REQUIRED_COLUMNS,
                                         start_date=start,
                                         norms_file=configuration.economics.norms_file)
        right_npv = economics.compute_npv(right_rows, headers=economics.REQUIRED_COLUMNS,
                                          start_date=start,
                                          norms_file=configuration.economics.norms_file)

        left_annual = {year["year"]: year for year in left_npv.annual}
        right_annual = {year["year"]: year for year in right_npv.annual}
        years = sorted(set(left_annual) | set(right_annual))

        deltas = margin_deltas(left_rows, right_rows, norms)

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

    @application.get("/run/{name}/cycle", response_class=HTMLResponse)
    def cycle_page(request: Request, name: str, model: str = "decline",
                   mode: str = "full", lookback: int | None = None) -> HTMLResponse:
        """Прогнать цикл МАС и показать решения.

        Запрос ничего не пишет на диск, поэтому он GET, а параметры запуска
        живут в адресе: ссылка повторяет прогон целиком — вместе с режимом и
        окном диагноза, а не только с именем прогона.
        """
        run = _find(runs(), name)
        configuration = settings()
        options = _cycle_options(model, mode, lookback, configuration)
        cycle = run_cycle_on(run, configuration, options)

        # слой интерпретируемости: модель излагает уже принятое решение по
        # посчитанным фактам. Отказ слоя не должен ронять страницу — при любой
        # проблеме возвращается шаблонное объяснение (ТЗ п. 3.1)
        try:
            explanation = explain_module.explain(
                explain_module.evidence_from_cycle(
                    cycle.result, cycle.surrogate_name,
                    limits=configuration.limits, fluids=configuration.fluids),
                configuration.explain,
            )
        except Exception:  # noqa: BLE001 — объяснение не должно уносить результат цикла
            explanation = None
        return TEMPLATES.TemplateResponse(
            request=request,
            name="cycle.html",
            context={
                "run": run,
                "result": cycle.result,
                "settings": configuration,
                "surrogate_name": cycle.surrogate_name,
                "warning_groups": fold_warnings(cycle.result.warnings),
                "explanation": explanation,
                "options": options,
                "case_dir": _case_directory(run.name, options),
                "case_ready": cycle.deck_wells is not None and not cycle.result.problems,
                "neural_available": _neural_available(project_root),
            },
        )

    @application.post("/run/{name}/cycle/save", response_class=HTMLResponse)
    def save_cycle(request: Request, name: str, model: str = "decline",
                   mode: str = "full", lookback: int | None = None) -> HTMLResponse:
        """Собрать кейс с расписанием, которое предложил цикл.

        Цикл считается заново теми же параметрами: он детерминирован, а держать
        результат предыдущего запроса между двумя обращениями значило бы завести
        в UI состояние, которого больше нигде в проекте нет.

        ЧДД собранного кейса остаётся прогнозом суррогата, пока кейс не посчитан
        на ГГДМ, — команда полного расчёта печатается на странице.
        """
        run = _find(runs(), name)
        configuration = settings()
        options = _cycle_options(model, mode, lookback, configuration)
        cycle = run_cycle_on(run, configuration, options)

        # три отказа сборки различаются кодом, а не только текстом: страницу на
        # защите трогают руками, и «почему не собралось» должно читаться из ответа
        if cycle.deck_wells is None:
            raise HTTPException(
                503,   # дека нет в окружении: повторять тот же запрос бессмысленно
                "дек модели недоступен: кейс собирается поверх дека организаторов, "
                f"а по пути {configuration.case.model_dir} его нет",
            )
        if cycle.result.problems:
            # то же правило, что в scripts/run_cycle.py: расписание с нарушениями
            # до файла не доходит, иначе нарушение уедет в сдаваемый пакет
            raise HTTPException(
                422,   # запрос понят, но расписание непригодно
                "расписание нарушает ограничения, кейс не собран: "
                + "; ".join(str(problem) for problem in cycle.result.problems[:3]),
            )

        target = project_root / "runs" / _case_directory(run.name, options)
        if target.exists():
            raise HTTPException(
                409,   # конфликт с уже существующим прогоном — единственный настоящий 409
                f"каталог runs/{target.name} уже есть. Прогон не перезаписывается: "
                "удалите или переименуйте его, если нужно пересобрать",
            )
        try:
            case = build_case(
                target,
                configuration.case.model_dir,
                steps=list(cycle.result.steps),
                control_start=configuration.case.control_start,
                wells=cycle.deck_wells,
                case_name=configuration.case.case_name,
            )
        except (ValueError, OSError) as error:
            # сборка сорвалась на нашей стороне: каталог свободен, запрос верен
            raise HTTPException(500, f"кейс собрать не удалось: {error}") from error

        # Подпись плотностей едет вместе с каталогом кейса: rows.csv здесь не
        # появится, его пишет export_rows.py уже по результату полного расчёта,
        # а кейс уедет считаться на другую машину (Р-47, хвост).
        deck_fluids().write_case_note(
            target,
            fluids=configuration.fluids,
            model_dir=configuration.case.model_dir,
            forecast_fluids=configuration.fluids,
        )

        data_file = case.data_file.resolve()
        return TEMPLATES.TemplateResponse(
            request=request,
            name="saved.html",
            context={
                "run": run,
                "result": cycle.result,
                "settings": configuration,
                "surrogate_name": cycle.surrogate_name,
                "options": options,
                "case": case,
                "case_dir": target.name,
                "data_path": (
                    data_file.relative_to(project_root)
                    if data_file.is_relative_to(project_root) else data_file
                ).as_posix(),
                "processes": configuration.run.processes,
            },
        )

    def run_cycle_on(run: Run, configuration, options: CycleOptions) -> CycleRun:
        """Пройти цикл МАС на прогоне — общее для показа и для сборки кейса."""
        rows = load_rows(run, configuration.fluids, deck_fluids().by_well)
        if not rows:
            raise HTTPException(404, f"В прогоне «{run.name}» нет результатов расчёта")
        if not run.schedule.is_file():
            raise HTTPException(404, f"В прогоне «{run.name}» нет wells_schedule.inc")

        assumptions, _ = economics.load_norms(configuration.economics.norms_file)
        base_steps = _steps_from(run.schedule, configuration)
        # SCHEDULE ищется по составу дека, а не по имени файла: на кейсе
        # организаторов он будет называться иначе (ТЗ п. 4.3)
        try:
            reference_schedule = deckfiles.layout(configuration.case.model_dir).one("SCHEDULE")
        except (ValueError, OSError):
            reference_schedule = None

        deck_wells: tuple[str, ...] | None = None
        if reference_schedule is not None and reference_schedule.is_file():
            parsed = parse_schedule(reference_schedule)
            reference_steps = [
                step for step in parsed.steps if step.date >= configuration.case.control_start
            ]
            deck_wells = parsed.wells
        else:
            reference_steps = _steps_from(run.schedule, configuration)

        try:
            selected = for_cycle(
                neural_path=(
                    project_root / "artifacts" / "surrogate.joblib"
                    if options.model == "neural" else None
                ),
                history=rows,
                reference_steps=reference_steps,
                fluids=configuration.fluids,
            )
        except SurrogateLoadError as error:
            raise HTTPException(409, str(error)) from error

        # Совместный поиск режимов на обученной модели считает полный ЧДД фонда
        # на каждый рассматриваемый множитель: 24.08 на машине Б это заняло
        # 1 ч 20 мин из командной строки. В веб-запросе такой расчёт означает
        # зависший браузер посреди доклада, поэтому он отвергается заранее и
        # называет команду, которой запускается (Б33).
        #
        # Молча подменить режим на «только остановки» нельзя: пользователь
        # просил одно, система сделала бы другое и не сказала — тот же класс,
        # что подстановка другого суррогата в подтверждение (Р-100).
        if options.tune_rates and selected.joint_rate_search:
            raise HTTPException(
                409,
                "совместный подбор режимов на обученной модели идёт больше часа — "
                "столько веб-запрос не живёт. Запустите его командой:\n"
                f"  python scripts/run_cycle.py --from runs/{run.name} "
                "--out runs/<куда> --surrogate-model artifacts/surrogate.joblib\n"
                "а на странице оставьте «только остановка убыточных» — она идёт "
                "секунды и показывает тот же слой решений.",
            )

        result = loop.run_cycle(
            selected.model,
            base_steps,
            Norms.from_assumptions(assumptions),
            control_start=configuration.case.control_start,
            start_date=configuration.economics.start_date,
            lookback_months=options.lookback_months,
            # фонд берётся из дека: расписание со скважиной, которой в модели
            # нет, симулятор не примет. Дека нет — сверять не с чем
            known_wells=frozenset(deck_wells) if deck_wells is not None else None,
            tune_rates=options.tune_rates,
            joint_rate_search=selected.joint_rate_search,
            limits=configuration.limits,
            norms_file=configuration.economics.norms_file,
            fluids=configuration.fluids,
        )
        return CycleRun(result=result, surrogate_name=selected.name, deck_wells=deck_wells)

    return application



def margin_deltas(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]],
                  norms: Norms, limit: int = 20) -> list[tuple[str, float | None, float | None]]:
    """Скважины, на которых разошлись деньги двух сценариев.

    Перебирается **объединение** фондов, а не фонд левого прогона. До 21.08
    считалось иначе, и страница с заголовком «где именно разошлись деньги» врала
    в обе стороны сразу: скважина, которой нет справа, показывалась как
    «справа +0.00, дельта −1.08» — то есть будто потеряла все деньги, — а
    скважина, которой нет слева, не показывалась вовсе.

    На двух прогонах одного дека фонд совпадает и разницы не видно. Она видна
    там, где сравнение и нужно: наш прогон против прогона на кейсе организаторов,
    где фонд другой.

    `None` вместо числа означает «в этом прогоне скважины нет». Ноль сюда не
    годится: ноль — это работающая скважина, не заработавшая ничего.
    """
    left = {verdict.well: verdict.total_margin_m for verdict in screen(left_rows, norms)}
    right = {verdict.well: verdict.total_margin_m for verdict in screen(right_rows, norms)}
    return sorted(
        ((well, left.get(well), right.get(well)) for well in set(left) | set(right)),
        key=lambda item: -abs((item[2] or 0.0) - (item[1] or 0.0)),
    )[:limit]

def load_rows(run: Run, fluids: summary.Fluids,
              well_fluids: dict[str, summary.Fluids] | None = None) -> list[dict[str, Any]]:
    """Строки прогона: сначала `rows.csv`, иначе `.UNSMRY` симулятора.

    `well_fluids` — плотности по PVT-регионам скважин (`aios.pvt`). Без них
    весь фонд считался бы по одной плотности из конфигурации, что на чужом
    пласте просто неверно (ТЗ п. 4.3).
    """
    if run.rows_csv.is_file():
        try:
            return read_rows_csv(run.rows_csv)
        except ValueError as error:
            raise HTTPException(
                422,
                f"выгрузка {run.rows_csv.name} не соответствует контракту расчётчика: {error}",
            ) from error
    case = run.summary_case
    if case is not None:
        return summary.monthly_rows(case, fluids=fluids, well_fluids=well_fluids)
    return []


def read_rows_csv(path: Path) -> list[dict[str, Any]]:
    """Прочитать выгрузку, приведя числовые колонки к float.

    Расчётчик и сам разбирает строки, но диагностика и графики считают
    арифметику, и строка «0.0» в сравнении с нулём ведёт себя не как число.
    Заголовок проверяется целиком: отсутствующая колонка — не нулевой показатель,
    а неполная выгрузка, которой нельзя доверять ЧДД.
    """
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        missing = [column for column in economics.REQUIRED_COLUMNS if column not in headers]
        duplicate = [column for column in economics.REQUIRED_COLUMNS if headers.count(column) > 1]
        if missing or duplicate:
            details = []
            if missing:
                details.append("нет колонок " + ", ".join(missing))
            if duplicate:
                details.append("колонки повторены " + ", ".join(duplicate))
            raise ValueError("; ".join(details))

        rows = []
        for raw in reader:
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


def _cycle_options(model: str, mode: str, lookback: int | None, configuration) -> CycleOptions:
    """Разобрать параметры запуска цикла, отвергнув то, чего мы не умеем.

    Молча подставлять умолчание вместо непонятого значения нельзя: страница
    показала бы прогон не тем, чем его просили запустить, и разница осталась бы
    только в адресе.
    """
    if model not in {"decline", "neural"}:
        raise HTTPException(400, "неизвестный тип суррогата")
    if mode not in {"full", "stops"}:
        raise HTTPException(400, "неизвестный режим цикла: бывает full или stops")
    months = configuration.policy.lookback_months if lookback is None else lookback
    if not 1 <= months <= 60:
        raise HTTPException(400, "окно диагноза задаётся в месяцах, от 1 до 60")
    return CycleOptions(model=model, tune_rates=mode == "full", lookback_months=months)


def _case_directory(run_name: str, options: CycleOptions) -> str:
    """Имя каталога под кейс цикла — производная от прогона и режима запуска.

    Имя собирается сервером, а не приходит из формы: путь на диске из
    пользовательского ввода — это запись куда угодно. Режим в имени нужен,
    чтобы два запуска на одном прогоне не претендовали на один каталог.
    """
    parts = [run_name, "mas", options.model]
    if not options.tune_rates:
        parts.append("stops")
    return "-".join(parts)


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
