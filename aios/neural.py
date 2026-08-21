"""Обучение нейросетевого суррогата на сценариях DoE.

Разбиение выполняется по целым сценариям, а не по строкам: соседние месяцы одного
прогона слишком похожи, и их попадание одновременно в train и test завышает
качество. Для каждой скважины обучается небольшой MLP, который видит время и
совместное управление всем добывающим фондом.

Прогноз разделён на **уровень** и **отклик** (`Р-53`):

    predict = level(скважина, месяц) + response(управление)

Уровень — среднее обучающих сценариев для пары «скважина × месяц»; он про
управление не знает вовсе и на выводе является константой. Отклик — разность
прогноза модели в запрошенной точке и в центре обучающих управлений, поэтому в
самом центре он равен нулю точно. Учится модель на остатке, и метрики считаются
на обеих величинах порознь: замер 21.08 на объединении четырёх пакетов показал,
что на уровень приходится 92 % дисперсии целей, а оптимизатор ходит по
оставшимся 7–8 % — по той части, качество которой до 21.08 не измерялось ни
разу.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aios.deck import InjectorControl, ProducerControl, ScheduleStep
from aios.summary import Fluids, sanity_problems
from aios.surrogate import TrainingEnvelope, expand

#: Чем кодируется «режим назначен там, где опорного режима не было» — запуск
#: скважины из простоя и перевод под закачку. Значение выбрано отрицательным
#: намеренно: множители по построению неотрицательны, ноль означает буквально
#: «скважина остановлена» (`aios/doe.py`, `Scenario.shut_wells`), и до 19.08 запуск
#: кодировался тем же нулём — то есть предложение запустить подавалось модели как
#: остановка. Отрицательное значение спутать не с чем.
#:
#: Числом это решение не становится: обучающая выборка запусков не содержит вовсе,
#: поэтому вход заведомо вне обученной области. Ровно поэтому кодирование идёт в
#: паре с обязательной пометкой такого месяца в `NeuralSurrogate.in_training_range`
#: — принцип 6 `CLAUDE.md`: сеть не падает, а тихо врёт, и решение обязано уйти на
#: полный расчёт, а не быть принятым по прогнозу.
RESTART_MULTIPLIER = -1.0

TARGET_COLUMNS = ("WLPT_Diff", "WOMT_Diff", "WWIT_Diff", "THP", "BHP")

#: Сколько первых признаков описывают время, а не управление: номер месяца и две
#: гармоники сезона. Всё, что дальше, — управляющие множители по `control_wells`.
TIME_FEATURES = 3
REQUIRED_COLUMNS = frozenset(
    {"scenario", "DATA", "well", "multiplier", "WLPR", "WWIR", *TARGET_COLUMNS}
)


@dataclass(frozen=True)
class TargetMetric:
    """Ошибка на отложенных сценариях: по уровню и по отклику порознь.

    `neural_r2` / `linear_r2` — `R²` по самой цели. В нём 92 % занимает уровень
    скважины, поэтому число получается высоким у модели, которая про управление
    не знает вовсе. Старые поля оставлены намеренно: в них написаны отчёт и
    `docs/experiments.md`, и без них прогоны разных дней перестанут сравниваться.

    `level_r2` — тот самый тривиальный предсказатель «среднее по (скважина,
    месяц) поперёк сценариев». Это нижняя отметка, ниже которой обученная модель
    не имеет права опускаться, а не достижение.

    `*_response_r2` — `R²` на остатке `y − level`, то есть ровно на той части,
    по градиенту которой ходит оптимизатор. `slope_response_r2` — тупой бейзлайн
    «у каждой скважины свой наклон на собственный множитель»; он планка
    (принцип 3 `CLAUDE.md`), а не справка.
    """

    neural_mae: float
    linear_mae: float
    neural_rmse: float | None = None
    linear_rmse: float | None = None
    neural_r2: float | None = None
    linear_r2: float | None = None
    level_r2: float | None = None
    neural_response_r2: float | None = None
    linear_response_r2: float | None = None
    slope_response_r2: float | None = None


@dataclass(frozen=True)
class TrainingReport:
    train_scenarios: tuple[str, ...]
    test_scenarios: tuple[str, ...]
    wells: int
    rows: int
    metrics: dict[str, TargetMetric]

    targets: str = "raw"
    """Как модель представляет нефть: `raw` — отдельной величиной, `watercut` —
    через долю воды, а нефть восстанавливается как `жидкость × (1 − доля)`.

    Прямое предсказание нефти ничем не связано с предсказанием жидкости, поэтому
    прогноз умеет нарушать физику: 20.08 на настоящем расписании 33 пометки из
    34 были «нефти больше, чем жидкости». Через обводнённость это невозможно по
    построению, а не ловится постфактум.
    """

    metric_rows: int = 0
    """Сколько строк отложенной выборки реально участвовало в метриках.

    Меньше, чем `rows`, когда часть выборки — история до начала управления:
    она одинакова у всех сценариев и по входу, и по отклику, поэтому мерить по
    ней значит мерить память, а не обобщение (ревизия 20.08, `d9d9750`).
    """

    metric_start: str = ""
    """С какой даты считались метрики; пусто — считались по всей выборке."""

    wells_without_model: tuple[str, ...] = ()
    """Скважины фонда, которым модели не досталось.

    Пропуск случается в двух местах `train`: у скважины нет строк по одну из
    сторон разбиения, либо все её строки лежат раньше `control_start`. Дальше она
    исчезает бесшумно — `predict` идёт по `models`, поэтому прогноза по ней нет
    ни одной строки, ЧДД считается по остатку фонда, а поиск режимов про неё не
    знает. Список нужен, чтобы это было видно, а не выводилось сравнением.
    """

    kind: str = "mlp"
    """Что именно сохранено в модели: `mlp` или `linear`.

    Линейный бейзлайн обучается всегда — им меряется сеть. До 19.08 он при этом
    выбрасывался: если MLP не проходил порог, в цикл ехал тривиальный суррогат,
    хотя измеренно лучшая модель на прогонах ГГДМ уже была обучена. Теперь её
    можно сохранить и отвечать за неё числом (`Р-16`).
    """


@dataclass
class NeuralBundle:
    """Обученные модели и точный порядок входных управляющих признаков."""

    control_wells: tuple[str, ...]
    models: dict[str, Any]
    start_month: pd.Timestamp
    month_span: int
    envelope: TrainingEnvelope
    report: TrainingReport
    metadata: dict[str, Any] = field(default_factory=dict)

    levels: dict[str, np.ndarray] = field(default_factory=dict)
    """Уровень скважины по месяцам: `(month_span + 1) × len(TARGET_COLUMNS)`.

    Среднее обучающих сценариев для пары «скважина × месяц» в том же
    представлении целей, в каком учится модель (`targets`). Месяцы, которых у
    скважины в обучении не было, заполнены её средним по всему горизонту.

    Пустой словарь — артефакт, обученный до `Р-53`: у такого модель предсказывает
    уровень сама, и разделять в нём нечего.
    """

    control_center: np.ndarray | None = None
    """Центр обучающих управлений по месяцам: `(month_span + 1) × len(control_wells)`.

    Средний управляющий вектор обучающих сценариев. Это центр **выборки DoE**, а
    не история: для расписания, далёкого от обучающего облака, отклик считается
    от смещённой точки, и настолько же смещён.
    """


@dataclass(frozen=True)
class NeuralSurrogate:
    """Обученный MLP в контракте общего цикла МАС."""

    bundle: NeuralBundle
    reference_steps: tuple[ScheduleStep, ...]
    fluids: Fluids = field(default_factory=Fluids)

    @classmethod
    def create(cls, bundle: NeuralBundle,
               reference_steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
               fluids: Fluids | None = None, *, require_quality: bool = True) -> NeuralSurrogate:
        if require_quality:
            assert_quality(bundle.report)
        return cls(bundle, tuple(reference_steps), fluids or Fluids())

    def predict(self,
                steps: list[ScheduleStep] | tuple[ScheduleStep, ...]) -> list[dict[str, Any]]:
        timeline = expand(steps)
        if not timeline:
            return []
        multipliers = self._multipliers(steps, timeline)
        months = list(timeline)
        feature_matrix = np.vstack([
            _inference_features(
                month_start, multipliers[month_start], self.bundle.start_month,
                self.bundle.month_span
            )[0]
            for month_start in months
        ])
        predictions = self._level_and_response(months, feature_matrix)
        cumulative: dict[str, list[float]] = {}
        rows: list[dict[str, Any]] = []

        for month_index, (month_start, controls) in enumerate(timeline.items()):
            days = month_days(month_start)
            for well in sorted(self.bundle.models, key=_well_key):
                predicted = predictions[well][month_index]
                control = controls.get(well)
                liquid_rate, injection_rate = _rates(control)
                liquid, oil, injection, thp, bhp = (float(value) for value in predicted)
                if liquid_rate <= 0:
                    liquid = oil = 0.0
                if injection_rate <= 0:
                    injection = 0.0
                if liquid_rate <= 0 and injection_rate <= 0:
                    thp = bhp = 0.0

                totals = cumulative.setdefault(well, [0.0, 0.0, 0.0])
                totals[0] += liquid
                totals[1] += oil
                totals[2] += injection
                rows.append(_contract_row(
                    month_start, well, liquid_rate, injection_rate, totals,
                    liquid, oil, injection, thp, bhp, days
                ))
        return rows

    def _level_and_response(self, months: list[date],
                            feature_matrix: np.ndarray) -> dict[str, np.ndarray]:
        """Прогноз каждой скважины: уровень месяца плюс отклик на управление.

        Отклик берётся разностью прогноза модели в запрошенной точке и в центре
        обучающих управлений. В центре обе точки совпадают побайтово, поэтому
        отклик там равен нулю точно, а не приблизительно — иначе часть уровня
        продолжала бы жить в отклике, и `R²` отклика снова мерил бы уровень.

        Пустой `levels` — артефакт, обученный до `Р-53`: у него модель
        предсказывает цель целиком, и разделять нечего.
        """
        targets = self.bundle.report.targets
        levels = getattr(self.bundle, "levels", None)
        centre = getattr(self.bundle, "control_center", None)
        if not levels or centre is None:
            return {well: _raw_space(model.predict(feature_matrix), targets)
                    for well, model in self.bundle.models.items()}

        index = np.clip([self._month_number(month) for month in months],
                        0, self.bundle.month_span)
        base_matrix = np.vstack([
            _inference_features(month, centre[position], self.bundle.start_month,
                                self.bundle.month_span)[0]
            for month, position in zip(months, index, strict=True)
        ])
        return {
            well: _raw_space(
                levels[well][index] + model.predict(feature_matrix)
                - model.predict(base_matrix),
                targets,
            )
            for well, model in self.bundle.models.items()
        }

    def _month_number(self, month_start: date) -> int:
        return ((month_start.year - self.bundle.start_month.year) * 12
                + month_start.month - self.bundle.start_month.month)

    def _multipliers(
        self,
        steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
        timeline: dict[date, dict[str, ProducerControl | InjectorControl]],
    ) -> dict[date, np.ndarray]:
        """Управляющие признаки расписания — ровно то, что видит модель."""
        return _schedule_multipliers(
            timeline,
            expand(self.reference_steps),
            self.bundle.control_wells,
            _expand_role_controls(steps),
            _expand_role_controls(self.reference_steps),
        )

    def covers_control(self, role: str, well: str, multiplier: float,
                       steps: list[ScheduleStep] | tuple[ScheduleStep, ...] | None = None) -> bool:
        """Был ли такой множитель у этого управления в обучении — до выбора, а не после.

        Роль — `production` или `injection`; ключ признака собирается здесь,
        чтобы вызывающему не приходилось знать про префиксы `prod:`/`inj:`.
        Скважина, которой в обучающем наборе нет вовсе, покрытой не считается.

        `steps` — расписание, которое поиск масштабирует (`base_steps`). Оно нужно
        потому, что множитель поиска отсчитывается **от него**, а обучающие
        множители DoE — от опорного расписания дека, и это разные величины всякий
        раз, когда цикл идёт поверх уже форсированного прогона: в интерфейсе
        `base_steps` берётся из открытого прогона, а опорное — из дека. Без
        перевода фильтр отвечал «покрыто» там, где детектор на собранном
        расписании говорит «вне диапазона» — две защиты об одном решении говорили
        разное. Опущенный аргумент означает «множитель уже отсчитан от опорного».
        """
        prefix = "prod:" if role == "production" else "inj:"
        key = f"{prefix}{well}"
        if key not in self.bundle.control_wells and well in self.bundle.control_wells:
            key = well   # датасет старого вида: одна колонка множителя на скважину
        if steps is None:
            return self.bundle.envelope.covers(key, multiplier)

        span = self._scale_span(role, well, steps)
        if span is None:
            return False
        # Область применимости — интервал, поэтому проверять достаточно концы:
        # всё между ними покрыто вместе с ними
        return all(self.bundle.envelope.covers(key, multiplier * scale) for scale in span)

    def _scale_span(self, role: str, well: str,
                    steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
                    ) -> tuple[float, float] | None:
        """Во сколько раз режим `steps` отличается от опорного — минимум и максимум.

        Один множитель на весь горизонт не годится: с периодами (`aios/doe.py`)
        режим скважины меняется по ходу сценария, и отношение к опорному вместе с
        ним. `None` означает «сопоставить не с чем»: скважина работает там, где
        опорное расписание её не запускает вовсе, — это не масштабирование режима,
        а запуск из простоя, которого в обучении нет ни одного.
        """
        planned = _expand_role_controls(steps)
        reference = _expand_role_controls(self.reference_steps)
        scales: list[float] = []
        for month_start, controls in planned.items():
            rate = _role_rate(controls.get((role, well)), role)
            base = _role_rate(reference.get(month_start, {}).get((role, well)), role)
            if base > 0:
                scales.append(rate / base)
            elif rate > 0:
                return None
        if not scales:
            # скважина стоит весь горизонт: множитель ничего не меняет, и
            # запрещать поиску ноль не за что
            return (1.0, 1.0)
        return (min(scales), max(scales))

    def in_training_range(self,
                          steps: list[ScheduleStep] | tuple[ScheduleStep, ...]) -> list[str]:
        timeline = expand(steps)
        warnings = self.bundle.envelope.warnings({
            month_start: dict(zip(self.bundle.control_wells, values, strict=True))
            for month_start, values in self._multipliers(steps, timeline).items()
        }) if timeline else []
        # Скважина, которой не досталось модели, исчезает из прогноза бесшумно:
        # `predict` идёт по `models`, поэтому строк по ней нет ни одной, ЧДД
        # считается по остатку фонда и подаётся как ЧДД сценария, а поиск режимов
        # про неё не знает. `getattr` — ради артефактов, обученных до появления
        # этого поля: у них список неизвестен, и это не повод падать.
        missing = getattr(self.bundle.report, "wells_without_model", ())
        if missing:
            listed = ", ".join(missing[:5]) + (f" и ещё {len(missing) - 5}"
                                               if len(missing) > 5 else "")
            warnings.append(
                f"скважин без модели: {len(missing)} ({listed}) — прогноза по ним "
                "нет ни одной строки, ЧДД посчитан по остатку фонда"
            )

        reference_dates = set(expand(self.reference_steps))
        warnings.extend(
            f"{step.date.isoformat()}: даты нет в опорном расписании обучающего датасета"
            for step in steps if step.date not in reference_dates
        )
        for step in steps:
            month_number = self._month_number(step.date)
            if month_number < 0 or month_number > self.bundle.month_span:
                warnings.append(
                    f"{step.date.isoformat()}: дата вне обучающего горизонта "
                    f"0..{self.bundle.month_span} месяцев"
                )
        warnings.extend(self._unlearned_controls(steps))
        warnings.extend(self._physical_problems(steps))
        return warnings

    def _physical_problems(
        self, steps: list[ScheduleStep] | tuple[ScheduleStep, ...]
    ) -> list[str]:
        """Нарушения физики — по строке на скважину, а не на скважино-месяц.

        Проверки те же (`aios.summary.sanity_problems`) и считаются на тех же
        строках: они и без того ведутся по скважине отдельно — накопленные
        величины сравниваются с предыдущим месяцем той же скважины, — поэтому
        разбиение прогноза по скважинам ничего не меняет в результате, но даёт
        имя скважины без разбора текста сообщения.

        Свёртка нужна по той же причине, что и у диапазонных пометок: на
        настоящем расписании Модели Z таких строк 16 084 у 68 скважин, и
        страница цикла печатает их все до одной. Счётчик, показывающий
        «Предупреждения (16542)», формально верен и бесполезен.
        """
        by_well: dict[str, list[dict[str, Any]]] = {}
        for row in self.predict(steps):
            by_well.setdefault(str(row["well"]), []).append(row)

        warnings: list[str] = []
        for well in sorted(by_well, key=_well_key):
            problems = sanity_problems(by_well[well])
            if not problems:
                continue
            warnings.append(
                f"скв. {well}: прогноз нарушает физические проверки, строк "
                f"{len(problems)}, первая — {problems[0]}"
            )
        return warnings

    def _unlearned_controls(
        self, steps: list[ScheduleStep] | tuple[ScheduleStep, ...]
    ) -> list[str]:
        """Решения, которых в обучающей выборке не было ни одного.

        Диапазонный детектор их не ловит по устройству: `producer_rates` хранит
        границы по скважине **за весь датасет**, поэтому запуск скважины,
        простаивающей именно в этом месяце, попадает внутрь её же диапазона и
        проходит молча. А перевод под закачку не ловится и подавно: у добывающей
        скважины колонки `inj:` в обучающем наборе нет вовсе, и предложение
        перевести её не искажается, а пропадает по дороге к модели.

        DoE масштабирует существующие режимы и умеет их обнулять, но запусков не
        содержит. Значит, оба класса — экстраполяция, и единственный честный ответ
        суррогата здесь: «не оценивал, считайте полностью».
        """
        planned = _expand_role_controls(steps)
        reference = _expand_role_controls(self.reference_steps)
        known = set(self.bundle.control_wells)
        warnings: list[str] = []
        started: dict[str, list[date]] = {}
        unknown: dict[str, dict[str, date]] = {}
        # Роли, размеченные в обучающем датасете. Пустое множество — датасет
        # старого вида, с одной колонкой на скважину: там о покрытии ролей судить
        # нельзя, и поведение остаётся прежним
        covered_roles = {key.split(":", 1)[0] for key in known if ":" in key}

        for month_start, controls in planned.items():
            for (role, well), control in controls.items():
                if _role_rate(control, role) <= 0:
                    continue
                key = f"{'prod' if role == 'production' else 'inj'}:{well}"
                if key not in known and well not in known:
                    unknown.setdefault(key.split(":", 1)[0], {}).setdefault(key, month_start)
                    continue
                base = _role_rate(reference.get(month_start, {}).get((role, well)), role)
                if base <= 0:
                    started.setdefault(key, []).append(month_start)

        # Целый класс управления, которого в датасете нет, — это одна причина
        # (пакет посчитан по одному виду скважин), а не находка на каждую
        # скважину. Ревизия 19.08 намерила 71 предупреждение на любое расписание,
        # включая историческое, у модели, обученной только на пакете по
        # нагнетателям (`docs/experiments.md`).
        for prefix, months_by_key in unknown.items():
            if covered_roles and prefix not in covered_roles:
                listed = ", ".join(list(months_by_key)[:5])
                if len(months_by_key) > 5:
                    listed += f" и ещё {len(months_by_key) - 5}"
                warnings.append(
                    f"управления вида {prefix}: нет в обучающем датасете вовсе — "
                    f"суррогат обучен на пакете одного вида, и эти решения прогноз "
                    f"не учитывает (скважин {len(months_by_key)}: {listed})"
                )
            else:
                warnings.extend(
                    f"{month.isoformat()}: управления {key} нет в обучающем "
                    f"датасете — прогноз этого решения не учитывает вовсе"
                    for key, month in months_by_key.items()
                )

        # Запуск живёт весь горизонт, а не один месяц: без свёртки девять
        # переведённых скважин дают 188 строк, из которых 180 — одна и та же
        # мысль об одной скважине (`docs/experiments.md`, ревизия 19.08).
        for key, months in started.items():
            first, last = min(months), max(months)
            span = (first.isoformat() if first == last
                    else f"{first.isoformat()}…{last.isoformat()} ({len(months)} мес.)")
            warnings.append(
                f"{span}: {key} запускается без опорного "
                f"режима — обучающая выборка таких решений не содержит"
            )
        return warnings


def train(frame: pd.DataFrame, *, seed: int, test_fraction: float = 0.25,
          hidden_layers: tuple[int, ...] = (32, 16), max_iter: int = 300,
          kind: str = "mlp", control_start: date | None = None,
          targets: str = "raw") -> NeuralBundle:
    """Обучить модель на каждую скважину и измерить held-out ошибку.

    `kind` выбирает, какая из двух обученных моделей поедет в цикл: `mlp` или
    `linear`. Меряются обе в любом случае — сравнение и есть смысл упражнения.

    Модель учится **на остатке** `y − level(скважина, месяц)`, а не на самой цели:
    на уровень приходится 92 % дисперсии (замер 21.08), и модель, потратившая
    ёмкость на него, получает высокий `R²`, ничего не зная про управление.
    Уровень считается по обучающим сценариям — среднее по всем занесло бы отклик
    отложенных в собственный прогноз.

    `control_start` отделяет управляемую часть горизонта от истории. **Учимся на
    всей выборке, меряемся только на управляемой.** История до этой даты у всех
    сценариев побайтово одна: сценарий DoE меняет расписание только с неё. После
    правки `42fec5d` вход там нейтрален, но отклик по-прежнему одинаков, поэтому
    каждая историческая строка отложенного сценария дословно встречается в
    обучающих — метрика по ней меряет память. Учить на них законно: на
    нейтральном управлении они задают базовый уровень скважины.
    """
    if kind not in ("mlp", "linear"):
        raise ValueError(f"неизвестный вид модели: {kind}")
    if targets not in ("raw", "watercut"):
        raise ValueError(f"неизвестное представление целей: {targets}")
    _validate(frame, test_fraction)
    data = frame.copy()
    data["scenario"] = data["scenario"].astype(str)
    data["well"] = data["well"].astype(str)
    data["DATA"] = pd.to_datetime(data["DATA"], errors="raise")

    train_scenarios, test_scenarios = split_scenarios(
        data["scenario"].unique().tolist(), seed=seed, test_fraction=test_fraction
    )
    controls = _control_matrix(data)
    start_month = data["DATA"].min()
    month_span = int(max(_month_number(data["DATA"], start_month).max(), 1))
    centre = _control_centre(controls, train_scenarios, start_month, month_span)

    models: dict[str, Any] = {}
    levels: dict[str, np.ndarray] = {}
    neural_errors: list[np.ndarray] = []
    linear_errors: list[np.ndarray] = []
    neural_predictions: list[np.ndarray] = []
    linear_predictions: list[np.ndarray] = []
    slope_predictions: list[np.ndarray] = []
    level_predictions: list[np.ndarray] = []
    test_targets: list[np.ndarray] = []
    own_controls = _own_controls(tuple(controls.columns))
    for well, rows in data.groupby("well", sort=True):
        train_rows = rows[rows["scenario"].isin(train_scenarios)]
        test_rows = rows[rows["scenario"].isin(test_scenarios)]
        if train_rows.empty or test_rows.empty:
            continue

        if control_start is not None:
            test_rows = test_rows[test_rows["DATA"] >= pd.Timestamp(control_start)]
            if test_rows.empty:
                continue

        x_train = _features(train_rows, controls, start_month, month_span)
        x_test = _features(test_rows, controls, start_month, month_span)
        base_test = _centre_features(test_rows, centre, start_month, month_span)
        y_test = test_rows.loc[:, TARGET_COLUMNS].astype(float).to_numpy()

        # уровень — среднее обучающих сценариев по паре «скважина × месяц»; учится
        # модель на остатке, то есть ровно на отклике (`Р-53`)
        level = _level_table(
            _month_number(train_rows["DATA"], start_month).to_numpy(),
            _model_space(train_rows.loc[:, TARGET_COLUMNS].astype(float).to_numpy(), targets),
            month_span,
        )
        level_train = level[_clipped_months(train_rows["DATA"], start_month, month_span)]
        level_test = level[_clipped_months(test_rows["DATA"], start_month, month_span)]
        y_train = _model_space(
            train_rows.loc[:, TARGET_COLUMNS].astype(float).to_numpy(), targets
        ) - level_train

        neural = _regressor(MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            random_state=seed,
            max_iter=max_iter,
            tol=1e-3,
            n_iter_no_change=20,
            early_stopping=len(train_rows) >= 40,
            validation_fraction=0.15,
        ))
        baseline = _regressor(Ridge(alpha=1.0))
        neural.fit(x_train, y_train)
        baseline.fit(x_train, y_train)

        # метрики считаются в исходных единицах при любом представлении целей,
        # иначе сравнение параметризаций перестаёт быть сравнением
        neural_prediction = _raw_space(level_test + _response(neural, x_test, base_test), targets)
        linear_prediction = _raw_space(level_test + _response(baseline, x_test, base_test), targets)
        slope_prediction = _raw_space(level_test + _slope_response(
            x_train, y_train, x_test, base_test, own_controls.get(str(well), ())), targets)
        neural_errors.append(np.abs(neural_prediction - y_test))
        linear_errors.append(np.abs(linear_prediction - y_test))
        neural_predictions.append(neural_prediction)
        linear_predictions.append(linear_prediction)
        slope_predictions.append(slope_prediction)
        level_predictions.append(_raw_space(level_test, targets))
        test_targets.append(y_test)
        models[well] = neural if kind == "mlp" else baseline
        levels[well] = level

    if not models:
        raise ValueError("после разбиения не осталось скважин с train и test строками")

    neural_mae = np.concatenate(neural_errors).mean(axis=0)
    linear_mae = np.concatenate(linear_errors).mean(axis=0)
    actual = np.concatenate(test_targets)
    neural_predicted = np.concatenate(neural_predictions)
    linear_predicted = np.concatenate(linear_predictions)
    slope_predicted = np.concatenate(slope_predictions)
    level_predicted = np.concatenate(level_predictions)
    # остаток — то, что видит оптимизатор: уровень одинаков у всех трёх моделей и
    # из разности уходит, поэтому `R²` остатка отличается от `R²` цели только
    # знаменателем
    residual = actual - level_predicted
    metrics = {
        target: TargetMetric(
            neural_mae=float(neural_mae[index]),
            linear_mae=float(linear_mae[index]),
            neural_rmse=float(mean_squared_error(actual[:, index], neural_predicted[:, index]) ** 0.5),
            linear_rmse=float(mean_squared_error(actual[:, index], linear_predicted[:, index]) ** 0.5),
            neural_r2=float(r2_score(actual[:, index], neural_predicted[:, index])),
            linear_r2=float(r2_score(actual[:, index], linear_predicted[:, index])),
            level_r2=_score(actual[:, index], level_predicted[:, index]),
            neural_response_r2=_score(
                residual[:, index], neural_predicted[:, index] - level_predicted[:, index]),
            linear_response_r2=_score(
                residual[:, index], linear_predicted[:, index] - level_predicted[:, index]),
            slope_response_r2=_score(
                residual[:, index], slope_predicted[:, index] - level_predicted[:, index]),
        )
        for index, target in enumerate(TARGET_COLUMNS)
    }
    report = TrainingReport(
        train_scenarios=tuple(sorted(train_scenarios)),
        test_scenarios=tuple(sorted(test_scenarios)),
        wells=len(models),
        rows=len(data),
        metrics=metrics,
        targets=targets,
        metric_rows=int(sum(len(part) for part in test_targets)),
        wells_without_model=tuple(
            sorted(set(data["well"].astype(str)) - set(models), key=_well_key)
        ),
        metric_start="" if control_start is None else control_start.isoformat(),
        kind=kind,
    )
    return NeuralBundle(
        tuple(controls.columns), models, start_month, month_span,
        _envelope(controls), report, levels=levels, control_center=centre
    )


def assert_quality(report: TrainingReport) -> None:
    """Не пустить в цикл модель, которая не годится вести оптимизацию.

    Проверок две, и они закрывают разные провалы.

    **Относительная** — только для сети: усложнение обязано доказать превосходство
    над линейным бейзлайном числом, иначе его не существует (принцип 3
    `CLAUDE.md`). К самой линейной модели она неприменима: бейзлайн отверг бы сам
    себя.

    **Абсолютная по цели** — для обеих: `R² > 0` на отложенных сценариях по
    жидкости и нефти. `R² ≤ 0` означает «хуже, чем среднее по этим же
    сценариям».

    **Абсолютная по отклику** — для обеих, и она здесь главная. Замер 21.08:
    92 % дисперсии целей приходится на уровень скважины, поэтому `R² > 0` по
    цели проходит модель, которая про управление не знает вовсе — тривиальный
    предсказатель «среднее по (скважина, месяц)» даёт 0.918 по жидкости и 0.933
    по нефти. Оптимизатор же ходит по градиенту отклика, то есть по оставшимся
    7–8 %, и качество там до 21.08 не измерялось ни разу. Порог, не смотрящий на
    отклик, — открытые ворота: 20.08 прошедшая его модель обещала в полном цикле
    +34 772,4 млн руб., а расчёт ГГДМ дал −9 409,2.

    Чего эта проверка по-прежнему не делает: она не требует бить тупой бейзлайн
    «свой наклон на собственный множитель» (`slope_response_r2`). Он печатается
    рядом и сравнивается человеком — ворота отвечают за «не хуже, чем ничего», а
    выбор между моделями остаётся решением с числом в журнале.

    До 20.08 абсолютная проверка стояла только на линейной модели, и это была
    асимметрия, а не строгость: на полном датасете MLP выиграл у линейной по MAE
    на всех пяти целях — и проехал бы в цикл с `R²` по нефти **−0.814**. По
    скважинам там `R² > 0` лишь у 7 из 90 при медиане −1.38, то есть модель ловит
    разницу между скважинами, а не отклик скважины на управление. Оптимизатор
    ищет максимум ЧДД именно по этому градиенту, поэтому цена ошибки здесь —
    не «неточный прогноз», а уверенно неверное решение.
    """
    if report.kind != "linear":
        failed = [
            target for target in ("WLPT_Diff", "WOMT_Diff")
            if report.metrics[target].neural_mae >= report.metrics[target].linear_mae
        ]
        if failed:
            raise ValueError(
                "MLP не прошёл порог качества против линейного бейзлайна: "
                + ", ".join(failed)
            )

    name = "линейный суррогат" if report.kind == "linear" else "MLP"
    # `None` — это «не измерено», а не «плохо»: у отчётов, собранных до появления
    # R² (и у синтетических в тестах), его нет, и отвергать их не за что.
    blind = _failing(report, "linear_r2" if report.kind == "linear" else "neural_r2")
    if blind:
        raise ValueError(
            f"{name} не объясняет отложенные сценарии (R² ≤ 0): " + ", ".join(blind)
        )

    deaf = _failing(
        report, "linear_response_r2" if report.kind == "linear" else "neural_response_r2"
    )
    if deaf:
        raise ValueError(
            f"{name} не объясняет отклик на управление (R² отклика ≤ 0): "
            + ", ".join(deaf)
            + " — прогноз держится на уровне скважины, а оптимизатор ходит "
            "по градиенту отклика"
        )


def _failing(report: TrainingReport, field_name: str) -> list[str]:
    """Цели, по которым измеренный `R²` не выше нуля."""
    failing = []
    for target in ("WLPT_Diff", "WOMT_Diff"):
        score = getattr(report.metrics[target], field_name, None)
        if score is not None and score <= 0.0:
            failing.append(target)
    return failing


def split_scenarios(scenarios: list[str], *, seed: int,
                    test_fraction: float) -> tuple[frozenset[str], frozenset[str]]:
    """Детерминированное group split без утечки строк одного прогона."""
    unique = np.array(sorted(set(scenarios)), dtype=object)
    if len(unique) < 3:
        raise ValueError("для train/test нужно минимум три сценария")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    test_count = min(max(1, round(len(unique) * test_fraction)), len(unique) - 1)
    test = frozenset(str(value) for value in unique[:test_count])
    train_set = frozenset(str(value) for value in unique[test_count:])
    return train_set, test


def _validate(frame: pd.DataFrame, test_fraction: float) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"в датасете нет колонок: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("обучающий датасет пуст")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction должен быть между 0 и 1")
    duplicates = frame.duplicated(["scenario", "DATA", "well"])
    if duplicates.any():
        raise ValueError("в датасете повторяются строки scenario x DATA x well")


def _control_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    index = ["scenario", "DATA"]
    full_index = pd.MultiIndex.from_frame(frame.loc[:, index].drop_duplicates().sort_values(index))

    if "production_multiplier" not in frame and "injection_multiplier" not in frame:
        controls = _pivot_controls(frame, frame["multiplier"], "", full_index)
    else:
        production = frame.get(
            "production_multiplier", pd.Series(np.nan, index=frame.index, dtype=float)
        ).copy()
        injection = frame.get(
            "injection_multiplier", pd.Series(np.nan, index=frame.index, dtype=float)
        )
        legacy = frame["multiplier"]
        production = production.where(~(production.isna() & injection.isna()), legacy)
        controls = pd.concat(
            (
                _pivot_controls(frame, production, "prod:", full_index),
                _pivot_controls(frame, injection, "inj:", full_index),
            ),
            axis=1,
        ).sort_index(axis=1)

    if controls.empty:
        raise ValueError("матрица управляющих множителей неполна")
    return controls.astype(float)


def _pivot_controls(
    frame: pd.DataFrame,
    values: pd.Series,
    prefix: str,
    full_index: pd.MultiIndex,
) -> pd.DataFrame:
    source = frame.loc[:, ["scenario", "DATA", "well"]].copy()
    source["value"] = pd.to_numeric(values, errors="coerce")
    pivoted = (
        source.dropna(subset=["value"])
        .groupby(["scenario", "DATA", "well"], sort=True)["value"]
        .first()
        .unstack("well")
    )
    pivoted.columns = [f"{prefix}{well}" for well in pivoted.columns]
    return pivoted.reindex(full_index).fillna(1.0)


def _envelope(controls: pd.DataFrame) -> TrainingEnvelope:
    """Область применимости снимается с той же матрицы, что уходит в признаки.

    Строк здесь ровно «сценарий × месяц» (у Модели Z это 12 × 370), а не
    «скважина × месяц», поэтому перевод в словари ничего не стоит.
    """
    return TrainingEnvelope.fit(controls.to_dict("records"))


def _features(rows: pd.DataFrame, controls: pd.DataFrame, start: pd.Timestamp,
              month_span: int) -> np.ndarray:
    keys = pd.MultiIndex.from_arrays(
        [rows["scenario"].to_numpy(), rows["DATA"].to_numpy()], names=controls.index.names
    )
    scenario_controls = controls.reindex(keys).to_numpy(dtype=float)
    return np.column_stack((_time_features(rows["DATA"], start, month_span),
                            scenario_controls))


def _centre_features(rows: pd.DataFrame, centre: np.ndarray, start: pd.Timestamp,
                     month_span: int) -> np.ndarray:
    """Те же строки признаков, но с управлением в центре обучающей выборки."""
    return np.column_stack((
        _time_features(rows["DATA"], start, month_span),
        centre[_clipped_months(rows["DATA"], start, month_span)],
    ))


def _time_features(dates: pd.Series, start: pd.Timestamp, month_span: int) -> np.ndarray:
    month = _month_number(dates, start).to_numpy(dtype=float) / month_span
    season = dates.dt.month.to_numpy(dtype=float)
    return np.column_stack((month, np.sin(2 * np.pi * season / 12),
                            np.cos(2 * np.pi * season / 12)))


def _clipped_months(dates: pd.Series, start: pd.Timestamp, month_span: int) -> np.ndarray:
    return np.clip(_month_number(dates, start).to_numpy(), 0, month_span)


def _control_centre(controls: pd.DataFrame, scenarios: frozenset[str],
                    start: pd.Timestamp, month_span: int) -> np.ndarray:
    """Средний управляющий вектор обучающих сценариев по месяцам.

    Это центр выборки DoE, а не история: месяц, которого в обучении нет, остаётся
    нейтральной единицей. Для расписания, далёкого от обучающего облака, отклик
    отсчитывается от смещённой точки и настолько же смещён — числом это здесь не
    закрывается, только пометкой `in_training_range`.
    """
    index = controls.index
    chosen = index.get_level_values("scenario").isin(scenarios)
    dates = index.get_level_values("DATA")
    months = np.clip(
        ((dates.year - start.year) * 12 + dates.month - start.month).to_numpy(),
        0, month_span,
    )
    centre = np.ones((month_span + 1, controls.shape[1]), dtype=float)
    mean = (pd.DataFrame(controls.to_numpy(dtype=float)[chosen], index=months[chosen])
            .groupby(level=0).mean())
    centre[mean.index.to_numpy()] = mean.to_numpy()
    return centre


def _level_table(months: np.ndarray, values: np.ndarray, month_span: int) -> np.ndarray:
    """Уровень скважины по месяцам — среднее обучающих сценариев.

    Зависит только от пары «скважина × месяц»: управление сюда не попадает по
    построению, иначе модель снова начнёт угадывать уровень. Месяц, которого у
    скважины в обучении не было, получает её среднее по всему горизонту —
    подставить ноль означало бы предсказать остановку.
    """
    table = (pd.DataFrame(values, index=np.clip(months, 0, month_span))
             .groupby(level=0).mean())
    level = np.repeat(values.mean(axis=0)[None, :], month_span + 1, axis=0)
    level[table.index.to_numpy()] = table.to_numpy()
    return level


def _response(model: Any, features: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Отклик модели: насколько прогноз отличается от прогноза в центре выборки.

    Разность нужна ради точного нуля в центре: смещения скрытых слоёв сами в ноль
    не обращаются, и остаток уровня продолжал бы жить в отклике.
    """
    return model.predict(features) - model.predict(centre)


def _own_controls(control_wells: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
    """Столбцы признаков, которыми скважина управляет сама.

    Роли (`prod:`/`inj:`) сводятся к одной скважине: собственное управление — это
    и отбор, и закачка, если и то и другое размечено в датасете.
    """
    own: dict[str, list[int]] = {}
    for position, key in enumerate(control_wells):
        _, well = _control_key(key)
        own.setdefault(well, []).append(TIME_FEATURES + position)
    return {well: tuple(columns) for well, columns in own.items()}


def _slope_response(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
                    x_centre: np.ndarray, columns: tuple[int, ...]) -> np.ndarray:
    """Отклик тупого бейзлайна: один наклон на собственный множитель скважины.

    Ни времени, ни соседей — это планка, которую обученная модель обязана бить,
    иначе её не существует (принцип 3 `CLAUDE.md`). Замер 21.08 на объединении
    четырёх пакетов: 53.0 % дисперсии отклика по жидкости и 28.5 % по нефти.

    Скважине без собственной колонки управления бейзлайн отвечает нулевым
    откликом: сказать про неё ему нечем.
    """
    if not columns:
        return np.zeros((len(x_test), y_train.shape[1]), dtype=float)
    model = _regressor(Ridge(alpha=1.0))
    model.fit(x_train[:, columns], y_train)
    return _response(model, x_test[:, columns], x_centre[:, columns])


def _score(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    """`R²` там, где он определён: у постоянной цели знаменатель нулевой."""
    if float(np.var(actual)) == 0.0:
        return None
    return float(r2_score(actual, predicted))


def _month_number(values: pd.Series, start: pd.Timestamp) -> pd.Series:
    return (values.dt.year - start.year) * 12 + values.dt.month - start.month


def _regressor(model: Any) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), model),
        transformer=StandardScaler(),
    )


def _schedule_multipliers(
    timeline: dict[date, dict[str, ProducerControl | InjectorControl]],
    reference: dict[date, dict[str, ProducerControl | InjectorControl]],
    wells: tuple[str, ...],
    role_timeline: dict[date, dict[tuple[str, str], ProducerControl | InjectorControl]],
    role_reference: dict[date, dict[tuple[str, str], ProducerControl | InjectorControl]],
) -> dict[date, np.ndarray]:
    result: dict[date, np.ndarray] = {}
    last = dict.fromkeys(wells, 1.0)
    for month_start, controls in timeline.items():
        reference_controls = reference.get(month_start, {})
        values = []
        for control_key in wells:
            role, well = _control_key(control_key)
            planned_control = (
                role_timeline.get(month_start, {}).get((role, well))
                if role is not None else controls.get(well)
            )
            reference_control = (
                role_reference.get(month_start, {}).get((role, well))
                if role is not None else reference_controls.get(well)
            )
            planned, base = _relative_rate(
                planned_control, reference_control, role
            )
            if base > 0:
                last[control_key] = planned / base
            elif planned > 0:
                # Опорного режима нет: это запуск из простоя или перевод под
                # закачку. Отношение не определено, inf sklearn справедливо
                # отвергает, а ноль занят остановкой — поэтому отдельное значение.
                last[control_key] = RESTART_MULTIPLIER
            values.append(last[control_key])
        result[month_start] = np.asarray(values, dtype=float)
    return result


def _expand_role_controls(
    steps: list[ScheduleStep] | tuple[ScheduleStep, ...],
) -> dict[date, dict[tuple[str, str], ProducerControl | InjectorControl]]:
    effective: dict[tuple[str, str], ProducerControl | InjectorControl] = {}
    timeline = {}
    for step in steps:
        effective.update({("production", control.well): control for control in step.producers})
        effective.update({("injection", control.well): control for control in step.injectors})
        timeline[step.date] = dict(effective)
    return timeline


def _relative_rate(
    planned: ProducerControl | InjectorControl | None,
    reference: ProducerControl | InjectorControl | None,
    role: str | None = None,
) -> tuple[float, float]:
    """Сопоставимый режим: добыча с добычей, закачка с закачкой."""
    planned_liquid, planned_injection = _rates(planned)
    base_liquid, base_injection = _rates(reference)
    if role == "production":
        return planned_liquid, base_liquid
    if role == "injection":
        return planned_injection, base_injection
    if base_liquid > 0:
        return planned_liquid, base_liquid
    if base_injection > 0:
        return planned_injection, base_injection
    return max(planned_liquid, planned_injection), 0.0


def _role_rate(control: ProducerControl | InjectorControl | None, role: str) -> float:
    """Режим, сопоставимый с ролью: добыча с добычей, закачка с закачкой."""
    liquid, injection = _rates(control)
    return liquid if role == "production" else injection


def _control_key(value: str) -> tuple[str | None, str]:
    if value.startswith("prod:"):
        return "production", value[5:]
    if value.startswith("inj:"):
        return "injection", value[4:]
    return None, value


def _inference_features(month_start: date, multipliers: np.ndarray,
                        start: pd.Timestamp, month_span: int) -> np.ndarray:
    month_number = (month_start.year - start.year) * 12 + month_start.month - start.month
    season = month_start.month
    time = np.array([
        month_number / month_span,
        np.sin(2 * np.pi * season / 12),
        np.cos(2 * np.pi * season / 12),
    ])
    return np.concatenate((time, multipliers)).reshape(1, -1)


def _rates(control: ProducerControl | InjectorControl | None) -> tuple[float, float]:
    if control is None or control.status != "OPEN":
        return 0.0, 0.0
    if isinstance(control, ProducerControl):
        return float(control.liquid_rate or 0.0), 0.0
    return 0.0, float(control.rate or 0.0)


def _model_space(values: np.ndarray, targets: str) -> np.ndarray:
    """Перевести цели в то представление, которому учится модель."""
    if targets != "watercut":
        return values
    converted = values.copy()
    liquid, oil = values[:, 0], values[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        watercut = np.where(liquid > 0, 1.0 - oil / liquid, 0.0)
    converted[:, 1] = np.clip(watercut, 0.0, 1.0)
    return converted


def _raw_space(values: np.ndarray, targets: str) -> np.ndarray:
    """Вернуть предсказание в единицы контракта: тонны нефти вместо доли воды.

    Нефть здесь не может превысить жидкость и не может стать отрицательной —
    это следует из формулы, а не из проверки после.
    """
    if targets != "watercut":
        return values
    converted = values.copy()
    liquid = np.maximum(values[:, 0], 0.0)
    watercut = np.clip(values[:, 1], 0.0, 1.0)
    converted[:, 0] = liquid
    converted[:, 1] = liquid * (1.0 - watercut)
    return converted


def _contract_row(month_start: date, well: str, liquid_rate: float,
                  injection_rate: float, totals: list[float], liquid: float,
                  oil: float, injection: float, thp: float, bhp: float,
                  days: int) -> dict[str, Any]:
    return {
        "DATA": month_start.isoformat(),
        "well": well,
        "WLPT": totals[0],
        "WLPR": liquid_rate,
        "WOMT": totals[1],
        "WOMR": oil / days,
        "WWIR": injection_rate,
        "WWIT": totals[2],
        "THP": thp,
        "BHP": bhp,
        "WEFF": 1.0,
        "WLPT_Diff": liquid,
        "WOMT_Diff": oil,
        "WWIT_Diff": injection,
    }


def month_days(value: date) -> int:
    return calendar.monthrange(value.year, value.month)[1]


def _well_key(name: str) -> tuple[int, int, str]:
    return (0, int(name), "") if name.isdigit() else (1, 0, name)
