"""Обучение нейросетевого суррогата на сценариях DoE.

Разбиение выполняется по целым сценариям, а не по строкам: соседние месяцы одного
прогона слишком похожи, и их попадание одновременно в train и test завышает
качество. Для каждой скважины обучается небольшой MLP, который видит время и
совместное управление всем добывающим фондом.
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
from aios.surrogate import NumericRange, TrainingEnvelope, expand

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
REQUIRED_COLUMNS = frozenset(
    {"scenario", "DATA", "well", "multiplier", "WLPR", "WWIR", *TARGET_COLUMNS}
)


@dataclass(frozen=True)
class TargetMetric:
    neural_mae: float
    linear_mae: float
    neural_rmse: float | None = None
    linear_rmse: float | None = None
    neural_r2: float | None = None
    linear_r2: float | None = None


@dataclass(frozen=True)
class TrainingReport:
    train_scenarios: tuple[str, ...]
    test_scenarios: tuple[str, ...]
    wells: int
    rows: int
    metrics: dict[str, TargetMetric]

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
        reference = expand(self.reference_steps)
        multipliers = _schedule_multipliers(
            timeline,
            reference,
            self.bundle.control_wells,
            _expand_role_controls(steps),
            _expand_role_controls(self.reference_steps),
        )
        months = list(timeline)
        feature_matrix = np.vstack([
            _inference_features(
                month_start, multipliers[month_start], self.bundle.start_month,
                self.bundle.month_span
            )[0]
            for month_start in months
        ])
        predictions = {
            well: model.predict(feature_matrix)
            for well, model in self.bundle.models.items()
        }
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

    def in_training_range(self,
                          steps: list[ScheduleStep] | tuple[ScheduleStep, ...]) -> list[str]:
        warnings = self.bundle.envelope.warnings(steps)
        reference_dates = set(expand(self.reference_steps))
        warnings.extend(
            f"{step.date.isoformat()}: даты нет в опорном расписании обучающего датасета"
            for step in steps if step.date not in reference_dates
        )
        for step in steps:
            month_number = (
                (step.date.year - self.bundle.start_month.year) * 12
                + step.date.month - self.bundle.start_month.month
            )
            if month_number < 0 or month_number > self.bundle.month_span:
                warnings.append(
                    f"{step.date.isoformat()}: дата вне обучающего горизонта "
                    f"0..{self.bundle.month_span} месяцев"
                )
        warnings.extend(self._unlearned_controls(steps))
        warnings.extend(
            f"прогноз нарушает физические проверки: {problem}"
            for problem in sanity_problems(self.predict(steps))
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
        seen: set[str] = set()
        warnings: list[str] = []

        for month_start, controls in planned.items():
            for (role, well), control in controls.items():
                if _role_rate(control, role) <= 0:
                    continue
                key = f"{'prod' if role == 'production' else 'inj'}:{well}"
                if key not in known and well not in known:
                    if key in seen:
                        continue
                    seen.add(key)
                    warnings.append(
                        f"{month_start.isoformat()}: управления {key} нет в обучающем "
                        f"датасете — прогноз этого решения не учитывает вовсе"
                    )
                    continue
                base = _role_rate(reference.get(month_start, {}).get((role, well)), role)
                if base <= 0:
                    warnings.append(
                        f"{month_start.isoformat()}: {key} запускается без опорного "
                        f"режима — обучающая выборка таких решений не содержит"
                    )
        return warnings


def train(frame: pd.DataFrame, *, seed: int, test_fraction: float = 0.25,
          hidden_layers: tuple[int, ...] = (32, 16), max_iter: int = 300,
          kind: str = "mlp") -> NeuralBundle:
    """Обучить модель на каждую скважину и измерить held-out ошибку.

    `kind` выбирает, какая из двух обученных моделей поедет в цикл: `mlp` или
    `linear`. Меряются обе в любом случае — сравнение и есть смысл упражнения.
    """
    if kind not in ("mlp", "linear"):
        raise ValueError(f"неизвестный вид модели: {kind}")
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
    month_span = max(_month_number(data["DATA"], start_month).max(), 1)

    models: dict[str, Any] = {}
    neural_errors: list[np.ndarray] = []
    linear_errors: list[np.ndarray] = []
    neural_predictions: list[np.ndarray] = []
    linear_predictions: list[np.ndarray] = []
    test_targets: list[np.ndarray] = []
    for well, rows in data.groupby("well", sort=True):
        train_rows = rows[rows["scenario"].isin(train_scenarios)]
        test_rows = rows[rows["scenario"].isin(test_scenarios)]
        if train_rows.empty or test_rows.empty:
            continue

        x_train = _features(train_rows, controls, start_month, month_span)
        x_test = _features(test_rows, controls, start_month, month_span)
        y_train = train_rows.loc[:, TARGET_COLUMNS].astype(float).to_numpy()
        y_test = test_rows.loc[:, TARGET_COLUMNS].astype(float).to_numpy()

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

        neural_prediction = neural.predict(x_test)
        linear_prediction = baseline.predict(x_test)
        neural_errors.append(np.abs(neural_prediction - y_test))
        linear_errors.append(np.abs(linear_prediction - y_test))
        neural_predictions.append(neural_prediction)
        linear_predictions.append(linear_prediction)
        test_targets.append(y_test)
        models[well] = neural if kind == "mlp" else baseline

    if not models:
        raise ValueError("после разбиения не осталось скважин с train и test строками")

    neural_mae = np.concatenate(neural_errors).mean(axis=0)
    linear_mae = np.concatenate(linear_errors).mean(axis=0)
    actual = np.concatenate(test_targets)
    neural_predicted = np.concatenate(neural_predictions)
    linear_predicted = np.concatenate(linear_predictions)
    metrics = {
        target: TargetMetric(
            neural_mae=float(neural_mae[index]),
            linear_mae=float(linear_mae[index]),
            neural_rmse=float(mean_squared_error(actual[:, index], neural_predicted[:, index]) ** 0.5),
            linear_rmse=float(mean_squared_error(actual[:, index], linear_predicted[:, index]) ** 0.5),
            neural_r2=float(r2_score(actual[:, index], neural_predicted[:, index])),
            linear_r2=float(r2_score(actual[:, index], linear_predicted[:, index])),
        )
        for index, target in enumerate(TARGET_COLUMNS)
    }
    report = TrainingReport(
        train_scenarios=tuple(sorted(train_scenarios)),
        test_scenarios=tuple(sorted(test_scenarios)),
        wells=len(models),
        rows=len(data),
        metrics=metrics,
        kind=kind,
    )
    return NeuralBundle(
        tuple(controls.columns), models, start_month, int(month_span), _envelope(data), report
    )


def assert_quality(report: TrainingReport) -> None:
    """Не пустить в цикл модель, которая не лучше того, чем её меряют.

    Для MLP «то, чем меряют» — линейный бейзлайн: усложнение обязано доказать
    превосходство числом, иначе его не существует (принцип 3 `CLAUDE.md`).

    Для линейной модели такой порог бессмыслен — она сама и есть бейзлайн, —
    поэтому проверяется другое и более грубое: объясняет ли она вообще хоть
    что-то на отложенных сценариях. `R² ≤ 0` означает «не лучше среднего», и
    такую модель в цикл пускать нельзя ничуть не больше, чем проигравший MLP.
    """
    if report.kind == "linear":
        failed = [
            target for target in ("WLPT_Diff", "WOMT_Diff")
            if (report.metrics[target].linear_r2 or 0.0) <= 0.0
        ]
        if failed:
            raise ValueError(
                "линейный суррогат не объясняет отложенные сценарии (R² ≤ 0): "
                + ", ".join(failed)
            )
        return

    failed = [
        target for target in ("WLPT_Diff", "WOMT_Diff")
        if report.metrics[target].neural_mae >= report.metrics[target].linear_mae
    ]
    if failed:
        raise ValueError(
            "MLP не прошёл порог качества против линейного бейзлайна: " + ", ".join(failed)
        )


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


def _envelope(frame: pd.DataFrame) -> TrainingEnvelope:
    positive_liquid = frame[frame["WLPR"].astype(float) > 0].groupby("well")["WLPR"]
    positive_injection = frame[frame["WWIR"].astype(float) > 0].groupby("well")["WWIR"]
    field = frame.groupby(["scenario", "DATA"])[["WLPR", "WWIR"]].sum()
    return TrainingEnvelope(
        producer_rates={
            str(well): NumericRange(float(values.min()), float(values.max()))
            for well, values in positive_liquid
        },
        injector_rates={
            str(well): NumericRange(float(values.min()), float(values.max()))
            for well, values in positive_injection
        },
        field_liquid_rate=NumericRange(float(field["WLPR"].min()), float(field["WLPR"].max())),
        field_injection_rate=NumericRange(float(field["WWIR"].min()), float(field["WWIR"].max())),
    )


def _features(rows: pd.DataFrame, controls: pd.DataFrame, start: pd.Timestamp,
              month_span: int) -> np.ndarray:
    dates = rows["DATA"]
    month = _month_number(dates, start).to_numpy(dtype=float) / month_span
    season = dates.dt.month.to_numpy(dtype=float)
    time = np.column_stack((month, np.sin(2 * np.pi * season / 12),
                            np.cos(2 * np.pi * season / 12)))
    keys = pd.MultiIndex.from_arrays(
        [rows["scenario"].to_numpy(), rows["DATA"].to_numpy()], names=controls.index.names
    )
    scenario_controls = controls.reindex(keys).to_numpy(dtype=float)
    return np.column_stack((time, scenario_controls))


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
