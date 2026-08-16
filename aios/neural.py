"""Обучение нейросетевого суррогата на сценариях DoE.

Разбиение выполняется по целым сценариям, а не по строкам: соседние месяцы одного
прогона слишком похожи, и их попадание одновременно в train и test завышает
качество. Для каждой скважины обучается небольшой MLP, который видит время и
совместное управление всем добывающим фондом.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COLUMNS = ("WLPT_Diff", "WOMT_Diff", "WWIT_Diff", "THP", "BHP")
REQUIRED_COLUMNS = frozenset({"scenario", "DATA", "well", "multiplier", *TARGET_COLUMNS})


@dataclass(frozen=True)
class TargetMetric:
    neural_mae: float
    linear_mae: float


@dataclass(frozen=True)
class TrainingReport:
    train_scenarios: tuple[str, ...]
    test_scenarios: tuple[str, ...]
    wells: int
    rows: int
    metrics: dict[str, TargetMetric]


@dataclass
class NeuralBundle:
    """Обученные модели и точный порядок входных управляющих признаков."""

    control_wells: tuple[str, ...]
    models: dict[str, Any]
    start_month: pd.Timestamp
    month_span: int
    report: TrainingReport


def train(frame: pd.DataFrame, *, seed: int, test_fraction: float = 0.25,
          hidden_layers: tuple[int, ...] = (32, 16), max_iter: int = 300) -> NeuralBundle:
    """Обучить отдельный MLP на каждую скважину и измерить held-out ошибку."""
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

        neural_errors.append(np.abs(neural.predict(x_test) - y_test))
        linear_errors.append(np.abs(baseline.predict(x_test) - y_test))
        models[well] = neural

    if not models:
        raise ValueError("после разбиения не осталось скважин с train и test строками")

    neural_mae = np.concatenate(neural_errors).mean(axis=0)
    linear_mae = np.concatenate(linear_errors).mean(axis=0)
    metrics = {
        target: TargetMetric(float(neural_mae[index]), float(linear_mae[index]))
        for index, target in enumerate(TARGET_COLUMNS)
    }
    report = TrainingReport(
        train_scenarios=tuple(sorted(train_scenarios)),
        test_scenarios=tuple(sorted(test_scenarios)),
        wells=len(models),
        rows=len(data),
        metrics=metrics,
    )
    return NeuralBundle(tuple(controls.columns), models, start_month, int(month_span), report)


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
    controls = (
        frame.dropna(subset=["multiplier"])
        .groupby(["scenario", "well"], sort=True)["multiplier"]
        .first()
        .unstack("well")
        .sort_index(axis=1)
    )
    if controls.empty or controls.isna().any().any():
        raise ValueError("матрица управляющих множителей неполна")
    return controls.astype(float)


def _features(rows: pd.DataFrame, controls: pd.DataFrame, start: pd.Timestamp,
              month_span: int) -> np.ndarray:
    dates = rows["DATA"]
    month = _month_number(dates, start).to_numpy(dtype=float) / month_span
    season = dates.dt.month.to_numpy(dtype=float)
    time = np.column_stack((month, np.sin(2 * np.pi * season / 12),
                            np.cos(2 * np.pi * season / 12)))
    scenario_controls = controls.loc[rows["scenario"]].to_numpy(dtype=float)
    return np.column_stack((time, scenario_controls))


def _month_number(values: pd.Series, start: pd.Timestamp) -> pd.Series:
    return (values.dt.year - start.year) * 12 + values.dt.month - start.month


def _regressor(model: Any) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), model),
        transformer=StandardScaler(),
    )
