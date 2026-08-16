"""Обучение суррогата без утечки между сценариями."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from aios import economics
from aios.deck import InjectorControl, ProducerControl, ScheduleStep
from aios.model import SurrogateLoadError, for_cycle
from aios.neural import (
    TARGET_COLUMNS,
    NeuralBundle,
    NeuralSurrogate,
    TargetMetric,
    TrainingReport,
    assert_quality,
    split_scenarios,
    train,
)
from aios.summary import Fluids
from aios.surrogate import NumericRange, TrainingEnvelope
from scripts.train_surrogate import load_datasets


def dataset() -> pd.DataFrame:
    rows = []
    for scenario_index in range(8):
        scenario = f"doe-{scenario_index:03d}"
        controls = {"1": 0.6 + scenario_index * 0.1, "2": 1.3 - scenario_index * 0.05}
        for month in range(1, 7):
            for well, multiplier in controls.items():
                interaction = controls["1"] * controls["2"]
                liquid = 100 * multiplier * month
                oil = liquid * (0.8 - 0.02 * month) + 10 * interaction
                rows.append({
                    "scenario": scenario,
                    "DATA": f"2014-{month:02d}-01",
                    "well": well,
                    "multiplier": multiplier,
                    "is_shut": 0,
                    "WLPR": 100 * multiplier,
                    "WWIR": 0.0,
                    "WLPT_Diff": liquid,
                    "WOMT_Diff": oil,
                    "WWIT_Diff": 0.0,
                    "THP": 200 + interaction,
                    "BHP": 50 + float(well),
                })
    return pd.DataFrame(rows)


def test_split_is_deterministic_and_has_no_scenario_leakage():
    scenarios = [f"s{index}" for index in range(10)]

    first = split_scenarios(scenarios, seed=7, test_fraction=0.2)
    second = split_scenarios(scenarios, seed=7, test_fraction=0.2)

    assert first == second
    assert first[0].isdisjoint(first[1])
    assert first[0] | first[1] == frozenset(scenarios)


def test_training_produces_models_and_finite_held_out_metrics():
    bundle = train(dataset(), seed=42, hidden_layers=(8,), max_iter=300)

    assert set(bundle.models) == {"1", "2"}
    assert set(bundle.report.train_scenarios).isdisjoint(bundle.report.test_scenarios)
    assert set(bundle.report.metrics) == set(TARGET_COLUMNS)
    assert all(np.isfinite(metric.neural_mae) for metric in bundle.report.metrics.values())
    assert all(np.isfinite(metric.neural_rmse) for metric in bundle.report.metrics.values())
    assert all(np.isfinite(metric.neural_r2) for metric in bundle.report.metrics.values())


def test_training_rejects_row_level_duplicates():
    frame = dataset()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="повторяются строки"):
        train(frame, seed=42)


class ConstantModel:
    def predict(self, features):
        assert features.shape[1] == 4
        return np.tile([900.0, 500.0, 0.0, 200.0, 50.0], (len(features), 1))


class InvalidModel:
    def predict(self, features):
        return np.tile([-1.0, 2.0, 0.0, 200.0, 50.0], (len(features), 1))


class RecordingModel(ConstantModel):
    def __init__(self):
        self.features = None

    def predict(self, features):
        self.features = features.copy()
        return np.tile([900.0, 500.0, 0.0, 200.0, 50.0], (len(features), 1))


def report(*, neural: float = 1.0, linear: float = 2.0) -> TrainingReport:
    return TrainingReport(
        ("train",), ("test",), 1, 2,
        {target: TargetMetric(neural, linear) for target in TARGET_COLUMNS},
    )


def bundle(metrics: TrainingReport | None = None) -> NeuralBundle:
    envelope = TrainingEnvelope(
        {"1": NumericRange(20.0, 40.0)}, {}, NumericRange(20.0, 40.0), NumericRange(0, 0)
    )
    return NeuralBundle(("1",), {"1": ConstantModel()}, pd.Timestamp("2014-01-01"), 1,
                        envelope, metrics or report())


def step(rate: float = 30.0) -> ScheduleStep:
    return ScheduleStep(
        pd.Timestamp("2014-01-01").date(),
        (ProducerControl("1", "OPEN", "LRAT", rate, 50.0),),
        (),
    )


def test_neural_surrogate_returns_the_reference_calculator_contract():
    surrogate = NeuralSurrogate.create(bundle(), [step()])

    rows = surrogate.predict([step()])

    assert list(rows[0]) == economics.REQUIRED_COLUMNS
    assert rows[0]["WLPT"] == pytest.approx(900.0)
    assert rows[0]["WOMT"] == pytest.approx(500.0)
    assert surrogate.in_training_range([step()]) == []


def test_quality_gate_rejects_a_network_worse_than_linear_baseline():
    with pytest.raises(ValueError, match="WLPT_Diff"):
        assert_quality(report(neural=3.0, linear=2.0))


def test_physical_prediction_failure_reaches_cycle_warnings():
    invalid = bundle()
    invalid.models["1"] = InvalidModel()
    surrogate = NeuralSurrogate.create(invalid, [step()])

    warnings = surrogate.in_training_range([step()])

    assert any("отрицательное WLPT_Diff" in warning for warning in warnings)


def test_date_outside_training_horizon_is_flagged():
    future = ScheduleStep(
        pd.Timestamp("2015-01-01").date(),
        (ProducerControl("1", "OPEN", "LRAT", 30.0, 50.0),),
        (),
    )
    surrogate = NeuralSurrogate.create(bundle(), [future])

    assert any("вне обучающего горизонта" in warning
               for warning in surrogate.in_training_range([future]))


def test_injector_multiplier_reaches_the_neural_features():
    recorder = RecordingModel()
    trained = bundle()
    trained.control_wells = ("1", "2")
    trained.models["1"] = recorder
    reference = ScheduleStep(
        step().date,
        step().producers,
        (InjectorControl("2", "WATER", "OPEN", "RATE", 40.0, 300.0),),
    )
    candidate = ScheduleStep(
        step().date,
        step().producers,
        (InjectorControl("2", "WATER", "OPEN", "RATE", 80.0, 300.0),),
    )

    NeuralSurrogate.create(trained, [reference]).predict([candidate])

    assert recorder.features is not None
    assert recorder.features[0, -2:].tolist() == pytest.approx([1.0, 2.0])


def test_multiple_doe_packages_keep_scenarios_distinct(tmp_path):
    first = tmp_path / "dataset.csv"
    second = tmp_path / "dataset-levels.csv"
    pd.DataFrame({"scenario": ["doe-000"], "value": [1]}).to_csv(first, index=False)
    pd.DataFrame({"scenario": ["doe-000"], "value": [2]}).to_csv(second, index=False)

    combined = load_datasets([str(first), str(second)])

    assert set(combined["scenario"]) == {"dataset:doe-000", "dataset-levels:doe-000"}


def test_cycle_loader_enables_joint_search_for_a_valid_neural_bundle(tmp_path):
    path = tmp_path / "surrogate.joblib"
    joblib.dump(bundle(), path)

    selected = for_cycle(
        neural_path=path, history=[], reference_steps=[step()], fluids=Fluids()
    )

    assert isinstance(selected.model, NeuralSurrogate)
    assert selected.joint_rate_search


def test_cycle_loader_rejects_an_unknown_artifact(tmp_path):
    path = tmp_path / "surrogate.joblib"
    joblib.dump({"not": "a model"}, path)

    with pytest.raises(SurrogateLoadError, match="NeuralBundle"):
        for_cycle(
            neural_path=path, history=[], reference_steps=[step()],
            fluids=Fluids(),
        )
