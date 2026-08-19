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
from scripts.train_surrogate import dataset_fingerprints, load_datasets


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


def test_monthly_control_matrix_keeps_production_and_injection_roles_separate():
    frame = pd.DataFrame(
        [
            {
                "scenario": scenario,
                "DATA": pd.Timestamp(f"2014-{month:02d}-01"),
                "well": well,
                "multiplier": value,
                "production_multiplier": value if well == "p" else np.nan,
                "injection_multiplier": value if well == "i" else np.nan,
            }
            for scenario, month, well, value in (
                ("a", 1, "p", 0.8), ("a", 1, "i", 1.2),
                ("a", 2, "p", 0.9), ("a", 2, "i", 1.4),
                ("b", 1, "p", 1.0), ("b", 1, "i", 0.7),
            )
        ]
    )

    controls = __import__("aios.neural", fromlist=["_control_matrix"])._control_matrix(frame)

    assert list(controls.columns) == ["inj:i", "prod:p"]
    assert controls.loc[("a", pd.Timestamp("2014-02-01")), "inj:i"] == pytest.approx(1.4)
    assert controls.loc[("b", pd.Timestamp("2014-01-01")), "prod:p"] == pytest.approx(1.0)


def test_role_features_follow_a_well_converted_to_injection():
    recorder = RecordingModel()
    trained = bundle()
    trained.control_wells = ("prod:9", "inj:9")
    trained.models = {"9": recorder}
    january = ScheduleStep(
        pd.Timestamp("2014-01-01").date(),
        (ProducerControl("9", "OPEN", "LRAT", 30.0, 50.0),),
        (InjectorControl("9", "WATER", "SHUT", "RATE", None, 300.0),),
    )
    february = ScheduleStep(
        pd.Timestamp("2014-02-01").date(),
        (ProducerControl("9", "SHUT", "LRAT", None, 50.0),),
        (InjectorControl("9", "WATER", "OPEN", "RATE", 40.0, 300.0),),
    )
    candidate = [
        january,
        ScheduleStep(
            february.date,
            february.producers,
            (InjectorControl("9", "WATER", "OPEN", "RATE", 80.0, 300.0),),
        ),
    ]

    NeuralSurrogate.create(trained, [january, february]).predict(candidate)

    np.testing.assert_allclose(recorder.features[:, -2:], [[1.0, 1.0], [1.0, 2.0]])


def test_multiple_doe_packages_keep_scenarios_distinct(tmp_path):
    first = tmp_path / "dataset.csv"
    second = tmp_path / "dataset-levels.csv"
    pd.DataFrame({"scenario": ["doe-000"], "value": [1]}).to_csv(first, index=False)
    pd.DataFrame({"scenario": ["doe-000"], "value": [2]}).to_csv(second, index=False)

    combined = load_datasets([str(first), str(second)])

    assert set(combined["scenario"]) == {"dataset:doe-000", "dataset-levels:doe-000"}


def test_dataset_fingerprint_records_exact_bytes(tmp_path):
    path = tmp_path / "dataset.csv"
    path.write_bytes(b"scenario,value\na,1\n")

    fingerprint = dataset_fingerprints([str(path)])[0]

    assert fingerprint["bytes"] == 19
    assert fingerprint["sha256"] == "a4c2b058680c4344c89636c9b268f6b8a90845ce36dfdc393643c1fe7d8a8956"


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


def test_starting_an_idle_well_is_not_encoded_as_a_shutdown():
    """Остановить скважину и запустить её из простоя — это не одно и то же.

    Признак строится как отношение планируемого режима к опорному. Когда опорный
    режим нулевой, отношение не определено, и `_schedule_multipliers` подставляет
    конечное `0.0`. Но в обучающей выборке `0.0` означает буквально «скважина
    остановлена» (`aios/doe.py`, `Scenario.shut_wells`), то есть предложение
    запустить скважину подаётся модели как её остановка.

    `CLAUDE.md` называет «запуск из простоя» одним из случаев, где суррогат врёт
    по построению и должен мериться отдельно. До 19.08 он не врал, а получал
    перевёрнутый вход.

    Закрыто 19.08: запуск кодируется `RESTART_MULTIPLIER`, спутать который с
    остановкой нельзя, потому что множители неотрицательны по построению. Само по
    себе это правильного числа не даёт — обучающая выборка запусков не содержит, —
    поэтому в паре с кодированием обязана срабатывать пометка
    `in_training_range`; она проверяется отдельным тестом ниже.
    """
    from datetime import date

    from aios import neural
    from aios.surrogate import expand

    def schedule(rate: float, status: str) -> list[ScheduleStep]:
        return [ScheduleStep(date(2014, 1, 1),
                             (ProducerControl("1", status, "LRAT", rate, 50.0),), ())]

    def multiplier(planned: list[ScheduleStep], reference: list[ScheduleStep]) -> float:
        values = neural._schedule_multipliers(
            expand(planned), expand(reference), ("prod:1",),
            neural._expand_role_controls(planned), neural._expand_role_controls(reference),
        )
        return float(values[date(2014, 1, 1)][0])

    shutdown = multiplier(schedule(0.0, "SHUT"), schedule(30.0, "OPEN"))
    start_up = multiplier(schedule(30.0, "OPEN"), schedule(0.0, "SHUT"))

    assert shutdown == 0.0, "остановка обязана кодироваться нулём"
    assert start_up != shutdown, "запуск из простоя не должен выглядеть как остановка"
    assert start_up < 0.0, (
        "значение обязано быть вне области множителей, а не ещё одним правдоподобным"
    )


def test_starting_an_idle_well_is_reported_as_outside_the_training_set():
    """Кодирование без пометки — замена одного неверного ответа другим.

    Обучающая выборка запусков не содержит вовсе (DoE масштабирует режимы и умеет
    их обнулять, но не запускает), поэтому любое число здесь — экстраполяция.
    Диапазонный детектор молчит по устройству: границы `producer_rates` собраны по
    скважине за весь датасет, и режим 30 м³/сут попадает внутрь них. Значит,
    сказать «не оценивал» обязан именно этот признак — иначе решение будет принято
    по прогнозу, а не отправлено на полный расчёт (принцип 6 `CLAUDE.md`).
    """
    idle = ScheduleStep(pd.Timestamp("2014-01-01").date(),
                        (ProducerControl("1", "SHUT", "LRAT", None, 50.0),), ())
    surrogate = NeuralSurrogate.create(bundle(), [idle])

    warnings = surrogate.in_training_range([step(30.0)])

    assert any("без опорного режима" in warning for warning in warnings), warnings


def test_conversion_to_injection_is_reported_as_not_covered_by_the_model():
    """Перевод под закачку не искажается, а пропадает — это надо сказать вслух.

    Признаки собираются по ролям: колонка `inj:<скважина>` появляется только у
    той, у которой в датасете есть непустой множитель закачки. У добывающей её нет
    ни в одном сценарии DoE, поэтому предложение перевести скважину под закачку до
    модели не доходит вовсе, а прогноз выглядит обычным.
    """
    trained = bundle()
    trained.control_wells = ("prod:1",)
    surrogate = NeuralSurrogate.create(trained, [step(30.0)])
    converted = ScheduleStep(
        pd.Timestamp("2014-01-01").date(), (),
        (InjectorControl("1", "WATER", "OPEN", "RATE", 120.0, 250.0),),
    )

    warnings = surrogate.in_training_range([converted])

    assert any("inj:1" in warning and "нет в обучающем датасете" in warning
               for warning in warnings), warnings


def test_training_script_reports_the_quality_verdict_next_to_the_metrics():
    """Об отказе порога надо узнавать при обучении, а не при запуске цикла.

    Порог применяется на входе в цикл (`NeuralSurrogate.create`), и это верно:
    непрошедшую модель полезно сохранить и разобрать. Но если при обучении об
    этом не сказано, метрики прогона уйдут в журнал как «сеть обучена».
    """
    from scripts.train_surrogate import quality_verdict

    assert "пройден" in quality_verdict(report(neural=1.0, linear=2.0))
    assert "НЕ ПРОЙДЕН" in quality_verdict(report(neural=3.0, linear=2.0))
