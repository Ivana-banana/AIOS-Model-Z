"""Обучение суррогата без утечки между сценариями."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import TransformedTargetRegressor

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
    """Запоминает каждый вызов: с разделением уровня и отклика их два на прогноз.

    Второй — точка центра (`_response` берёт разность), и он ничего не говорит о
    поданном расписании. Поэтому вызовы копятся списком, а `features` остаётся
    последним для тестов, где путь без разделения и вызов один.
    """

    def __init__(self):
        self.features = None
        self.calls = []

    def predict(self, features):
        self.features = features.copy()
        self.calls.append(features.copy())
        return np.tile([900.0, 500.0, 0.0, 200.0, 50.0], (len(features), 1))


def report(*, neural: float = 1.0, linear: float = 2.0) -> TrainingReport:
    return TrainingReport(
        ("train",), ("test",), 1, 2,
        {target: TargetMetric(neural, linear) for target in TARGET_COLUMNS},
    )


def bundle(metrics: TrainingReport | None = None) -> NeuralBundle:
    envelope = TrainingEnvelope(
        {"1": NumericRange(0.5, 2.0)}, NumericRange(0.5, 2.0), NumericRange(0.0, 0.0)
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


def test_negative_increments_never_reach_the_economics():
    """Отрицательное приращение обнуляется в строках и остаётся в диагностике.

    Эталонный расчётчик выбрасывает строку с отрицательным `*_Diff` целиком, и
    прогнозный ЧДД считался на обрезанном наборе: на кандидатах 24.08 исключалось
    60–86 строк, то есть поиск сравнивал числа разного состава (ревизия 03.09,
    2.2 #5). Обнулять надо в строках, а не в прогнозе: иначе исчезнет и
    предупреждение о самом нарушении.
    """
    invalid = bundle()
    invalid.models["1"] = InvalidModel()
    months = [ScheduleStep(pd.Timestamp(f"2014-{month:02d}-01").date(),
                           (ProducerControl("1", "OPEN", "LRAT", 30.0, 50.0),), ())
              for month in range(1, 4)]
    surrogate = NeuralSurrogate.create(invalid, months)

    rows = surrogate.predict(months)
    assert rows, "прогноз не пуст"
    for row in rows:
        for key in ("WLPT_Diff", "WOMT_Diff", "WWIT_Diff"):
            assert row[key] >= 0.0, f"{key} ушёл в экономику отрицательным"
        assert row["WOMT_Diff"] <= row["WLPT_Diff"] + 1e-9, "нефти больше, чем жидкости"

    # накопленные величины остаются монотонными — это следствие того же обнуления
    totals = [row["WLPT"] for row in rows]
    assert totals == sorted(totals)

    # и при этом само нарушение по-прежнему названо
    assert any("отрицательное WLPT_Diff" in warning
               for warning in surrogate.in_training_range(months))


def test_physical_failures_are_folded_to_one_line_per_well():
    """Одна скважина — одна строка с числом нарушений, а не строка на месяц.

    На настоящем расписании Модели Z таких строк 16 084 у 68 скважин, и страница
    цикла печатает их все: `{% for warning in result.warnings %}` в cycle.html.
    Счётчик «Предупреждения (16542)» формально верен и бесполезен.
    """
    invalid = bundle()
    invalid.models["1"] = InvalidModel()
    months = [ScheduleStep(pd.Timestamp(f"2014-{month:02d}-01").date(),
                           (ProducerControl("1", "OPEN", "LRAT", 30.0, 50.0),), ())
              for month in range(1, 13)]
    surrogate = NeuralSurrogate.create(invalid, months)

    physical = [warning for warning in surrogate.in_training_range(months)
                if "физические проверки" in warning]

    assert len(physical) == 1, physical
    assert "скв. 1" in physical[0]
    assert "отрицательное WLPT_Diff" in physical[0]


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


def test_multiple_doe_packages_have_a_canonical_order(tmp_path):
    first = tmp_path / "b.csv"
    second = tmp_path / "a.csv"
    pd.DataFrame({"scenario": ["one"], "value": [1]}).to_csv(first, index=False)
    pd.DataFrame({"scenario": ["two"], "value": [2]}).to_csv(second, index=False)

    forward = load_datasets([str(first), str(second)])
    reverse = load_datasets([str(second), str(first)])

    pd.testing.assert_frame_equal(forward, reverse)
    assert forward["scenario"].tolist() == ["a:two", "b:one"]


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


def test_the_linear_baseline_can_be_shipped_when_the_network_loses():
    """Обученный бейзлайн не должен выбрасываться вместе с проигравшей сетью.

    До 19.08 путь был такой: MLP не прошёл порог — в цикл едет тривиальный
    суррогат (экстраполяция кривой падения), хотя измеренно лучшая модель на
    прогонах ГГДМ уже обучена и тут же выброшена. Проигрыш сети — это довод
    против сети, а не против данных.

    Здесь проверяется, что модель линейного вида действительно попадает в
    артефакт и что порог для неё — свой: «лучше бейзлайна» к самому бейзлайну
    неприменим, поэтому спрашивается объяснённая дисперсия на отложенных
    сценариях.
    """
    frame = _dataset_frame()

    bundle = train(frame, seed=7, kind="linear")

    assert bundle.report.kind == "linear"
    assert all(isinstance(model, TransformedTargetRegressor) for model in bundle.models.values())
    # порог MLP к линейной модели не применяется, иначе она отвергала бы сама себя
    assert_quality(bundle.report)


def test_a_linear_surrogate_explaining_nothing_is_still_refused():
    """Свой порог не значит «никакого порога»: R² ≤ 0 — это «не лучше среднего»."""
    metrics = {
        target: TargetMetric(1.0, 1.0, 1.0, 1.0, neural_r2=0.5, linear_r2=-0.2)
        for target in TARGET_COLUMNS
    }
    report = TrainingReport(("train",), ("test",), 1, 2, metrics, kind="linear")

    with pytest.raises(ValueError, match="R²"):
        assert_quality(report)


def _dataset_frame():
    """Крошечный датасет в контракте обучения: шесть сценариев, две скважины.

    Шесть, а не четыре, — чтобы в отложенную часть попал не один сценарий, а два.
    При одном остаток `y − уровень` оказывается одной константой на всю
    отложенную часть, дисперсия отклика — нулевой, и `R²` отклика мерить не на
    чем: фикстура вырождается ровно в той величине, ради которой заведена.

    Диапазон множителей 0.8…1.5 сохранён: на нём стоят проверки области
    применимости в этом же файле.
    """
    rows = []
    for index, (scenario, multiplier) in enumerate(
        (("doe-000", 0.8), ("doe-001", 1.0), ("doe-002", 1.3), ("doe-003", 1.5),
         ("doe-004", 0.9), ("doe-005", 1.2))
    ):
        for month in range(1, 13):
            for well in ("1", "2"):
                base = 100.0 * multiplier + 3.0 * month + 10.0 * int(well)
                rows.append({
                    "scenario": scenario, "DATA": f"2014-{month:02d}-01", "well": well,
                    "multiplier": multiplier, "production_multiplier": multiplier,
                    "injection_multiplier": float("nan"), "is_shut": 0,
                    "WLPR": 30.0 * multiplier, "WWIR": 0.0,
                    "WLPT_Diff": base, "WOMT_Diff": base * 0.6,
                    "WWIT_Diff": 0.0, "THP": 100.0 + index, "BHP": 80.0 + index,
                })
    return pd.DataFrame(rows)


def test_an_idle_well_started_for_many_months_is_reported_once():
    """Одна мысль об одной скважине не должна повторяться по числу месяцев.

    Измерено ревизией 19.08 на настоящем расписании: сценарий с девятью
    переведёнными скважинами даёт 188 предупреждений, из которых 180 — одна и та
    же скважина, повторённая помесячно. Первый блок дедуплицирован `seen`, второй
    нет, поэтому защита читаема там, где она не нужна, и нечитаема ровно на том
    классе решений, ради которого заводилась (`docs/experiments.md`).
    """
    from datetime import date

    idle = [ScheduleStep(date(2014, month, 1),
                         (ProducerControl("1", "SHUT", "LRAT", None, 50.0),), ())
            for month in range(1, 13)]
    planned = [ScheduleStep(date(2014, month, 1),
                            (ProducerControl("1", "OPEN", "LRAT", 30.0, 50.0),), ())
               for month in range(1, 13)]
    surrogate = NeuralSurrogate.create(bundle(), idle)

    warnings = [w for w in surrogate.in_training_range(planned) if "без опорного режима" in w]

    assert len(warnings) == 1, warnings
    assert "2014-01-01" in warnings[0] and "2014-12-01" in warnings[0], (
        "диапазон месяцев обязан остаться в тексте: без него неясен масштаб")
    assert "12" in warnings[0]


def test_a_model_trained_on_one_kind_of_control_says_so_once():
    """Пакет одного вида — это одна причина, а не предупреждение на каждую скважину.

    Измерено ревизией 19.08: суррогат, обученный только на пакете по
    нагнетателям, выдаёт 71 предупреждение на **любое** расписание, включая
    историческое, — по одному на каждую добывающую скважину, управления которой
    в датасете нет вовсе. Условие «обучать на объединении пакетов» нигде не
    записано и из кода не следует: `assert_quality` смотрит метрики, а не
    покрытие координат управления. Защита в таком виде превращается в шум
    (`docs/experiments.md`).
    """
    from datetime import date

    trained = bundle()
    trained.control_wells = ("inj:9",)
    producers = [ScheduleStep(date(2014, 1, 1),
                              tuple(ProducerControl(well, "OPEN", "LRAT", 30.0, 50.0)
                                    for well in ("1", "2", "3")), ())]
    surrogate = NeuralSurrogate.create(trained, producers)

    warnings = [w for w in surrogate.in_training_range(producers)
                if "обучающем датасете" in w]

    assert len(warnings) == 1, warnings
    assert "3" in warnings[0], "число затронутых скважин обязано остаться видимым"
    assert "prod" in warnings[0]


def test_a_network_worse_than_the_mean_is_refused_even_if_it_beats_the_baseline():
    """Порог «лучше бейзлайна» — относительный, и этого мало.

    Измерено 20.08 на полном датасете: MLP выигрывает у линейной модели по MAE на
    всех пяти целях и потому проходит относительный порог, но `R²` по нефти у него
    **−0.814** — на невиданных сценариях он предсказывает нефть хуже, чем среднее
    по этим же сценариям. По скважинам картина ещё резче: `R² > 0` лишь у 7 из 90,
    медиана −1.38.

    Такая модель не годится вести оптимизацию: `choose_joint_rates` ищет максимум
    ЧДД по её градиенту. Абсолютный порог `R² > 0` был введён 19.08 только для
    линейного вида и не применялся к сети — асимметрия проверки, а не заслуга
    модели.
    """
    metrics = {
        target: TargetMetric(1.0, 2.0, 1.0, 2.0, neural_r2=0.5, linear_r2=0.4)
        for target in TARGET_COLUMNS
    }
    metrics["WOMT_Diff"] = TargetMetric(1.0, 2.0, 1.0, 2.0, neural_r2=-0.814, linear_r2=-2.0)
    losing = TrainingReport(("train",), ("test",), 1, 2, metrics)

    with pytest.raises(ValueError, match="R²"):
        assert_quality(losing)


def test_history_rows_do_not_count_towards_held_out_metrics():
    """История одинакова во всех сценариях, поэтому в метриках она — память.

    Находка ревизии 20.08 (`d9d9750`): фикс `42fec5d` убрал управляющий вектор с
    исторических строк, но сами строки остались — и теперь совпадают у всех
    сценариев **и по входу, и по отклику**. Разбиение по сценариям их не
    разделяет: каждая историческая строка отложенного сценария дословно
    встречается в обучающих. Две пятых held-out предсказываются точным повтором
    обучающей строки, то есть метрика меряет память, а не обобщение.

    Обучать на них по-прежнему законно — на нейтральном управлении они учат
    базовому уровню скважины. Мерить по ним нельзя.
    """
    rows = []
    for scenario, multiplier in (("a", 0.8), ("b", 1.0), ("c", 1.2), ("d", 1.4)):
        for month in range(1, 13):
            date = pd.Timestamp(f"2006-{month:02d}-01")
            # история: вход нейтрален, отклик одинаков у всех сценариев
            rows.append({
                "scenario": scenario, "DATA": date, "well": "1",
                "multiplier": np.nan, "production_multiplier": np.nan,
                "injection_multiplier": np.nan, "is_shut": "",
                "WLPR": 30.0, "WWIR": 0.0, "WLPT_Diff": 900.0, "WOMT_Diff": 600.0,
                "WWIT_Diff": 0.0, "THP": 100.0, "BHP": 80.0,
            })
        for month in range(1, 13):
            date = pd.Timestamp(f"2007-{month:02d}-01")
            rows.append({
                "scenario": scenario, "DATA": date, "well": "1",
                "multiplier": multiplier, "production_multiplier": multiplier,
                "injection_multiplier": np.nan, "is_shut": 0,
                "WLPR": 30.0 * multiplier, "WWIR": 0.0,
                "WLPT_Diff": 900.0 * multiplier, "WOMT_Diff": 600.0 * multiplier,
                "WWIT_Diff": 0.0, "THP": 100.0, "BHP": 80.0,
            })

    bundle = train(pd.DataFrame(rows), seed=3, control_start=date_(2007, 1, 1))

    assert bundle.report.metric_rows > 0
    assert bundle.report.metric_rows < bundle.report.rows, (
        "метрики обязаны считаться на управляемой части, а не на всей выборке"
    )
    assert bundle.report.metric_start == "2007-01-01"


def date_(year: int, month: int, day: int):
    from datetime import date

    return date(year, month, day)


def _schedule(rate: float, months: int = 12) -> list[ScheduleStep]:
    """Расписание, запрашивающее один и тот же режим каждый месяц 2014 года."""
    return [
        ScheduleStep(
            pd.Timestamp(f"2014-{month:02d}-01").date(),
            (ProducerControl("1", "OPEN", "LRAT", rate, 50.0),
             ProducerControl("2", "OPEN", "LRAT", rate, 50.0)),
            (),
        )
        for month in range(1, months + 1)
    ]


def _range_warnings(warnings: list[str]) -> list[str]:
    return [warning for warning in warnings if "вне обучающего диапазона" in warning]


def test_reference_schedule_is_inside_the_envelope_though_rates_never_reach_it():
    """Опорное расписание обязано проходить молча, даже если режим не достигался.

    Скважина сплошь не выходит на заданный режим: симулятор переключает её с
    `LRAT` на `BHP`, и достигнутый дебит ниже запрошенного в разы. До 20.08
    область применимости снималась с достигнутых `WLPR`, а проверялся
    запрошенный `liquid_rate` — на историческом расписании Модели Z это давало
    4 796 пометок «режим вне диапазона» при ожидаемом нуле
    (`docs/experiments.md`, 20.08). Защита, срабатывающая на любом расписании,
    не отличает опасное решение от безобидного.
    """
    bundle = train(_dataset_frame(), seed=7, hidden_layers=(4,), max_iter=50)
    reference = _schedule(rate=95.0)   # запрошено 95 при достигнутых 24…45

    surrogate = NeuralSurrogate.create(bundle, reference, require_quality=False)

    assert _range_warnings(surrogate.in_training_range(reference)) == []


def test_forcing_beyond_the_doe_is_still_flagged():
    """Форсаж за пределы обучающих множителей обязан оставаться видимым."""
    bundle = train(_dataset_frame(), seed=7, hidden_layers=(4,), max_iter=50)
    reference = _schedule(rate=95.0)

    surrogate = NeuralSurrogate.create(bundle, reference, require_quality=False)
    forced = _schedule(rate=95.0 * 2.0)   # датасет кончается на 1.5

    assert _range_warnings(surrogate.in_training_range(forced))


def test_coverage_is_asked_about_the_schedule_being_scaled_not_about_the_bare_multiplier():
    """Множитель поиска отсчитывается от базового расписания, область — от опорного.

    Это те же две несопоставимые величины, ради которых 20.08 переписан детектор
    (`cfa7153`), только на входе в поиск. `choose_joint_rates` масштабирует
    `base_steps`, а обучающие множители DoE отсчитаны от расписания дека. В
    интерфейсе это разные расписания: `base_steps` берётся из открытого прогона
    (`aios/web/app.py`), а опорное — из дека, поэтому цикл поверх любого
    форсированного прогона спрашивал покрытие не про тот режим, который собирался
    выбрать. Фильтр Р-43 при этом отвечал «покрыто», а детектор на собранном
    расписании — «вне диапазона»: две защиты об одном решении говорили разное.
    """
    bundle = train(_dataset_frame(), seed=7, hidden_layers=(4,), max_iter=50)
    reference = _schedule(rate=100.0)          # обучение: множители 0.8…1.5
    forced = _schedule(rate=140.0)             # открытый прогон уже ×1.4 к деку

    surrogate = NeuralSurrogate.create(bundle, reference, require_quality=False)

    # ×1.2 к форсированному прогону — это ×1.68 к деку, вне обучающего диапазона
    assert surrogate.covers_control("production", "1", 1.2, forced) is False
    assert _range_warnings(surrogate.in_training_range(_schedule(rate=168.0)))

    # то же ×1.2 поверх опорного расписания остаётся покрытым
    assert surrogate.covers_control("production", "1", 1.2, reference) is True
    assert surrogate.covers_control("production", "1", 1.2) is True

def test_watercut_parameterisation_cannot_predict_more_oil_than_liquid():
    """Нефть предсказывается через обводнённость, а не отдельной величиной.

    Измерено 20.08 на настоящем расписании: из 34 оставшихся пометок 33 —
    «нефти больше, чем жидкости» в самом прогнозе (скв. 1: 437,2 против 427,6 т).
    Это физически невозможно, и причина в параметризации: модель предсказывает
    `WLPT_Diff` и `WOMT_Diff` как две независимые величины, ничем не связанные
    между собой.

    Если предсказывать долю воды и восстанавливать нефть как
    `жидкость × (1 − обводнённость)`, нарушение становится невозможным по
    построению, а не отлавливается постфактум. Заодно у модели появляется
    величина в понятном диапазоне 0…1 вместо тонн, отличающихся по скважинам на
    порядки.
    """
    frame = _watercut_frame()

    bundle = train(frame, seed=11, targets="watercut")
    surrogate = NeuralSurrogate.create(
        bundle, [step(30.0)], require_quality=False
    )
    rows = surrogate.predict([step(30.0)])

    assert bundle.report.targets == "watercut"
    assert rows, "прогноз пуст"
    for row in rows:
        assert row["WOMT_Diff"] <= row["WLPT_Diff"] + 1e-9, (
            f"нефти больше жидкости: {row['WOMT_Diff']} > {row['WLPT_Diff']}"
        )
        assert row["WOMT_Diff"] >= 0.0


def _watercut_frame() -> pd.DataFrame:
    """Датасет, на котором наивная параметризация нарушает физику.

    Обводнённость растёт со временем, поэтому нефть падает быстрее жидкости, и
    модель, предсказывающая их порознь, легко ставит нефть выше жидкости на
    краях диапазона.
    """
    rows = []
    for scenario, multiplier in (("a", 0.7), ("b", 1.0), ("c", 1.3), ("d", 1.6)):
        for month in range(1, 25):
            liquid = 1000.0 * multiplier
            watercut = min(0.05 + 0.03 * month, 0.95)
            rows.append({
                "scenario": scenario,
                "DATA": pd.Timestamp(f"2007-{(month - 1) % 12 + 1:02d}-01")
                        + pd.DateOffset(years=(month - 1) // 12),
                "well": "1",
                "multiplier": multiplier, "production_multiplier": multiplier,
                "injection_multiplier": np.nan, "is_shut": 0,
                "WLPR": 30.0 * multiplier, "WWIR": 0.0,
                "WLPT_Diff": liquid, "WOMT_Diff": liquid * (1.0 - watercut),
                "WWIT_Diff": 0.0, "THP": 100.0, "BHP": 80.0,
            })
    return pd.DataFrame(rows)


def test_a_model_whose_quality_was_never_measured_says_so_in_its_label(tmp_path):
    """Артефакт без единого `R²` проходит ворота — значит он обязан быть подписан.

    `assert_quality` терпит отсутствующий `R²` намеренно: `None` означает «не
    измерено», а не «плохо», и отвергать за это отчёты, собранные до появления
    метрики, не за что. Но у модели вида `linear` — а с 20.08 в цикл идёт именно
    она — относительная проверка неприменима по построению, а абсолютная при
    пустом `R²` молчит. Замер 20.08: такой артефакт проходит `assert_quality`
    целиком, то есть ведёт оптимизацию наравне с проверенным и выглядит везде
    так же.

    Ворота при этом не трогаем: их поведение описано и на нём стоят фикстуры
    этого же файла. Закрываем молчание — подпись модели читает человек, который
    смотрит, чем посчитан прирост.
    """
    unmeasured = TrainingReport(
        ("train",), ("test",), 1, 2,
        {target: TargetMetric(1.0, 2.0) for target in TARGET_COLUMNS},
        kind="linear",
    )
    path = tmp_path / "surrogate.joblib"
    joblib.dump(bundle(unmeasured), path)

    selected = for_cycle(
        neural_path=path, history=[], reference_steps=[step()], fluids=Fluids()
    )

    assert "качество не измерено" in selected.name, selected.name


def test_a_measured_model_is_not_labelled_as_unmeasured(tmp_path):
    """Обратная сторона: подпись про неизмеренное качество не должна липнуть ко всем."""
    measured = TrainingReport(
        ("train",), ("test",), 1, 2,
        {target: TargetMetric(1.0, 2.0, neural_r2=0.7, linear_r2=0.8)
         for target in TARGET_COLUMNS},
        kind="linear",
    )
    path = tmp_path / "surrogate.joblib"
    joblib.dump(bundle(measured), path)

    selected = for_cycle(
        neural_path=path, history=[], reference_steps=[step()], fluids=Fluids()
    )

    assert "качество не измерено" not in selected.name


def test_a_well_without_a_model_is_named_not_silently_dropped():
    """Скважина, которой не досталось модели, обязана быть названа.

    `train` пропускает скважину, у которой нет строк по одну из сторон
    разбиения: `if train_rows.empty or test_rows.empty: continue`. Второй такой
    пропуск — скважина, все строки которой лежат раньше `control_start`.

    Дальше она исчезает бесшумно: `predict` идёт по `bundle.models`, поэтому
    прогноза по ней нет **ни одной строки**, ЧДД считается по остатку фонда и
    подаётся как ЧДД сценария, а поиск режимов про эту скважину не знает вовсе.
    Замер 21.08: датасет из двух скважин, обучена одна, `control_wells` содержит
    обе, прогноз содержит одну, предупреждений — ноль.

    На настоящем пакете DoE фонд у всех сценариев один, поэтому первый пропуск
    не срабатывает. Второй — вполне: скважина, остановленная до начала
    управления, управляемых строк не имеет.
    """
    frame = _dataset_frame()
    # скважина «2» встречается ровно в одном сценарии — при любом разбиении она
    # окажется либо только в обучении, либо только в отложенной части
    frame = frame[(frame["well"] == "1") | (frame["scenario"] == "doe-001")]

    bundle = train(frame, seed=7, hidden_layers=(4,), max_iter=50)
    surrogate = NeuralSurrogate.create(bundle, _schedule(rate=95.0), require_quality=False)

    assert set(bundle.models) == {"1"}, "проверка вырождена: пропуска не случилось"
    warnings = surrogate.in_training_range(_schedule(rate=95.0))

    assert any("2" in warning and "модел" in warning for warning in warnings), (
        f"скважина без модели не названа ни одним предупреждением: {warnings}"
    )


def test_a_well_absent_from_training_is_named_though_it_vanishes_from_the_forecast():
    """Скважина, которой в обучении не было вовсе, обязана быть названа.

    Это не тот случай, что в тесте выше. Там скважина есть в датасете, но модели
    ей не досталось. Здесь её нет нигде: ни в `models`, ни в `control_wells`, ни
    в опорном расписании — она приходит только в расписании, которое просят
    оценить. Ровно так выглядит перенос на новую скважину: кейс, чужой дек, ввод
    из бурения.

    Поведение проверено прогоном 02.09 и оно двойственное, поэтому и закреплено:
    `predict` идёт по `bundle.models`, поэтому строк по новой скважине нет **ни
    одной** и ЧДД считается по остатку фонда, — но `_unlearned_controls` называет
    её отдельной пометкой. Молчаливой потери нет; есть прогноз по остатку фонда с
    названной причиной, и решение о доверии принимает тот, кто читает пометку.

    Тест стережёт обе половины. Исчезновение из прогноза — поведение по
    построению, а не дефект: сказать про скважину, которой модель не видела,
    суррогату нечем. Дефектом было бы молчание.
    """
    frame = _dataset_frame()
    bundle = train(frame, seed=7, hidden_layers=(4,), max_iter=50)
    surrogate = NeuralSurrogate.create(bundle, _schedule(rate=95.0), require_quality=False)

    newcomer = [
        ScheduleStep(
            step.date,
            step.producers + (ProducerControl("99", "OPEN", "LRAT", 95.0, 50.0),),
            step.injectors,
        )
        for step in _schedule(rate=95.0)
    ]

    assert "99" not in set(bundle.models), "проверка вырождена: скважина известна модели"

    forecast_wells = {str(row["well"]) for row in surrogate.predict(newcomer)}
    assert "99" not in forecast_wells, (
        "поведение изменилось: незнакомая скважина попала в прогноз — "
        "проверить, чем именно её посчитали"
    )

    warnings = surrogate.in_training_range(newcomer)
    assert any("99" in warning for warning in warnings), (
        f"незнакомая скважина исчезла из прогноза молча: {warnings}"
    )


class SlopeModel:
    """Модель, чей прогноз зависит от управления, а не от одного времени.

    Нужна там, где проверяется само разделение: с постоянным прогнозом отклик
    равен нулю при любом управлении, и проверка вырождается в тавтологию.
    """

    def predict(self, features):
        control = features[:, 3:].sum(axis=1)
        return np.column_stack([
            100.0 * control, 60.0 * control, np.zeros_like(control),
            np.zeros_like(control), np.zeros_like(control),
        ])


def _decomposed(model=None) -> NeuralBundle:
    """Пучок, обученный с разделением уровня и отклика (`Р-55`).

    Уровень задан константой по обоим месяцам горизонта, центр управлений —
    единица, то есть опорный режим и есть центр.
    """
    trained = bundle()
    trained.models["1"] = model or SlopeModel()
    trained.levels = {"1": np.tile([900.0, 500.0, 0.0, 200.0, 50.0], (2, 1))}
    trained.control_center = np.ones((2, 1))
    return trained


def test_the_gate_refuses_a_model_that_only_knows_the_well_level():
    """Порог мерил уровень, а оптимизатор ходит по отклику — и порог его пускал.

    Замер 21.08 (`docs/experiments.md`): на объединении четырёх пакетов 92 %
    дисперсии целей приходится на уровень скважины в этом месяце, и тривиальный
    предсказатель «среднее по (скважина, месяц) поперёк сценариев», который про
    управление не знает вовсе, даёт `R² = 0.918` по жидкости и `0.933` по нефти.
    Абсолютная проверка `R² > 0` на такой величине не отсеивает ничего: её
    проходит модель, у которой отклика нет вовсе.

    Цена ошибки известна: прошедшая порог модель 20.08 обещала в полном цикле
    +34 772,4 млн руб., а расчёт ГГДМ дал −9 409,2 (`docs/report.md`).
    """
    good = {"neural_r2": 0.918, "linear_r2": 0.900, "level_r2": 0.918,
            "neural_response_r2": 0.40, "linear_response_r2": 0.30,
            "slope_response_r2": 0.530}
    metrics = {target: TargetMetric(1.0, 2.0, 1.0, 2.0, **good) for target in TARGET_COLUMNS}
    metrics["WOMT_Diff"] = TargetMetric(
        1.0, 2.0, 1.0, 2.0, neural_r2=0.933, linear_r2=0.900, level_r2=0.933,
        neural_response_r2=-0.024, linear_response_r2=0.10, slope_response_r2=0.285,
    )
    deaf = TrainingReport(("train",), ("test",), 1, 2, metrics)

    assert all(metrics[target].neural_r2 > 0 for target in ("WLPT_Diff", "WOMT_Diff")), (
        "проверка вырождена: старый порог обязан такую модель пропускать"
    )
    with pytest.raises(ValueError, match="отклик"):
        assert_quality(deaf)


def test_the_response_is_exactly_zero_at_the_centre_of_the_training_controls():
    """В центре обучающих управлений прогноз равен уровню — точно, а не примерно.

    Отклик берётся разностью прогноза модели в запрошенной точке и в центре, и
    в самом центре обе точки совпадают побайтово. Свойство нужно именно точное:
    приблизительный ноль означал бы, что часть уровня продолжает жить в отклике,
    а значит `R²` отклика снова меряет уровень.
    """
    surrogate = NeuralSurrogate.create(_decomposed(), [step(30.0)])

    centred = surrogate.predict([step(30.0)])    # множитель ровно 1.0 — это центр
    forced = surrogate.predict([step(45.0)])     # ×1.5 — отклик обязан появиться

    assert centred[0]["WLPT_Diff"] == 900.0
    assert centred[0]["WOMT_Diff"] == 500.0
    assert forced[0]["WLPT_Diff"] != 900.0, "проверка вырождена: модель не реагирует"


def test_the_decomposed_prediction_keeps_the_reference_calculator_contract():
    """Параметризация внутренняя: наружу идут те же 14 колонок в тех же единицах.

    Экономика и санити-чеки стоят ниже по течению и о разделении уровня и
    отклика знать не должны — накопленные величины остаются накопленными, а
    `WOMR` остаётся суточным дебитом, а не месячной добычей.
    """
    february = ScheduleStep(
        pd.Timestamp("2014-02-01").date(),
        (ProducerControl("1", "OPEN", "LRAT", 30.0, 50.0),),
        (),
    )
    surrogate = NeuralSurrogate.create(_decomposed(), [step(30.0)])

    rows = surrogate.predict([step(30.0), february])

    assert [list(row) for row in rows] == [economics.REQUIRED_COLUMNS] * 2
    assert rows[0]["WLPT_Diff"] == pytest.approx(900.0), "уровень не дошёл до контракта"
    assert rows[0]["WLPT"] == pytest.approx(rows[0]["WLPT_Diff"])
    assert rows[1]["WLPT"] == pytest.approx(rows[0]["WLPT_Diff"] + rows[1]["WLPT_Diff"])
    assert rows[0]["WOMR"] == pytest.approx(rows[0]["WOMT_Diff"] / 31)
    assert rows[1]["WOMR"] == pytest.approx(rows[1]["WOMT_Diff"] / 28)


def test_the_level_is_the_predictor_that_knows_nothing_about_control():
    """Уровень зависит только от пары «скважина × месяц» — и ни от чего больше.

    Это тот самый тривиальный предсказатель, которым 21.08 измерена планка. Если
    в него просочится что-нибудь от кандидата, модель снова начнёт угадывать
    уровень, и `R²` отклика станет таким же красивым и бессмысленным, каким был
    `R²` по уровню.

    Считается он по обучающим сценариям: взять среднее по всем — значит занести
    отклик отложенных сценариев в собственный прогноз.
    """
    frame = _dataset_frame()

    trained = train(frame, seed=7, hidden_layers=(4,), max_iter=50)

    learned = frame[frame["scenario"].isin(trained.report.train_scenarios)]
    expected = (learned[learned["well"] == "1"]
                .groupby("DATA")[list(TARGET_COLUMNS)].mean().to_numpy())
    np.testing.assert_allclose(trained.levels["1"], expected)


def test_response_metrics_are_measured_for_both_models_and_the_slope_baseline():
    """Оба набора метрик печатаются: по уровню — для сравнимости с журналом.

    Старые числа не выбрасываются, потому что в них написаны отчёт и
    `docs/experiments.md`, и без них прогоны разных дней перестанут
    сравниваться. Новые добавляются рядом.
    """
    trained = train(_dataset_frame(), seed=7, hidden_layers=(4,), max_iter=50)

    for target in ("WLPT_Diff", "WOMT_Diff"):
        metric = trained.report.metrics[target]
        measured = (metric.level_r2, metric.neural_response_r2,
                    metric.linear_response_r2, metric.slope_response_r2)
        assert all(value is not None for value in measured), (target, measured)
        assert all(np.isfinite(value) for value in measured), (target, measured)


def test_the_training_script_says_whether_the_response_beats_the_dumb_slope():
    """Превосходство над предыдущим уровнем — число при обучении, а не намерение.

    Порог его не требует, и это осознанно: ворота отвечают за «не хуже, чем
    ничего». Но узнавать, что обученная модель проиграла бейзлайну «один наклон
    на собственный множитель», надо там же, где считаются метрики, — иначе
    прогон уйдёт в журнал как «сеть обучена», а сравнение придётся
    восстанавливать задним числом.
    """
    from scripts.train_surrogate import response_verdict

    metrics = {target: TargetMetric(1.0, 2.0, neural_response_r2=0.60,
                                    slope_response_r2=0.530)
               for target in TARGET_COLUMNS}
    metrics["WOMT_Diff"] = TargetMetric(1.0, 2.0, neural_response_r2=0.10,
                                        slope_response_r2=0.285)

    verdict = response_verdict(TrainingReport(("train",), ("test",), 1, 2, metrics))

    assert "WLPT_Diff: 0.6 против 0.53 — бьёт" in verdict, verdict
    assert "WOMT_Diff: 0.1 против 0.285 — НЕ бьёт" in verdict, verdict


def _own_feature_bundle(model) -> NeuralBundle:
    """Пучок с признаками `own`: скважина-добытчик и скважина-нагнетатель."""
    trained = bundle(TrainingReport(
        ("train",), ("test",), 1, 2,
        {target: TargetMetric(1.0, 2.0) for target in TARGET_COLUMNS},
        features="own",
    ))
    trained.control_wells = ("prod:1", "inj:2")
    trained.models = {"1": model}
    trained.levels = {"1": np.tile([900.0, 500.0, 0.0, 200.0, 50.0], (2, 1))}
    trained.control_center = np.ones((2, 2))
    return trained


def _pair(producer: float, injector: float) -> ScheduleStep:
    return ScheduleStep(
        pd.Timestamp("2014-01-01").date(),
        (ProducerControl("1", "OPEN", "LRAT", producer, 50.0),),
        (InjectorControl("2", "WATER", "OPEN", "RATE", injector, 300.0),),
    )


def test_own_features_are_seven_columns_and_carry_the_wells_own_control():
    """При `own` модель видит семь столбцов, а не весь фонд.

    Три временных, два собственных множителя скважины и два агрегата. Столбца
    `inj:1` в датасете нет вовсе, и на его месте стоит нейтральная единица — то
    же значение, каким `_pivot_controls` заполняет пропуски: «нет управления» и
    «управление нейтрально» обязаны кодироваться одинаково, иначе отсутствующая
    роль читается как остановка.
    """
    recorder = RecordingModel()
    surrogate = NeuralSurrogate.create(
        _own_feature_bundle(recorder), [_pair(30.0, 40.0)], require_quality=False
    )

    surrogate.predict([_pair(45.0, 40.0)])

    asked = recorder.calls[0]         # второй вызов — центр, про расписание он молчит
    assert asked.shape[1] == 7, asked.shape
    own_production, own_injection = asked[0, 3], asked[0, 4]
    assert own_production == pytest.approx(1.5), "собственный множитель не дошёл"
    assert own_injection == pytest.approx(1.0), "отсутствующая роль обязана быть нейтральной"


def test_a_neighbour_reaches_the_model_through_the_field_aggregate():
    """Сосед обязан менять вход модели, иначе взаимодействие через пласт потеряно.

    Ради этого агрегаты и заведены. Цена их отсутствия измерена 20.08: суррогат
    предсказал, что при закачке, срезанной в 28 раз, нефть почти не пострадает,
    — она упала вчетверо, и цикл на суррогате дал −79,4 % ЧДД. Сокращая 112
    признаков до собственных, канал влияния соседа легко потерять целиком, и
    тогда правка чинила бы метрику ценой ровно того, что 20.08 стоило прогона.
    """
    recorder = RecordingModel()
    surrogate = NeuralSurrogate.create(
        _own_feature_bundle(recorder), [_pair(30.0, 40.0)], require_quality=False
    )

    surrogate.predict([_pair(30.0, 40.0)])
    unchanged = recorder.calls[0][0].copy()
    recorder.calls.clear()
    surrogate.predict([_pair(30.0, 80.0)])   # сосед-нагнетатель ×2, своя скважина та же
    with_neighbour = recorder.calls[0][0].copy()

    assert with_neighbour[3] == pytest.approx(unchanged[3]), "свой множитель не менялся"
    assert with_neighbour[6] == pytest.approx(2.0 * unchanged[6]), (
        "агрегат закачки по фонду обязан отследить соседа"
    )


def test_own_features_keep_the_contract_and_the_exact_zero_at_the_centre():
    """Сокращение признаков — внутреннее дело модели: наружу тот же контракт."""
    surrogate = NeuralSurrogate.create(
        _own_feature_bundle(SlopeModel()), [_pair(30.0, 40.0)], require_quality=False
    )

    centred = surrogate.predict([_pair(30.0, 40.0)])
    forced = surrogate.predict([_pair(45.0, 40.0)])

    assert list(centred[0]) == economics.REQUIRED_COLUMNS
    assert centred[0]["WLPT_Diff"] == 900.0, "в центре обязан остаться уровень"
    assert forced[0]["WLPT_Diff"] != 900.0, "проверка вырождена: модель не реагирует"


def test_an_unknown_feature_set_is_refused_by_name():
    with pytest.raises(ValueError, match="набор признаков"):
        train(_dataset_frame(), seed=7, features="соседи")


def test_the_default_feature_set_is_the_one_that_passes_the_gate():
    """Умолчание обучения — `own`: на `all` порог не проходит ни одна модель.

    Замер 21.08 на четырёх пакетах: `R²` отклика по нефти у сети −0,189, у
    линейной −11,96, то есть выбора между наборами по существу нет — `all` в
    цикл не попадает вовсе. Проверяется именно умолчание, а не наличие
    параметра: в артефакт уходит то, чем обучили по умолчанию, и молчаливый
    возврат к 112 признакам не должен быть возможен опечаткой в команде.

    Умолчание флага `--features` этим тестом не покрыто: парсер собирается
    внутри `main`, и вытащить его значение можно только разобрав текст файла.
    """
    bundle = train(_dataset_frame(), seed=7, hidden_layers=(4,), max_iter=50)

    assert bundle.report.features == "own"


def test_a_model_that_wins_on_average_but_loses_on_most_scenarios_is_refused():
    """Порог по медиане ловит то, чего не ловит порог по среднему.

    Замер 22.08 на четырёх пакетах: по нефти `R²` отклика 0,4215 — порог пройден,
    средний `Skill_MAE` +0,0444 — тоже, а медиана по сценариям −0,0581. Выигрыш
    создаётся меньшинством сценариев, решение принимается на каждом.
    """
    report = _report_with(neural_response_r2=0.42, neural_skill=0.044,
                          neural_skill_median=-0.058)

    with pytest.raises(ValueError, match="хуже бездействия"):
        assert_quality(report)


def test_a_positive_median_skill_passes():
    report = _report_with(neural_response_r2=0.62, neural_skill=0.354,
                          neural_skill_median=0.365)

    assert_quality(report)


def test_an_unmeasured_median_skill_is_not_treated_as_failure():
    """`None` — «не измерено»: артефакты до 22.08 этого поля не несут."""
    report = _report_with(neural_response_r2=0.62, neural_skill=None,
                          neural_skill_median=None)

    assert_quality(report)


def _report_with(**values) -> TrainingReport:
    metric = TargetMetric(
        neural_mae=1.0, linear_mae=2.0, neural_r2=0.95, linear_r2=0.9,
        level_r2=0.91, linear_response_r2=0.5, **values,
    )
    return TrainingReport(
        train_scenarios=("s1",), test_scenarios=("s2",), wells=1, rows=10,
        metrics=dict.fromkeys(TARGET_COLUMNS, metric),
    )


def test_the_trust_region_is_built_from_training_scenarios_only():
    """Отложенные сценарии в порог не попадают: иначе он мерит то, чего модель не видела."""
    bundle = train(dataset(), seed=7, control_start=None)

    region = bundle.trust_region
    assert region is not None
    assert set(region.scenarios) == set(bundle.report.train_scenarios)
    assert not set(region.scenarios) & set(bundle.report.test_scenarios)


def test_the_trust_region_survives_the_artifact_round_trip(tmp_path):
    """Область обязана лежать в том же файле, что и модель, а не рядом."""
    bundle = train(dataset(), seed=7, control_start=None)
    path = tmp_path / "bundle.joblib"
    joblib.dump(bundle, path)

    loaded = joblib.load(path)

    assert loaded.trust_region is not None
    assert loaded.trust_region.threshold == pytest.approx(bundle.trust_region.threshold)
    assert loaded.trust_region.scenarios == bundle.trust_region.scenarios


def test_an_artifact_saved_before_the_trust_region_existed_still_loads():
    """Артефакт до 22.08 обязан читаться как «области нет», а не падать.

    Проверено и на настоящем артефакте (`artifacts/srg-own.joblib`, обучен до
    правки): `trust_region` возвращает `None`. Тест воспроизводит тот же случай
    синтетически — у распакованного объекта поля в `__dict__` просто нет, и
    чтение обязано попасть в умолчание класса.
    """
    bundle = train(dataset(), seed=7, control_start=None)
    object.__delattr__(bundle, "trust_region")

    assert bundle.trust_region is None


def test_frozen_scenarios_reach_neither_training_nor_the_selection_split():
    """Внешний тест обязан быть исключён целиком, а не попасть в отложенную часть.

    Иначе он перестаёт быть внешним: по отложенной части мы выбираем модель,
    признаки и пороги, и каждый такой выбор понемногу подгоняет решение под неё.
    """
    frame = dataset()
    every = sorted(frame["scenario"].unique())
    frozen = frozenset(every[:2])

    bundle = train(frame, seed=7, control_start=None, holdout=frozen)

    assert bundle.report.holdout_scenarios == tuple(sorted(frozen))
    assert not frozen & set(bundle.report.train_scenarios)
    assert not frozen & set(bundle.report.test_scenarios)


def test_freezing_everything_refuses_instead_of_training_on_nothing():
    frame = dataset()
    every = frozenset(frame["scenario"].unique())

    with pytest.raises(ValueError, match="обучать не на чем"):
        train(frame, seed=7, control_start=None, holdout=every)


def test_without_a_holdout_the_report_says_so_rather_than_pretending():
    bundle = train(dataset(), seed=7, control_start=None)

    assert bundle.report.holdout_scenarios == ()


def test_the_median_skill_gate_judges_the_model_that_actually_goes_into_the_cycle():
    """Для линейной модели порог обязан смотреть на её медиану, а не на медиану сети.

    Дефект был тихим ровно в ту сторону, которая опасна: линейная модель могла
    пройти допуск по чужому показателю — по медиане MLP, которого в цикле нет.
    """
    metric = TargetMetric(
        neural_mae=1.0, linear_mae=2.0, neural_r2=0.95, linear_r2=0.9,
        neural_response_r2=0.6, linear_response_r2=0.5,
        neural_skill_median=0.4, linear_skill_median=-0.1,
    )
    report = TrainingReport(
        train_scenarios=("s1",), test_scenarios=("s2",), wells=1, rows=10,
        metrics=dict.fromkeys(TARGET_COLUMNS, metric), kind="linear",
    )

    with pytest.raises(ValueError, match="хуже бездействия"):
        assert_quality(report)


def test_the_bootstrap_interval_is_taken_over_scenarios_not_rows():
    """Интервал обязан быть широким, когда сценариев мало, — в этом весь смысл."""
    from aios.neural import skill_interval

    values = (-0.4, -0.2, -0.1, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4)
    low, high = skill_interval(values)

    assert low < 0 < high, "на таких данных знак медианы не определён, и это надо видеть"
    assert skill_interval(values) == (low, high), "seed фиксирован — результат повторяем"


def test_a_clearly_negative_sample_gives_an_interval_below_zero():
    from aios.neural import skill_interval

    low, high = skill_interval(tuple(-0.5 + 0.01 * index for index in range(12)))

    assert high < 0


def test_too_few_scenarios_give_no_interval_rather_than_a_narrow_one():
    from aios.neural import skill_interval

    assert skill_interval((0.1, 0.2)) is None


def test_per_scenario_skills_are_kept_so_the_interval_can_be_recomputed():
    """Сводка не восстанавливается обратно в сырьё — поэтому хранится сырьё."""
    bundle = train(dataset(), seed=7, control_start=None)

    values = bundle.report.metrics["WLPT_Diff"].neural_skill_scenarios
    assert values, "по сценариям должно быть хоть что-то"
    assert list(values) == sorted(values)


# --- донорский перенос: скважина без своей модели -----------------------------
# П. 10 плана ревизии 03.09. Замечание трекера на КТ-2: «если обучаете MLP, то
# переносим на другое месторождение — у вас всё сломается». Веса не переносятся,
# и спорить не с чем; между «своя модель» и «ничего» есть измеренная середина —
# медиана отклика доноров той же роли, 85 % качества по нефти (Р-157).


def _schedule_with_three(rate: float, months: int = 12) -> list[ScheduleStep]:
    """Расписание, где скважина донорского переноса тоже присутствует.

    Без неё проверка вырождается: `predict` обнуляет строки скважины, которой в
    расписании не назначен режим (`liquid_rate <= 0` → жидкость и нефть в ноль),
    и «отклик не перенесён» получилось бы у любой реализации.
    """
    return [
        ScheduleStep(
            pd.Timestamp(f"2014-{month:02d}-01").date(),
            tuple(ProducerControl(well, "OPEN", "LRAT", rate, 50.0)
                  for well in ("1", "2", "3")),
            (),
        )
        for month in range(1, months + 1)
    ]


def _frame_with_a_well_only_in_training(seed: int = 7):
    """Датасет, где скважина «3» встречается только в обучающих сценариях.

    Разбиение детерминировано, поэтому обучающая часть спрашивается у того же
    `split_scenarios`, а не угадывается: иначе тест зависел бы от того, куда
    ляжет сценарий, и «пропуска не случилось» приходилось бы ловить наугад.
    """
    frame = _dataset_frame()
    train_scenarios, _ = split_scenarios(sorted(frame["scenario"].unique()),
                                         seed=seed, test_fraction=0.25)

    # скважина «3» — добывающая, встречается только в обучающих сценариях
    donor_rows = frame[(frame["well"] == "1")
                       & (frame["scenario"].isin(train_scenarios))].copy()
    donor_rows["well"] = "3"

    # скважина «4» — нагнетательная и с моделью. Без неё проверка отбора доноров
    # по роли пуста: когда весь фонд добывающий, «доноры вперемешку» и «доноры
    # своей роли» дают один и тот же список, и мутация правила не роняет тест.
    injector = frame[frame["well"] == "1"].copy()
    injector["well"] = "4"
    injector["production_multiplier"] = float("nan")
    injector["injection_multiplier"] = injector["multiplier"]
    injector["WLPR"] = 0.0
    injector["WWIR"] = 40.0 * injector["multiplier"]
    injector["WWIT_Diff"] = injector["WLPT_Diff"]
    injector["WLPT_Diff"] = 0.0
    injector["WOMT_Diff"] = 0.0

    return (pd.concat([frame, donor_rows, injector], ignore_index=True),
            train_scenarios)


def test_a_well_without_its_own_model_is_carried_by_donors_of_the_same_role():
    """Прогноз по такой скважине есть — и он не пустой ряд и не голый уровень.

    До правки `predict` шёл по `bundle.models`, поэтому строк по ней не было ни
    одной: ЧДД считался по остатку фонда и подавался как ЧДД сценария. Теперь
    отклик берётся медианой доноров её роли на её же признаках, а уровень
    остаётся собственный — переносится реакция на управление, а не добыча.
    """
    frame, _ = _frame_with_a_well_only_in_training()

    bundle = train(frame, seed=7, hidden_layers=(4,), max_iter=50)

    assert "3" not in bundle.models, "проверка вырождена: скважине досталась модель"
    assert bundle.report.wells_without_model == ("3",)
    assert "3" in bundle.levels, "уровень скважины без модели не сохранён — переносить не на чем"

    surrogate = NeuralSurrogate.create(bundle, _schedule_with_three(rate=95.0), require_quality=False)
    assert surrogate.donor_wells() == {"3": ("1", "2")}, (
        "в доноры добывающей скважины попала нагнетательная «4»"
    )

    rows = surrogate.predict(_schedule_with_three(rate=140.0))
    carried = [row for row in rows if row["well"] == "3"]

    assert len(carried) == len([row for row in rows if row["well"] == "1"]), (
        "у скважины донорского переноса строк меньше, чем у скважины с моделью"
    )
    assert all(row["WLPT_Diff"] >= 0 for row in carried)
    assert any(row["WLPT_Diff"] > 0 for row in carried), (
        "строки есть, но все нулевые — это не прогноз, а тишина под другим именем"
    )


def test_the_donor_response_is_not_the_level_alone():
    """Перенос обязан нести отклик, а не повторять уровень.

    Без этой проверки «донорский прогноз» неотличим от «прогноза без модели
    вообще»: уровень есть у скважины и так, и молчаливая подстановка одного
    уровня выглядела бы работающим механизмом, ничего не перенося.
    """
    frame, _ = _frame_with_a_well_only_in_training()
    bundle = train(frame, seed=7, hidden_layers=(4,), max_iter=50)
    surrogate = NeuralSurrogate.create(bundle, _schedule_with_three(rate=95.0), require_quality=False)

    quiet = {row["DATA"]: row["WLPT_Diff"]
             for row in surrogate.predict(_schedule_with_three(rate=95.0)) if row["well"] == "3"}
    forced = {row["DATA"]: row["WLPT_Diff"]
              for row in surrogate.predict(_schedule_with_three(rate=160.0)) if row["well"] == "3"}

    assert quiet and forced
    assert any(abs(forced[stamp] - quiet[stamp]) > 1e-9 for stamp in quiet), (
        "прогноз скважины не реагирует на смену режима — отклик не перенесён"
    )


def test_a_donor_must_share_the_role_of_the_well_it_carries():
    """Доноры отбираются по роли: нагнетательную не переносят с добывающих.

    Первый прогон замера 02.09 брал доноров вперемешку и давал по закачке ровно
    нулевой Skill — механизм выглядел работающим и не работал (Р-157).
    """
    frame, _ = _frame_with_a_well_only_in_training()
    bundle = train(frame, seed=7, hidden_layers=(4,), max_iter=50)
    surrogate = NeuralSurrogate.create(bundle, _schedule_with_three(rate=95.0), require_quality=False)

    donors = surrogate.donor_wells()["3"]

    assert donors, "доноров не нашлось — переноса не будет"
    assert all(donor in bundle.models for donor in donors)
    assert "3" not in donors, "скважина не может быть донором самой себе"
    assert "4" in bundle.models, "фикстура вырождена: нагнетательной модели не досталось"
    assert "4" not in donors, (
        "нагнетательная попала в доноры добывающей — ровно та ошибка, которая "
        "02.09 давала по закачке нулевой Skill"
    )


def test_the_warning_names_the_transfer_and_its_measured_price():
    """Перенос — не бесплатный: предупреждение обязано нести его цену числом.

    Скорость без точности — подача; то же и здесь. 85 % по нефти и 25 % по
    закачке названы прямо в пометке, чтобы читающий видел, чем именно заполнен
    прогноз этой скважины.
    """
    frame, _ = _frame_with_a_well_only_in_training()
    bundle = train(frame, seed=7, hidden_layers=(4,), max_iter=50)
    surrogate = NeuralSurrogate.create(bundle, _schedule_with_three(rate=95.0), require_quality=False)

    warnings = surrogate.in_training_range(_schedule_with_three(rate=95.0))
    about = [one for one in warnings if "донорск" in one]

    assert about, f"перенос не назван ни одной пометкой: {warnings}"
    assert "85 %" in about[0] and "25 %" in about[0]
    assert "нет ни одной строки" not in about[0]


def test_the_clamp_keeps_unphysical_rows_out_of_the_economics():
    """В экономику не попадает ни одной строки «нефти больше жидкости».

    Модель предсказывает пять целей независимо, и на почти безводной скважине
    нефть перескакивает жидкость. Замер 05.09 на опорном расписании: 910 строк
    из 23 175 (3,93 %), медиана превышения 2,71 %, максимум 29,2 %. Путь в
    экономику (`clamp=True`) обязан обнулять этот класс полностью — иначе ЧДД
    считается на данных, которых не бывает.

    Проверяется на синтетическом наборе: настоящий артефакт есть не на всякой
    машине, а свойство клампа от него не зависит.
    """
    frame = _dataset_frame()
    bundle = train(frame, seed=7, hidden_layers=(4,), max_iter=50)
    surrogate = NeuralSurrogate.create(bundle, _schedule(rate=95.0),
                                       require_quality=False)

    raw = surrogate.predict(_schedule(rate=140.0), clamp=False)
    clamped = surrogate.predict(_schedule(rate=140.0), clamp=True)

    assert raw and clamped
    assert not [one for one in clamped
                if one["WOMT_Diff"] > one["WLPT_Diff"] > 0], (
        "нефизичная строка дошла до экономики"
    )
    assert not [one for one in clamped if one["WLPT_Diff"] < 0
                or one["WOMT_Diff"] < 0 or one["WWIT_Diff"] < 0], (
        "отрицательное приращение дошло до экономики"
    )


def test_the_diagnostics_still_see_what_the_clamp_hides():
    """Кламп правит прогноз, но не прячет нарушение от диагностики.

    `_physical_problems` смотрит прогноз **до** клампа: иначе обнуление скрыло бы
    само нарушение, и пометки «нефти больше жидкости» исчезли бы вместе с ним.

    Проверяется на **сдаваемом** артефакте: на синтетической фикстуре модель
    физична, и тест выродился бы в пропуск. Замер 05.09 на опорном расписании:
    910 нефизичных строк из 23 175 до клампа, 0 после, 31 пометка диагностики.
    """
    from pathlib import Path

    from aios.deck import parse_schedule

    artefact = Path(__file__).resolve().parent.parent / "artifacts" / "surrogate.joblib"
    schedule = Path(__file__).resolve().parent.parent / "runs" / "base" / "wells_schedule.inc"
    if not artefact.is_file() or not schedule.is_file():
        pytest.skip("нет сдаваемого артефакта или выгрузки runs/base")

    steps = [one for one in parse_schedule(schedule).steps
             if one.date >= date_(2007, 1, 1)]
    surrogate = NeuralSurrogate.create(joblib.load(artefact), steps,
                                       require_quality=False)

    raw = surrogate.predict(steps, clamp=False)
    unphysical = [one for one in raw if one["WOMT_Diff"] > one["WLPT_Diff"] > 0]
    clamped = surrogate.predict(steps, clamp=True)

    assert unphysical, "у сдаваемого артефакта нарушений нет — обновить замер в docs/surrogate.md"
    assert not [one for one in clamped if one["WOMT_Diff"] > one["WLPT_Diff"] > 0], (
        "нефизичная строка дошла до экономики"
    )
    # Именно это нарушение, а не «хоть какое-то»: при клампе строки обнуляются,
    # и диагностика всё равно найдёт другие поводы — тогда проверка ничего не
    # стережёт. Ищем пометку про нефть больше жидкости.
    said = [one for one in surrogate.in_training_range(steps)
            if "нефти больше" in one]
    assert said, (
        "кламп спрятал от диагностики именно «нефти больше жидкости» — "
        "она смотрит уже исправленный прогноз"
    )


# ── что нашёл мутационный прогон 05.09 (Р-221) ───────────────────────────────


def test_the_time_features_encode_the_month_and_the_season():
    """Признаки времени: номер месяца от начала и синус-косинус сезона.

    Пять мутаций сидели прямо здесь — в арифметике номера месяца, в делении на
    длину горизонта, в множителе `2π`. Сеть на этих признаках учится, и ошибка в
    них не падает, а тихо врёт: модель получает не тот момент времени и не тот
    сезон, а метрики на тех же признаках выглядят прежними.

    Проверяется числом, а не формой: январь 2015 при старте января 2014 — это
    двенадцатый месяц, и его сезон совпадает с январём 2014 по синусу и косинусу.
    """
    from aios.neural import _inference_features

    start = pd.Timestamp("2014-01-01")
    span = 24
    multipliers = np.array([1.0, 1.5])

    first = _inference_features(date(2014, 1, 1), multipliers, start, span)[0]
    year_later = _inference_features(date(2015, 1, 1), multipliers, start, span)[0]
    mid = _inference_features(date(2014, 7, 1), multipliers, start, span)[0]

    assert first[0] == 0.0, "старт горизонта обязан быть нулём по времени"
    assert year_later[0] == pytest.approx(12 / span), (
        f"январь через год получил время {year_later[0]} вместо {12 / span}: "
        "номер месяца или деление на длину горизонта считаются неверно"
    )
    assert mid[0] == pytest.approx(6 / span)

    # сезон цикличен: тот же месяц другого года даёт те же синус и косинус
    assert year_later[1] == pytest.approx(first[1])
    assert year_later[2] == pytest.approx(first[2])
    # синус и косинус проверяются **порознь**: подмена множителя в одном из них
    # проскакивает мимо проверки «различаются хоть чем-то» (05.09)
    assert first[1] == pytest.approx(np.sin(2 * np.pi / 12))
    assert first[2] == pytest.approx(np.cos(2 * np.pi / 12))
    assert mid[1] == pytest.approx(np.sin(2 * np.pi * 7 / 12))
    assert mid[2] == pytest.approx(np.cos(2 * np.pi * 7 / 12))
    assert mid[1] != pytest.approx(first[1]), "январь и июль совпали по синусу сезона"
    assert mid[2] != pytest.approx(first[2]), "январь и июль совпали по косинусу сезона"

    # множители идут следом за тремя признаками времени
    assert list(first[3:]) == [1.0, 1.5]


def test_watercut_of_a_month_without_liquid_is_zero_not_a_division():
    """Обводнённость при нулевой жидкости — ноль, а не деление на ноль.

    Мутация `liquid > 0` на `>= 0` переживала сюиту: строки с нулевой жидкостью
    в фикстурах не было. На выгрузке она появляется сама — остановленная
    скважина даёт ровно такую, и `1 - 0/0` дало бы `nan`, который тихо разъедет
    по обучению.
    """
    from aios.neural import _model_space

    values = np.array([[0.0, 0.0], [100.0, 30.0]])

    converted = _model_space(values, "watercut")

    assert not np.isnan(converted).any(), f"в целях появился nan: {converted}"
    assert converted[0, 1] == 0.0, "месяц без жидкости получил ненулевую обводнённость"
    assert converted[1, 1] == pytest.approx(0.7)


def test_a_missing_control_is_filled_with_the_neutral_multiplier():
    """Пропуск в таблице управлений заполняется единицей, а не нулём.

    Тот же класс, что в `trust` и `optimise`, и третье место, где он нашёлся:
    1,0 означает режим без изменения, 0 — закрытую скважину. С нулём месяц, о
    котором в датасете нет строки, учит сеть, что фонд стоял.
    """
    from aios.neural import _pivot_controls

    # в феврале строки нет — её и должен заполнить нейтральный множитель
    frame = pd.DataFrame({
        "scenario": ["s1", "s1"],
        "DATA": pd.to_datetime(["2014-01-01", "2014-03-01"]),
        "well": ["a", "a"],
    })
    values = pd.Series([1.5, 0.5], index=frame.index)
    full = pd.MultiIndex.from_product(
        [["s1"], pd.date_range("2014-01-01", "2014-03-01", freq="MS")],
        names=["scenario", "DATA"],
    )

    pivoted = _pivot_controls(frame, values, "m:", full)

    assert pivoted.loc[("s1", pd.Timestamp("2014-02-01")), "m:a"] == 1.0, (
        "пропущенный месяц заполнен не нейтральным множителем: "
        "сеть учится тому, что скважина стояла"
    )
    assert pivoted.loc[("s1", pd.Timestamp("2014-01-01")), "m:a"] == 1.5


def test_quality_is_required_unless_asked_otherwise():
    """Ворота качества суррогата включены по умолчанию.

    Мутация умолчания `require_quality = True` на `False` переживала сюиту:
    вызовы передают флаг явно. Умолчание, равное `False`, означало бы, что
    непроверенная модель проходит в цикл везде, где о воротах забыли, — и
    заявляемое число опёрлось бы на сеть, которую никто не допускал.
    """
    import inspect

    from aios.neural import NeuralSurrogate

    assert inspect.signature(NeuralSurrogate.create)         .parameters["require_quality"].default is True


def test_the_surrogate_bundle_cannot_be_edited_after_loading():
    """Загруженный суррогат неизменяем: артефакт подписан хэшем."""
    import dataclasses

    from aios.neural import NeuralSurrogate

    assert dataclasses.is_dataclass(NeuralSurrogate)
    assert any(one.name for one in dataclasses.fields(NeuralSurrogate))
    assert NeuralSurrogate.__dataclass_params__.frozen, (
        "суррогат стал изменяемым: модель можно подменить между загрузкой и прогнозом"
    )


def test_the_control_centre_puts_each_month_in_its_own_row():
    """Центр обучающего облака раскладывается по номеру месяца от старта.

    Две мутации сидят в этой арифметике: `год × 12 + месяц − месяц старта`.
    Ошибка здесь не падает — она сдвигает центр, от которого отсчитывается
    отклик, и прогноз оказывается смещён на столько же. Ровно тот случай, о
    котором предупреждает принцип «сеть не падает, а тихо врёт».

    Месяцы, которых в обучении нет, остаются нейтральной единицей — это тоже
    закреплено: иначе пропуск читался бы как «фонд стоял».
    """
    from aios.neural import _control_centre

    index = pd.MultiIndex.from_tuples(
        [("s1", pd.Timestamp("2014-01-01")), ("s1", pd.Timestamp("2014-03-01"))],
        names=["scenario", "DATA"],
    )
    controls = pd.DataFrame({"m:a": [1.5, 0.5]}, index=index)

    centre = _control_centre(controls, frozenset({"s1"}),
                             pd.Timestamp("2014-01-01"), 3)

    assert centre.shape == (4, 1)
    assert centre[0, 0] == pytest.approx(1.5), "январь не встал в нулевую строку"
    assert centre[2, 0] == pytest.approx(0.5), (
        f"март не встал во вторую строку: центр {centre.ravel()} — номер месяца "
        "считается неверно, и отклик отсчитывается от смещённой точки"
    )
    assert centre[1, 0] == 1.0, "месяц без обучения заполнен не единицей"
    assert centre[3, 0] == 1.0


def test_a_split_needs_at_least_three_scenarios():
    """Меньше трёх сценариев — отказ: train/test на двух не делится осмысленно.

    Мутация `< 3` на `< 0` переживала сюиту. С ней сплит на двух сценариях
    проходит молча, и «тест» оказывается либо пустым, либо тем же сценарием,
    что и обучение, — то есть метрика меряет запоминание.
    """
    with pytest.raises(ValueError, match="три сценария"):
        split_scenarios(["s1", "s2"], seed=1, test_fraction=0.25)

    train_set, test_set = split_scenarios(["s1", "s2", "s3"], seed=1,
                                          test_fraction=0.25)
    assert train_set and test_set
    assert not (train_set & test_set), "сценарий попал и в обучение, и в тест"


def test_the_prediction_adds_the_response_to_the_level():
    """Прогноз — уровень **плюс** отклик, а не минус.

    Мутация знака переживала сюиту: метрики считались на тех же данных, и
    разница уходила в число, которое никто не сверял с определением. Между тем
    это и есть устройство суррогата — «уровень скважины плюс отклик на
    управление», и знак здесь меняет прогноз на противоположный по смыслу.
    """
    from aios.neural import _raw_space

    level = np.array([[100.0, 30.0]])
    response = np.array([[10.0, 3.0]])

    combined = _raw_space(level + response, "raw")

    assert combined[0, 0] == pytest.approx(110.0), (
        "прогноз собран не как «уровень плюс отклик»"
    )
    assert combined[0, 1] == pytest.approx(33.0)


def test_the_quality_gate_names_the_kind_and_checks_its_own_metric():
    """Ворота качества зовут модель своим именем и смотрят свою метрику.

    Две мутации подменяли сравнение с именем вида на обратное. Первая меняет имя
    в отказе — линейная модель называлась бы MLP и наоборот; вторая страшнее:
    ворота проверяли бы `R²` **чужой** модели, то есть допускали бы сеть по
    качеству линейного бейзлайна.
    """
    metrics = {
        "WOMT_Diff": TargetMetric(neural_mae=1.0, linear_mae=2.0,
                                  neural_r2=-1.0, linear_r2=0.9),
    }
    report = TrainingReport(train_scenarios=("s1",), test_scenarios=("s2",),
                            wells=1, rows=10, metrics=metrics, kind="mlp")

    with pytest.raises(ValueError) as raised:
        assert_quality(report)

    text = str(raised.value)
    assert "MLP" in text, f"вид модели назван неверно: {text}"
    assert "линейный суррогат" not in text
    assert "WOMT_Diff" in text, (
        f"ворота смотрят чужую метрику: у MLP R² = -1,0, у линейной 0,9 — "
        f"отказ обязан прийти по первой. {text}"
    )

    # линейная модель с теми же числами проходит: её собственный R² выше нуля
    assert_quality(TrainingReport(train_scenarios=("s1",), test_scenarios=("s2",),
                                  wells=1, rows=10, metrics=metrics, kind="linear"))


def test_training_stores_the_model_kind_it_was_asked_for():
    """`kind="mlp"` кладёт в артефакт нейросеть, `kind="linear"` — линейную.

    Мутация `kind == "mlp"` на `!=` переживала сюиту: тесты смотрели на метрики
    и состав скважин, но не на **тип** сохранённой модели. С подменой обучение
    считает и сравнивает обе, а в артефакт кладёт не ту — заявляемое число
    опёрлось бы на модель, которую никто не выбирал, и метрики в отчёте
    относились бы к другой.
    """
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor

    def inner(model):
        """Достать регрессор из-под обёрток пайплайна."""
        while hasattr(model, "regressor_") or hasattr(model, "steps"):
            model = (model.regressor_ if hasattr(model, "regressor_")
                     else model.steps[-1][1])
        return model

    neural = train(dataset(), seed=42, hidden_layers=(8,), max_iter=300, kind="mlp")
    linear = train(dataset(), seed=42, kind="linear")

    assert neural.report.kind == "mlp"
    assert linear.report.kind == "linear"

    picked = inner(next(iter(neural.models.values())))
    assert isinstance(picked, MLPRegressor), (
        f"под видом MLP сохранено {type(picked).__name__}: артефакт несёт не ту "
        "модель, а метрики отчёта относятся к другой"
    )
    picked = inner(next(iter(linear.models.values())))
    assert isinstance(picked, Ridge), f"под видом линейной сохранено {type(picked).__name__}"


def test_the_warning_about_lost_wells_counts_the_hidden_ones():
    """Предупреждение о скважинах без модели честно считает скрытых.

    Мутация `len(lost) - 5` на `+ 5` переживала сюиту: список длиннее пяти в
    фикстурах не встречался, а первая редакция этого теста повторяла формулу
    вместо вызова — и мутацию, разумеется, не убивала. Здесь строка берётся из
    самого суррогата.

    Строку читает человек: «и ещё 13» вместо трёх меняет представление о том,
    насколько прогноз неполон, а прогноз по этим скважинам отсутствует целиком.
    """
    frame, _ = _frame_with_a_well_only_in_training()
    bundle = train(frame, seed=7, hidden_layers=(4,), max_iter=50)

    # восемь скважин без модели и без доноров: перенос невозможен, все «потеряны»
    lost = tuple(f"нет-{index}" for index in range(8))
    report = replace(bundle.report, wells_without_model=lost)
    surrogate = NeuralSurrogate.create(replace(bundle, report=report),
                                       _schedule_with_three(rate=95.0),
                                       require_quality=False)

    about = [one for one in surrogate.in_training_range(_schedule_with_three(rate=95.0))
             if "без модели и без уровня" in one]

    assert about, surrogate.in_training_range(_schedule_with_three(rate=95.0))
    assert "и ещё 3" in about[0], (
        f"скрытых скважин посчитано неверно: {about[0]}"
    )
    assert "скважин без модели и без уровня: 8" in about[0], about[0]


def test_the_metric_is_computed_as_level_plus_response():
    """Метрика собирается как «уровень **плюс** отклик» — знак проверяем числом.

    Мутация знака в этой сборке переживала сюиту: метрики считались, выглядели
    конечными, и никто не сверял их с определением суррогата. Замер 05.09
    показывает цену: по жидкости `neural_mae` растёт с 4,2 до 23,3, по нефти с
    3,8 до 16,6 — и MLP становится **хуже** линейного бейзлайна, на котором
    стоят ворота качества.

    Отсюда и форма проверки: не абсолютное число (оно зависит от seed и данных),
    а отношение к бейзлайну. Именно им меряется Skill, и именно оно ломается при
    неверном знаке.
    """
    bundle = train(dataset(), seed=42, hidden_layers=(8,), max_iter=300)

    liquid = bundle.report.metrics["WLPT_Diff"]
    assert liquid.neural_mae < liquid.linear_mae, (
        f"MLP хуже линейного бейзлайна по жидкости: {liquid.neural_mae:.4g} против "
        f"{liquid.linear_mae:.4g}. Так выглядит прогноз, собранный не как "
        "«уровень плюс отклик»"
    )
    assert liquid.neural_r2 > 0.9, (
        f"R² по жидкости {liquid.neural_r2:.4g} — модель не объясняет отложенные "
        "сценарии на данных, где отклик задан явно"
    )


def test_the_profile_covers_every_control_well_exactly_once():
    """Профиль покрывает каждое управление ровно один раз.

    `strict=True` в сборке профиля мутационный прогон помечает эквивалентным, и
    это верный вердикт: `_schedule_multipliers` строит вектор циклом **по тому
    же** списку `control_wells`, поэтому длины совпадают по построению —
    расхождение недостижимо даже на укороченном списке (проверено 05.09).
    Строгость оставлена как защита от будущей правки, а здесь закреплён сам
    инвариант, ради которого она стоит.

    Инвариант не косметический: по профилю шлюз доверия меряет расстояние до
    обучающих сценариев, и потерянное управление означало бы, что судится не
    то, что сдаётся.
    """
    bundle = train(dataset(), seed=42, hidden_layers=(8,), max_iter=300)
    steps = _schedule_with_three(rate=95.0)

    surrogate = NeuralSurrogate.create(bundle, steps, require_quality=False)
    profile = surrogate.control_profile(steps)

    assert profile, "профиль пуст"
    named = {key.rsplit(":", 1)[0] for key in profile}
    for well in bundle.control_wells:
        assert any(key.startswith(well) for key in named) or well in named, (
            f"управление {well} не попало в профиль: {sorted(named)[:5]}"
        )
    # укороченный список даёт укороченный профиль — длина следует за списком,
    # а не задаётся отдельно; отсюда и недостижимость расхождения
    short = replace(bundle, control_wells=bundle.control_wells[:-1])
    shorter = NeuralSurrogate.create(short, steps, require_quality=False)

    assert len(shorter.control_profile(steps)) < len(profile)


def test_the_trivial_baseline_does_not_get_the_joint_search():
    """У тривиального бейзлайна совместный поиск выключен — и это Р-198.

    Мутация умолчания переживала сюиту: проверялся только случай с обученной
    сетью. Совместный поиск оценивает кандидата целиком, с откликом соседей;
    у тривиального бейзлайна отклика соседей нет **по построению**, и включить
    ему этот путь значит вернуть ровно то, что стоило −63…−68 % ЧДД на полном
    расчёте.
    """
    from conftest import make_row

    history = [make_row("2014-01-01", "1", wlpr=30.0, womr=1.0,
                        liquid=900.0, oil=30.0)]

    selected = for_cycle(neural_path=None, history=history,
                         reference_steps=[step()], fluids=Fluids())

    assert selected.name == "тривиальный бейзлайн"
    assert not selected.joint_rate_search, (
        "тривиальному бейзлайну включён совместный поиск: оценка действия "
        "пойдёт по прогнозу, в котором отклика соседей нет"
    )


def test_the_chosen_model_is_labelled_by_its_actual_kind(tmp_path):
    """Подпись модели называет её вид, а не противоположный.

    Мутация сравнения переживала сюиту: линейную модель подпись звала бы MLP и
    наоборот. Подписью человек судит, чем посчитан прирост, — и на защите
    отвечает по ней.
    """
    path = tmp_path / "surrogate.joblib"
    joblib.dump(bundle(), path)

    selected = for_cycle(neural_path=path, history=[], reference_steps=[step()],
                         fluids=Fluids())

    kind = selected.model.bundle.report.kind
    if kind == "linear":
        assert "линейный" in selected.name, selected.name
        assert "MLP" not in selected.name
    else:
        assert "MLP" in selected.name, selected.name
        assert "линейный" not in selected.name


def test_an_unmeasured_quality_is_named_only_when_nothing_was_measured(tmp_path):
    """Подпись «качество не измерено» ставится, когда не измерено **ничего**.

    Мутация `and` на `or` переживала сюиту. С подменой подпись появлялась бы у
    модели, у которой измерена одна из двух метрик, — то есть у проверенной, и
    человек читал бы её как непроверенную. Ворота качества при этом не
    трогаются: `None` означает «не измерено», а не «плохо».
    """
    from dataclasses import replace as _replace

    ready = bundle()
    metrics = dict(ready.report.metrics)
    half = _replace(metrics["WLPT_Diff"], linear_r2=0.9, neural_r2=None)
    blind = _replace(metrics["WOMT_Diff"], linear_r2=None, neural_r2=None)

    measured = _replace(ready, report=_replace(
        ready.report, metrics={**metrics, "WLPT_Diff": half, "WOMT_Diff": half}))
    unmeasured = _replace(ready, report=_replace(
        ready.report, metrics={**metrics, "WLPT_Diff": blind, "WOMT_Diff": blind}))

    for artefact, expected in ((measured, False), (unmeasured, True)):
        path = tmp_path / f"{expected}.joblib"
        joblib.dump(artefact, path)
        selected = for_cycle(neural_path=path, history=[],
                             reference_steps=[step()], fluids=Fluids())
        assert ("качество не измерено" in selected.name) is expected, (
            f"подпись «{selected.name}» не отвечает измеренности метрик"
        )


def test_the_chosen_model_cannot_be_swapped_after_loading(tmp_path):
    """Выбранная модель неизменяема: подменить её между загрузкой и прогнозом нельзя."""
    import dataclasses

    path = tmp_path / "surrogate.joblib"
    joblib.dump(bundle(), path)
    selected = for_cycle(neural_path=path, history=[], reference_steps=[step()],
                         fluids=Fluids())

    with pytest.raises(dataclasses.FrozenInstanceError):
        selected.joint_rate_search = False


def test_the_trust_region_is_built_in_the_window_where_it_is_applied():
    """Обучающие профили и профиль кандидата считаются одним окном месяцев.

    Профиль кандидата берётся по расписанию периода управления
    (`optimise._distance_of`), а обучающие профили строились по всей матрице
    управлений — вместе с историей, где все множители нейтральны. На Модели Z
    это 145 нейтральных месяцев из 370: уровень сценария с множителем 1,5
    превращался в (145·1,0 + 225·1,5)/370 ≈ 1,30, а кандидат приходил со своими
    1,5. Систематический сдвиг раздувал расстояние — равномерный кандидат ×1,1
    давал 28,58 при пороге 24,13 и выбывал из поиска (Н-26).
    """
    from datetime import date as date_type

    import pandas as pd

    from aios import neural

    control_start = date_type(2007, 1, 1)
    # два года нейтральной истории и два года управления: ровно та пропорция,
    # что и на Модели Z, где истории 145 месяцев из 370
    months = pd.date_range("2005-01-01", "2008-12-01", freq="MS")
    scenarios = {f"s{number}": 1.2 + 0.1 * number for number in range(1, 9)}
    rows, index = [], []
    for scenario, multiplier in scenarios.items():
        for stamp in months:
            rows.append({"prod:1": multiplier if stamp.date() >= control_start else 1.0})
            index.append((scenario, stamp))
    controls = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(
        index, names=["scenario", "DATA"]))

    whole = neural._trust_region(controls, set(scenarios))
    windowed = neural._trust_region(controls, set(scenarios), control_start)

    assert whole is not None and windowed is not None
    level = windowed.features.index("prod:1|уровень")
    average = sum(scenarios.values()) / len(scenarios)

    assert windowed.centre[level] == pytest.approx(average, abs=0.01), (
        "в окне управления центр уровня — среднее самих множителей"
    )
    assert whole.centre[level] < average - 0.1, (
        "история обязана разбавлять профиль: именно этот сдвиг и раздувал "
        f"расстояние (центр по всей матрице {whole.centre[level]:.3f})"
    )


def test_a_holdout_that_matches_nothing_is_refused_not_ignored():
    """Заморозка, не нашедшая ни одного сценария, — отказ, а не тишина.

    Имя сценария несёт имя пакета: `load_datasets` приписывает «<stem файла>:».
    Заморозили по `dataset-all-wide.csv`, обучили по умолчанию
    (`dataset.csv dataset-levels.csv`) — совпадений ноль, все «замороженные»
    ушли в обучение, а `train_surrogate.py` напечатал «замороженный внешний
    тест: N сценариев» и отчёт их перечислил. Дальше любое «стало лучше»
    меряется по строкам, на которых модель обучалась, и заметить это нечем
    (Н-38).
    """
    import pandas as pd

    from aios import neural

    rows = []
    for scenario in ("пакет:s1", "пакет:s2", "пакет:s3", "пакет:s4"):
        for month in range(1, 13):
            rows.append({"scenario": scenario, "well": "1",
                         "DATA": f"2007-{month:02d}-01", "multiplier": 1.0,
                         "WLPT_Diff": 100.0, "WOMT_Diff": 20.0, "WWIT_Diff": 0.0,
                         "WLPR": 10.0, "WWIR": 0.0, "THP": 100.0, "BHP": 80.0})
    frame = pd.DataFrame(rows)

    with pytest.raises(ValueError, match="ни один замороженный сценарий"):
        neural.train(frame, seed=20260909, holdout=frozenset({"другой-пакет:s1"}))
