"""Обучение суррогата без утечки между сценариями."""

from __future__ import annotations

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
    """Крошечный датасет в контракте обучения: три сценария, две скважины."""
    rows = []
    for index, (scenario, multiplier) in enumerate(
        (("doe-000", 0.8), ("doe-001", 1.0), ("doe-002", 1.3), ("doe-003", 1.5))
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
