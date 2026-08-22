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
