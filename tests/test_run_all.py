"""Сквозной прогон одной командой — требование ТЗ п. 3.2.3.

Симулятор здесь не запускается: проверяется состав и порядок шагов, поведение
при отказе и то, что флаги симулятора остаются видимыми в команде. Сам расчёт
проверяется прогоном на ГГДМ и записывается в `docs/experiments.md`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_all import execute, plan, refuse_surrogate  # noqa: E402


def steps(**overrides):
    kwargs = {
        "case_name": "RUN",
        "processes": 4,
        "flow_args": ["--parsing-strictness=low"],
        "config_path": "config.toml",
        "skip_base": False,
    }
    kwargs.update(overrides)
    return plan(Path("runs/base"), Path("runs/policy"), **kwargs)


def commands(**overrides) -> list[str]:
    return [" ".join(command) for _title, command in steps(**overrides)]


def test_full_run_has_five_steps():
    assert len(steps()) == 5


def test_simulator_runs_twice_around_the_cycle():
    """Расчёт базового и расчёт сценария — иначе прирост не с чем сравнивать."""
    flow_steps = [c for c in commands() if "run_flow.sh" in c]
    assert len(flow_steps) == 2


def test_confirmation_is_the_last_step():
    """Пока не прошёл полный расчёт, прирост — прогноз, а не результат."""
    assert "--confirm" in commands()[-1]


def test_forecast_step_precedes_the_confirmation():
    order = commands()
    forecast = next(i for i, c in enumerate(order) if "--out" in c)
    confirm = next(i for i, c in enumerate(order) if "--confirm" in c)
    assert forecast < confirm


def test_flow_arguments_stay_visible_in_the_command():
    """`--parsing-strictness=low` меняет физику и обязан быть виден, а не зашит."""
    for command in commands():
        if "run_flow.sh" in command:
            assert "--parsing-strictness=low" in command


def test_no_flow_arguments_by_default():
    for command in commands(flow_args=()):
        assert "parsing-strictness" not in command


def test_skip_base_starts_from_the_cycle():
    shortened = commands(skip_base=True)
    assert len(shortened) == 3
    assert "build_case.py" not in " ".join(shortened)


def test_process_count_reaches_the_simulator():
    assert any("run_flow.sh" in c and " 8 " in f"{c} " for c in commands(processes=8))


def test_config_path_is_passed_to_every_python_step():
    for command in commands(config_path="config-nobalance.toml"):
        if ".py" in command:
            assert "config-nobalance.toml" in command


# --- поведение при отказе ---------------------------------------------------

def test_all_steps_run_when_each_succeeds():
    seen: list[list[str]] = []

    def runner(command):
        seen.append(list(command))
        return 0

    assert execute(steps(), runner) == 0
    assert len(seen) == 5


def test_failure_stops_the_run_immediately():
    """Шаг 4 на непосчитанном шаге 3 дал бы не ошибку, а тихо неверное число."""
    seen: list[list[str]] = []

    def runner(command):
        seen.append(list(command))
        return 0 if len(seen) < 2 else 3

    assert execute(steps(), runner) == 3
    assert len(seen) == 2, "после отказа прогон обязан остановиться"


@pytest.mark.parametrize("code", [1, 2, 137])
def test_failure_code_is_propagated(code: int):
    assert execute(steps(), lambda _command: code) == code


def test_cycle_runs_on_the_baseline_unless_a_model_is_given():
    """Без ключа шаг 3 идёт на тривиальном бейзлайне — и это видно в команде."""
    assert not any("--surrogate-model" in command for command in commands())


def test_trained_surrogate_reaches_the_single_command_run():
    """Обученный суррогат обязан попадать в сквозной прогон, а не только в отдельную команду.

    Иначе «одна команда» ТЗ п. 3.2.3 всегда считает тривиальным бейзлайном, и
    нейросетевая часть трека 2 в сквозном прогоне не участвует вовсе.
    """
    cycle = [command for command in commands(surrogate_model="artifacts/surrogate.joblib")
             if "run_cycle.py" in command and "--confirm" not in command]

    assert len(cycle) == 1
    assert "--surrogate-model artifacts/surrogate.joblib" in cycle[0]


def test_a_missing_model_is_refused_before_the_simulator(tmp_path):
    """Отказ обязан приходить до расчёта: базовый прогон стоит 11 минут."""
    assert refuse_surrogate(tmp_path / "нет.joblib") is not None


def test_a_foreign_artifact_is_refused_by_name(tmp_path):
    import joblib

    path = tmp_path / "чужое.joblib"
    joblib.dump({"это": "не модель"}, path)

    refusal = refuse_surrogate(path)

    assert refusal is not None and "dict" in refusal


def test_a_model_below_the_quality_gate_is_refused_before_the_simulator(tmp_path):
    """Тот же порог, что и в цикле, только применённый до первого расчёта."""
    import joblib
    import pandas as pd

    from aios.neural import NeuralBundle, TargetMetric, TrainingReport
    from aios.surrogate import NumericRange, TrainingEnvelope

    # сеть хуже линейного бейзлайна по обеим целям порога
    metrics = {target: TargetMetric(neural_mae=3.0, linear_mae=1.0)
               for target in ("WLPT_Diff", "WOMT_Diff", "WWIT_Diff", "THP", "BHP")}
    path = tmp_path / "слабая.joblib"
    joblib.dump(
        NeuralBundle(
            ("prod:1",), {}, pd.Timestamp("2007-01-01"), 12,
            TrainingEnvelope({"prod:1": NumericRange(0.5, 1.5)},
                             NumericRange(0.5, 1.5), NumericRange(0.0, 0.0)),
            TrainingReport(("a",), ("b",), 1, 2, metrics),
        ),
        path,
    )

    refusal = refuse_surrogate(path)

    assert refusal is not None and "порог" in refusal
