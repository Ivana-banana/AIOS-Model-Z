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

from run_all import execute, plan  # noqa: E402


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
