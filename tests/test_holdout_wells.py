"""Замер переноса на скважину без модели: доноры берутся по роли, Skill — от уровня."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "holdout_wells",
    Path(__file__).resolve().parent.parent / "scripts" / "holdout_wells.py")
holdout_wells = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(holdout_wells)


def test_roles_are_read_from_control_columns_not_from_names():
    """Роль скважины — это её колонки управления, а не догадка по имени.

    Скважина может быть и добывающей, и нагнетательной одновременно: на Модели Z
    таких в отложенной выборке 02.09 нашлась одна (скв. 52).
    """
    roles = holdout_wells.roles_of(("prod:1", "prod:2", "inj:2", "inj:3"))

    assert roles["1"] == frozenset({"prod"})
    assert roles["2"] == frozenset({"prod", "inj"})
    assert roles["3"] == frozenset({"inj"})


def test_skill_is_measured_against_the_level_and_zero_means_no_better_than_history():
    """`Skill = 1 − MAE / MAE(уровень)`: ноль — «не лучше истории», минус — хуже.

    Знак здесь несёт весь смысл замера, поэтому закреплён обеими сторонами:
    первый прогон 02.09 дал по закачке ровно ноль, и именно ноль сказал, что
    «общая модель фонда» там молча выродилась в уровень.
    """
    results = [
        {"mae": {"уровень": {"WOMT_Diff": 100.0}, "своя": {"WOMT_Diff": 50.0},
                 "фонд": {"WOMT_Diff": 100.0}}},
        {"mae": {"уровень": {"WOMT_Diff": 100.0}, "своя": {"WOMT_Diff": 50.0},
                 "фонд": {"WOMT_Diff": 120.0}}},
    ]

    summary = holdout_wells.summarise(results, "WOMT_Diff")

    assert summary["уровень"] == 100.0
    assert summary["skill:своя"] == pytest.approx(0.5)
    # 100 и 120 в среднем 110 — хуже, чем не предсказывать вовсе
    assert summary["skill:фонд"] == pytest.approx(-0.1)


def test_a_zero_level_error_does_not_divide_by_zero():
    """Скважина, у которой уровень идеален, не должна ронять замер."""
    results = [{"mae": {"уровень": {"THP": 0.0}, "своя": {"THP": 0.0},
                        "фонд": {"THP": 0.0}}}]

    assert holdout_wells.summarise(results, "THP")["skill:фонд"] == 0.0


class _ConstantModel:
    """Модель, чей отклик — заданная константа на каждую цель."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, x: np.ndarray) -> np.ndarray:
        # отклик считается разностью двух вызовов, поэтому константа обязана
        # зависеть от входа: иначе _response вернёт ноль у любой модели
        return np.outer(x[:, 3], np.full(5, self.value))


def _bundle(models: dict[str, object]) -> object:
    """Утиный `NeuralBundle`: `evaluate` берёт у него только эти поля."""
    report = type("Report", (), {"targets": "raw", "test_scenarios": ("s1",)})()
    return type("Bundle", (), {
        "models": models,
        "levels": {well: np.zeros((13, 5)) for well in models},
        "start_month": pd.Timestamp("2014-01-01"),
        "month_span": 12,
        "control_center": np.ones((13, 4)),
        "report": report,
    })()


def _frame() -> pd.DataFrame:
    rows = []
    for month in range(1, 4):
        for well, produces in (("1", True), ("2", True), ("3", False)):
            rows.append({
                "scenario": "s1", "DATA": pd.Timestamp(f"2014-{month:02d}-01"),
                "well": well, "multiplier": 1.2,
                "production_multiplier": 1.2 if produces else float("nan"),
                "injection_multiplier": float("nan") if produces else 1.4,
                "is_shut": 0,
                "WLPT_Diff": 10.0, "WOMT_Diff": 4.0, "WWIT_Diff": 0.0,
                "THP": 20.0, "BHP": 50.0,
            })
    return pd.DataFrame(rows)


def test_donors_are_restricted_to_the_same_role():
    """Донором нагнетательной скважины не может быть добывающая.

    Это не косметика, а исправление дефекта, найденного первым же прогоном
    02.09: фонд Модели Z — 71 добывающая против 41 нагнетательной, случайные
    доноры оказывались на две трети добывающими, модель добывающей отвечает по
    закачке нулём, и медиана такой смеси давала по `WWIT_Diff` ровно уровень.
    `Skill` выходил 0,0000 — не потому, что перенос невозможен, а потому, что
    замер спрашивал не тех.

    Проверяется по результату, а не по внутренностям: у скважины «3» (закачка)
    единственный донор её роли — она сама исключена, значит доноров нет вовсе, и
    отбор обязан не подсунуть ей добывающие модели.
    """
    models = {"1": _ConstantModel(1.0), "2": _ConstantModel(1.0),
              "3": _ConstantModel(9.0)}

    matched = holdout_wells.evaluate(
        _bundle(models), _frame(), ["1"], donors=8, seed=1,
        control_start=None, role_matched=True)
    mixed = holdout_wells.evaluate(
        _bundle(models), _frame(), ["1"], donors=8, seed=1,
        control_start=None, role_matched=False)

    # у скважины «1» роль «добыча»: донор её роли ровно один — скважина «2»
    assert matched[0]["donors"] == 1, "в доноры попала скважина чужой роли"
    assert mixed[0]["donors"] == 2, "проверка вырождена: без отбора доноров тоже один"
    assert matched[0]["mae"]["фонд"] != mixed[0]["mae"]["фонд"], (
        "отбор по роли не изменил прогноз — значит он ни на что не влияет"
    )
