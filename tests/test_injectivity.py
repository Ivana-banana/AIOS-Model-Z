"""Предел приёмистости: скважина на пределе давления не примет прибавку уставки."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aios.injectivity import assess, bhp_limits, problems

SINCE = "2007-01-01"
LIMIT = 300.0


def _history(rate: np.ndarray, pressure: np.ndarray, well: str = "1") -> pd.DataFrame:
    dates = pd.date_range(SINCE, periods=len(rate), freq="MS")
    return pd.DataFrame({
        "DATA": dates, "well": well,
        "WWIR": rate, "BHP": pressure, "THP": pressure - 100.0,
        "WWIT_Diff": rate * 30.0,
    })


def test_the_matrix_branch_recovers_the_planted_injectivity_and_pressure():
    reservoir, index = 120.0, 2.0
    rate = 50.0 + 40.0 * np.sin(np.linspace(0.0, 12.0, 60))
    pressure = reservoir + rate / index

    state = assess(_history(rate, pressure), {"1": LIMIT}, SINCE)["1"]

    assert state.index == pytest.approx(index, rel=1e-6)
    assert state.reservoir == pytest.approx(reservoir, rel=1e-6)
    assert state.deliverable == pytest.approx(index * (LIMIT - reservoir))


def test_a_well_sitting_at_the_pressure_limit_is_marked_as_limited():
    rate = np.full(60, 40.0)
    state = assess(_history(rate, np.full(60, LIMIT)), {"1": LIMIT}, SINCE)["1"]

    assert state.months_at_limit == 60
    assert state.limited
    assert state.index is None

    found = problems({"1": state}, {"1": 80.0})
    assert len(found) == 1
    assert "останется в файле" in found[0]


def test_a_target_above_the_deliverable_rate_is_refused_with_a_number():
    reservoir, index = 120.0, 1.0
    rate = 50.0 + 20.0 * np.sin(np.linspace(0.0, 12.0, 60))
    state = assess(_history(rate, reservoir + rate / index), {"1": LIMIT}, SINCE)["1"]

    assert problems({"1": state}, {"1": 150.0}) == []
    refused = problems({"1": state}, {"1": 200.0})
    assert len(refused) == 1
    assert "выше предела 180" in refused[0]


def test_reservoir_pressure_drifting_over_the_history_does_not_fake_a_branch():
    """За 18 лет пластовое давление меняется — регрессия по всей истории врёт.

    Здесь расход постоянен, а давление падает вместе с пластом: связи «больше
    качаем — выше давление» нет вообще, и оценка обязана вернуть «не знаю».
    """
    rate = np.full(120, 60.0)
    pressure = np.linspace(280.0, 200.0, 120)

    state = assess(_history(rate, pressure), {"1": LIMIT}, SINCE)["1"]

    assert state.index is None
    assert problems({"1": state}, {"1": 120.0})[0].endswith(
        "принять или отвергнуть цель нечем")


def test_a_target_within_history_raises_no_objection():
    rate = np.full(60, 40.0)
    state = assess(_history(rate, np.full(60, LIMIT)), {"1": LIMIT}, SINCE)["1"]

    assert problems({"1": state}, {"1": 40.0}) == []


def test_limits_come_from_the_deck_and_not_from_a_constant_in_the_code():
    from aios.deck import parse_schedule_text

    schedule = parse_schedule_text(
        "DATES\n 1 JAN 2007 / \n/\n\n"
        "WCONINJE\n '42' 'WATER' 'OPEN' 'RATE' 15.0 1* 250 1* 1* /\n"
        " '49' 'WATER' 'OPEN' 'RATE' 200.0 1* 300 1* 1* /\n/\n"
    )

    assert bhp_limits(schedule) == {"42": 250.0, "49": 300.0}
