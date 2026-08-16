"""Обучение суррогата без утечки между сценариями."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aios.neural import TARGET_COLUMNS, split_scenarios, train


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


def test_training_rejects_row_level_duplicates():
    frame = dataset()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="повторяются строки"):
        train(frame, seed=42)
