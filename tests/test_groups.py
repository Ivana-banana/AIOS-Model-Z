"""Группы фонда: разбиение обязано быть воспроизводимым и объяснимым."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aios.groups import (
    Group,
    build_groups,
    link_matrix,
    problems,
    stability,
    watercut,
)

SINCE = "2007-01-01"


def _rows(records: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    frame["DATA"] = pd.to_datetime(frame["DATA"])
    for column in ("WLPR", "WWIR", "WLPT_Diff", "WOMT_Diff", "WWIT_Diff"):
        if column not in frame:
            frame[column] = 0.0
    return frame


def _history(months: int, wells: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    dates = pd.date_range(SINCE, periods=months, freq="MS")
    records = []
    for well, series in wells.items():
        for index, date in enumerate(dates):
            liquid = float(series.get("WLPR", np.zeros(months))[index])
            injection = float(series.get("WWIR", np.zeros(months))[index])
            oil = float(series.get("WOMR", np.zeros(months))[index])
            records.append({
                "DATA": date, "well": well,
                "WLPR": liquid, "WWIR": injection,
                "WLPT_Diff": liquid * 30.0, "WOMT_Diff": oil * 30.0,
                "WWIT_Diff": injection * 30.0,
            })
    return _rows(records)


def test_a_planted_lagged_link_is_found_and_an_unrelated_well_is_not():
    months = 120
    noise = np.random.default_rng(0).normal(0.0, 1.0, months)
    injection = 100.0 + np.cumsum(noise)
    lag = 3
    response = np.concatenate([np.full(lag, 50.0), 50.0 + 0.4 * injection[:-lag]])
    unrelated = 50.0 + np.cumsum(np.random.default_rng(1).normal(0.0, 1.0, months))

    rows = _history(months, {
        "1": {"WWIR": injection},
        "2": {"WLPR": response, "WOMR": response * 0.1},
        "3": {"WLPR": unrelated, "WOMR": unrelated * 0.1},
    })

    links = link_matrix(rows, SINCE)
    assert links.loc["1", "2"] > 0.9
    assert links.loc["1", "3"] == 0.0


def test_producers_and_injectors_never_share_a_group():
    months = 60
    rate = np.full(months, 100.0)
    rows = _history(months, {
        str(well): ({"WLPR": rate, "WOMR": rate * (0.5 if well < 5 else 0.05)}
                    if well < 9 else {"WWIR": rate})
        for well in range(1, 17)
    })

    groups = build_groups(rows, SINCE)
    producers = {w for g in groups if g.kind == "producer" for w in g.wells}
    injectors = {w for g in groups if g.kind == "injector" for w in g.wells}
    assert not producers & injectors
    assert injectors == {str(w) for w in range(9, 17)}


def test_dry_and_watered_producers_land_in_different_groups():
    months = 60
    rate = np.full(months, 100.0)
    rows = _history(months, {
        "1": {"WLPR": rate, "WOMR": rate * 0.6},
        "2": {"WLPR": rate, "WOMR": rate * 0.5},
        "3": {"WLPR": rate, "WOMR": rate * 0.02},
        "4": {"WLPR": rate, "WOMR": rate * 0.01},
    })

    cut = watercut(rows, SINCE)
    assert cut["1"] < cut["4"]

    dry = next(g for g in build_groups(rows, SINCE) if g.name == "добывающие сухие")
    assert set(dry.wells) == {"1", "2"}


def test_weights_are_shares_of_historical_rate_and_sum_to_one():
    months = 60
    rows = _history(months, {
        "1": {"WLPR": np.full(months, 300.0), "WOMR": np.full(months, 30.0)},
        "2": {"WLPR": np.full(months, 100.0), "WOMR": np.full(months, 10.0)},
    })

    group = next(g for g in build_groups(rows, SINCE) if g.wells)
    assert sum(group.weights.values()) == pytest.approx(1.0)
    assert group.weights["1"] == pytest.approx(0.75)
    assert group.weights["2"] == pytest.approx(0.25)


def test_a_group_of_two_wells_is_reported_as_too_small():
    months = 60
    rate = np.full(months, 100.0)
    rows = _history(months, {
        "1": {"WLPR": rate, "WOMR": rate * 0.5},
        "2": {"WLPR": rate, "WOMR": rate * 0.02},
    })

    found = problems(build_groups(rows, SINCE), rows, SINCE)
    assert any("меньше четырёх" in message for message in found)


def test_a_group_holding_a_sliver_of_the_volume_is_reported():
    months = 60
    rows = _history(months, {
        **{str(w): {"WLPR": np.full(months, 1000.0), "WOMR": np.full(months, 500.0)}
           for w in range(1, 6)},
        **{str(w): {"WLPR": np.full(months, 1.0), "WOMR": np.full(months, 0.01)}
           for w in range(6, 11)},
    })

    found = problems(build_groups(rows, SINCE), rows, SINCE)
    assert any("базового объёма" in message for message in found)


def test_an_empty_window_yields_empty_groups_without_dividing_by_zero():
    rows = _history(12, {"1": {"WLPR": np.zeros(12)}})
    groups = build_groups(rows, SINCE)
    assert all(isinstance(group, Group) and group.weights == {} for group in groups)


def test_stability_is_high_when_the_split_does_not_depend_on_the_window():
    months = 120
    rate = np.full(months, 100.0)
    rows = _history(months, {
        str(well): {"WLPR": rate, "WOMR": rate * (0.5 if well < 5 else 0.02)}
        for well in range(1, 9)
    })

    assert stability(rows, SINCE, draws=5, seed=3) == pytest.approx(1.0)


def test_stability_does_not_pass_when_bootstrap_loses_comparable_wells():
    """ARI пустой выборки равен 1 формально, но устойчивость группы не доказывает."""
    rows = _history(5, {
        "1": {
            "WLPR": np.array([100.0, 0.0, 0.0, 0.0, 0.0]),
            "WOMR": np.array([50.0, 0.0, 0.0, 0.0, 0.0]),
        },
    })

    assert stability(rows, SINCE, draws=10, seed=0) == 0.0


def test_stability_requires_at_least_one_bootstrap_draw():
    rows = _history(3, {"1": {"WLPR": np.full(3, 100.0)}})

    with pytest.raises(ValueError, match="число выборок"):
        stability(rows, SINCE, draws=0, seed=0)
