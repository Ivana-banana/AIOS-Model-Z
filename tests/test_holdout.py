"""Замороженный внешний тест: главное его свойство — что его нельзя пересобрать."""

from __future__ import annotations

import json

import pytest

from aios import holdout as holdout_module

SCENARIOS = [f"doe-{index:03d}" for index in range(40)]


def test_freezing_is_deterministic_by_seed():
    first = holdout_module.freeze(SCENARIOS, seed=42, count=8)
    second = holdout_module.freeze(SCENARIOS, seed=42, count=8)

    assert first.scenarios == second.scenarios
    assert len(first.scenarios) == 8


def test_a_different_seed_freezes_a_different_set():
    first = holdout_module.freeze(SCENARIOS, seed=1, count=8)
    second = holdout_module.freeze(SCENARIOS, seed=2, count=8)

    assert first.scenarios != second.scenarios


def test_freezing_too_much_leaves_nothing_to_learn_on():
    with pytest.raises(ValueError, match="осталось бы меньше"):
        holdout_module.freeze(SCENARIOS[:16], seed=1, count=8)


def test_a_frozen_test_cannot_be_rebuilt_over_itself(tmp_path):
    """Замороженный тест, который можно пересобрать, не заморожен."""
    path = tmp_path / "holdout.json"
    holdout_module.save(holdout_module.freeze(SCENARIOS, seed=1, count=8), path)

    with pytest.raises(FileExistsError, match="не пересобирают"):
        holdout_module.save(holdout_module.freeze(SCENARIOS, seed=2, count=8), path)


def test_the_file_carries_the_rule_not_just_the_names(tmp_path):
    """Файл читает человек, и он обязан объяснять сам себя."""
    path = tmp_path / "holdout.json"
    holdout_module.save(holdout_module.freeze(SCENARIOS, seed=1, count=8, note="48 сценариев"), path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert "не участвуют ни в обучении" in raw["rule"]
    assert raw["seed"] == 1 and raw["frozen_at"]
    assert len(raw["source"]) == len(SCENARIOS), "видно, из какого набора отбирали"


def test_loading_a_missing_file_is_a_state_not_an_error(tmp_path):
    assert holdout_module.load(tmp_path / "nope.json") is None


def test_a_round_trip_keeps_the_scenarios(tmp_path):
    path = tmp_path / "holdout.json"
    frozen = holdout_module.freeze(SCENARIOS, seed=1, count=8)
    holdout_module.save(frozen, path)

    loaded = holdout_module.load(path)

    assert loaded is not None
    assert loaded.scenarios == frozen.scenarios
    assert loaded.seed == frozen.seed


def test_a_leak_is_named_because_it_is_silent_otherwise():
    """Модель, обученная на замороженном сценарии, подтвердит любое улучшение."""
    frozen = holdout_module.freeze(SCENARIOS, seed=1, count=8)
    used = sorted(frozen.scenarios)[:2] + ["doe-999"]

    assert holdout_module.leaked(frozen, used) == sorted(frozen.scenarios)[:2]
    assert holdout_module.leaked(frozen, ["doe-999"]) == []
    assert holdout_module.leaked(None, used) == []
