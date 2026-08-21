"""Точный вклад групп в ЧДД по полному плану коалиций (разбор 4-8).

Математика проверяется без симулятора: значения коалиций подставляются руками,
и от функции требуется ровно то, ради чего Шепли и берут — сумма вкладов равна
фактическому приросту, а не «примерно равна».
"""

from __future__ import annotations

import pytest

from aios import shapley

GROUPS = ["A", "B", "C"]


def game(values: dict[str, float], source: str = shapley.FULL_RUN) -> list[shapley.Coalition]:
    """Игра из записи «AB: 120» — пустая коалиция обозначается пустой строкой."""
    return [
        shapley.Coalition(frozenset(name), npv, source)
        for name, npv in values.items()
    ]


ADDITIVE = {"": 100.0, "A": 110.0, "B": 130.0, "C": 140.0,
            "AB": 140.0, "AC": 150.0, "BC": 170.0, "ABC": 180.0}


# ── план ─────────────────────────────────────────────────────────────────────

def test_the_plan_is_every_coalition_in_a_fixed_order():
    found = shapley.plan(GROUPS)

    assert len(found) == 8
    assert found[0] == frozenset()
    assert found[-1] == frozenset(GROUPS)
    assert shapley.plan(["C", "A", "B"]) == found, "порядок не зависит от порядка ввода"


def test_four_groups_cost_sixteen_runs():
    """Наши четыре группы (Р-64) — это 16 полных прогонов, а не 8."""
    assert len(shapley.plan(["A", "B", "C", "D"])) == 16


def test_repeated_group_names_are_refused():
    with pytest.raises(ValueError, match="повторяются"):
        shapley.plan(["A", "A", "B"])


def test_an_empty_game_is_refused():
    with pytest.raises(ValueError, match="игра не определена"):
        shapley.plan([])


# ── вклады ───────────────────────────────────────────────────────────────────

def test_an_additive_game_gives_each_group_exactly_its_own_effect():
    """Без взаимодействий Шепли обязан вернуть просто `v({i}) − v(∅)`."""
    found = shapley.shapley(GROUPS, game(ADDITIVE))

    assert found["A"] == pytest.approx(10.0)
    assert found["B"] == pytest.approx(30.0)
    assert found["C"] == pytest.approx(40.0)


def test_the_contributions_add_up_to_the_actual_uplift():
    """Свойство, ради которого метод и взят: сумма равна заявляемому приросту."""
    known = game(ADDITIVE | {"ABC": 200.0, "AB": 155.0})
    found = shapley.shapley(GROUPS, known)

    assert shapley.efficiency_gap(GROUPS, known, found) == pytest.approx(0.0, abs=1e-9)
    assert sum(found.values()) == pytest.approx(100.0)


def test_synergy_is_split_evenly_between_the_pair_that_creates_it():
    """Пара даёт +20 сверх суммы одиночных — каждому по половине, это и есть Шепли."""
    known = game(ADDITIVE | {"AB": 160.0, "ABC": 200.0})
    found = shapley.shapley(GROUPS, known)

    assert found["A"] > 10.0 and found["B"] > 30.0
    assert found["A"] - 10.0 == pytest.approx(found["B"] - 30.0)


def test_a_group_that_only_destroys_value_gets_a_negative_contribution():
    known = game(ADDITIVE | {"C": 80.0, "AC": 90.0, "BC": 110.0, "ABC": 120.0})
    found = shapley.shapley(GROUPS, known)

    assert found["C"] < 0


def test_an_incomplete_game_has_no_contributions_at_all():
    """Половина посчитанных коалиций не даёт половины ответа — она не даёт никакого."""
    partial = game({key: value for key, value in ADDITIVE.items() if key != "BC"})

    with pytest.raises(ValueError, match="не определён на неполной игре"):
        shapley.shapley(GROUPS, partial)

    assert shapley.missing(GROUPS, partial) == [frozenset("BC")]


# ── синергии ─────────────────────────────────────────────────────────────────

def test_pairwise_synergy_is_measured_against_the_state_of_the_others():
    known = game(ADDITIVE | {"AB": 160.0})

    assert shapley.interaction(known, "A", "B", {"C": False}) == pytest.approx(20.0)


def test_a_synergy_that_flips_sign_is_not_averaged_away():
    """Ровно то, из-за чего разбор требует обе условные синергии, а не одну.

    При выключенной C пара A×B даёт +20, при включённой — −30. Одно усреднённое
    число описало бы несуществующее среднее и спрятало бы обе ситуации.
    """
    known = game(ADDITIVE | {"AB": 160.0, "ABC": 150.0})

    found = shapley.conditional_interactions(GROUPS, known, "A", "B")

    assert found[frozenset()] == pytest.approx(20.0)
    assert found[frozenset("C")] < 0
    assert "меняет знак" in shapley.explain(GROUPS, known)


def test_a_group_cannot_be_both_the_pair_and_the_background():
    with pytest.raises(ValueError, match="одновременно"):
        shapley.interaction(game(ADDITIVE), "A", "B", {"A": True})


# ── честность вывода ─────────────────────────────────────────────────────────

def test_surrogate_values_are_never_presented_as_money():
    """Маркировка обязательна: денежным вкладом называется только полный расчёт."""
    known = game(ADDITIVE, source=shapley.SURROGATE)

    text = shapley.explain(GROUPS, known)

    assert "ВНИМАНИЕ" in text and "прогноз, а не денежный вклад" in text
    assert shapley.as_dict(GROUPS, known)["all_from_full_runs"] is False


def test_full_run_values_carry_no_warning():
    text = shapley.explain(GROUPS, game(ADDITIVE))

    assert "ВНИМАНИЕ" not in text
    assert shapley.as_dict(GROUPS, game(ADDITIVE))["all_from_full_runs"] is True


def test_the_explanation_says_the_contribution_is_not_a_property_of_the_reservoir():
    written = shapley.as_dict(GROUPS, game(ADDITIVE))

    assert "смена baseline меняет знак" in written["scope"]
    assert "не считался" in written["confidence_interval"]


def test_an_unfinished_plan_explains_what_is_still_missing():
    partial = game({key: value for key, value in ADDITIVE.items() if key not in ("BC", "C")})

    text = shapley.explain(GROUPS, partial)

    assert "вкладов нет" in text
    assert "не посчитано 2 коалиций" in text
