"""Подбор режимов поверх суррогата."""

from __future__ import annotations

from datetime import date

import pytest
from conftest import make_row

from aios import economics, optimise, summary
from aios.deck import InjectorControl, ProducerControl, ScheduleStep
from aios.margin import Norms
from aios.schedule import MAX_LIQUID_RATE_M3_PER_DAY, validate
from aios.surrogate import DeclineSurrogate

FLUIDS = summary.Fluids()
START = date(2014, 1, 1)


@pytest.fixture
def norms():
    assumptions, _ = economics.load_norms()
    return Norms.from_assumptions(assumptions)


def history(well: str, watercut_start: float, watercut_step: float, rate: float = 30.0):
    rows = []
    for index in range(12):
        liquid_m3 = rate * 30
        watercut = min(watercut_start + watercut_step * index, 0.999)
        oil_m3 = liquid_m3 * (1 - watercut)
        oil_t = oil_m3 * FLUIDS.oil_density / 1000
        liquid_t = oil_t + (liquid_m3 - oil_m3) * FLUIDS.water_density / 1000
        rows.append(make_row(date(2013, index + 1, 1).isoformat(), well, wlpr=rate,
                             womr=oil_t / 30, liquid=liquid_t, oil=oil_t))
    return rows


def plan(wells, months: int = 24, rate: float = 30.0):
    steps = []
    for index in range(months):
        steps.append(
            ScheduleStep(
                date(2014 + index // 12, index % 12 + 1, 1),
                tuple(ProducerControl(well, "OPEN", "LRAT", rate, 50.0) for well in wells),
                (),
            )
        )
    return steps


def test_wacc_comes_from_the_organisers_file(norms):
    assert norms.wacc_rate == pytest.approx(0.10)


def test_discounted_margin_matches_the_reference_ebitda(norms):
    """Наша целевая функция — та же EBITDA методики, только со скидкой по годам."""
    rows = (
        [make_row(f"{year}-06-01", "A", wlpr=30, womr=5, liquid=900, oil=150)
         for year in (2014, 2015, 2016)]
    )
    result = economics.compute_npv(rows, headers=economics.REQUIRED_COLUMNS,
                                   start_date="2014-01-01")

    ours = sum(optimise.discounted_margin_by_well(rows, norms, 2014).values())
    theirs = sum(year["ebitdaM"] * year["discountFactor"] for year in result.raw["annual"])

    assert ours == pytest.approx(theirs, abs=1e-9)


def test_scale_shuts_the_well_at_zero():
    steps = optimise.scale(plan(["1"], months=1), 0.0)
    control = steps[0].producers[0]

    assert control.status == "SHUT"
    assert control.liquid_rate is None


def test_scale_respects_the_methodology_ceiling():
    steps = optimise.scale(plan(["1"], months=1, rate=400.0), 2.0)
    assert steps[0].producers[0].liquid_rate == MAX_LIQUID_RATE_M3_PER_DAY
    assert validate(steps) == []


def test_scale_leaves_injectors_and_shut_wells_alone():
    steps = [
        ScheduleStep(
            START,
            (ProducerControl("1", "SHUT", "LRAT", None, 50.0),),
            (InjectorControl("2", "WATER", "OPEN", "RATE", 40.0, 300.0),),
        )
    ]
    scaled = optimise.scale(steps, 0.5)

    assert scaled[0].producers[0].status == "SHUT"
    assert scaled[0].injectors[0].rate == 40.0


def test_a_watered_out_well_is_stopped_and_a_good_one_is_kept(norms):
    model = DeclineSurrogate.fit(history("good", 0.50, 0.005) + history("bad", 0.99, 0.0002),
                                 FLUIDS)
    choices = {c.well: c for c in optimise.choose_rates(
        model, plan(["good", "bad"]), norms, start_date=START)}

    assert choices["bad"].multiplier == 0.0
    assert choices["good"].multiplier > 0.0
    assert choices["bad"].gain_m > 0


def test_multipliers_must_include_the_baseline(norms):
    model = DeclineSurrogate.fit(history("1", 0.5, 0.005), FLUIDS)
    with pytest.raises(ValueError, match="1.0"):
        optimise.choose_rates(model, plan(["1"]), norms, start_date=START,
                              multipliers=(0.5, 2.0))


def test_choice_never_loses_to_the_baseline(norms):
    """Худшее, что может выбрать подбор, — оставить как есть."""
    model = DeclineSurrogate.fit(
        history("a", 0.50, 0.005) + history("b", 0.90, 0.004) + history("c", 0.99, 0.0002),
        FLUIDS,
    )
    for choice in optimise.choose_rates(model, plan(["a", "b", "c"]), norms, start_date=START):
        assert choice.chosen_margin_m >= choice.baseline_margin_m - 1e-9


def test_the_assembled_plan_raises_the_reference_npv(norms):
    """Главная проверка: подбор растит то самое число, которым нас оценивают."""
    wells = ["good", "bad"]
    model = DeclineSurrogate.fit(history("good", 0.50, 0.005) + history("bad", 0.99, 0.0002),
                                 FLUIDS)
    base_steps = plan(wells)

    choices = optimise.choose_rates(model, base_steps, norms, start_date=START)
    tuned = optimise.apply_choices(base_steps, choices)

    before = economics.compute_npv(model.predict(base_steps),
                                   headers=economics.REQUIRED_COLUMNS, start_date="2014-01-01")
    after = economics.compute_npv(model.predict(tuned),
                                  headers=economics.REQUIRED_COLUMNS, start_date="2014-01-01")

    assert after.chdd_m > before.chdd_m
    assert validate(tuned) == []


def test_apply_choices_touches_only_the_chosen_wells(norms):
    base_steps = plan(["a", "b"], months=2)
    choices = [
        optimise.RateChoice("a", 0.5, baseline_margin_m=1.0, chosen_margin_m=2.0),
        optimise.RateChoice("b", 1.0, baseline_margin_m=1.0, chosen_margin_m=1.0),
    ]
    tuned = optimise.apply_choices(base_steps, choices)
    rates = {c.well: c.liquid_rate for c in tuned[0].producers}

    assert rates["a"] == pytest.approx(15.0)
    assert rates["b"] == pytest.approx(30.0)


def test_explain_reports_what_changed_and_what_did_not(norms):
    model = DeclineSurrogate.fit(history("good", 0.50, 0.005) + history("bad", 0.99, 0.0002),
                                 FLUIDS)
    text = optimise.explain(optimise.choose_rates(model, plan(["good", "bad"]), norms,
                                                  start_date=START))

    assert "Изменено режимов" in text
    assert "скв. bad" in text
    assert "млн руб" in text


def test_explain_says_so_when_nothing_beats_the_baseline():
    choices = [optimise.RateChoice("1", 1.0, 5.0, 5.0)]
    assert "Режимы не менялись" in optimise.explain(choices)


class InteractingSurrogate:
    """Синтетический отклик, в котором результат A зависит от режима B."""

    def predict(self, steps):
        rates = {control.well: control.liquid_rate or 0.0 for control in steps[0].producers}
        a, b = rates["a"], rates["b"]
        oil = max(100.0 + b - (a - 15.0) ** 2 / 10.0 + 0.02 * a * b, 0.0)
        return [make_row("2014-01-01", "field", wlpr=oil, womr=oil, liquid=oil, oil=oil)]

    def in_training_range(self, steps):
        return []


def test_joint_search_never_reduces_field_npv():
    model = InteractingSurrogate()
    base = plan(["a", "b"], months=1)
    choices = optimise.choose_joint_rates(
        model, base, start_date=START, multipliers=(0.0, 0.5, 1.0)
    )
    tuned = optimise.apply_choices(base, choices)

    before = economics.compute_npv(
        model.predict(base), headers=economics.REQUIRED_COLUMNS, start_date=START.isoformat()
    )
    after = economics.compute_npv(
        model.predict(tuned), headers=economics.REQUIRED_COLUMNS, start_date=START.isoformat()
    )

    assert after.chdd_m >= before.chdd_m
    assert all(choice.chosen_chdd_m >= choice.baseline_chdd_m for choice in choices)
    assert any(choice.changed for choice in choices)
