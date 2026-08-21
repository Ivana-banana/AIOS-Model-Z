"""Подбор режимов поверх суррогата."""

from __future__ import annotations

from datetime import date

import pytest
from conftest import make_row

from aios import economics, optimise, planner, summary
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


class InjectionSurrogate:
    def predict(self, steps):
        production = steps[0].producers[0].liquid_rate or 0.0
        injection = steps[0].injectors[0].rate or 0.0
        oil = max(100.0 + production + injection - (injection - 60.0) ** 2 / 20.0, 0.0)
        return [make_row("2014-01-01", "field", wlpr=oil, womr=oil, liquid=oil, oil=oil)]

    def in_training_range(self, steps):
        return []


def test_joint_search_can_change_injection_and_apply_the_choice():
    base = [
        ScheduleStep(
            START,
            (ProducerControl("p", "OPEN", "LRAT", 30.0, 50.0),),
            (InjectorControl("i", "WATER", "OPEN", "RATE", 40.0, 300.0),),
        )
    ]

    choices = optimise.choose_joint_rates(
        InjectionSurrogate(), base, start_date=START, multipliers=(0.5, 1.0, 1.5)
    )
    tuned = optimise.apply_choices(base, choices)
    injection = next(choice for choice in choices if choice.kind == "injection")

    assert injection.multiplier == 1.5
    assert tuned[0].injectors[0].rate == pytest.approx(60.0)
    assert validate(tuned) == []


# ── область применимости проверяется до выбора ───────────────────────────────


class ForcingSurrogate:
    """Прогноз растёт с дебитом без предела — так суррогат и врёт за границей обучения."""

    def predict(self, steps):
        oil = 100.0 + (steps[0].producers[0].liquid_rate or 0.0)
        return [make_row("2014-01-01", "a", wlpr=oil, womr=oil, liquid=oil, oil=oil)]

    def in_training_range(self, steps):
        return []


class CoveredForcingSurrogate(ForcingSurrogate):
    """Тот же прогноз плюс запрос покрытия: выше 1.25 в обучении такого режима не было."""

    def covers_control(self, role, well, multiplier, steps=None):
        # `steps` — расписание, которое масштабирует поиск: настоящий суррогат
        # переводит по нему множитель в отсчёт от опорного расписания обучения.
        # Здесь база и опорное совпадают, поэтому перевод тождественный
        return multiplier <= 1.25 + 1e-9


def test_joint_search_does_not_choose_a_candidate_outside_training():
    """Непокрытый кандидат не выигрывает, даже когда обещает больше покрытого.

    Требование сильнее равенства прогнозов: 1.5 здесь обещает строго больше, чем
    1.25, и всё равно не выбирается — обещанию нечем быть подтверждённым.
    """
    choices = optimise.choose_joint_rates(
        CoveredForcingSurrogate(), plan(["a"], months=1), start_date=START
    )

    assert choices[0].multiplier == 1.25
    assert choices[0].outside_training == (1.5,)
    assert "1.5" in str(choices[0])


def test_joint_search_keeps_forcing_when_the_surrogate_cannot_answer_about_coverage():
    """Без `covers_control` поиск ведёт себя ровно как раньше — берёт лучший прогноз."""
    choices = optimise.choose_joint_rates(
        ForcingSurrogate(), plan(["a"], months=1), start_date=START
    )

    assert choices[0].multiplier == 1.5
    assert choices[0].outside_training == ()


def test_joint_search_runs_on_the_baseline_surrogate(norms):
    """У тривиального бейзлайна покрытия нет вовсе, и цикл на нём не должен ломаться."""
    model = DeclineSurrogate.fit(history("good", 0.50, 0.005) + history("bad", 0.99, 0.0002),
                                 FLUIDS)
    assert not hasattr(model, "covers_control")

    choices = optimise.choose_joint_rates(model, plan(["good", "bad"]), start_date=START)

    assert [choice.well for choice in choices] == ["bad", "good"]
    assert all(choice.outside_training == () for choice in choices)
    assert all(choice.chosen_chdd_m >= choice.baseline_chdd_m for choice in choices)


# ── перевод под закачку (ВНС) ────────────────────────────────────────────────
#
# Механизм нужен не сам по себе: суррогат сегодня не видит ВНС вовсе. Признаки
# собираются по ролям, и колонка `inj:<скважина>` появляется только у той, у кого
# в датасете есть непустой множитель закачки. У добывающей его нет ни в одном
# сценарии, поэтому предложение перевести её под закачку не искажается, а
# исчезает. Чтобы вход перестал быть немым, сценарий обязан записать по
# переведённой скважине **обе** величины: добычу в ноль и закачку больше нуля.


def _producing(day: date, wells: dict[str, float]) -> ScheduleStep:
    return ScheduleStep(
        day,
        tuple(ProducerControl(w, "OPEN", "LRAT", rate, 50.0) for w, rate in wells.items()),
        (),
    )


def test_conversion_shuts_production_and_opens_injection():
    steps = [_producing(date(2007, 1, 1), {"1": 40.0, "2": 30.0}),
             _producing(date(2007, 2, 1), {"1": 40.0, "2": 30.0})]

    result = optimise.convert_to_injection(steps, {"1"}, multiplier=1.0)

    first = result[0]
    shut = [c for c in first.producers if c.well == "1"]
    assert shut and shut[0].status == "SHUT", "добыча обязана закрыться"
    assert [c.well for c in first.injectors] == ["1"], "и открыться закачка"
    assert first.injectors[0].rate == pytest.approx(40.0), "по её же дебиту"
    # соседняя скважина не тронута
    assert [c.well for c in first.producers if c.status == "OPEN"] == ["2"]


def test_conversion_holds_for_every_later_step():
    """Дек переоткрывает добычу на каждом шаге — закрывать надо на всех."""
    steps = [_producing(date(2007, month, 1), {"1": 40.0}) for month in (1, 2, 3)]

    result = optimise.convert_to_injection(steps, {"1"}, multiplier=1.0)

    assert all(c.status == "SHUT" for step in result for c in step.producers)


def test_conversion_writes_the_injector_once():
    """Eclipse держит режим до следующей записи — дублировать её незачем."""
    steps = [_producing(date(2007, month, 1), {"1": 40.0}) for month in (1, 2, 3)]

    result = optimise.convert_to_injection(steps, {"1"}, multiplier=1.0)

    assert sum(len(step.injectors) for step in result) == 1


def test_conversion_multiplier_scales_the_injection_rate():
    steps = [_producing(date(2007, 1, 1), {"1": 40.0})]
    result = optimise.convert_to_injection(steps, {"1"}, multiplier=1.5)
    assert result[0].injectors[0].rate == pytest.approx(60.0)


def test_a_well_that_never_produced_is_not_converted():
    """Приёмистость выводится из её же дебита — без него выдумывать нечего."""
    steps = [ScheduleStep(date(2007, 1, 1),
                          (ProducerControl("1", "SHUT", "LRAT", None, 50.0),), ())]

    result = optimise.convert_to_injection(steps, {"1"}, multiplier=1.0)

    assert all(not step.injectors for step in result)


def test_converted_schedule_stays_valid():
    steps = [_producing(date(2007, month, 1), {"1": 40.0, "2": 30.0}) for month in (1, 2, 3)]
    result = optimise.convert_to_injection(steps, {"1"}, multiplier=1.0)
    assert validate(result) == []


# ── циклическое заводнение ───────────────────────────────────────────────────
#
# Организаторы рекомендуют его прямо. Это не «переменный режим по периодам»:
# смысл в **перепаде давления** между группами, поэтому одна группа работает,
# пока другая стоит, и на границе полупериода они меняются местами. Сценарий, где
# все нагнетатели пульсируют синхронно, физически бессмыслен — он просто снижает
# среднюю закачку.


def _injecting(day: date, wells: dict[str, float]) -> ScheduleStep:
    return ScheduleStep(
        day, (),
        tuple(InjectorControl(w, "WATER", "OPEN", "RATE", rate, 300.0)
              for w, rate in wells.items()),
    )


def _months(count: int, wells: dict[str, float]) -> list[ScheduleStep]:
    return [_injecting(date(2007, m + 1, 1), wells) for m in range(count)]


def _rate(step: ScheduleStep, well: str) -> float:
    control = next(c for c in step.injectors if c.well == well)
    return control.rate or 0.0


def test_groups_alternate_and_are_never_on_together():
    steps = _months(6, {"A": 100.0, "B": 200.0})

    result = optimise.cyclic_injection(steps, {"A"}, {"B"}, half_period=2)

    for step in result:
        on = [c.well for c in step.injectors if c.status == "OPEN"]
        assert len(on) == 1, f"на шаге {step.date} работают {on} — противофазы нет"


def test_the_half_period_sets_how_long_a_group_stays_on():
    steps = _months(6, {"A": 100.0, "B": 200.0})

    result = optimise.cyclic_injection(steps, {"A"}, {"B"}, half_period=2)

    working = ["A" if _rate(step, "A") > 0 else "B" for step in result]
    assert working == ["A", "A", "B", "B", "A", "A"]


def test_amplitude_raises_the_rate_of_the_working_group():
    steps = _months(2, {"A": 100.0, "B": 200.0})

    result = optimise.cyclic_injection(steps, {"A"}, {"B"}, half_period=2, amplitude=1.5)

    assert _rate(result[0], "A") == pytest.approx(150.0)
    assert _rate(result[0], "B") == 0.0


def test_wells_outside_both_groups_are_untouched():
    steps = _months(2, {"A": 100.0, "B": 200.0, "C": 50.0})

    result = optimise.cyclic_injection(steps, {"A"}, {"B"}, half_period=1)

    assert all(_rate(step, "C") == pytest.approx(50.0) for step in result)


def test_a_synchronous_pulse_is_refused():
    """Общая группа означала бы синхронную пульсацию — это не заводнение."""
    with pytest.raises(ValueError, match="противофаз"):
        optimise.cyclic_injection(_months(2, {"A": 100.0}), {"A"}, {"A"}, half_period=1)


def test_half_period_must_be_at_least_a_month():
    with pytest.raises(ValueError, match="полупериод"):
        optimise.cyclic_injection(_months(2, {"A": 100.0}), {"A"}, set(), half_period=0)


def test_cyclic_schedule_stays_valid():
    steps = _months(6, {"A": 100.0, "B": 200.0})
    assert validate(optimise.cyclic_injection(steps, {"A"}, {"B"}, half_period=3)) == []


# ── кандидаты вне обучающего покрытия: сводка ────────────────────────────────
#
# Отбор кандидатов проверяется тестами выше. Здесь — то, чего они не видят:
# `explain` перечисляет только ИЗМЕНЁННЫЕ режимы, поэтому управление, у которого
# кандидатов отбросили, а решение осталось прежним, не попадает в отчёт вовсе.
# А это как раз тот случай, когда стоит знать, что выбор был неполным.


def test_explain_reports_candidates_withheld_even_without_a_change():
    unchanged = optimise.JointRateChoice("1", 1.0, 100.0, 100.0, "production",
                                         outside_training=(1.5,))
    changed = optimise.JointRateChoice("2", 1.25, 100.0, 130.0, "production")

    text = optimise.explain([unchanged, changed])

    assert "вне обучения" in text
    assert "1" in text


# ── режим по периодам горизонта ──────────────────────────────────────────────


class SeasonalSurrogate:
    """Отклик, у которого выгодный режим меняется по ходу горизонта.

    Первый год пласт отдаёт нефть пропорционально отбору, второй — обводнён:
    нефти столько же при любом дебите, а жидкости тем больше, чем выше режим.
    Один множитель на весь горизонт вынужден выбирать между этими годами.
    """

    def predict(self, steps):
        rows = []
        for step in steps:
            for control in step.producers:
                rate = (control.liquid_rate or 0.0) if control.status == "OPEN" else 0.0
                liquid_t = rate * 30.0 * FLUIDS.water_density / 1000.0
                oil_t = liquid_t * 0.5 if step.date.year == 2014 else min(liquid_t, 200.0) * 0.02
                rows.append(make_row(step.date.isoformat(), control.well, wlpr=rate,
                                     womr=oil_t / 30.0, liquid=liquid_t, oil=oil_t))
        return rows

    def in_training_range(self, steps):
        return []


def _field_npv(model, steps):
    return economics.compute_npv(
        model.predict(steps), headers=economics.REQUIRED_COLUMNS, start_date=START.isoformat()
    ).chdd_m


def test_periods_let_the_search_change_the_regime_along_the_horizon():
    """Пять периодов DoE учили модель управлению, которое меняется во времени.

    Поиск с одним множителем на 19 лет беднее обучающего пространства: часть
    того, что суррогат умеет предсказывать, ему недоступна. Здесь выгодный режим
    меняется на границе года, и два периода находят то, чего один найти не может.
    """
    model = SeasonalSurrogate()
    base = plan(["a"], months=24)

    one = optimise.choose_joint_rates(model, base, start_date=START)
    two = optimise.choose_joint_rates(model, base, start_date=START, periods=2)

    assert len({choice.multiplier for choice in two}) > 1, "оба периода выбрали одно и то же"
    assert _field_npv(model, optimise.apply_choices(base, two)) >            _field_npv(model, optimise.apply_choices(base, one))


def test_coverage_of_a_period_is_asked_about_that_period_only():
    """Покрытие спрашивается про тот кусок расписания, который масштабируется (Р-50).

    Множитель периода меняет режим только внутри своих месяцев, поэтому и вопрос
    «было ли такое в обучении» относится к ним, а не ко всему горизонту.
    """
    asked: list[tuple[float, ...]] = []

    class Recording(SeasonalSurrogate):
        def covers_control(self, role, well, multiplier, steps=None):
            asked.append(tuple(sorted({step.date.year for step in steps})))
            return True

    optimise.choose_joint_rates(Recording(), plan(["a"], months=24),
                                start_date=START, periods=2, multipliers=(1.0, 1.5))

    assert asked == [(2014,), (2015,)]


def test_one_period_is_the_default_and_keeps_todays_behaviour():
    model = SeasonalSurrogate()
    base = plan(["a"], months=24)

    default = optimise.choose_joint_rates(model, base, start_date=START)
    explicit = optimise.choose_joint_rates(model, base, start_date=START, periods=1)

    assert [(c.well, c.multiplier, c.since) for c in default] ==            [(c.well, c.multiplier, c.since) for c in explicit]
    assert all(choice.since is None for choice in default)


def test_the_choice_of_a_period_is_applied_only_to_its_part_of_the_horizon():
    """Множитель периода действует с его даты и до начала следующего."""
    base = plan(["a"], months=24, rate=30.0)
    choices = [
        optimise.JointRateChoice("a", 0.5, 0.0, 0.0),
        optimise.JointRateChoice("a", 1.5, 0.0, 0.0, since=date(2015, 1, 1)),
    ]

    tuned = optimise.apply_choices(base, choices)

    assert tuned[0].producers[0].liquid_rate == pytest.approx(15.0)
    assert tuned[11].producers[0].liquid_rate == pytest.approx(15.0)
    assert tuned[12].producers[0].liquid_rate == pytest.approx(45.0)


# ── общий ресурс учитывается до выбора, а не после ───────────────────────────


class WaterBoundSurrogate:
    """Отклик, где закачка двух скважин упирается в общий ресурс — добытую воду.

    Нефть растёт от закачки, но у `i2` она в двенадцать раз полезнее, чем у `i1`. Воды на
    объекте ровно столько, сколько подняли (`docs/method/qa-organisers.md`),
    поэтому лишняя закачка одной скважины отнимается у другой: планировщик режет
    всех пропорционально.
    """

    def predict(self, steps):
        rows = []
        for step in steps:
            injection = {control.well: (control.rate or 0.0) if control.status == "OPEN" else 0.0
                         for control in step.injectors}
            oil = 100.0 + injection.get("i1", 0.0) + 12.0 * injection.get("i2", 0.0)
            rows.append(make_row(step.date.isoformat(), "p", wlpr=100.0, womr=oil / 30.0,
                                 liquid=1000.0, oil=oil))
            rows.extend(
                make_row(step.date.isoformat(), well, wwir=rate, injection=rate * 30.0)
                for well, rate in injection.items()
            )
        return rows

    def in_training_range(self, steps):
        return []


def water_bound_plan(months: int = 12):
    return [
        ScheduleStep(
            date(2014 + index // 12, index % 12 + 1, 1),
            (ProducerControl("p", "OPEN", "LRAT", 100.0, 50.0),),
            (InjectorControl("i1", "WATER", "OPEN", "RATE", 10.0, 300.0),
             InjectorControl("i2", "WATER", "OPEN", "RATE", 10.0, 300.0)),
        )
        for index in range(months)
    ]


def _after_planner(model, steps, norms, limits):
    fitted, _decisions = planner.enforce(steps, model.predict(steps), norms, limits, FLUIDS)
    return economics.compute_npv(
        model.predict(fitted), headers=economics.REQUIRED_COLUMNS, start_date=START.isoformat()
    ).chdd_m


def test_a_candidate_the_planner_will_cut_does_not_win(norms):
    """Кандидат оценивается уже с учётом общего ресурса, а не до срезки.

    Замер 21.08 на Модели Z и обученном суррогате: поиск выбрал режимы на
    110 391,5 млн руб., после планировщика осталось 84 740,8 — минус 25 650,8
    (−23,2 %), и 23 нагнетательных режима из выбранных поиском изменены чужим
    решением. Значит поиск принимал решение по числу, которого не будет.

    Здесь то же самое на синтетике, где ответ известен: слепой поиск обещает
    18,80 и отдаёт 14,02 при исходных 13,99 — то есть весь его прирост срезается;
    согласованный обещает 18,55 и отдаёт ровно столько же.
    """
    model = WaterBoundSurrogate()
    base = water_bound_plan()
    limits = planner.Limits(enforce_water_balance=True, water_tolerance=1.0)

    blind = optimise.choose_joint_rates(model, base, start_date=START)
    aware = optimise.choose_joint_rates(model, base, start_date=START,
                                        limits=limits, norms=norms, fluids=FLUIDS)

    baseline = _after_planner(model, base, norms, limits)
    blind_chdd = _after_planner(model, optimise.apply_choices(base, blind), norms, limits)
    aware_chdd = _after_planner(model, optimise.apply_choices(base, aware), norms, limits)

    # слепой поиск форсирует ту закачку, которую планировщик потом урежет, —
    # и обещает за это прирост, которого не будет
    assert next(c for c in blind if c.well == "i1").multiplier > 1.0
    assert blind[-1].chosen_chdd_m > blind_chdd

    # согласованный видит общий ресурс и отдаёт воду i1 более полезному i2 —
    # то есть распоряжается дефицитом, а не соревнуется за него
    assert next(c for c in aware if c.well == "i1").multiplier < 1.0
    assert next(c for c in aware if c.well == "i2").multiplier > 1.0
    assert aware_chdd > blind_chdd
    assert aware_chdd > baseline


def test_the_gain_the_coordinated_search_reports_is_the_one_that_survives(norms):
    """Обещанное число — то же, что останется после планировщика, а не до него."""
    model = WaterBoundSurrogate()
    base = water_bound_plan()
    limits = planner.Limits(enforce_water_balance=True, water_tolerance=1.0)

    choices = optimise.choose_joint_rates(model, base, start_date=START,
                                          limits=limits, norms=norms, fluids=FLUIDS)
    realised = _after_planner(model, optimise.apply_choices(base, choices), norms, limits)

    assert choices[-1].chosen_chdd_m == pytest.approx(realised)


def test_limits_without_norms_are_refused():
    """Планировщику нужна маржа, чтобы решать, кого резать: молча обойтись нечем."""
    with pytest.raises(ValueError, match="норматив"):
        optimise.choose_joint_rates(
            WaterBoundSurrogate(), water_bound_plan(months=1), start_date=START,
            limits=planner.Limits(),
        )


def test_coverage_is_asked_about_the_schedule_the_search_scales():
    """Фильтр обязан спрашивать покрытие про базовое расписание, а не про голый множитель.

    Множитель поиска отсчитан от `base_steps`, обучающие множители DoE — от
    опорного расписания дека, и в интерфейсе это разные расписания (цикл
    запускается поверх открытого прогона). Без передачи базы суррогат отвечал про
    режим, которого поиск не рассматривает: замер 20.08 на форсированном ×1.4
    прогоне показал «покрыто» для ×1.68 к деку при обучении на 0.8…1.5.
    """
    asked: list[object] = []

    class RecordingSurrogate(ForcingSurrogate):
        def covers_control(self, role, well, multiplier, steps=None):
            asked.append(steps)
            return True

    base = plan(["a"], months=1)
    optimise.choose_joint_rates(RecordingSurrogate(), base, start_date=START)

    assert asked, "покрытие не спрашивалось вовсе"
    assert all(steps == base for steps in asked)


def test_the_objective_ignores_rows_the_methodology_drops(norms):
    """Целевая функция поиска не должна считать деньги, которых нет.

    Дисконтированная маржа — это та же EBITDA методики со скидкой по годам, и
    равенство проверяется соседним тестом. Оно держалось только на чистой
    выгрузке: строку с отрицательным приращением расчётчик выбрасывает целиком,
    а поиск складывал её как есть и получал другую величину, чем та, по которой
    нас оценивают.
    """
    from conftest import make_row

    from aios.optimise import discounted_margin_by_well

    rows = [make_row(f"2007-{month:02d}-01", "1", wlpr=30, womr=5, liquid=900, oil=150)
            for month in range(1, 13)]
    clean = discounted_margin_by_well(rows, norms, 2007)["1"]

    rows[5]["WOMT_Diff"] = -150.0
    with_dropped = discounted_margin_by_well(rows, norms, 2007)["1"]
    without_month = discounted_margin_by_well(
        [row for index, row in enumerate(rows) if index != 5], norms, 2007
    )["1"]

    assert with_dropped == pytest.approx(without_month), (
        "исключённая строка попала в целевую функцию"
    )
    assert with_dropped < clean, "проверка вырождена: строка ничего не меняла"
