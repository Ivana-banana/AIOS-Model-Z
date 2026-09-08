"""Сквозной цикл управления на тривиальном суррогате.

Ценность этих тестов в том, что контур целиком работает без единого расчёта
ГГДМ: пока товарищи поднимают симулятор, агенты и UI строятся не вслепую.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from conftest import make_row

from aios import economics, loop, optimise, summary, trust
from aios.deck import ProducerControl, ScheduleStep
from aios.margin import Norms
from aios.surrogate import DeclineSurrogate

FLUIDS = summary.Fluids()
CONTROL_START = date(2014, 1, 1)


@pytest.fixture
def norms():
    assumptions, _ = economics.load_norms()
    return Norms.from_assumptions(assumptions)


def well_history(well: str, watercut_start: float, watercut_step: float, rate: float = 30.0):
    rows = []
    for index in range(12):
        month = date(2013, index + 1, 1)
        days = 30
        liquid_m3 = rate * days
        watercut = min(watercut_start + watercut_step * index, 0.999)
        oil_m3 = liquid_m3 * (1 - watercut)
        oil_t = oil_m3 * FLUIDS.oil_density / 1000
        liquid_t = oil_t + (liquid_m3 - oil_m3) * FLUIDS.water_density / 1000
        rows.append(make_row(month.isoformat(), well, wlpr=rate, womr=oil_t / days,
                             liquid=liquid_t, oil=oil_t))
    return rows


@pytest.fixture
def surrogate():
    """Два фонда: «хороший» с обводнённостью 0,5 и «плохой» на грани 0,99."""
    history = well_history("good", 0.50, 0.005) + well_history("bad", 0.985, 0.0005)
    return DeclineSurrogate.fit(history, FLUIDS)


def plan(months: int = 18):
    steps = []
    for index in range(months):
        year = 2014 + index // 12
        month = index % 12 + 1
        steps.append(
            ScheduleStep(
                date(year, month, 1),
                (
                    ProducerControl("good", "OPEN", "LRAT", 30.0, 50.0),
                    ProducerControl("bad", "OPEN", "LRAT", 30.0, 50.0),
                ),
                (),
            )
        )
    return steps


class _Responsive:
    """Суррогат, чей прогноз следует за уставками расписания.

    Нужен ровно для одного свойства: после того как планировщик срежет отбор,
    следующий прогноз обязан показать **меньший** отбор. На тривиальном
    бейзлайне этого не видно — он предсказывает по истории скважины, а не по
    поданной уставке, — а именно это свойство и делает коридор баланса воды
    неподвижной точкой, а не одним проходом (Р-189).
    """

    def predict(self, steps):
        rows = []
        for step in steps:
            for control in step.producers:
                rate = (control.liquid_rate or 0.0) if control.status == "OPEN" else 0.0
                oil_t = rate * 0.1 * FLUIDS.oil_density / 1000
                water_t = rate * 0.9 * FLUIDS.water_density / 1000
                rows.append(make_row(step.date.isoformat(), control.well, wlpr=rate,
                                     womr=oil_t / 30, liquid=oil_t + water_t, oil=oil_t))
            for control in step.injectors:
                rate = (control.rate or 0.0) if control.status == "OPEN" else 0.0
                rows.append(make_row(step.date.isoformat(), control.well,
                                     wwir=rate, injection=rate * 30.0))
        return rows

    def in_training_range(self, steps):
        return []


def test_an_empty_search_keeps_the_starting_scenario(norms, surrogate, monkeypatch):
    """Пустой ответ поиска — «ничего не менять», а не «вернуться к деку».

    `apply_choices(base_steps, [])` отдаёт опорное расписание и молча теряет
    карту старта из `--base-schedule`: прирост считался бы от одного сценария, а
    расписание собиралось из другого. Поиск возвращает пустой список законно —
    когда стартовая точка вне области доверия (Р-174, Р-190).
    """
    monkeypatch.setattr(loop.optimise, "choose_rates", lambda *a, **k: [])

    start = {("production", "good"): 1.5}
    result = loop.run_cycle(
        surrogate, plan(), norms,
        control_start=CONTROL_START, start_date=CONTROL_START,
        start_multipliers=start,
    )

    rates = {control.well: control.liquid_rate
             for step in result.steps for control in step.producers
             if control.status == "OPEN"}
    assert rates.get("good") == pytest.approx(45.0), (
        f"стартовый сценарий обязан сохраниться, получено {rates}"
    )
    # Прирост при этом нулевым быть не обязан: цикл, кроме подбора режимов,
    # ещё останавливает убыточные скважины — и это отдельное решение, которое
    # пустой ответ поиска не отменяет.
    assert result.base_chdd_m == pytest.approx(
        loop.run_cycle(surrogate, plan(), norms, control_start=CONTROL_START,
                       start_date=CONTROL_START, start_multipliers=start,
                       tune_rates=False).base_chdd_m
    ), "точка отсчёта остаётся стартовым сценарием"


def test_the_corridor_holds_after_the_cuts_that_follow_it(norms):
    """Срезки отбора поднимают компенсацию — цикл обязан довести её до коридора.

    Компенсация есть отношение закачки к отобранному объёму, и каждая срезка
    отбора её поднимает. Пока коридор подбирался первым и одним проходом,
    расписание выходило из планировщика нарушающим тот самый коридор: замер
    ревизии — 16 нарушений и 133–141 % на полном цикле, то есть при включённой
    проверке fail-closed (Р-165) ни один кандидат не проходил сертификацию.
    Лечится порядком **и** пересчётом прогноза (Р-189).
    """
    from aios.deck import InjectorControl, ProducerControl, ScheduleStep
    from aios.planner import Limits

    steps = [ScheduleStep(date(2014, month, 1),
                          (ProducerControl("P1", "OPEN", "LRAT", 100.0, 50.0),),
                          (InjectorControl("I1", "WATER", "OPEN", "RATE", 100.0, 300.0),))
             for month in range(1, 13)]

    result = loop.run_cycle(
        _Responsive(), steps, norms,
        control_start=CONTROL_START, start_date=CONTROL_START,
        tune_rates=False,
        # предел по жидкости срежет отбор вдвое уже ПОСЛЕ того, как коридор
        # подобран по исходному отбору
        limits=Limits(max_field_liquid_m3_per_day=50.0,
                      enforce_water_balance=True,
                      water_balance_rule="compensation",
                      water_tolerance=1.3),
        fluids=FLUIDS,
    )

    assert result.case_problems == (), (
        "расписание, только что приведённое к ограничениям, их нарушает: "
        + "; ".join(result.case_problems)
    )
    assert result.allocations, "срезки обязаны быть названы, а не потеряны"


def test_cycle_stops_the_watered_out_well_and_keeps_the_good_one(surrogate, norms):
    """Скважину останавливает причинное решение, а не множитель перебора.

    До 04.09 остановку давал одиночный перебор режимов: множитель 0,0 выигрывал
    по марже **одной** скважины на прогнозе без отклика соседей. Замер (Б3)
    показал цену: 33 из 72 добывающих на историческом расписании, −63…−68 % ЧДД
    по полному расчёту. Ноль убран из умолчания перебора, и остановка пришла
    оттуда, где ей место, — из `policy.plan_shutdowns`, который смотрит на
    накопленный убыток и **объясняет** решение.
    """
    result = loop.run_cycle(surrogate, plan(), norms,
                            control_start=CONTROL_START, start_date=CONTROL_START)

    chosen = {choice.well: choice.multiplier for choice in result.rate_choices}
    assert chosen["bad"] > 0.0, "остановка вернулась в перебор режимов"
    assert chosen["bad"] < chosen["good"], "обводнённая скважина даже не заглушена"

    shut = [action for action in result.actions
            if action.well == "bad" and action.kind == "shut"]
    assert shut, f"скважину никто не остановил: {[(a.well, a.kind) for a in result.actions]}"
    assert "убыток" in shut[0].reason, "остановка без причины в тексте решения"

    assert result.gain_m > 0
    assert result.is_improvement


def test_the_causal_policy_catches_what_rate_tuning_missed(surrogate, norms):
    """Подбор смотрит на горизонт целиком, политика — только назад.

    Скважина, которая становится убыточной по ходу горизонта уже на выбранном
    режиме, ловится именно политикой, и её решение датировано.
    """
    result = loop.run_cycle(surrogate, plan(), norms, control_start=CONTROL_START,
                            start_date=CONTROL_START, tune_rates=False)

    assert [action.well for action in result.actions] == ["bad"]
    assert result.actions[0].date > CONTROL_START
    assert result.rate_choices == ()


def test_cycle_does_not_force_wells_beyond_their_history(surrogate, norms):
    """Форсаж бейзлайн подтвердить не может, поэтому вверх режимы не идут."""
    result = loop.run_cycle(surrogate, plan(), norms,
                            control_start=CONTROL_START, start_date=CONTROL_START)

    assert all(choice.multiplier <= 1.0 for choice in result.rate_choices)
    assert result.warnings == ()


def test_cycle_leaves_a_profitable_fund_alone(norms):
    history = well_history("good", 0.50, 0.005) + well_history("also-good", 0.55, 0.005)
    model = DeclineSurrogate.fit(history, FLUIDS)
    steps = [
        ScheduleStep(
            date(2014, month, 1),
            (
                ProducerControl("good", "OPEN", "LRAT", 30.0, 50.0),
                ProducerControl("also-good", "OPEN", "LRAT", 30.0, 50.0),
            ),
            (),
        )
        for month in range(1, 13)
    ]

    result = loop.run_cycle(model, steps, norms,
                            control_start=CONTROL_START, start_date=CONTROL_START)

    assert result.actions == ()
    assert all(choice.multiplier == 1.0 for choice in result.rate_choices)
    assert result.gain_m == pytest.approx(0.0)


def test_resulting_schedule_breaks_no_constraints(surrogate, norms):
    result = loop.run_cycle(surrogate, plan(), norms,
                            control_start=CONTROL_START, start_date=CONTROL_START)
    assert result.problems == ()


def test_forecast_is_never_reported_as_confirmed(surrogate, norms):
    """Прирост на суррогате — прогноз. Пока не подтверждён расчётом, так и пишем."""
    result = loop.run_cycle(surrogate, plan(), norms,
                            control_start=CONTROL_START, start_date=CONTROL_START)

    assert result.confirmed is False
    assert "ПРОГНОЗ, не подтверждён" in result.summary()


def test_uncalibrated_margin_is_visible_in_the_cycle_report(surrogate, norms):
    margin = trust.OptimismMargin.fit([(100.0, 90.0)])
    result = loop.run_cycle(surrogate, plan(), norms, control_start=CONTROL_START,
                            start_date=CONTROL_START, margin=margin)

    assert "НЕ КАЛИБРОВАН" in result.summary()


def test_cycle_report_shows_the_raw_and_reserved_forecast(surrogate, norms):
    margin = trust.OptimismMargin.fit([(100.0 + gap, 100.0) for gap in range(1, 10)])
    result = loop.run_cycle(surrogate, plan(), norms, control_start=CONTROL_START,
                            start_date=CONTROL_START, margin=margin)

    assert margin.calibrated
    assert "Прогноз после запаса" in result.summary()
    assert "сырой" in result.summary()


def test_confirmation_replaces_the_forecast_with_the_real_calculation(surrogate, norms):
    result = loop.run_cycle(surrogate, plan(), norms,
                            control_start=CONTROL_START, start_date=CONTROL_START)

    # «выгрузка ГГДМ»: тот же фонд, но нефти вдвое меньше прогноза
    measured = [dict(row, WOMT_Diff=row["WOMT_Diff"] / 2) for row in result.rows]
    confirmed = loop.confirm(result, measured, CONTROL_START, base_rows=result.rows)

    assert confirmed.confirmed is True
    assert confirmed.chdd_m < result.chdd_m
    assert confirmed.actions == result.actions
    assert "подтверждено полным расчётом" in confirmed.summary()


def test_extrapolation_warnings_reach_the_result(surrogate, norms):
    """Форсаж выше исторического обязан дойти до решения, а не осесть в логе."""
    forced = [
        ScheduleStep(
            step.date,
            tuple(
                ProducerControl(c.well, c.status, c.mode, 200.0, c.bhp) for c in step.producers
            ),
            step.injectors,
        )
        for step in plan(6)
    ]

    result = loop.run_cycle(surrogate, forced, norms,
                            control_start=CONTROL_START, start_date=CONTROL_START)

    assert result.warnings
    assert any("исторического максимума" in warning for warning in result.warnings)
    assert "Предупреждения" in result.summary()


def test_summary_names_both_numbers(surrogate, norms):
    result = loop.run_cycle(surrogate, plan(), norms,
                            control_start=CONTROL_START, start_date=CONTROL_START)
    text = result.summary()

    assert "ЧДД базового сценария" in text
    assert "ЧДД после решений" in text
    assert "Прирост" in text
    assert "скв. bad" in text


def test_planner_has_the_last_word_in_the_cycle(surrogate, norms):
    """Ограничения объекта применяются после скважинных решений, а не вместо них."""
    from aios.planner import Limits

    result = loop.run_cycle(
        surrogate, plan(), norms,
        control_start=CONTROL_START, start_date=CONTROL_START,
        limits=Limits(max_field_liquid_m3_per_day=20.0, enforce_water_balance=False),
    )

    assert result.allocations
    assert {item.resource for item in result.allocations} == {"жидкость"}
    for step in result.steps:
        working = sum(c.liquid_rate or 0.0 for c in step.producers if c.status == "OPEN")
        assert working <= 20.0 + 1e-9
    assert "Согласование" in result.summary() or "жидкость:" in result.summary()


def test_without_limits_the_planner_stays_out(surrogate, norms):
    result = loop.run_cycle(surrogate, plan(), norms,
                            control_start=CONTROL_START, start_date=CONTROL_START)
    assert result.allocations == ()


def test_joint_search_sees_the_same_limits_the_planner_enforces(surrogate, norms, monkeypatch):
    """Поиск режимов обязан узнавать об ограничении раньше планировщика (Р-54, `docs/decisions.md`).

    Иначе он выбирает режимы по ЧДД, которого не будет: планировщик режет их
    после. `choose_joint_rates` умеет учитывать `limits`/`norms`/`fluids` с
    21.08 (`aios/optimise.py`), но `run_cycle` их не передавал.
    """
    from aios.planner import Limits

    limits = Limits(max_field_liquid_m3_per_day=20.0, enforce_water_balance=False)
    seen = {}
    original = optimise.choose_joint_rates

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(loop.optimise, "choose_joint_rates", spy)

    loop.run_cycle(
        surrogate, plan(), norms,
        control_start=CONTROL_START, start_date=CONTROL_START,
        joint_rate_search=True, limits=limits, fluids=FLUIDS,
    )

    assert seen.get("limits") is limits
    assert seen.get("norms") is norms
    assert seen.get("fluids") is FLUIDS


def test_forecast_reports_rows_the_calculator_threw_away():
    """Прогноз считается по строкам суррогата, а негодные расчётчик выбрасывает.

    Методика исключает строку с отрицательным приростом целиком, поэтому
    прогнозный ЧДД падает почти пропорционально доле таких строк: 5 % негодных
    дают −5,3 %, 20 % — −20,0 % (измерено 20.08). Само по себе это правильно, но
    молча: цикл берёт из расчёта только число денег и теряет счётчик.

    На странице настоящего прогона счётчик показывается, а в прогнозе — нет, то
    есть честный указатель стоит ровно там, где он наименее нужен. Замер А4 на
    сценарии с ВНС дал 22 563 нарушения физических проверок, так что случай не
    гипотетический.
    """
    from datetime import date

    from conftest import make_row

    from aios import loop as loop_module

    rows = []
    for month in range(6):
        broken = month % 3 == 0
        rows.append(make_row(f"2014-{month + 1:02d}-01", "1", wlpr=30.0,
                             womr=-1.0 if broken else 5.0,
                             liquid=-30.0 if broken else 900.0,
                             oil=-30.0 if broken else 150.0))

    result = loop_module._npv(rows, date(2014, 1, 1))
    assert result.excluded_negative_rows == 2, "строки с отрицательным приростом исключены"


def test_summary_names_the_discarded_rows_when_there_are_any():
    """Прогноз, посчитанный на обрезанном наборе строк, обязан это сказать."""
    from aios.loop import CycleResult

    clean = CycleResult(base_chdd_m=100.0, chdd_m=110.0, actions=(), rate_choices=(),
                        allocations=(), steps=(), warnings=(), problems=())
    assert "исключено" not in clean.summary().lower()

    spoiled = CycleResult(base_chdd_m=100.0, chdd_m=110.0, actions=(), rate_choices=(),
                          allocations=(), steps=(), warnings=(), problems=(),
                          excluded_rows=4231)
    text = spoiled.summary()
    assert "4231" in text
    assert "исключ" in text.lower()


def test_confirmed_gain_is_a_difference_of_two_full_calculations(surrogate, norms):
    """Подтверждённый прирост нельзя считать от прогнозного базового.

    `confirm` заменял только `chdd_m`, а `base_chdd_m` оставлял прогнозным — и
    `gain_m` становился разностью двух разных источников, подписанной
    «подтверждено полным расчётом». Замер 21.08 на синтетике: прогноз обещал
    прирост −100,0, после подтверждения печаталось −999,2, потому что базовое
    число осталось от суррогата.

    `scripts/run_cycle.py` эту ловушку обходил вручную — считал базовый ЧДД сам и
    печатал разность мимо `gain_m`. То есть защита держалась на том, что
    единственный вызывающий про неё знает.
    """
    result = loop.run_cycle(surrogate, plan(), norms,
                            control_start=CONTROL_START, start_date=CONTROL_START)
    measured = [dict(row, WOMT_Diff=row["WOMT_Diff"] / 2) for row in result.rows]
    base_measured = [dict(row, WOMT_Diff=row["WOMT_Diff"] * 0.9) for row in result.rows]

    confirmed = loop.confirm(result, measured, CONTROL_START, base_rows=base_measured)

    from aios import economics
    expected_base = economics.compute_npv(
        base_measured, headers=economics.REQUIRED_COLUMNS,
        start_date=CONTROL_START.isoformat(),
    ).chdd_m
    assert confirmed.base_chdd_m == pytest.approx(expected_base), (
        "базовое число осталось прогнозным"
    )
    assert confirmed.gain_m == pytest.approx(confirmed.chdd_m - expected_base)


# --- С4: запас под потолком компенсации считался не тем счётом ---------------


def _rows_two_years(first_injection: float, second_injection: float):
    """Два года: в первом закачки почти нет, во втором много.

    Годовой максимум и накопленное отношение на таких данных расходятся: второй
    год сам по себе далеко за потолком, а накопленный итог — нет.
    """
    from conftest import make_row

    rows = []
    for year, injection in ((2014, first_injection), (2015, second_injection)):
        rows.append(make_row(f"{year}-01-01", "1", wlpr=100.0, womr=10.0,
                             liquid=1000.0, oil=100.0))
        rows.append(make_row(f"{year}-01-01", "2", wwir=injection / 30,
                             injection=injection))
    return rows


def test_headroom_is_measured_by_the_same_count_the_planner_cuts_by():
    """Накопленное отношение, а не годовой максимум.

    Планировщик режет и аудит судит по накопленному на конец года; запас же
    считался внутри одного года. Две разные величины на одном пороге расходятся
    в обе стороны — отсюда переводы под закачку, которые коридор потом срезает.
    """
    from aios.loop import _water_headroom
    from aios.planner import Limits

    limits = Limits(enforce_water_balance=True, water_balance_rule="compensation",
                    water_tolerance=1.3)

    # первый год почти без закачки, второй — втрое выше потолка по себе
    rows = _rows_two_years(first_injection=100.0, second_injection=3000.0)

    # накопленное отношение: (100 + 3000) / (1100 + 1100) = 1,41 — запаса нет
    assert _water_headroom(rows, limits, FLUIDS) is False


def test_headroom_follows_the_rule_the_case_declared():
    """При правиле `produced_water` знаменатель — добытая вода, а не жидкость.

    Правило бралось всегда как `compensation`, даже когда кейс объявил другое:
    число получалось меньше в разы, и запас объявлялся там, где его нет.
    """
    from aios.loop import _water_headroom
    from aios.planner import Limits

    # 1 000 т жидкости и 100 т нефти в год дают 877,4 м³ жидкости и 767,9 м³
    # воды (плотности дека). За два года: 1 754,9 против 1 535,8 — закачка 1 600
    # ложится между ними, и правило решает всё
    rows = _rows_two_years(first_injection=800.0, second_injection=800.0)

    by_liquid = Limits(enforce_water_balance=True, water_balance_rule="compensation",
                       water_tolerance=1.0)
    by_water = Limits(enforce_water_balance=True, water_balance_rule="produced_water",
                      water_tolerance=1.0)

    # 1 600 / 1 754,9 = 0,91 — запас есть; 1 600 / 1 535,8 = 1,04 — запаса нет
    assert _water_headroom(rows, by_liquid, FLUIDS) is True
    assert _water_headroom(rows, by_water, FLUIDS) is False


# --- С1: один потолок в трёх окнах -------------------------------------------
# Выбор окна — вопрос к организаторам и решение владельца. Здесь он не
# принимается; здесь он перестаёт быть незаметным.


def test_a_baseline_that_breaks_the_ceiling_is_named_before_candidates_are_judged():
    """Если бездействие уже вне коридора, об этом обязано быть сказано.

    Иначе fail-closed отвергает **любого** кандидата, а оператор читает «условия
    кейса нарушены», не зная, что ничего не делать — тоже нарушение. Замер 04.09
    на `runs/base/rows.csv`: с 1994 накопленная компенсация 1,239, с 2007 — 1,401
    в 2011, при потолке 1,3. Один потолок, разные окна, разные вердикты (С1).
    """
    from aios.loop import _baseline_outside_the_corridor
    from aios.planner import Limits

    limits = Limits(enforce_water_balance=True, water_balance_rule="compensation",
                    water_tolerance=1.3)

    # закачка вдвое выше отбора — опорное расписание заведомо за потолком
    rows = _rows_two_years(first_injection=2000.0, second_injection=2000.0)
    said = _baseline_outside_the_corridor(rows, limits, FLUIDS)

    assert said, "нарушение потолка самим опорным расписанием осталось незамеченным"
    assert "опорное расписание" in said[0]
    assert "1.30" in said[0] or "1,30" in said[0]


def test_a_baseline_inside_the_corridor_says_nothing():
    """Проверка не должна шуметь там, где всё в порядке."""
    from aios.loop import _baseline_outside_the_corridor
    from aios.planner import Limits

    limits = Limits(enforce_water_balance=True, water_balance_rule="compensation",
                    water_tolerance=1.3)

    rows = _rows_two_years(first_injection=800.0, second_injection=800.0)

    assert _baseline_outside_the_corridor(rows, limits, FLUIDS) == []


def test_the_measured_windows_of_the_real_export_still_disagree():
    """Числа замера С1 закреплены: они и есть содержание находки.

    Пропуск, если выгрузки нет: на машине без прогонов сравнивать нечего.
    """
    import pytest

    from aios import balance
    from aios.web.app import read_rows_csv

    export = Path(__file__).resolve().parent.parent / "runs" / "base" / "rows.csv"
    if not export.is_file():
        pytest.skip("нет выгрузки runs/base на этой машине")

    rows = read_rows_csv(export)
    whole = balance.worst_cumulative_compensation(rows, FLUIDS)
    window = balance.worst_cumulative_compensation(
        [row for row in rows if str(row["DATA"])[:4] >= "2007"], FLUIDS)

    assert whole is not None and window is not None
    assert round(whole[1], 3) == 1.239, "изменилась выгрузка или счёт — обновить С1"
    assert round(window[1], 3) == 1.401
    assert whole[1] < 1.3 < window[1], (
        "окна перестали расходиться — находка С1 закрыта, снять её из документов"
    )


def test_the_cycle_itself_says_the_baseline_is_outside_the_corridor(norms):
    """Предупреждение обязано дойти до результата цикла, а не жить в помощнике.

    Оператор читает `result.warnings`; проверка, которая зовёт помощника
    напрямую, не стережёт того, что цикл его вызывает.
    """
    from aios.deck import InjectorControl, ProducerControl, ScheduleStep
    from aios.planner import Limits

    # закачка вдвое выше отбора: опорное расписание заведомо за потолком 1,3
    steps = [ScheduleStep(date(2014, month, 1),
                          (ProducerControl("P1", "OPEN", "LRAT", 100.0, 50.0),),
                          (InjectorControl("I1", "WATER", "OPEN", "RATE", 200.0, 300.0),))
             for month in range(1, 13)]

    result = loop.run_cycle(
        _Responsive(), steps, norms,
        control_start=CONTROL_START, start_date=CONTROL_START, tune_rates=False,
        limits=Limits(enforce_water_balance=True, water_balance_rule="compensation",
                      water_tolerance=1.3),
        fluids=FLUIDS,
    )

    said = [one for one in result.warnings if "опорное расписание" in one]
    assert said, (
        "цикл не сказал, что бездействие само вне коридора: "
        + "; ".join(one[:60] for one in result.warnings[:5])
    )


def test_the_single_search_starts_from_the_start_scenario_not_the_deck(surrogate, norms):
    """Одиночный перебор ведётся от стартового сценария, а не от дека.

    У `choose_rates` нет параметра `start`, каким пользуется совместный перебор,
    поэтому карта из `--base-schedule` терялась: поиск шёл по деку, а
    `base_chdd_m` считался по стартовому сценарию — «прирост» сравнивал две
    разные точки отсчёта (2.1 #8).

    Набор множителей сведён к одному `1.0` намеренно: тогда результат зависит
    **только** от системы координат, а не от того, что выбрал поиск. В системе
    старта режим обязан остаться равным `дек × 0,9`, в системе дека вышел бы сам
    дек.
    """
    start = {("production", "good"): 0.9}

    result = loop.run_cycle(surrogate, plan(), norms,
                            control_start=CONTROL_START, start_date=CONTROL_START,
                            start_multipliers=start, joint_rate_search=False)

    rates = {control.well: control.liquid_rate
             for step in result.steps for control in step.producers
             if control.status == "OPEN"}
    deck = {control.well: control.liquid_rate
            for step in plan() for control in step.producers
            if control.status == "OPEN"}
    chosen = {one.well: one.multiplier for one in result.rate_choices}

    assert "good" in rates and deck.get("good"), "фикстура не даёт режима «good»"
    assert "good" in chosen, "перебор не вернул решения по «good» — проверка вырождена"

    # уставка обязана быть множителем от **стартового** режима (дек × 0,9), а не
    # от дековского: в системе дека вышло бы deck × chosen
    expected = deck["good"] * 0.9 * chosen["good"]
    assert rates["good"] == pytest.approx(expected), (
        f"режим «good» {rates['good']} против ожидаемых {expected} "
        f"(дек {deck['good']} × старт 0,9 × выбор {chosen['good']}): "
        f"перебор шёл по деку, а не по стартовому сценарию"
    )


def test_the_single_search_is_handed_the_start_scenario_itself(surrogate, norms, monkeypatch):
    """Прямая проверка передачи: перебор получает стартовый сценарий, не дек.

    Проверка результата этого не ловит, пока фикстура даёт один и тот же выбор в
    обеих системах координат, — а различаться они начнут ровно тогда, когда
    экономика режимов разойдётся. Поэтому здесь проверяется сама передача.
    """
    seen = {}
    original = optimise.choose_rates

    def spy(surrogate_, steps, norms_, **kwargs):
        seen["rates"] = [control.liquid_rate
                         for step in steps for control in step.producers
                         if control.status == "OPEN"]
        return original(surrogate_, steps, norms_, **kwargs)

    monkeypatch.setattr(optimise, "choose_rates", spy)

    deck = [control.liquid_rate for step in plan() for control in step.producers
            if control.status == "OPEN"]
    loop.run_cycle(surrogate, plan(), norms,
                   control_start=CONTROL_START, start_date=CONTROL_START,
                   start_multipliers={("production", "good"): 0.9},
                   joint_rate_search=False)

    assert seen, "перебор не вызывался — проверка вырождена"
    assert seen["rates"] != deck, (
        "перебору отдали дек, а не стартовый сценарий: карта старта потеряна"
    )


def test_the_corridor_carries_the_history_before_the_horizon():
    """Накопленное до горизонта переносится, а не начинается с нуля (Р-204).

    Владелец решил 04.09: коридор судится тем же окном, что и подтверждение.
    История в прогнозе отсутствует, поэтому переносится числом; без переноса цикл
    считал компенсацию с начала управления, где сама история вне потолка.
    """
    from aios.loop import _carried_water
    from aios.planner import Limits

    limits = Limits(enforce_water_balance=True, water_balance_rule="compensation",
                    water_tolerance=1.3)
    history = _rows_two_years(first_injection=900.0, second_injection=900.0)

    # горизонт начинается позже обоих лет истории — переносится всё
    carried = _carried_water(history, 2016, limits, FLUIDS)
    assert carried is not None
    withdrawn, injected = carried
    assert injected == pytest.approx(1800.0)
    assert withdrawn > 0

    # горизонт начинается с первого года истории — переносить нечего
    assert _carried_water(history, 2014, limits, FLUIDS) is None


def test_without_an_export_the_corridor_counts_as_before():
    """Вызов без выгрузки — законный случай «истории нет», а не тихое умолчание.

    Так работают тесты, разбор и веб-страница: переноса нет, счёт начинается с
    нуля, и это не подмена решения владельца, а его отсутствие данных.
    """
    from aios.loop import _carried_water
    from aios.planner import Limits

    limits = Limits(enforce_water_balance=True, water_balance_rule="compensation")

    assert _carried_water(None, 2016, limits, FLUIDS) is None
    assert _carried_water([], 2016, limits, FLUIDS) is None


def test_the_carry_follows_the_rule_too():
    """Знаменатель переноса — тот же, что у правила: жидкость или добытая вода."""
    from aios.loop import _carried_water
    from aios.planner import Limits

    history = _rows_two_years(first_injection=900.0, second_injection=900.0)
    by_liquid = _carried_water(
        history, 2016,
        Limits(enforce_water_balance=True, water_balance_rule="compensation"), FLUIDS)
    by_water = _carried_water(
        history, 2016,
        Limits(enforce_water_balance=True, water_balance_rule="produced_water"), FLUIDS)

    assert by_liquid[0] > by_water[0], "жидкость обязана быть больше добытой воды"
    assert by_liquid[1] == by_water[1] == pytest.approx(1800.0)


def test_the_history_changes_what_the_corridor_cuts(norms):
    """Перенос обязан доходить до планировщика, а не оставаться в помощнике.

    Проверка самого `_carried_water` этого не стережёт: мутация «не передавать
    перенос в `enforce`» её не роняет. Здесь один и тот же цикл гоняется с
    историей и без неё — решения коридора обязаны различаться.
    """
    from aios.deck import InjectorControl, ProducerControl, ScheduleStep
    from aios.planner import Limits

    steps = [ScheduleStep(date(2014, month, 1),
                          (ProducerControl("P1", "OPEN", "LRAT", 100.0, 50.0),),
                          (InjectorControl("I1", "WATER", "OPEN", "RATE", 100.0, 300.0),))
             for month in range(1, 13)]
    limits = Limits(enforce_water_balance=True, water_balance_rule="compensation",
                    water_tolerance=1.3)

    # история 2013 года: закачки много, отбора мало — накопленное сразу за потолком
    history = [
        make_row("2013-01-01", "P1", wlpr=10.0, womr=1.0, liquid=10.0, oil=1.0),
        make_row("2013-01-01", "I1", wwir=500.0, injection=15000.0),
    ]

    without = loop.run_cycle(_Responsive(), steps, norms,
                             control_start=CONTROL_START, start_date=CONTROL_START,
                             tune_rates=False, limits=limits, fluids=FLUIDS)
    with_history = loop.run_cycle(_Responsive(), steps, norms,
                                  control_start=CONTROL_START, start_date=CONTROL_START,
                                  tune_rates=False, limits=limits, fluids=FLUIDS,
                                  history_rows=history)

    def accumulated(result):
        cuts = [one for one in result.allocations if one.resource == "закачка"]
        assert cuts, "коридор не срезал ничего — проверка вырождена"
        return cuts[0].reason

    # Наблюдаемое — не число срезок, а **накопленные величины** в обосновании:
    # с историей к закачке добавляются её 15 000 м³, к отбору — её же отбор.
    assert "36,000 м³" in accumulated(without), accumulated(without)
    assert "51,000 м³" in accumulated(with_history), accumulated(with_history)


def test_the_cycle_reports_its_stages_as_it_goes(surrogate, norms):
    """Цикл подаёт признаки жизни: этапы, а не тишина на десятки минут.

    Замер 05.09: на обученном суррогате цикл идёт свыше сорока минут при полном
    наборе множителей и до этой правки не печатал между «конформный запас» и
    концом ни строки. Отличить работу от зависания было нечем — ни на демо, ни
    оператору.

    Печатает по-прежнему вызывающий: модуль зовёт колбэк, а не `print`. Без
    колбэка поведение прежнее, и это тоже проверяется — иначе прогресс стал бы
    обязательным для всех, кто зовёт цикл из кода.
    """
    seen: list[str] = []

    result = loop.run_cycle(surrogate, plan(), norms,
                            control_start=CONTROL_START, start_date=CONTROL_START,
                            progress=seen.append)

    assert len(seen) >= 4, f"этапов слишком мало, чтобы понять, где стоим: {seen}"
    assert seen[0] == "прогноз базы", f"первым обязан идти прогноз базы: {seen}"
    assert any("подбор режимов" in one for one in seen), seen
    assert any("остановок" in one for one in seen), seen
    assert result.gain_m is not None, "отметки не должны менять сам результат"

    quiet = loop.run_cycle(surrogate, plan(), norms,
                           control_start=CONTROL_START, start_date=CONTROL_START)
    assert quiet.gain_m == result.gain_m, "без колбэка цикл обязан считать то же самое"


def test_the_summary_names_the_penalty_that_dwarfs_the_margin():
    """Число «после запаса» разбирается на слагаемые, когда штраф их подавляет.

    Замер 05.09 на настоящем прогоне: сырой прогноз 2 647,2, запас 927,8, а
    «после запаса» -214 989,8. Разница -216 709,2 — это штраф за удаление от
    обучающих сценариев, и подпись, называющая только запас, превращала сигнал
    «безнадёжно вне области» в бессмыслицу.

    Заодно проверяется, что строка выводима в cp1251: первая редакция несла
    типографский минус U+2212 и роняла вывод на консоли Windows
    `UnicodeEncodeError` — наступил на это сам при проверке.
    """
    margin = trust.OptimismMargin(value=927.8, samples=9, reason="",
                                  alpha=0.1, index=8)
    result = loop.CycleResult(
        base_chdd_m=0.0, chdd_m=2647.2, actions=(), rate_choices=(),
        allocations=(), steps=(), warnings=(), problems=(),
        raw_chdd_m=2647.2, safe_chdd_m=-214989.8, margin=margin)

    text = result.summary()

    assert "штраф за удаление" in text, "штраф не назван — число читается как бессмыслица"
    assert "216,709.2" in text, f"штраф посчитан неверно:{chr(10)}{text}"
    assert "вне доверенной области" in text, "не сказано, что число означает"
    text.encode("cp1251")  # падение здесь = вывод сломается на консоли Windows

    # без штрафа лишней строки быть не должно: она объясняет расхождение,
    # которого в этом случае нет
    calm = loop.CycleResult(
        base_chdd_m=0.0, chdd_m=2647.2, actions=(), rate_choices=(),
        allocations=(), steps=(), warnings=(), problems=(),
        raw_chdd_m=2647.2, safe_chdd_m=2647.2 - 927.8, margin=margin)
    assert "штраф за удаление" not in calm.summary()


def test_the_case_audit_judges_the_corridor_in_the_same_window_as_the_planner():
    """Аудит коридора и его приведение обязаны считать одним окном.

    Планировщик подбирает коридор с переносом накопленного за историю (решение
    владельца 04.09, Р-204), запас под потолком считается так же, и
    подтверждение судит по всей выгрузке. А `_case_problems` считал управляемые
    годы **с нуля** — и на нетронутой истории Модели Z находил нарушения там, где
    их нет. С fail-closed (Р-165) это означает, что непригодным объявляется любой
    кандидат, включая бездействие, и оператор читает «условия кейса нарушены».

    Замер 09.09 на `runs/base/rows.csv`: 16 нарушений в окне цикла против нуля с
    переносом и нуля по всей выгрузке; худший год 1,401 в 2011 против 1,239 в
    2020 при потолке 1,3. Здесь то же самое на синтетике: история закачивает
    мало, горизонт много, и без переноса горизонт выглядит нарушителем.
    """
    from aios import planner as planner_module

    limits = planner_module.Limits(water_tolerance=1.3, enforce_water_balance=True,
                                   water_balance_rule="compensation")
    history = [make_row(f"200{year}-01-01", "1", liquid=1000.0) for year in range(1, 7)]
    history += [make_row(f"200{year}-01-01", "2", injection=100.0) for year in range(1, 7)]
    horizon = [make_row("2007-01-01", "1", liquid=100.0),
               make_row("2007-01-01", "2", injection=1000.0)]

    without = loop._case_problems([], horizon, limits, None, [])
    with_history = loop._case_problems([], horizon, limits, None, [], history)

    assert without, "синтетика подобрана неверно: без переноса нарушения обязаны быть"
    assert not with_history, (
        "аудит судит окном, в котором коридор не приводился: " + "; ".join(with_history)
    )


def test_the_cycle_measures_trust_distance_on_the_intent_not_on_the_cut_schedule(
        norms, monkeypatch):
    """Расстояние до доверенной области меряется по замыслу, а не после срезки.

    Обучающие точки области собраны из множителей DoE, то есть из **замыслов**.
    Планировщик режет закачку по годам, у нагнетателей появляется признак
    «размах», которого в обучении не было ни у одного сценария, — и расстояние
    улетает в 1,25·10¹⁵ при пороге 18: вне области объявляется любой кандидат,
    включая стартовый.

    `optimise._value_of` считает по замыслу с 03.09 (Р-190), а цикл продолжал
    мерить после планировщика — одна и та же функция с двух разных входов
    (Н-15). Здесь проверяется, что вход у неё один и тот же.
    """
    from aios import optimise as optimise_module
    from aios.deck import InjectorControl, ProducerControl, ScheduleStep
    from aios.planner import Limits

    steps = [ScheduleStep(date(2014, month, 1),
                          (ProducerControl("P1", "OPEN", "LRAT", 100.0, 50.0),),
                          (InjectorControl("I1", "WATER", "OPEN", "RATE", 100.0, 300.0),))
             for month in range(1, 13)]

    seen: list[tuple] = []
    original = optimise_module._distance_of

    def spy(surrogate_, measured, region=None):
        seen.append(tuple(measured))
        return original(surrogate_, measured, region)

    monkeypatch.setattr(optimise_module, "_distance_of", spy)

    result = loop.run_cycle(
        _Responsive(), steps, norms,
        control_start=CONTROL_START, start_date=CONTROL_START,
        tune_rates=False,
        limits=Limits(max_field_liquid_m3_per_day=50.0,
                      enforce_water_balance=True,
                      water_balance_rule="compensation",
                      water_tolerance=1.3),
        fluids=FLUIDS,
    )

    assert seen, "цикл не спросил расстояние вовсе"
    assert result.allocations, "срезки не случилось — проверять нечего"
    assert seen[-1] != tuple(result.steps), (
        "расстояние померено по расписанию после планировщика: именно это "
        "объявляло вне области любого кандидата, включая стартовый"
    )


def test_a_concession_in_one_year_does_not_excuse_a_quota_breach_in_another():
    """Уступка засчитывается в том периоде, где объявлена, а не во всех сразу.

    Цикл спрашивал «была ли уступка вообще» и выбрасывал из списка все проверки
    квоты мероприятий. Одна объявленная уступка гасила квоту целиком: 22
    мероприятия против предела 1 в 2019 проходили fail-closed, если планировщик
    уступил хоть в каком-нибудь году (Н-19). Репетиция кейса умела считать
    правильно с 05.09 — теперь правило у них общее.
    """
    from aios import planner as planner_module

    limits = planner_module.Limits(max_events_per_year=1)
    steps = [
        ScheduleStep(date(2019, month, 1),
                     (ProducerControl("1", "OPEN" if month % 2 else "SHUT", "LRAT",
                                      30.0 if month % 2 else None, 50.0),), ())
        for month in range(1, 13)
    ]
    elsewhere = planner_module.Allocation(
        well="1", resource="квота мероприятий, год 2021", before=0.0, after=0.0,
        reason="остановка оставлен вопреки квоте 1 на год 2021: он порождён "
               "ограничением объекта",
    )
    here = planner_module.Allocation(
        well="1", resource="квота мероприятий, год 2019", before=0.0, after=0.0,
        reason="остановка оставлен вопреки квоте 1 на год 2019: он порождён "
               "ограничением объекта",
    )

    assert loop._case_problems(steps, [], limits, None, []), (
        "синтетика подобрана неверно: без уступки нарушение обязано быть"
    )
    assert loop._case_problems(steps, [], limits, None, [elsewhere]), (
        "уступка в другом году погасила нарушение — это и был дефект"
    )
    assert not loop._case_problems(steps, [], limits, None, [here])
