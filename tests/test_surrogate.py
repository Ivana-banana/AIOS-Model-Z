"""Тривиальный суррогат: планка, которую обязана побить нейросеть."""

from __future__ import annotations

from datetime import date

import pytest
from conftest import make_row

from aios import economics, summary
from aios.deck import InjectorControl, ProducerControl, ScheduleStep
from aios.surrogate import DeclineSurrogate, TrainingEnvelope, expand

FLUIDS = summary.Fluids()


def history(months: int = 12, *, rate: float = 30.0, watercut_start: float = 0.5,
            watercut_step: float = 0.01, well: str = "1") -> list[dict]:
    """История с линейно растущей обводнённостью — то, что бейзлайн умеет поймать."""
    rows = []
    for index in range(months):
        month = date(2013, 1, 1).replace(month=(index % 12) + 1)
        days = 30
        liquid_m3 = rate * days
        watercut = watercut_start + watercut_step * index
        oil_m3 = liquid_m3 * (1 - watercut)
        water_m3 = liquid_m3 - oil_m3
        oil_t = oil_m3 * FLUIDS.oil_density / 1000
        liquid_t = oil_t + water_m3 * FLUIDS.water_density / 1000
        rows.append(
            make_row(month.isoformat(), well, wlpr=rate, womr=oil_t / days,
                     liquid=liquid_t, oil=oil_t)
        )
    return rows


def producing(day: date, rate: float = 30.0, well: str = "1", status: str = "OPEN"):
    return ScheduleStep(day, (ProducerControl(well, status, "LRAT", rate, 50.0),), ())


def test_expand_carries_control_forward():
    """Скважину, не перечисленную на шаге, не останавливают — у неё прежний режим."""
    steps = [
        producing(date(2014, 1, 1), 30.0),
        ScheduleStep(date(2014, 2, 1), (), ()),
        producing(date(2014, 3, 1), 45.0),
    ]
    timeline = expand(steps)

    assert timeline[date(2014, 2, 1)]["1"].liquid_rate == 30.0
    assert timeline[date(2014, 3, 1)]["1"].liquid_rate == 45.0


def test_expand_keeps_producers_and_injectors_together():
    steps = [
        ScheduleStep(
            date(2014, 1, 1),
            (ProducerControl("1", "OPEN", "LRAT", 30.0, 50.0),),
            (InjectorControl("2", "WATER", "OPEN", "RATE", 40.0, 300.0),),
        ),
        ScheduleStep(date(2014, 2, 1), (), ()),
    ]
    later = expand(steps)[date(2014, 2, 1)]
    assert set(later) == {"1", "2"}


def test_fit_recovers_a_linear_watercut_trend():
    model = DeclineSurrogate.fit(history(), FLUIDS)
    decline = model.wells["1"]

    assert decline.months_fitted == 12
    assert decline.watercut_at_start == pytest.approx(0.5 + 0.01 * 11, abs=1e-6)
    assert decline.watercut_slope > 0
    assert decline.max_liquid_rate_seen == pytest.approx(30.0)


def test_a_well_with_too_short_a_history_is_not_fitted():
    model = DeclineSurrogate.fit(history(months=2), FLUIDS)
    assert model.wells == {}


def test_prediction_continues_the_trend():
    """Обводнённость растёт дальше, дебит нефти падает от месяца к месяцу.

    Сравниваем именно дебиты: месячная добыча немонотонна просто потому, что в
    феврале 28 суток, а в январе 31 — и это не имеет отношения к физике.
    """
    model = DeclineSurrogate.fit(history(), FLUIDS)
    steps = [producing(date(2014, month, 1)) for month in (1, 2, 3)]

    rows = model.predict(steps)

    oil_rate = [row["WOMR"] for row in rows]
    assert oil_rate[0] > oil_rate[1] > oil_rate[2] > 0
    assert all(row["WOMT_Diff"] < row["WLPT_Diff"] for row in rows)
    # а вот за месяц февраль честно даёт меньше января при том же дебите
    assert rows[1]["WOMT_Diff"] < rows[0]["WOMT_Diff"]


def test_higher_planned_rate_gives_more_liquid():
    model = DeclineSurrogate.fit(history(), FLUIDS)

    slow = model.predict([producing(date(2014, 1, 1), 30.0)])[0]
    fast = model.predict([producing(date(2014, 1, 1), 60.0)])[0]

    assert fast["WLPR"] == pytest.approx(2 * slow["WLPR"])
    assert fast["WLPT_Diff"] > slow["WLPT_Diff"]
    # нефти больше, но не вдвое: обводнённость растёт быстрее при форсаже
    assert slow["WOMT_Diff"] < fast["WOMT_Diff"] < 2 * slow["WOMT_Diff"]


def test_shut_well_produces_nothing():
    model = DeclineSurrogate.fit(history(), FLUIDS)
    row = model.predict([producing(date(2014, 1, 1), status="SHUT")])[0]

    assert row["WLPR"] == 0.0
    assert row["WOMT_Diff"] == 0.0
    assert row["WLPT_Diff"] == 0.0


def test_injector_is_passed_through():
    model = DeclineSurrogate.fit(history(), FLUIDS)
    steps = [
        ScheduleStep(
            date(2014, 1, 1), (), (InjectorControl("9", "WATER", "OPEN", "RATE", 50.0, 300.0),)
        )
    ]
    row = model.predict(steps)[0]

    assert row["WWIR"] == pytest.approx(50.0)
    assert row["WWIT_Diff"] == pytest.approx(50.0 * 31)
    assert row["WOMT_Diff"] == 0.0


def test_prediction_matches_the_calculator_contract_and_is_sane():
    model = DeclineSurrogate.fit(history(), FLUIDS)
    rows = model.predict([producing(date(2014, month, 1)) for month in range(1, 13)])

    assert list(rows[0].keys()) == economics.REQUIRED_COLUMNS
    assert summary.sanity_problems(rows) == []


def test_prediction_feeds_the_reference_calculator():
    """Цикл замыкается без единого расчёта ГГДМ: прогноз оценивается теми же деньгами."""
    model = DeclineSurrogate.fit(history(), FLUIDS)
    rows = model.predict([producing(date(2014, month, 1)) for month in range(1, 13)])

    result = economics.compute_npv(
        rows, headers=economics.REQUIRED_COLUMNS, start_date="2014-01-01"
    )
    assert result.chdd_m > 0
    assert result.raw["annual"][0]["activeWellMonths"] == pytest.approx(12.0)


def test_shutting_a_watered_out_well_raises_the_forecast_npv():
    """Проверка бейзлайн-политики на суррогате — до всякой ГГДМ."""
    model = DeclineSurrogate.fit(history(watercut_start=0.985, watercut_step=0.0005), FLUIDS)
    months = [date(2014, month, 1) for month in range(1, 13)]

    working = model.predict([producing(month) for month in months])
    stopped = model.predict([producing(months[0], status="SHUT")])

    as_is = economics.compute_npv(working, headers=economics.REQUIRED_COLUMNS,
                                  start_date="2014-01-01")
    after = economics.compute_npv(stopped, headers=economics.REQUIRED_COLUMNS,
                                  start_date="2014-01-01")
    assert after.chdd_m > as_is.chdd_m


def injection_history(months: int = 12, *, rate: float = 100.0, well: str = "2") -> list[dict]:
    """История нагнетательной скважины — то, относительно чего считается изменение."""
    rows = []
    for index in range(months):
        month = date(2013, 1, 1).replace(month=(index % 12) + 1)
        rows.append(make_row(month.isoformat(), well, wwir=rate, injection=rate * 30))
    return rows


def injecting(day: date, rate: float, well: str = "2"):
    return ScheduleStep(day, (), (InjectorControl(well, "WATER", "OPEN", "RATE", rate, 300.0),))


def test_injection_on_the_historical_level_is_not_flagged():
    model = DeclineSurrogate.fit(history() + injection_history(rate=100.0), FLUIDS)
    assert model.in_training_range([injecting(date(2014, 1, 1), 100.0)]) == []


def test_changed_injection_is_flagged_because_the_baseline_cannot_see_it():
    """Бейзлайн не моделирует отклик на закачку — менять её вслепую нельзя.

    Проверка не теоретическая, она из уже случившегося. 16.08 планировщик урезал
    закачку в 4,7 раза, опираясь на прогноз этого суррогата: тот пообещал
    +646,5 млн руб., полный расчёт дал −9295,4. Прогноз не изменился вовсе —
    закачка в него не входит, — и предупреждения не было ни одного.
    """
    model = DeclineSurrogate.fit(history() + injection_history(rate=100.0), FLUIDS)

    assert any("закачк" in warning for warning in model.in_training_range(
        [injecting(date(2014, 1, 1), 300.0)]))
    assert any("закачк" in warning for warning in model.in_training_range(
        [injecting(date(2014, 1, 1), 20.0)]))


def test_a_quiet_month_inside_the_historical_spread_is_not_flagged():
    """Сравнение идёт с диапазоном, а не с максимумом.

    Закачка по скважине за историю гуляет. Если опираться на один максимум, любой
    спокойный месяц базового сценария объявляется изменением — и предупреждение
    перестаёт что-либо значить, потому что горит всегда.
    """
    varied = (injection_history(months=6, rate=50.0)
              + injection_history(months=6, rate=180.0))
    model = DeclineSurrogate.fit(history() + varied, FLUIDS)

    assert model.in_training_range([injecting(date(2014, 1, 1), 50.0)]) == []
    assert model.in_training_range([injecting(date(2014, 1, 1), 120.0)]) == []
    assert model.in_training_range([injecting(date(2014, 1, 1), 400.0)]) != []


def test_injection_into_a_well_without_history_is_flagged():
    model = DeclineSurrogate.fit(history(), FLUIDS)
    warnings = model.in_training_range([injecting(date(2014, 1, 1), 50.0, well="9")])
    assert warnings and "закачк" in warnings[0]


def test_rate_within_the_history_is_not_flagged():
    """Дебит на историческом уровне и ниже — то, на чём подгонка честна."""
    model = DeclineSurrogate.fit(history(rate=30.0), FLUIDS)

    assert model.in_training_range([producing(date(2014, 1, 1), 30.0)]) == []
    assert model.in_training_range([producing(date(2014, 1, 1), 15.0)]) == []


def test_any_rate_above_the_history_is_flagged_as_forcing():
    """Скважина никогда так не работала — значит, это не прогноз, а догадка.

    Порог именно в исторический максимум, а не «в полтора раза выше»: бейзлайн не
    знает пластового давления, поэтому форсаж в нём выгоден всегда и обоснован
    никогда.
    """
    model = DeclineSurrogate.fit(history(rate=30.0), FLUIDS)

    warnings = model.in_training_range([producing(date(2014, 1, 1), 37.5)])
    assert len(warnings) == 1
    assert "форсаж" in warnings[0]
    assert "30" in warnings[0]


def test_a_well_without_history_is_flagged_rather_than_guessed():
    model = DeclineSurrogate.fit(history(), FLUIDS)
    warnings = model.in_training_range([producing(date(2014, 1, 1), 30.0, well="404")])

    assert len(warnings) == 1
    assert "истории работы нет" in warnings[0]
    # и прогноза по ней тоже нет, а не нулевая выдумка с ненулевым дебитом
    row = model.predict([producing(date(2014, 1, 1), 30.0, well="404")])[0]
    assert row["WLPR"] == 0.0


def training_rows() -> list[dict]:
    return [
        {"scenario": "low", "DATA": "2014-01-01", "well": "1", "WLPR": 20, "WWIR": 0},
        {"scenario": "low", "DATA": "2014-01-01", "well": "2", "WLPR": 0, "WWIR": 30},
        {"scenario": "high", "DATA": "2014-01-01", "well": "1", "WLPR": 40, "WWIR": 0},
        {"scenario": "high", "DATA": "2014-01-01", "well": "2", "WLPR": 0, "WWIR": 50},
    ]


def test_training_envelope_accepts_seen_local_and_field_ranges():
    envelope = TrainingEnvelope.fit(training_rows())
    steps = [ScheduleStep(
        date(2014, 1, 1),
        (ProducerControl("1", "OPEN", "LRAT", 30.0, 50.0),),
        (InjectorControl("2", "WATER", "OPEN", "RATE", 40.0, 300.0),),
    )]

    assert envelope.warnings(steps) == []


def test_training_envelope_flags_unknown_role_and_field_combination():
    envelope = TrainingEnvelope.fit(training_rows())
    steps = [ScheduleStep(
        date(2014, 1, 1),
        (
            ProducerControl("1", "OPEN", "LRAT", 40.0, 50.0),
            ProducerControl("2", "OPEN", "LRAT", 20.0, 50.0),
        ),
        (),
    )]

    warnings = envelope.warnings(steps)

    assert any("скв. 2" in warning and "режим отсутствует" in warning for warning in warnings)
    assert any("суммарный дебит 60" in warning for warning in warnings)


def test_training_envelope_rejects_empty_dataset():
    with pytest.raises(ValueError, match="датасет пуст"):
        TrainingEnvelope.fit([])
