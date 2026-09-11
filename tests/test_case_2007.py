"""Валидационный кейс трека 2: ограничения кейса выражаются конфигом, а не кодом.

Кейс меняет условия задачи, но не объект. Эти тесты закрепляют ровно это
утверждение: файл `config-case.toml` читается штатным загрузчиком, и каждое
условие из текста кейса оказывается в структуре ограничений тем числом, которое
написано в кейсе. Без такого теста опечатка в ключе или дате прошла бы молча —
незнакомый ключ загрузчик отвергает, а вот дата 2007-06-01 вместо 2007-04-01
выглядит правдоподобно и меняет обязательный ремонт.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from aios import balance, config

CASE = Path(__file__).resolve().parents[1] / "config.toml"


@pytest.fixture(scope="module")
def limits():
    return config.load(CASE).limits


def test_case_config_loads(limits):
    assert limits is not None


def test_field_caps_are_600_each(limits):
    assert limits.max_field_liquid_m3_per_day == 600
    assert limits.max_field_injection_m3_per_day == 600


def test_field_caps_apply_to_every_month_not_to_the_annual_average(limits):
    # «в любой момент времени» — годовое усреднение разрешило бы пик вдвое выше.
    assert limits.field_limits_are_annual_average is False


def test_compensation_corridor_is_085_115_over_a_three_month_window(limits):
    assert limits.water_balance_rule == "compensation"
    assert limits.water_tolerance == pytest.approx(1.15)
    assert limits.water_tolerance_min == pytest.approx(0.85)
    assert limits.compensation_window_months == 3


def test_pressures_come_from_the_case(limits):
    assert limits.min_reservoir_pressure_atm == pytest.approx(108.0)
    assert limits.min_producer_bhp_atm == pytest.approx(50.0)
    assert limits.max_injector_bhp_atm == pytest.approx(300.0)


def test_all_sixteen_mandatory_workovers_are_present(limits):
    windows = limits.forced_shut_windows
    assert len(windows) == 16
    # Скв. 54 останавливается дважды — в 2008 и в 2023.
    assert sum(1 for item in windows if item.well == "54") == 2
    first = next(item for item in windows if item.well == "90")
    assert (first.start, first.end) == (date(2007, 4, 1), date(2007, 6, 1))
    last = next(item for item in windows if item.well == "7")
    assert (last.start, last.end) == (date(2025, 5, 1), date(2025, 9, 1))


def test_a_workover_window_closes_whole_months(limits):
    # Окно «с 01.04 по 01.06» кончается первым числом, но шаг расписания —
    # месяц: июнь закрывается целиком. Огрубление в сторону остановки.
    window = next(item for item in limits.forced_shut_windows if item.well == "90")
    assert window.covers(date(2007, 4, 1))
    assert window.covers(date(2007, 6, 1))
    assert not window.covers(date(2007, 3, 1))
    assert not window.covers(date(2007, 7, 1))


def _row(stamp: str, oil_t: float, liquid_t: float, injected_m3: float) -> dict:
    return {"DATA": stamp, "well": "1", "WOMT_Diff": oil_t, "WLPT_Diff": liquid_t,
            "WWIT_Diff": injected_m3}


def test_a_single_month_outage_does_not_break_the_rolling_corridor():
    """Смысл окна: единичный простой нагнетательной скважины — не нарушение.

    Три месяца ровной компенсации, в среднем из которых закачка провалена
    полностью. По месяцу это −100 %, по окну — в коридоре.
    """
    rows = [_row("2007-01-01", 100.0, 100.0, 164.0),
            _row("2007-02-01", 100.0, 100.0, 0.0),
            _row("2007-03-01", 100.0, 100.0, 164.0)]
    monthly = balance.rolling_compensation_problems(rows, window=1)
    assert monthly, "по одному месяцу провал закачки обязан читаться нарушением"
    assert not balance.rolling_compensation_problems(rows, window=3)


def test_a_sustained_shortfall_is_still_caught():
    """И обратное: окно не должно прощать систематическую недокомпенсацию."""
    rows = [_row(f"2007-{month:02d}-01", 100.0, 100.0, 10.0) for month in range(1, 7)]
    problems = balance.rolling_compensation_problems(rows, window=3)
    assert len(problems) == 4
    assert "ниже границы 85%" in problems[0]


def test_the_rolling_ratio_is_a_ratio_of_sums_not_a_mean_of_ratios():
    """Месяц с почти нулевым отбором не должен уводить среднее в бесконечность."""
    rows = [_row("2007-01-01", 100.0, 100.0, 109.0),
            _row("2007-02-01", 0.001, 0.001, 0.0),
            _row("2007-03-01", 100.0, 100.0, 109.0)]
    assert not balance.rolling_compensation_problems(rows, window=3)


def test_short_windows_at_the_start_are_not_judged():
    rows = [_row("2007-01-01", 100.0, 100.0, 0.0),
            _row("2007-02-01", 100.0, 100.0, 0.0)]
    assert balance.rolling_compensation_problems(rows, window=3) == []


def test_an_empty_corridor_is_refused():
    with pytest.raises(ValueError, match="пустой коридор"):
        balance.rolling_compensation_problems([], ceiling=0.5, floor=0.9)
