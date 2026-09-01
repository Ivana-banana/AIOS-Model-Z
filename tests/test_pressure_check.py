"""Сверка прогноза давления с фактом: считается решение гейта, а не красивый R².

Замер 01.09 (`docs/experiments.md`) показал, зачем это разделение: на заявляемом
кандидате при пределе 290 атм факт нарушают 17 строк, прогноз называет 19 — и
**ни одной общей**. Средняя ошибка при этом 11,5 атм, то есть «приличная». Метрика
ошибки такого не показывает вовсе, а таблица ложных тревог и пропусков показывает.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "pressure_check", Path(__file__).resolve().parent.parent / "scripts" / "pressure_check.py")
pressure_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pressure_check)


def _row(date: str, well: str, thp: float, liquid: float = 0.0, injection: float = 0.0) -> dict:
    return {"DATA": date, "well": well, "THP": thp, "WLPR": liquid, "WWIR": injection}


def test_only_rows_present_in_both_are_compared():
    """Скважина, которой нет в прогнозе, не сравнивается ни с чем.

    Обученный суррогат знает свой список `control_wells`, и он не обязан
    совпадать с фондом выгрузки. Молча приписать такой строке нулевое давление
    значило бы объявить нарушение там, где прогноза не было вовсе.
    """
    fact = [_row("2007-01-01", "1", 100.0, liquid=10.0),
            _row("2007-01-01", "2", 200.0, injection=10.0)]
    predicted = [_row("2007-01-01", "1", 105.0, liquid=10.0)]

    matched = pressure_check.pairs(fact, predicted)

    assert len(matched) == 1
    assert matched[0][:4] == ("2007-01-01", "1", 100.0, 105.0)


def test_a_missed_violation_and_a_false_alarm_are_counted_apart():
    """Пропуск и ложная тревога — разные беды, и цена у них разная."""
    matched = [
        # факт нарушает, прогноз молчит — пропуск
        ("2007-01-01", "1", 295.0, 285.0, 0.0, 50.0),
        # прогноз кричит, факт спокоен — ложная тревога
        ("2007-02-01", "1", 285.0, 295.0, 0.0, 50.0),
        # оба спокойны
        ("2007-03-01", "1", 280.0, 281.0, 0.0, 50.0),
    ]

    counts = pressure_check.gate_table(matched, 290.0, injectors=True)

    assert counts["строк"] == 3
    assert counts["нарушает по факту"] == 1
    assert counts["нарушает по прогнозу"] == 1
    assert counts["пропусков"] == 1
    assert counts["ложных тревог"] == 1


def test_a_shut_well_carries_no_pressure_and_is_not_counted():
    """Остановленная скважина даёт нулевое давление в контракте — не нарушение."""
    matched = [("2007-01-01", "1", 0.0, 0.0, 0.0, 0.0)]

    injectors = pressure_check.gate_table(matched, 290.0, injectors=True)
    producers = pressure_check.gate_table(matched, 50.0, injectors=False)

    assert injectors["строк"] == 0
    assert producers["строк"] == 0
    assert producers["нарушает по факту"] == 0


def test_the_role_of_a_row_is_taken_from_the_fact_not_from_the_forecast():
    """Роль берётся по факту: другого признака в контракте расчётчика нет.

    Строка нагнетательной скважины не должна попасть в проверку добывающих
    только потому, что там низкое давление, — иначе каждая закачка ниже 50 атм
    станет «нарушением предела добывающей».
    """
    matched = [("2007-01-01", "1", 20.0, 20.0, 0.0, 50.0)]

    producers = pressure_check.gate_table(matched, 50.0, injectors=False)

    assert producers["строк"] == 0
    assert producers["нарушает по факту"] == 0
