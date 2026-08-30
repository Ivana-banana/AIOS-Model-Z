"""Скрипт Manus 5-1 сохраняет сумму ставок, а не объём — и это не одно и то же.

`docs/manus/cyclic-schedules.py` пришёл как эталон постановки эксперимента
«равномерная закачка против циклической при равной накопленной за год». Он
нормирует сумму месячных ставок. Ставка `WCONINJE` — м³/сут, месяцы разной
длины, поэтому годовой **объём** у двух расписаний расходится, а знак
расхождения переворачивается вместе с фазой.

Тест закрепляет дефект, а не чинит его: скрипт лежит в репозитории побайтово
как пришёл, и эти проверки объясняют, почему им нельзя пользоваться на нашем
расписании. Разбор — `docs/manus/cyclic-waterflooding.md`, решение Р-112.
"""

from __future__ import annotations

import calendar
import csv
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "cyclic_schedules", ROOT / "docs" / "manus" / "cyclic-schedules.py")
cyclic_schedules = importlib.util.module_from_spec(SPEC)
# `dataclass` внутри скрипта ищет свой модуль в sys.modules — без регистрации
# импорт падает на разборе аннотаций.
sys.modules[SPEC.name] = cyclic_schedules
SPEC.loader.exec_module(cyclic_schedules)


def _year(tmp_path: Path, year: int, rate: float = 100.0) -> Path:
    path = tmp_path / "input.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["well", "date", "rate"])
        for month in range(1, 13):
            writer.writerow(["W1", f"{year}-{month:02d}-01", rate])
    return path


def _schedules(path: Path, half_period: int, amplitude: float, phase: int):
    records = cyclic_schedules.read_input(path)
    return cyclic_schedules.build_schedules(records, half_period, amplitude, phase)


def _volume(records: list) -> float:
    """Объём за год: ставка в м³/сут умножается на длину своего месяца."""
    return sum(item.rate * calendar.monthrange(item.year, item.month_index)[1]
               for item in records)


def test_the_sum_of_monthly_rates_is_preserved_exactly(tmp_path):
    uniform, cyclic = _schedules(_year(tmp_path, 2025), half_period=1,
                                 amplitude=0.75, phase=0)

    assert sum(item.rate for item in cyclic) == pytest.approx(
        sum(item.rate for item in uniform), abs=1e-9)


def test_the_annual_volume_is_not_preserved_because_months_differ_in_length(tmp_path):
    uniform, cyclic = _schedules(_year(tmp_path, 2025), half_period=1,
                                 amplitude=0.75, phase=0)

    assert _volume(cyclic) / _volume(uniform) - 1 == pytest.approx(-0.00616, abs=5e-5)


def test_the_sign_of_the_volume_error_flips_with_the_phase(tmp_path):
    """Перестановка фаз — сценарий C6 справки; он сравнивал бы разные объёмы воды."""
    path = _year(tmp_path, 2025)
    uniform, first = _schedules(path, half_period=1, amplitude=0.75, phase=0)
    _, second = _schedules(path, half_period=1, amplitude=0.75, phase=1)

    base = _volume(uniform)
    assert _volume(first) < base < _volume(second)
    assert (_volume(second) - _volume(first)) / base == pytest.approx(0.0123, abs=5e-5)


def test_a_leap_year_hides_the_defect_at_a_three_month_half_period(tmp_path):
    """Низкие и высокие кварталы 2024 набирают по 183 дня — ноль случайный."""
    uniform, cyclic = _schedules(_year(tmp_path, 2024), half_period=3,
                                 amplitude=0.75, phase=0)

    assert _volume(cyclic) == pytest.approx(_volume(uniform), abs=1e-6)
