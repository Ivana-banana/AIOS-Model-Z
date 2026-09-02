"""Замер ускорения прогноза: число «во сколько раз» должно быть честным.

П. 3.2.2 ТЗ требует ускоренного прогнозирования, то есть скорость — предмет
требования. Само время меряется на машине и в тесте воспроизведено быть не может;
проверяется здесь то, что легко испортить незаметно: какая величина попадает в
отчёт (медиана, а не лучший из повторов), считается ли ускорение от медианы и не
превращается ли отсутствующее время ГГДМ в ускорение «в ноль раз».
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "forecast_speed", Path(__file__).resolve().parent.parent / "scripts" / "forecast_speed.py")
forecast_speed = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(forecast_speed)


def test_the_reported_time_is_the_median_and_not_the_luckiest_repeat():
    """Лучший повтор — реклама, медиана — то, что получит проверяющий."""
    entry = forecast_speed.report("модель", [0.9, 0.5, 0.7], rows=100, ggdm_seconds=None)

    assert entry["медиана, с"] == 0.7
    assert entry["минимум, с"] == 0.5
    assert entry["первый вызов, с"] == 0.9
    assert entry["повторов"] == 3


def test_speedup_is_counted_from_the_median_time():
    entry = forecast_speed.report("модель", [2.0, 1.0, 1.0], rows=100, ggdm_seconds=1000.0)

    # медиана 1,0 с против 1000 с полного расчёта
    assert entry["быстрее ГГДМ, раз"] == 1000.0


def test_without_a_measured_full_run_there_is_no_speedup_at_all():
    """Пустое `--ggdm-seconds` означает «не измеряли», а не «в ноль раз быстрее».

    Ускорение — отношение к числу, полученному на **той же машине**. Подставить
    сюда чужое время (18 мин 43 с машины А, 11 мин из журнала) значило бы
    сравнивать два разных железа и выдавать это за замер.
    """
    entry = forecast_speed.report("модель", [0.5], rows=100, ggdm_seconds=None)

    assert "быстрее ГГДМ, раз" not in entry


def test_every_repeat_is_timed_including_the_first():
    calls = []
    times = forecast_speed.measure(lambda: calls.append(1), repeats=4)

    assert len(calls) == 4
    assert len(times) == 4
    assert all(value >= 0.0 for value in times)
