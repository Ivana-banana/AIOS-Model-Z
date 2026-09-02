"""Репликация фонда для замера масштабируемости: копии независимы и полны."""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

from aios.deck import InjectorControl, ProducerControl, ScheduleStep

SPEC = importlib.util.spec_from_file_location(
    "benchmark_scale",
    Path(__file__).resolve().parent.parent / "scripts" / "benchmark_scale.py")
benchmark_scale = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark_scale)


def _rows() -> list[dict[str, object]]:
    return [
        {"DATA": "2007-01-01", "well": "1", "WLPT_Diff": 10.0, "WOMT_Diff": 4.0},
        {"DATA": "2007-02-01", "well": "1", "WLPT_Diff": 11.0, "WOMT_Diff": 3.0},
        {"DATA": "2007-01-01", "well": "2", "WLPT_Diff": 20.0, "WOMT_Diff": 8.0},
    ]


def _step() -> ScheduleStep:
    return ScheduleStep(
        date=date(2007, 1, 1),
        producers=(ProducerControl(well="1", status="OPEN", mode="LRAT",
                                   liquid_rate=100.0, bhp=50.0),),
        injectors=(InjectorControl(well="2", fluid="WATER", status="OPEN",
                                   mode="RATE", rate=200.0, bhp=300.0),),
    )


def test_replication_keeps_the_original_fund_under_its_own_names():
    """Нулевая копия — исходный фонд побайтово: иначе ×1 не точка отсчёта."""
    replicated = benchmark_scale.replicate_rows(_rows(), 3)

    assert replicated[:3] == _rows()


def test_every_copy_of_the_fund_gets_the_whole_history():
    """Копия неполна — время счёта занижено, а замер этого не увидит."""
    replicated = benchmark_scale.replicate_rows(_rows(), 4)

    assert len(replicated) == 12
    for copy in range(4):
        suffix = "" if copy == 0 else f"_k{copy}"
        wells = {row["well"] for row in replicated if str(row["well"]).endswith(suffix)}
        assert wells >= {f"1{suffix}", f"2{suffix}"}


def test_replicated_wells_carry_the_values_of_their_original():
    """Копия отличается только именем: ЧДД обязан выйти кратным."""
    replicated = benchmark_scale.replicate_rows(_rows(), 2)

    copy = [row for row in replicated if row["well"] == "1_k1"]
    assert [row["WLPT_Diff"] for row in copy] == [10.0, 11.0]
    assert [row["WOMT_Diff"] for row in copy] == [4.0, 3.0]


def test_schedule_keeps_its_dates_while_the_fund_widens():
    """Раздувается фонд, а не горизонт: шагов столько же, скважин в них больше."""
    replicated = benchmark_scale.replicate_steps([_step()], 3)

    assert len(replicated) == 1
    assert replicated[0].date == date(2007, 1, 1)
    assert [control.well for control in replicated[0].producers] == ["1", "1_k1", "1_k2"]
    assert [control.well for control in replicated[0].injectors] == ["2", "2_k1", "2_k2"]


def test_replicated_controls_keep_their_regime():
    """Скопировать имя, но потерять режим — значит мерить другой сценарий."""
    replicated = benchmark_scale.replicate_steps([_step()], 2)

    producer = replicated[0].producers[1]
    assert (producer.status, producer.mode, producer.liquid_rate, producer.bhp) == (
        "OPEN", "LRAT", 100.0, 50.0)
    injector = replicated[0].injectors[1]
    assert (injector.fluid, injector.mode, injector.rate) == ("WATER", "RATE", 200.0)


def test_single_factor_changes_nothing():
    """×1 обязан быть исходным фондом: с ним сравниваются остальные точки."""
    assert benchmark_scale.replicate_rows(_rows(), 1) == _rows()
    assert benchmark_scale.replicate_steps([_step()], 1) == [_step()]
