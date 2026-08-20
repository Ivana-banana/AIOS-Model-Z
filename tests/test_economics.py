"""Обвязка ЧДД обязана давать ровно то же, что эталонный расчётчик организаторов.

Расхождение заявленного ЧДД с их пересчётом = дисквалификация, поэтому паритет
проверяется на каждом прогоне, а контрольные числа заморожены.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from conftest import EXAMPLE_INPUT

from aios import economics

#: ЧДД примера организаторов при старте по умолчанию (минимальная дата данных).
GOLDEN_CHDD_DEFAULT_START = 52.05489669421487

#: Тот же пример, но год дисконтирования задан явно.
GOLDEN_CHDD_START_2014 = 54.10613636363636


def test_vendor_version_is_pinned():
    """Обновился архив организаторов — контрольные числа надо перепроверить, а не подогнать."""
    assert economics.VENDOR_VERSION == "7.0.2-negative-row-filter"


def test_required_columns_are_the_organisers_contract():
    assert economics.REQUIRED_COLUMNS == [
        "DATA", "well", "WLPT", "WLPR", "WOMT", "WOMR", "WWIR", "WWIT",
        "THP", "BHP", "WEFF", "WLPT_Diff", "WOMT_Diff", "WWIT_Diff",
    ]


@pytest.mark.skipif(not EXAMPLE_INPUT.is_file(), reason="нет примера исходных данных организаторов")
def test_example_npv_matches_frozen_value():
    result = economics.compute_npv_from_file(EXAMPLE_INPUT)
    assert result.chdd_m == pytest.approx(GOLDEN_CHDD_DEFAULT_START, abs=1e-9)


@pytest.mark.skipif(not EXAMPLE_INPUT.is_file(), reason="нет примера исходных данных организаторов")
def test_start_date_shifts_discounting_base():
    result = economics.compute_npv_from_file(EXAMPLE_INPUT, start_date="2014-01-01")
    assert result.chdd_m == pytest.approx(GOLDEN_CHDD_START_2014, abs=1e-9)
    assert result.chdd_m != pytest.approx(GOLDEN_CHDD_DEFAULT_START, abs=1e-9)


@pytest.mark.skipif(not EXAMPLE_INPUT.is_file(), reason="нет примера исходных данных организаторов")
def test_wrapper_is_bit_identical_to_direct_vendor_call():
    """Наша обвязка ничего не добавляет от себя — сверяем с прямым вызовом эталона."""
    import chdd_model  # вендор попадает в sys.path при импорте aios.economics

    headers, records = economics.load_records(EXAMPLE_INPUT)
    assumptions, pumps = economics.load_norms()
    direct = chdd_model.compute_calculation(
        records, headers=headers, assumptions=assumptions, pumps=pumps, start_date=None
    )
    wrapped = economics.compute_npv(
        records, headers=headers, assumptions=assumptions, pumps=pumps, start_date=None
    )
    assert wrapped.chdd_m == direct["summary"]["totalChddM"]
    assert wrapped.raw["annual"] == direct["annual"]
    assert wrapped.raw["fieldMonthly"] == direct["fieldMonthly"]


@pytest.mark.skipif(not EXAMPLE_INPUT.is_file(), reason="нет примера исходных данных организаторов")
@pytest.mark.parametrize(
    ("start_year", "golden"),
    [(None, GOLDEN_CHDD_DEFAULT_START), (2014, GOLDEN_CHDD_START_2014)],
)
def test_organisers_own_script_gives_the_same_number(tmp_path, start_year, golden):
    """Прямой запуск `РАСЧЕТ_ЧДД.py` — то, чем считают нас.

    Сверка выше идёт с `chdd_model` внутри процесса, то есть проверяет, что мы
    не добавили ничего поверх модели. Но организаторы запускают не модель, а свою
    точку входа: она сама читает исходный файл и сама строит `start_date` из
    года. Именно там расхождение и осталось бы незамеченным, а расхождение
    заявленного ЧДД с их пересчётом — дисквалификация без второй попытки.
    """
    json_file = tmp_path / "result.json"
    command = [
        sys.executable,
        str(economics.VENDOR_DIR / "РАСЧЕТ_ЧДД.py"),
        "--input", str(EXAMPLE_INPUT),
        "--output", str(tmp_path / "report.xlsx"),
        "--json", str(json_file),
    ]
    if start_year is not None:
        command += ["--start-year", str(start_year)]

    completed = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    assert completed.returncode == 0, completed.stderr

    theirs = json.loads(json_file.read_text(encoding="utf-8"))["summary"]["totalChddM"]
    start_date = f"{start_year}-01-01" if start_year is not None else None
    ours = economics.compute_npv_from_file(EXAMPLE_INPUT, start_date=start_date).chdd_m

    assert theirs == pytest.approx(golden, abs=1e-9)
    # сравнение по модулю разности, а не по round: округление скрыло бы дрейф
    assert abs(theirs - ours) <= 1e-9


@pytest.mark.skipif(not EXAMPLE_INPUT.is_file(), reason="нет примера исходных данных организаторов")
def test_norms_come_from_the_organisers_file_not_from_literals():
    assumptions, pumps = economics.load_norms()
    assert assumptions["oilPriceRubT"] == 28_000.0
    assert assumptions["waccRate"] == 10.0
    assert assumptions["profitTaxRate"] == 25.0
    assert assumptions["propertyTaxRate"] == 2.2
    assert len(pumps) == 12
    assert pumps[-1]["nominal"] == 400 and pumps[-1]["max"] == 500


def test_start_date_has_no_default_because_forgetting_it_changes_the_declared_number():
    """Базовый год дисконтирования нельзя забыть молча.

    У эталонного расчётчика умолчание есть — минимальная дата данных. Оно не
    ошибка, но тихо меняет базовый год: выгрузка с историей от 1994 и она же с
    2007 дисконтируются по-разному и обе считаются без единой жалобы. Заявляемое
    число при этом другое, а расхождение с пересчётом организаторов — это
    дисквалификация с одной попытки.

    Поэтому у нашей обвязки умолчания нет: аргумент обязателен, и передать `None`
    осознанно по-прежнему можно — но это решение будет видно в коде вызова.
    Проверено 21.08: все десять вызовов в `aios/` и `scripts/` и все семнадцать
    в тестах дату передают, так что требование ничего не ломает.
    """
    import inspect

    signature = inspect.signature(economics.compute_npv)
    start_date = signature.parameters["start_date"]

    assert start_date.default is inspect.Parameter.empty, (
        "у start_date появилось умолчание — забытый аргумент снова станет тихим"
    )
    assert start_date.kind is inspect.Parameter.KEYWORD_ONLY

    with pytest.raises(TypeError):
        economics.compute_npv([], headers=economics.REQUIRED_COLUMNS)
