"""Обвязка ЧДД обязана давать ровно то же, что эталонный расчётчик организаторов.

Расхождение заявленного ЧДД с их пересчётом = дисквалификация, поэтому паритет
проверяется на каждом прогоне, а контрольные числа заморожены.
"""

from __future__ import annotations

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
def test_norms_come_from_the_organisers_file_not_from_literals():
    assumptions, pumps = economics.load_norms()
    assert assumptions["oilPriceRubT"] == 28_000.0
    assert assumptions["waccRate"] == 10.0
    assert assumptions["profitTaxRate"] == 25.0
    assert assumptions["propertyTaxRate"] == 2.2
    assert len(pumps) == 12
    assert pumps[-1]["nominal"] == 400 and pumps[-1]["max"] == 500
