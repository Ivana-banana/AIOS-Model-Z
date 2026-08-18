"""PVT-регионы: какая плотность нефти относится к какой скважине.

Плотность переводит объём в тонны, тонны идут в выручку — ошибка здесь стоит
денег в ЧДД. Проверки идут на маленьких синтетических деках, а не на Модели Z:
на валидационном кейсе организаторов (ТЗ п. 4.3) дек будет другой, и разбор
обязан работать на любом, а не на нашем.
"""

from __future__ import annotations

import pytest

from aios.pvt import (
    assign_regions,
    parse_completions,
    parse_density,
    parse_int_array,
    parse_volume_factors,
    report,
    well_fluids,
)

DENSITY = """\
DENSITY
-- oil               water gas
   913.0765883224138 1172  1.195725159619077  /
   928.1613457381528 1173  0.9360160248733308 /
"""

#: Сетка 2×2×2: восемь ячеек, нижний слой — второй регион.
SMALL_GRID = (2, 2, 2)
SMALL_REGIONS = """\
PVTNUM
 1 1 1 1
 2 2 2 2
/
"""

COMPDAT = """\
COMPDAT
-- WELL I J K1 K2 STATE
 'A' 1 1 1 1 OPEN 1* 1* 0.146 1* 2 1* Z /
 'B' 2 2 2 2 OPEN 1* 1* 0.146 1* 2 1* Z /
/
"""


# --- разбор DENSITY ---------------------------------------------------------

def test_density_rows_become_numbered_regions():
    regions = parse_density(DENSITY)
    assert [r.number for r in regions] == [1, 2]
    assert regions[0].oil_density == pytest.approx(913.0765883224138)
    assert regions[1].oil_density == pytest.approx(928.1613457381528)


def test_water_density_is_read_too():
    """Вода тоже идёт в массу жидкости, а регионы по ней отличаются."""
    regions = parse_density(DENSITY)
    assert regions[0].water_density == pytest.approx(1172.0)
    assert regions[1].water_density == pytest.approx(1173.0)


def test_missing_density_is_an_error_not_a_default():
    with pytest.raises(ValueError, match="DENSITY"):
        parse_density("PVTNUM\n 1 /\n")


# --- разбор массива регионов ------------------------------------------------

def test_repeat_counts_are_expanded():
    assert parse_int_array("PVTNUM\n 3*1 2*2 /\n", "PVTNUM") == [1, 1, 1, 2, 2]


def test_plain_values_and_repeats_mix():
    assert parse_int_array("PVTNUM\n 1 2*0 2 /\n", "PVTNUM") == [1, 0, 0, 2]


def test_comments_inside_the_array_are_ignored():
    assert parse_int_array("PVTNUM\n-- слой 1\n 2*1\n-- слой 2\n 2*2 /\n", "PVTNUM") == [1, 1, 2, 2]


def test_missing_keyword_is_an_error():
    with pytest.raises(ValueError, match="SATNUM"):
        parse_int_array("PVTNUM\n 1 /\n", "SATNUM")


# --- разбор вскрытий --------------------------------------------------------

def test_completions_are_grouped_by_well():
    completions = parse_completions(COMPDAT)
    assert completions == {"A": [(1, 1, 1, 1)], "B": [(2, 2, 2, 2)]}


def test_several_blocks_of_one_well_are_kept():
    text = COMPDAT + "\nCOMPDAT\n 'A' 1 2 1 2 OPEN 1* 1* 0.146 /\n/\n"
    assert parse_completions(text)["A"] == [(1, 1, 1, 1), (1, 2, 1, 2)]


def test_defaulted_coordinates_are_skipped_not_guessed():
    """`1*` в координатах — не ноль и не единица; такую запись пропускаем."""
    text = "COMPDAT\n 'A' 1* 1* 1 1 OPEN /\n/\n"
    assert parse_completions(text) == {}


# --- отнесение скважины к региону -------------------------------------------

def test_well_is_assigned_to_the_region_of_its_cells():
    assigned = assign_regions(parse_completions(COMPDAT), parse_int_array(SMALL_REGIONS, "PVTNUM"),
                              SMALL_GRID)
    assert assigned["A"].region == 1
    assert assigned["B"].region == 2


def test_majority_of_completed_cells_wins():
    completions = {"A": [(1, 1, 1, 1), (2, 1, 1, 1), (1, 1, 2, 2)]}  # два в первом, один во втором
    assigned = assign_regions(completions, parse_int_array(SMALL_REGIONS, "PVTNUM"), SMALL_GRID)
    assert assigned["A"].region == 1
    assert assigned["A"].counts == {1: 2, 2: 1}
    assert assigned["A"].mixed is True
    assert assigned["A"].share == pytest.approx(2 / 3)


def test_a_well_inside_one_region_is_not_marked_mixed():
    assigned = assign_regions({"A": [(1, 1, 1, 1)]}, parse_int_array(SMALL_REGIONS, "PVTNUM"),
                              SMALL_GRID)
    assert assigned["A"].mixed is False
    assert assigned["A"].share == 1.0


def test_a_tie_is_resolved_by_the_lower_region_number():
    """Выбор обязан быть воспроизводимым, а не зависеть от порядка словаря."""
    completions = {"A": [(1, 1, 1, 1), (1, 1, 2, 2)]}
    assigned = assign_regions(completions, parse_int_array(SMALL_REGIONS, "PVTNUM"), SMALL_GRID)
    assert assigned["A"].region == 1


def test_inactive_cells_do_not_form_a_region():
    """`PVTNUM = 0` — ячейка вне пласта; в Модели Z таких 306 974 из 547 638."""
    zeros = parse_int_array("PVTNUM\n 4*0 4*2 /\n", "PVTNUM")
    assigned = assign_regions({"A": [(1, 1, 1, 1)]}, zeros, SMALL_GRID)
    assert "A" not in assigned, "скважина только в неактивных ячейках региона не имеет"


def test_completions_outside_the_grid_are_ignored():
    assigned = assign_regions({"A": [(99, 99, 1, 1)]}, parse_int_array(SMALL_REGIONS, "PVTNUM"),
                              SMALL_GRID)
    assert "A" not in assigned


def test_grid_mismatch_fails_loudly():
    """Массив не на ту сетку — это чужой дек, а не повод считать дальше."""
    with pytest.raises(ValueError, match="не совпадает с сеткой"):
        assign_regions({"A": [(1, 1, 1, 1)]}, [1, 2, 3], SMALL_GRID)


def test_cell_index_follows_eclipse_natural_order():
    """I меняется быстрее J, J быстрее K. Перепутанный порядок — чужой регион."""
    array = parse_int_array("PVTNUM\n 1 2 2 2 2 2 2 2 /\n", "PVTNUM")
    assert assign_regions({"A": [(1, 1, 1, 1)]}, array, SMALL_GRID)["A"].region == 1
    assert assign_regions({"B": [(2, 1, 1, 1)]}, array, SMALL_GRID)["B"].region == 2


# --- плотности по скважинам -------------------------------------------------

def test_well_fluids_carry_the_density_of_their_region():
    regions = parse_density(DENSITY)
    assigned = assign_regions(parse_completions(COMPDAT), parse_int_array(SMALL_REGIONS, "PVTNUM"),
                              SMALL_GRID)
    fluids = well_fluids(regions, assigned)
    assert fluids["A"].oil_density == pytest.approx(913.0765883224138)
    assert fluids["B"].oil_density == pytest.approx(928.1613457381528)
    assert fluids["B"].water_density == pytest.approx(1173.0)


def test_report_names_regions_and_counts_mixed_wells():
    regions = parse_density(DENSITY)
    assigned = assign_regions(
        {"A": [(1, 1, 1, 2)]}, parse_int_array(SMALL_REGIONS, "PVTNUM"), SMALL_GRID
    )
    text = report(regions, assigned)
    assert "913.08" in text and "928.16" in text
    assert "несколько регионов: 1" in text


# --- объёмные коэффициенты --------------------------------------------------

PVT_TABLES = """\
PVTO
-- gor      pressure  fvf      viscosity
   0.005     1.13     1.0014   34.19
             10.29    1.0011   34.58
             19.45    0.9900   34.97
/
PVTW
-- pref fvf compressibility viscosity viscosibility
   123  1   0.000046 1.6 /
   125  1   0.000046 1.6 /
"""


def test_oil_volume_factor_range_comes_from_pvto():
    factors = parse_volume_factors(PVT_TABLES)
    assert factors.oil_min == pytest.approx(0.9900)
    assert factors.oil_max == pytest.approx(1.0014)


def test_continuation_lines_of_pvto_are_read():
    """У PVTO первая строка узла длиннее продолжений — иначе теряются точки."""
    assert parse_volume_factors(PVT_TABLES).oil_max == pytest.approx(1.0014)


def test_water_volume_factor_comes_from_pvtw():
    assert parse_volume_factors(PVT_TABLES).water == pytest.approx(1.0)


def test_spread_shows_how_much_the_factor_moves():
    factors = parse_volume_factors(PVT_TABLES)
    assert factors.oil_spread == pytest.approx((1.0014 - 0.99) / 0.99)


def test_missing_pvto_is_an_error():
    with pytest.raises(ValueError, match="PVTO"):
        parse_volume_factors("PVTW\n 123 1 0.000046 1.6 /\n")


def test_missing_pvtw_is_an_error():
    with pytest.raises(ValueError, match="PVTW"):
        parse_volume_factors("PVTO\n 0.005 1.13 1.0014 34.19 /\n")
