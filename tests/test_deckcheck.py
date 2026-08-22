"""Входной контроль чужого дека: каждая ловушка каталога закреплена отказом.

Проверки ставятся не ради Модели Z — на ней предположения генератора выполняются
и потому не видны. Они ставятся ради валидационного кейса (ТЗ п. 4.3), где любое
из них может не выполниться, а генератор сегодня напишет физически другой график
без единого сообщения.
"""

from __future__ import annotations

import pytest
from conftest import MODEL_Z_DIR

from aios import deckcheck

METRIC_HEAD = """RUNSPEC
METRIC

START
  1 'JAN' 2007 /

GRID
"""


def deck(tmp_path, body: str, name: str = "CASE.DATA"):
    path = tmp_path / name
    path.write_text(METRIC_HEAD + body, encoding="ascii")
    return path


def codes(found) -> list[str]:
    return sorted({refusal.code for refusal in found})


# ── ECL-001…003: файлы, включения, лексика ───────────────────────────────────

def test_non_ascii_outside_a_comment_is_refused(tmp_path):
    """Кириллица в имени скважины — там, где её разбирает парсер, а не читает человек."""
    path = tmp_path / "CASE.DATA"
    path.write_bytes(
        (METRIC_HEAD + "SCHEDULE\nWELSPECS\n  'СКВ1' 'G1' 10 10 1* OIL /\n/\n").encode("utf-8")
    )

    assert "ECL-001" in codes(deckcheck.check(path))


def test_non_ascii_only_in_comments_is_a_note_not_a_refusal(tmp_path):
    """Замер на самой Модели Z: её дек несёт кириллицу в заголовках, и Flow его считает.

    Входной контроль, отвергающий работающий кейс, стоит ровно столько же,
    сколько контроль, пропускающий сломанный.
    """
    path = tmp_path / "CASE.DATA"
    path.write_bytes(
        ("-- Copyright (C) Интегрированные разработки\n" + METRIC_HEAD).encode("utf-8")
    )

    assert "ECL-001" not in codes(deckcheck.check(path))
    assert deckcheck.notes(path), "но молчать об этом нельзя — при переносе кодировка поедет"


def test_a_missing_include_stops_the_check_before_anything_else(tmp_path):
    path = deck(tmp_path, "SCHEDULE\nINCLUDE\n  'sch.inc' /\n")

    found = deckcheck.check(path)

    assert codes(found) == ["ECL-002"]
    assert "неполному деку" in found[0].message


def test_nested_relative_includes_resolve_against_the_data_root(tmp_path):
    """Совместимостная особенность opm-common, а не произвол.

    Вложенный относительный INCLUDE разрешается от каталога `.DATA`, а не от
    файла, в котором записан. Разрешать «как удобнее» значит собрать другой дек.
    """
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "inner.inc").write_text("-- nothing to declare\n", encoding="ascii")
    (tmp_path / "outer.inc").write_text("INCLUDE\n  'sub/inner.inc' /\n", encoding="ascii")
    path = deck(tmp_path, "SCHEDULE\nINCLUDE\n  'outer.inc' /\n")

    assert "ECL-002" not in codes(deckcheck.check(path))


def test_a_cyclic_include_is_named_not_hung_on(tmp_path):
    (tmp_path / "loop.inc").write_text("INCLUDE\n  'loop.inc' /\n", encoding="ascii")
    path = deck(tmp_path, "SCHEDULE\nINCLUDE\n  'loop.inc' /\n")

    found = deckcheck.check(path)

    assert codes(found) == ["ECL-002"]
    assert "цикл" in found[0].message


def test_paths_aliases_are_refused_while_we_cannot_expand_them(tmp_path):
    path = deck(tmp_path, "PATHS\n  'DECK' '/data/deck' /\n/\n")

    assert "ECL-002" in codes(deckcheck.check(path))


def test_an_unbalanced_quote_makes_substitution_unsafe(tmp_path):
    path = deck(tmp_path, "SCHEDULE\nWELSPECS\n  'W1 'G1' 10 10 1* OIL /\n/\n")

    assert "ECL-003" in codes(deckcheck.check(path))


# ── ECL-005…008: единицы, PVT, время, рестарт ────────────────────────────────

def test_field_units_are_refused_because_the_numbers_would_stay_plausible(tmp_path):
    path = tmp_path / "CASE.DATA"
    path.write_text("RUNSPEC\nFIELD\n\nSTART\n  1 'JAN' 2007 /\n\nGRID\n", encoding="ascii")

    found = deckcheck.check(path)

    assert "ECL-005" in codes(found)
    assert "размерностное преобразование" in found[0].message


def test_two_unit_systems_at_once_are_refused_as_ambiguous(tmp_path):
    path = tmp_path / "CASE.DATA"
    path.write_text("RUNSPEC\nMETRIC\nLAB\n\nSTART\n  1 'JAN' 2007 /\n", encoding="ascii")

    assert "ECL-005" in codes(deckcheck.check(path))


def test_metric_alone_passes_the_unit_check(tmp_path):
    assert "ECL-005" not in codes(deckcheck.check(deck(tmp_path, "")))


def test_several_pvt_regions_are_refused(tmp_path):
    path = deck(tmp_path, "TABDIMS\n  1 3 30 30 /\n")

    found = deckcheck.check(path)

    assert "ECL-006" in codes(found)
    assert "3 PVT" in found[0].message


def test_dates_without_start_cannot_be_placed_on_the_calendar(tmp_path):
    path = tmp_path / "CASE.DATA"
    path.write_text("RUNSPEC\nMETRIC\n\nSCHEDULE\nDATES\n  1 'FEB' 2007 /\n/\n",
                    encoding="ascii")

    found = deckcheck.check(path)

    assert "ECL-007" in codes(found)
    assert any("нет START" in refusal.message for refusal in found)


def test_tstep_is_refused_because_a_quarter_year_is_not_a_calendar_date(tmp_path):
    path = deck(tmp_path, "SCHEDULE\nTSTEP\n  10*365.25 /\n")

    assert "ECL-007" in codes(deckcheck.check(path))


def test_a_mid_month_step_is_refused(tmp_path):
    path = deck(tmp_path, "SCHEDULE\nDATES\n  15 'FEB' 2007 /\n/\n")

    found = deckcheck.check(path)

    assert "ECL-007" in codes(found)
    assert any("месячной сетке" in refusal.message for refusal in found)


def test_monthly_dates_pass(tmp_path):
    path = deck(tmp_path, "SCHEDULE\nDATES\n  1 'FEB' 2007 /\n  1 'MAR' 2007 /\n/\n")

    assert "ECL-007" not in codes(deckcheck.check(path))


def test_a_restart_deck_is_refused_without_its_restart_file(tmp_path):
    path = deck(tmp_path, "SOLUTION\nRESTART\n  'BASE' 120 /\n")

    assert "ECL-008" in codes(deckcheck.check(path))


# ── ECL-009…011: скважины, управление, группы ────────────────────────────────

def test_a_control_without_a_preceding_welspecs_is_refused(tmp_path):
    path = deck(tmp_path, "SCHEDULE\nWCONPROD\n  'W1' OPEN LRAT 5* 30.0 /\n/\n")

    found = deckcheck.check(path)

    assert "ECL-009" in codes(found)
    assert "без предшествующей WELSPECS" in found[0].message


def test_a_well_name_longer_than_eight_characters_is_not_portable(tmp_path):
    path = deck(tmp_path, "SCHEDULE\nWELSPECS\n  'VERYLONGNAME' 'G1' 10 10 1* OIL /\n/\n")

    assert "ECL-009" in codes(deckcheck.check(path))


def test_a_declared_well_used_afterwards_is_fine(tmp_path):
    path = deck(
        tmp_path,
        "SCHEDULE\nWELSPECS\n  'W1' 'G1' 10 10 1* OIL /\n/\n"
        "WCONPROD\n  'W1' OPEN LRAT 5* 30.0 /\n/\n",
    )

    assert "ECL-009" not in codes(deckcheck.check(path))


def test_history_control_is_refused_because_it_would_take_our_keyword_place(tmp_path):
    path = deck(
        tmp_path,
        "SCHEDULE\nWELSPECS\n  'W1' 'G1' 10 10 1* OIL /\n/\n"
        "WCONHIST\n  'W1' OPEN ORAT 20.0 /\n/\n",
    )

    found = deckcheck.check(path)

    assert "ECL-010" in codes(found)


def test_group_controlled_wells_are_refused(tmp_path):
    path = deck(
        tmp_path,
        "SCHEDULE\nWELSPECS\n  'W1' 'G1' 10 10 1* OIL /\n/\n"
        "WCONPROD\n  'W1' OPEN GRUP 5* 30.0 /\n/\n",
    )

    found = deckcheck.check(path)

    assert "ECL-010" in codes(found)
    assert any("группой" in refusal.message for refusal in found)


def test_group_allocation_keywords_stop_the_generator(tmp_path):
    path = deck(tmp_path, "SCHEDULE\nGCONPROD\n  'FIELD' ORAT 1000.0 /\n/\n")

    assert "ECL-011" in codes(deckcheck.check(path))


# ── ECL-012…014: геометрия, логика, имена ────────────────────────────────────

@pytest.mark.parametrize("keyword", ["WELSEGS", "COMPSEGS", "COMPDATL", "CARFIN", "COMPDATMD"])
def test_geometry_beyond_a_literal_compdat_is_refused(tmp_path, keyword):
    path = deck(tmp_path, f"SCHEDULE\n{keyword}\n  'W1' 1 1 1 /\n/\n")

    assert "ECL-012" in codes(deckcheck.check(path))


@pytest.mark.parametrize("keyword", ["ACTIONX", "UDQ", "PYACTION", "ACTIONW"])
def test_runtime_logic_is_refused_because_static_replacement_deletes_it(tmp_path, keyword):
    path = deck(tmp_path, f"SCHEDULE\n{keyword}\n  'A1' 1 /\n/\n")

    assert "ECL-013" in codes(deckcheck.check(path))


def test_a_well_template_does_not_define_a_definite_set_of_wells(tmp_path):
    path = deck(
        tmp_path,
        "SCHEDULE\nWELSPECS\n  'W1' 'G1' 10 10 1* OIL /\n/\n"
        "WCONPROD\n  'W*' OPEN LRAT 5* 30.0 /\n/\n",
    )

    found = deckcheck.check(path)

    assert "ECL-014" in codes(found)
    assert "шаблоном" in " ".join(refusal.message for refusal in found)


def test_names_differing_only_in_case_are_not_merged_silently(tmp_path):
    path = deck(
        tmp_path,
        "SCHEDULE\nWELSPECS\n  'W1' 'G1' 10 10 1* OIL /\n"
        "  'w1' 'G1' 11 11 1* OIL /\n/\n",
    )

    assert "ECL-014" in codes(deckcheck.check(path))


# ── честность отчёта ─────────────────────────────────────────────────────────

def test_the_report_names_what_it_did_not_check(tmp_path):
    """«Не проверяли» и «проверили, всё хорошо» — разные утверждения."""
    text = deckcheck.report(deck(tmp_path, ""))

    assert "отказов нет" in text
    assert "ECL-004" in text and "ECL-015" in text
    assert "нужен симулятор" in text


def test_a_clean_metric_deck_passes_every_static_check(tmp_path):
    path = deck(
        tmp_path,
        "SCHEDULE\nWELSPECS\n  'W1' 'G1' 10 10 1* OIL /\n/\n"
        "COMPDAT\n  'W1' 10 10 1 3 OPEN 1* 1* 0.1 /\n/\n"
        "WCONPROD\n  'W1' OPEN LRAT 5* 30.0 /\n/\n"
        "DATES\n  1 'FEB' 2007 /\n/\n",
    )

    assert deckcheck.check(path) == []


MODEL_Z = MODEL_Z_DIR / "Model_Z.data"


@pytest.mark.skipif(not MODEL_Z.exists(), reason="дек Модели Z не распакован")
def test_the_deck_we_actually_compute_passes_every_static_check():
    """Калибровка контроля: рабочий кейс обязан проходить.

    Модель Z посчитана полным прогоном (18 мин 43 с, ЧДД 11 851,8), значит любой
    отказ на ней — ложный по построению. Именно этот тест дважды поправил
    проверки: `TABDIMS 2 2 …` (два PVT-региона, которые `aios.pvt` умеет) и
    `01 NOV 1994` (день сравнивался строкой, а не числом).

    Заметки при этом есть и остаются: кириллица в заголовках четырёх файлов.
    """
    assert deckcheck.check(MODEL_Z) == []
    assert len(deckcheck.notes(MODEL_Z)) == 4


# ── три ложных срабатывания, найденных на открытых деках (docs/portability.md) ──

def test_an_include_without_quotes_is_a_legal_form(tmp_path):
    """SPE9 из opm-data пишет путь голым токеном; OPM считает эти деки регрессией.

    Требование кавычек безусловно давало отказ ECL-002 и — хуже — останавливало
    все последующие проверки.
    """
    (tmp_path / "TOPSVALUES.DATA").write_text("-- nothing\n", encoding="ascii")
    path = deck(tmp_path, "GRID\nINCLUDE\nTOPSVALUES.DATA /\n")

    assert "ECL-002" not in codes(deckcheck.check(path))


def test_a_quoted_include_with_spaces_still_works(tmp_path):
    """Кавычки нужны именно для таких путей, и они обязаны продолжать работать."""
    (tmp_path / "a b").mkdir()
    (tmp_path / "a b" / "grid.inc").write_text("-- nothing\n", encoding="ascii")
    path = deck(tmp_path, "GRID\nINCLUDE\n  'a b/grid.inc' /\n")

    assert "ECL-002" not in codes(deckcheck.check(path))


def test_a_well_name_padded_with_spaces_is_the_same_well(tmp_path):
    """Eclipse дополняет имена до восьми символов: 'INJE1 ' и 'INJE1' — одна скважина.

    Сравнение как есть давало 10 ложных отказов ECL-009 на SPE9group.
    """
    path = deck(
        tmp_path,
        "SCHEDULE\nWELSPECS\n  'INJE1 ' 'G1' 10 10 1* WATER /\n/\n"
        "WCONINJE\n  'INJE1' WATER OPEN RATE 100.0 /\n/\n",
    )

    assert "ECL-009" not in codes(deckcheck.check(path))
    assert "ECL-014" not in codes(deckcheck.check(path)), "и не коллизия регистра"


def test_pvtnum_assigned_through_equals_counts_as_defined(tmp_path):
    """Norne задаёт регион строкой внутри EQUALS, а не своим ключевым словом.

    Регион при этом полностью выводим, и отказ был отказом на рабочем деке.
    """
    path = deck(
        tmp_path,
        "TABDIMS\n  2 2 50 50 /\n\nREGIONS\nEQUALS\n"
        "  'PVTNUM' 1 1 46 1 112 1 22 /\n/\n",
    )

    assert "ECL-006" not in codes(deckcheck.check(path))


def test_several_pvt_regions_with_no_assignment_anywhere_are_still_refused(tmp_path):
    """Смягчение не должно превратиться в отсутствие проверки."""
    path = deck(tmp_path, "TABDIMS\n  2 2 50 50 /\n")

    assert "ECL-006" in codes(deckcheck.check(path))
