"""Чтение и сверка эталонной раскладки организаторов (`aios.reference`).

Файл организаторов лежит в репозитории, поэтому контрольные числа берутся из
него самого: если книгу заменят другой версией, тесты упадут — и это правильно,
сверка на подменённом эталоне не значит ничего.
"""

from __future__ import annotations

from pathlib import Path

import openpyxl
import pytest

from aios import reference

REFERENCE_FILE = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "reference"
    / "Расчет ЧДД через OPM Flow Model_Z.xlsx"
)

#: ЧДД базового сценария Модели Z по OPM Flow, опубликованный организаторами 30.08.
REFERENCE_CHDD_M = 5157.016329811206


@pytest.fixture(scope="module")
def table() -> reference.ReferenceTable:
    return reference.load(REFERENCE_FILE)


def test_reference_file_present() -> None:
    assert REFERENCE_FILE.is_file(), "книга организаторов пропала из репозитория"


def test_years_cover_full_history(table: reference.ReferenceTable) -> None:
    assert table.years[0] == 1991
    assert table.years[-1] == 2025
    assert len(table.years) == 35


def test_cumulative_chdd_is_reference_number(table: reference.ReferenceTable) -> None:
    assert abs(table.cumulative_chdd_m - REFERENCE_CHDD_M) <= 1e-9


def test_start_year_is_1991(table: reference.ReferenceTable) -> None:
    """База дисконтирования эталона — 1991, а не 2007 (открытый вопрос В-01)."""
    assert table.start_year == 1991
    assert table.rows["Коэффициент дисконтирования"][1991] == 1.0


def test_discount_factors_follow_wacc_10(table: reference.ReferenceTable) -> None:
    factors = table.rows["Коэффициент дисконтирования"]
    for year in table.years:
        expected = 1 / 1.1 ** (year - 1991)
        assert abs(factors[year] - expected) <= 1e-12


def test_all_mapped_rows_present(table: reference.ReferenceTable) -> None:
    for label in reference.ROW_TO_ANNUAL:
        assert label in table.rows


def test_events_totals(table: reference.ReferenceTable) -> None:
    """Числа событий эталона — то, во что упирается сверка затрат."""
    assert table.total("Количество установленных новых насосов") == 145
    assert table.total("Количество остановок или запусков") == 131
    assert table.total("Количество переводов под закачку") == 28


def test_series_rejects_unknown_label(table: reference.ReferenceTable) -> None:
    with pytest.raises(KeyError):
        table.series("Такой строки нет")


def test_compare_matches_years_and_skips_missing() -> None:
    table = reference.ReferenceTable(
        path=Path("нет"),
        sheet="лист",
        years=(1991, 1992, 1993),
        rows={
            "ЧДД": {1991: 10.0, 1992: 20.0, 1993: 30.0},
            "Коэффициент дисконтирования": {1991: 1.0, 1992: 0.5, 1993: 0.25},
        },
    )
    annual = [
        {"year": 1992, "chddM": 22.0},
        {"year": 1993, "chddM": 30.0},
    ]
    diffs = reference.compare(table, annual, labels=("ЧДД",))
    assert [d.year for d in diffs] == [1992, 1993]
    assert diffs[0].delta == pytest.approx(2.0)
    assert diffs[0].relative == pytest.approx(0.1)
    assert diffs[1].delta == pytest.approx(0.0)


def test_compare_relative_is_none_on_zero_reference() -> None:
    table = reference.ReferenceTable(
        path=Path("нет"),
        sheet="лист",
        years=(1991,),
        rows={"ЧДД": {1991: 0.0}},
    )
    diff = reference.compare(table, [{"year": 1991, "chddM": 5.0}], labels=("ЧДД",))[0]
    assert diff.relative is None
    assert diff.delta == pytest.approx(5.0)


def test_start_year_requires_unit_factor() -> None:
    table = reference.ReferenceTable(
        path=Path("нет"),
        sheet="лист",
        years=(2000,),
        rows={"Коэффициент дисконтирования": {2000: 0.5}},
    )
    with pytest.raises(ValueError, match="не определяется однозначно"):
        assert table.start_year


def test_load_rejects_foreign_layout(tmp_path: Path) -> None:
    """Чужая книга не должна молча прочитаться как эталон."""
    book = openpyxl.Workbook()
    sheet = book.active
    sheet["A1"] = "Что-то другое"
    path = tmp_path / "чужая.xlsx"
    book.save(path)
    with pytest.raises(ValueError, match="формат книги изменился"):
        reference.load(path)


def test_load_reports_missing_rows(tmp_path: Path) -> None:
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append([reference.HEADER_LABEL, "Ед. изм.", 1991, 1992])
    sheet.append(["ЧДД", "млн руб.", 1.0, 2.0])
    path = tmp_path / "обрезок.xlsx"
    book.save(path)
    with pytest.raises(ValueError, match="не найдены строки"):
        reference.load(path)
