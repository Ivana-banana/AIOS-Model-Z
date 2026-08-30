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

#: BOM, с которого начинается CSV организаторов.
BOM = "\ufeff"

LINE_END = "\n"

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "docs" / "reference"
REFERENCE_FILE = REFERENCE_DIR / "Расчет ЧДД через OPM Flow Model_Z.xlsx"
TNAVIGATOR_FILE = REFERENCE_DIR / "Расчет ЧДД через тНавигатор Model_Z.csv"

#: ЧДД базового сценария Модели Z, опубликованный организаторами 30.08:
#: два симулятора, один и тот же сценарий.
REFERENCE_CHDD_M = 5157.016329811206
TNAVIGATOR_CHDD_M = 5218.9449080841405

#: Разрыв между симуляторами по ЧДД — то, что наш стенд измерить не может.
SIMULATOR_GAP_SHARE = (TNAVIGATOR_CHDD_M - REFERENCE_CHDD_M) / REFERENCE_CHDD_M


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
    with pytest.raises(ValueError, match="формат раскладки изменился"):
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


@pytest.fixture(scope="module")
def tnavigator() -> reference.ReferenceTable:
    return reference.load(TNAVIGATOR_FILE)


def test_tnavigator_file_present() -> None:
    assert TNAVIGATOR_FILE.is_file(), "раскладка тНавигатора пропала из репозитория"


def test_csv_reference_parses(tnavigator: reference.ReferenceTable) -> None:
    """CSV читается тем же кодом: BOM, точка с запятой, десятичная запятая."""
    assert abs(tnavigator.cumulative_chdd_m - TNAVIGATOR_CHDD_M) <= 1e-9
    assert tnavigator.years[0] == 1991
    assert tnavigator.years[-1] == 2025


def test_csv_reference_skips_empty_years(tnavigator: reference.ReferenceTable) -> None:
    """В CSV нет колонок 1992 и 1993 — раскладка короче на два года."""
    assert 1992 not in tnavigator.years
    assert len(tnavigator.years) == 33


def test_both_references_share_discount_base(
    table: reference.ReferenceTable, tnavigator: reference.ReferenceTable
) -> None:
    """Оба симулятора дисконтируют от одного года — иначе итоги несопоставимы."""
    assert table.start_year == tnavigator.start_year == 1991


def test_simulator_gap_is_measured_and_small(
    table: reference.ReferenceTable, tnavigator: reference.ReferenceTable
) -> None:
    """Разрыв тНавигатор ↔ OPM Flow на одном сценарии — 1,2 % по ЧДД.

    Сторож на число, которое мы выносим в отчёт как измеренное. Совпадение
    физики (закачка сходится в ноль) отличает разрыв численных схем от
    расхождения сценариев.
    """
    assert abs(SIMULATOR_GAP_SHARE - 0.0120) < 0.0005
    assert abs(tnavigator.total("Закачка") - table.total("Закачка")) < 1.0
    oil_gap = tnavigator.total("Добыча нефти") / table.total("Добыча нефти") - 1
    assert abs(oil_gap - 0.0050) < 0.0005


def test_labels_are_normalised_across_formats(
    table: reference.ReferenceTable, tnavigator: reference.ReferenceTable
) -> None:
    """Подписи из xlsx и csv совпадают после нормализации, а не буквально."""
    for label in reference.ROW_TO_ANNUAL:
        assert label in table.rows
        assert label in tnavigator.rows


def test_normalize_label_strips_csv_punctuation() -> None:
    assert reference.normalize_label(" Показатель ") == "Показатель"
    assert reference.normalize_label("Ед, изм.") == "Ед, изм"
    assert reference.normalize_label("Добыча  нефти") == "Добыча нефти"


def test_csv_reader_handles_decimal_comma(tmp_path: Path) -> None:
    path = tmp_path / "раскладка.csv"
    body = [
        f"{reference.HEADER_LABEL};Ед, изм.;1991;1992",
        *(f"{label};ед;1,5;2,5" for label in reference.ROW_TO_ANNUAL),
    ]
    path.write_text(BOM + LINE_END.join(body), encoding="utf-8")
    loaded = reference.load(path)
    assert loaded.series("ЧДД") == {1991: 1.5, 1992: 2.5}


def test_csv_reader_rejects_foreign_layout(tmp_path: Path) -> None:
    path = tmp_path / "чужая.csv"
    path.write_text("Что-то другое;ед;1991" + LINE_END, encoding="utf-8")
    with pytest.raises(ValueError, match="формат раскладки изменился"):
        reference.load(path)
