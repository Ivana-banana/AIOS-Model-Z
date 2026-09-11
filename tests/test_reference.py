"""Чтение и сверка эталонной раскладки организаторов (`aios.reference`).

Файл организаторов лежит в репозитории, поэтому контрольные числа берутся из
него самого: если книгу заменят другой версией, тесты упадут — и это правильно,
сверка на подменённом эталоне не значит ничего.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import openpyxl
import pytest

from aios import reference

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import compare_reference  # noqa: E402
from compare_reference import report  # noqa: E402

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


def test_decompose_sums_to_chdd_gap() -> None:
    """Сумма вкладов слагаемых обязана дать разницу ЧДД, а не «примерно».

    Это сторож на само разложение: если знак или состав `FCF_TERMS` разъедется,
    невязка перестанет быть нулевой, и объяснение расхождения станет вымыслом.
    """
    table = reference.ReferenceTable(
        path=Path("нет"),
        sheet="лист",
        years=(1991, 1992),
        rows={
            "Коэффициент дисконтирования": {1991: 1.0, 1992: 0.5},
            "Выручка от основной деятельности": {1991: 100.0, 1992: 200.0},
            "Полный OPEX": {1991: 10.0, 1992: 20.0},
            "CAPEX новых ЭЦН": {1991: 1.0, 1992: 2.0},
            "Отчисления": {1991: 70.0, 1992: 140.0},
            "Налог на прибыль": {1991: 5.0, 1992: 10.0},
            "ЧДД": {1991: 14.0, 1992: 14.0},
        },
    )
    annual = [
        {"year": 1991, "revenueM": 110.0, "opexM": 10.0, "capexM": 1.0,
         "deductionsM": 70.0, "profitTaxM": 5.0, "chddM": 24.0},
        {"year": 1992, "revenueM": 200.0, "opexM": 24.0, "capexM": 2.0,
         "deductionsM": 140.0, "profitTaxM": 10.0, "chddM": 12.0},
    ]
    parts = reference.decompose(table, annual)
    # выручка выросла на 10 в год с коэффициентом 1, OPEX — на 4 в год с 0,5
    assert parts["Выручка от основной деятельности"] == pytest.approx(10.0)
    assert parts["Полный OPEX"] == pytest.approx(-2.0)
    ours_chdd = sum(item["chddM"] for item in annual)
    gap = ours_chdd - sum(table.series("ЧДД").values())
    assert sum(parts.values()) == pytest.approx(gap)


def test_decompose_skips_years_absent_from_our_run() -> None:
    table = reference.ReferenceTable(
        path=Path("нет"),
        sheet="лист",
        years=(1991, 1992),
        rows={
            "Коэффициент дисконтирования": {1991: 1.0, 1992: 0.5},
            "Выручка от основной деятельности": {1991: 100.0, 1992: 200.0},
        },
    )
    parts = reference.decompose(
        table,
        [{"year": 1992, "revenueM": 210.0}],
        terms={"Выручка от основной деятельности": +1},
    )
    assert parts["Выручка от основной деятельности"] == pytest.approx(5.0)


def test_explained_report_returns_reference_chdd(capsys: pytest.CaptureFixture[str]) -> None:
    """`--explain` не должен менять тип итога, нужного для сравнения эталонов."""
    table = reference.ReferenceTable(
        path=Path("эталон.xlsx"),
        sheet="лист",
        years=(1991,),
        rows={
            "ЧДД": {1991: 10.0},
            "Накопленный ЧДД": {1991: 10.0},
            "Коэффициент дисконтирования": {1991: 1.0},
            "Добыча нефти": {1991: 1.0},
            "Выручка от основной деятельности": {1991: 10.0},
            "Полный OPEX": {1991: 2.0},
            "CAPEX новых ЭЦН": {1991: 1.0},
            "Отчисления": {1991: 3.0},
            "Налог на прибыль": {1991: 1.0},
        },
    )
    result = SimpleNamespace(
        chdd_m=12.0,
        annual=[{
            "year": 1991, "chddM": 12.0, "oilKt": 1.5, "revenueM": 12.0,
            "opexM": 2.0, "capexM": 1.0, "deductionsM": 3.0, "profitTaxM": 1.0,
        }],
    )

    returned = report(table, result, labels=("ЧДД",), detailed=False, explain=True)

    assert returned == 10.0
    assert "откуда расхождение" in capsys.readouterr().out


def test_compare_reference_rejects_a_different_economic_start_year(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Сменить базу без новой раскладки организаторов означает сравнить разное."""
    rows = tmp_path / "rows.csv"
    rows.write_text("DATA,well\n", encoding="utf-8")

    def must_not_run(*args, **kwargs):
        pytest.fail("расчёт не должен запускаться при несовместимой базе эталона")

    monkeypatch.setattr(compare_reference.economics, "compute_npv_from_file", must_not_run)

    assert compare_reference.main(["--rows", str(rows), "--start-year", "2007"]) == 2
    assert "1991" in capsys.readouterr().err


def test_reference_coverage_rejects_a_nonzero_year_missing_from_our_run() -> None:
    """Итог ЧДД нельзя сравнивать с раскладкой, у которой пропущен живой год."""
    table = reference.ReferenceTable(
        path=Path("эталон.xlsx"),
        sheet="лист",
        years=(1991, 1992),
        rows={
            "ЧДД": {1991: 0.0, 1992: 10.0},
            "Накопленный ЧДД": {1991: 0.0, 1992: 10.0},
        },
    )

    problem = compare_reference.coverage_problem(table, [{"year": 1991}])

    assert problem is not None
    assert "1992" in problem


def test_events_agree_with_both_references_after_2004(
    table: reference.ReferenceTable, tnavigator: reference.ReferenceTable
) -> None:
    """С 2004 счётчик остановок-пусков у обоих симуляторов один и тот же.

    Всё расхождение по событиям живёт в 1994–2003, когда фонд вводился в
    работу. Там сами эталоны расходятся между собой на 35 % — то есть счётчик
    событий в этот период неустойчив у организаторов, а не только у нас.
    """
    key = "Количество остановок или запусков"
    late = [year for year in table.years if year >= 2004]
    assert sum(table.rows[key][y] for y in late) == 23
    assert sum(tnavigator.rows[key][y] for y in late) == 23

    early = [year for year in table.years if 1994 <= year <= 2003]
    opm_early = sum(table.rows[key][y] for y in early)
    tn_early = sum(tnavigator.rows[key][y] for y in early)
    assert opm_early == 108
    assert tn_early == 80


# ── разрыв схем в окне, за которое заявляется число ──────────────────────────


def test_the_window_of_the_gap_is_the_window_of_the_declared_number():
    """Разрыв схем считается в том окне, в котором заявляется ЧДД.

    До 09.09 он был известен только накопленным с начала книги — с 1991:
    «1,20 % между расчётами организаторов». Но заявляется число за период
    управления, а у эталонного расчётчика окно и база дисконтирования — один
    параметр `--start-year`, и он равен 2007 (Р-249). Годы 1991–2006 в
    заявляемое число не входят вовсе, а разрыв там втрое больше: на
    разбуривании он доходит до 2,7 % по нефти. Считать его с 1991 значит
    приписывать заявке чужую погрешность.
    """
    import simulator_gap

    tables = simulator_gap.books()
    assert len(tables) >= 2, "в docs/reference/ меньше двух раскладок"

    for table in tables:
        # тождество, которым проверяется само перенесение базы: сумма ЧДД с
        # года, где коэффициент дисконтирования равен единице, и есть
        # накопленное значение книги
        assert simulator_gap.npv_in_window(table, table.start_year) == pytest.approx(
            table.cumulative_chdd_m, abs=1e-9), table.path.name

    wide = [simulator_gap.npv_in_window(one, 1991) for one in tables]
    narrow = [simulator_gap.npv_in_window(one, 2007) for one in tables]

    gap_wide = abs(wide[1] - wide[0]) / wide[0] * 100
    gap_narrow = abs(narrow[1] - narrow[0]) / narrow[0] * 100

    assert gap_wide == pytest.approx(1.20, abs=0.02), gap_wide
    assert gap_narrow == pytest.approx(0.29, abs=0.02), gap_narrow
    assert gap_narrow < gap_wide, (
        "окно 2007 обязано давать разрыв меньше: годы разбуривания в заявку не входят"
    )


def test_injection_agrees_between_the_two_schemes_and_that_is_the_control():
    """Закачка совпадает в ноль — это разделяющее наблюдение, а не совпадение.

    Закачку задаёт расписание напрямую, нефть и жидкость получаются решением
    уравнений. Если бы разошлась и она, перед нами были бы разные сценарии, а
    не разные численные схемы, и сравнивать было бы нечего.
    """
    import simulator_gap

    tables = simulator_gap.books()
    years = [year for year in tables[0].years if year >= 2007]

    worst = max(
        abs(simulator_gap.relative(tables[1].series("Закачка")[year],
                                   tables[0].series("Закачка")[year]))
        for year in years
    )
    assert worst < 0.2, f"закачка разошлась на {worst:.2f} % — это разные сценарии"
