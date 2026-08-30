"""Эталонные годовые раскладки ЧДД, опубликованные организаторами.

30.08 организаторы выложили расчёт ЧДД базового сценария по Модели Z **в двух
симуляторах**, и это две разные роли:

- **OPM Flow** — тот же симулятор, которым считаем мы. Годится как построчная
  сверка нашего контура: физика, состав OPEX, события и дисконт по отдельности.
- **тНавигатор** — симулятор, которым организаторы будут проверять сдачу. Разница
  между двумя их файлами и есть тот самый разрыв «наш симулятор ↔ тНавигатор»,
  который до 30.08 у нас числился неизмеримым своими силами.

Здесь только чтение раскладки в типизированную структуру. Сравнение — в
`scripts/compare_reference.py`; своей формулы ЧДД тут нет и быть не может
(`CLAUDE.md`, главное правило проекта).

Раскладка устроена как «показатель × год»: первая колонка — название строки,
вторая — единица измерения, дальше по колонке на год. Названия строк совпадают с
шапкой годового отчёта эталонного расчётчика, поэтому сопоставление с нашей
выгрузкой идёт по словарю `ROW_TO_ANNUAL`, а не по порядку строк.

Форматов пришло два — `.xlsx` и `.csv`, — и различаются они не только разделителем:
в CSV десятичная запятая, а точки в подписях («Ед. изм.») заменены на запятые.
Поэтому названия строк при чтении **нормализуются**, а не сравниваются буквально;
иначе один и тот же показатель из двух файлов не сошёлся бы сам с собой.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openpyxl

#: Строка книги → ключ годовой раскладки эталонного расчётчика (`NpvResult.annual`).
#: Сверяются только величины, которые считает наш контур; счётчики оборудования
#: («ЭЦН 15 м3/сут — установлено новых») оставлены за рамками: они разложены по
#: типоразмерам, а нам важна сумма затрат, а не гистограмма.
ROW_TO_ANNUAL: dict[str, str] = {
    "Добыча нефти": "oilKt",
    "Добыча жидкости": "liquidKt",
    "Закачка": "injectionKm3",
    "Выручка от основной деятельности": "revenueM",
    "OPEX на нефть": "oilOpexM",
    "OPEX на жидкость": "liquidOpexM",
    "OPEX на закачку": "injectionOpexM",
    "Содержание действующего фонда": "fundOpexM",
    "Операции по смене ЭЦН": "pumpOperationM",
    "Остановки и запуски": "startStopCostM",
    "Переводы под закачку": "conversionOpexM",
    "Налог на имущество": "propertyTaxM",
    "Полный OPEX": "opexM",
    "CAPEX новых ЭЦН": "capexM",
    "Амортизация": "depreciationM",
    "Отчисления": "deductionsM",
    "Прибыль до налогообложения": "taxableProfitM",
    "Налог на прибыль": "profitTaxM",
    "EBITDA": "ebitdaM",
    "FCF": "fcfM",
    "ЧДД": "chddM",
    "Накопленный ЧДД": "cumulativeChddM",
    "Коэффициент дисконтирования": "discountFactor",
    "Количество остановок или запусков": "startStopCount",
    "Количество переводов под закачку": "conversionCount",
}

#: Заголовок первой колонки — им опознаётся строка с годами.
HEADER_LABEL = "Показатель"

#: Разделитель колонок в CSV-выгрузке организаторов.
CSV_DELIMITER = ";"


def normalize_label(label: str) -> str:
    """Привести подпись строки к виду, одинаковому для `.xlsx` и `.csv`.

    В CSV точки в подписях заменены запятыми («Ед, изм.», «млн руб,»), а сами
    названия показателей совпадают. Схлопываем пробелы и убираем хвостовую
    пунктуацию — сравнивать подписи буквально нельзя.
    """

    return " ".join(label.replace(" ", " ").split()).strip(" .,;")


def _to_float(value: Any) -> float:
    """Число из ячейки. В CSV десятичный разделитель — запятая."""

    if value is None or value == "":
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    return float(str(value).replace(" ", "").replace(" ", "").replace(",", "."))


def _table_from_grid(grid: list[list[Any]], *, path: Path, sheet: str) -> ReferenceTable:
    """Собрать раскладку из уже прочитанной сетки «строка × колонка»."""

    if not grid:
        raise ValueError(f"{path}: пустая раскладка")

    header = grid[0]
    if normalize_label(str(header[0] or "")) != HEADER_LABEL:
        raise ValueError(
            f"{path}: первая ячейка {header[0]!r}, ожидалось {HEADER_LABEL!r} — "
            "формат раскладки изменился, сверку нельзя считать корректной."
        )
    years = tuple(int(value) for value in header[2:] if value not in (None, ""))

    rows: dict[str, dict[int, float]] = {}
    for line in grid[1:]:
        if not line or line[0] in (None, ""):
            continue
        label = normalize_label(str(line[0]))
        values = line[2 : 2 + len(years)]
        rows[label] = {
            year: _to_float(value)
            for year, value in zip(years, values, strict=False)
        }

    missing = sorted(set(ROW_TO_ANNUAL) - set(rows))
    if missing:
        raise ValueError(
            f"{path}: в раскладке не найдены строки {missing} — "
            "сопоставление с нашей выгрузкой неполно."
        )
    return ReferenceTable(path=path, sheet=sheet, years=years, rows=rows)


@dataclass(frozen=True)
class ReferenceTable:
    """Годовая раскладка эталона: `rows[название][год] = значение`."""

    path: Path
    sheet: str
    years: tuple[int, ...]
    rows: dict[str, dict[int, float]]

    @property
    def cumulative_chdd_m(self) -> float:
        """ЧДД сценария — накопленное значение последнего года книги."""
        return self.rows["Накопленный ЧДД"][self.years[-1]]

    @property
    def start_year(self) -> int:
        """Год, от которого организаторы дисконтируют: коэффициент равен 1."""
        factors = self.rows["Коэффициент дисконтирования"]
        for year in self.years:
            if abs(factors[year] - 1.0) <= 1e-12:
                return year
        raise ValueError(
            "В книге нет года с коэффициентом дисконтирования 1 — "
            "база дисконтирования не определяется однозначно."
        )

    def series(self, label: str) -> dict[int, float]:
        if label not in self.rows:
            raise KeyError(f"в книге нет строки {label!r}")
        return self.rows[label]

    def total(self, label: str) -> float:
        return sum(self.series(label).values())


def load(path: Path) -> ReferenceTable:
    """Прочитать раскладку организаторов: `.xlsx` или `.csv`.

    Из книги берутся посчитанные значения, а не формулы.
    """

    if path.suffix.lower() == ".csv":
        # utf-8-sig: файл приходит с BOM, иначе первая подпись не опознаётся
        with path.open(encoding="utf-8-sig", newline="") as handle:
            grid = [list(row) for row in csv.reader(handle, delimiter=CSV_DELIMITER)]
        return _table_from_grid(grid, path=path, sheet=path.stem)

    book = openpyxl.load_workbook(path, data_only=True, read_only=True)
    sheet = book.worksheets[0]
    grid = [list(row) for row in sheet.iter_rows(values_only=True)]
    title = sheet.title
    book.close()
    return _table_from_grid(grid, path=path, sheet=title)


@dataclass(frozen=True)
class YearDiff:
    """Расхождение по одному показателю за один год."""

    year: int
    label: str
    reference: float
    ours: float

    @property
    def delta(self) -> float:
        return self.ours - self.reference

    @property
    def relative(self) -> float | None:
        """Относительное расхождение; `None`, когда эталон равен нулю."""
        if abs(self.reference) <= 1e-12:
            return None
        return self.delta / abs(self.reference)


def compare(
    table: ReferenceTable,
    annual: list[dict[str, Any]],
    *,
    labels: tuple[str, ...] | None = None,
) -> list[YearDiff]:
    """Сопоставить годовую раскладку нашего расчёта с книгой организаторов.

    Годы, которых нет в нашей раскладке, пропускаются молча: до 1994 модель ещё
    не считается, и нулевые колонки книги сравнивать не с чем.
    """

    wanted = labels if labels is not None else tuple(ROW_TO_ANNUAL)
    ours = {int(item["year"]): item for item in annual}
    result: list[YearDiff] = []
    for label in wanted:
        key = ROW_TO_ANNUAL[label]
        reference = table.series(label)
        for year in table.years:
            item = ours.get(year)
            if item is None:
                continue
            result.append(
                YearDiff(
                    year=year,
                    label=label,
                    reference=reference[year],
                    ours=float(item[key]),
                )
            )
    return result
