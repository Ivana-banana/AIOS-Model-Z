"""Эталонная годовая раскладка ЧДД, опубликованная организаторами.

30.08 организаторы выложили расчёт ЧДД базового сценария по Модели Z в двух
симуляторах. Файл по OPM Flow — тот же симулятор, которым считаем мы, поэтому
он годится не как «ориентир по порядку величин», а как построчная сверка нашего
контура: физика, состав OPEX, события и дисконт проверяются по отдельности.

Здесь только чтение книги в типизированную структуру. Сравнение — в
`scripts/compare_reference.py`; своей формулы ЧДД тут нет и быть не может
(`CLAUDE.md`, главное правило проекта).

Книга устроена как «показатель × год»: первая колонка — название строки, вторая —
единица измерения, дальше по колонке на год. Названия строк совпадают с шапкой
годового отчёта эталонного расчётчика, поэтому сопоставление с нашей выгрузкой
идёт по словарю `ROW_TO_ANNUAL`, а не по порядку строк.
"""

from __future__ import annotations

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
    """Прочитать книгу организаторов. Значения берутся посчитанные, не формулы."""

    book = openpyxl.load_workbook(path, data_only=True, read_only=True)
    sheet = book.worksheets[0]
    grid = [list(row) for row in sheet.iter_rows(values_only=True)]
    book.close()
    if not grid:
        raise ValueError(f"{path}: пустой лист")

    header = grid[0]
    if header[0] != HEADER_LABEL:
        raise ValueError(
            f"{path}: первая ячейка {header[0]!r}, ожидалось {HEADER_LABEL!r} — "
            "формат книги изменился, сверку нельзя считать корректной."
        )
    years = tuple(int(value) for value in header[2:] if value is not None)

    rows: dict[str, dict[int, float]] = {}
    for line in grid[1:]:
        label = line[0]
        if label is None:
            continue
        values = line[2 : 2 + len(years)]
        rows[str(label)] = {
            year: float(value if value is not None else 0.0)
            for year, value in zip(years, values, strict=False)
        }

    missing = sorted(set(ROW_TO_ANNUAL) - set(rows))
    if missing:
        raise ValueError(
            f"{path}: в книге не найдены строки {missing} — "
            "сопоставление с нашей выгрузкой неполно."
        )
    return ReferenceTable(path=path, sheet=sheet.title, years=years, rows=rows)


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
