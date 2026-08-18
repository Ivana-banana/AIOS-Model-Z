"""Расчёт ЧДД — тонкая обвязка над эталонным расчётчиком организаторов.

Методику ЧДД мы не реализуем. Единственный источник числа — вендоренный как есть
`vendor/CHDD_PYTHON/` (версия 7.0.2-negative-row-filter). Здесь только загрузка
этого кода, приведение наших данных к его 14 колонкам и типизированный результат.

Расхождение заявленного ЧДД с пересчётом организаторов = дисквалификация, поэтому
любая своя формула ЧДД в проекте — дефект. См. CLAUDE.md.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor" / "CHDD_PYTHON"
NORMS_FILE = VENDOR_DIR / "input" / "Нормативы_ЧДД.xlsx"

if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

import chdd_model  # noqa: E402  (импорт после правки sys.path — иначе вендор не виден)
import excel_io  # noqa: E402

#: Контракт «симулятор → экономика». Порядок и написание — как у расчётчика.
REQUIRED_COLUMNS: list[str] = list(chdd_model.REQUIRED_COLUMNS)

#: Версия эталона, под которую написана обвязка. Обновили архив — обновить и тесты.
VENDOR_VERSION: str = chdd_model.VERSION


@dataclass(frozen=True)
class NpvResult:
    """Итог расчёта. `raw` — полный ответ эталона, включая годовую раскладку."""

    chdd_m: float
    """ЧДД сценария, млн руб. Это и есть заявляемое значение."""

    revenue_m: float
    opex_m: float
    capex_m: float
    fcf_m: float
    oil_kt: float
    liquid_kt: float
    injection_km3: float
    pump_changes: float
    start_stop_count: int
    conversion_count: int
    years: int
    excluded_negative_rows: int
    warnings: tuple[str, ...]
    raw: dict[str, Any]

    @property
    def annual(self) -> list[dict[str, Any]]:
        return self.raw["annual"]

    @property
    def field_monthly(self) -> list[dict[str, Any]]:
        return self.raw["fieldMonthly"]

    @property
    def events(self) -> list[dict[str, Any]]:
        return self.raw["events"]


def load_norms(norms_file: Path = NORMS_FILE) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Нормативы и таблица ЭЦН из файла организаторов — не из литералов в коде."""
    return excel_io.load_norms(norms_file)


def load_records(source_file: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Прочитать исходные данные (.xlsx или .csv) в формате эталонного расчётчика."""
    return excel_io.load_source_records(source_file)


def compute_npv(
    records: Sequence[dict[str, Any]],
    *,
    headers: Iterable[str] | None = None,
    start_date: Any = None,
    assumptions: dict[str, Any] | None = None,
    pumps: list[dict[str, Any]] | None = None,
    name: str = "Расчет ЧДД",
    source_file: str = "",
) -> NpvResult:
    """Посчитать ЧДД сценария эталонным расчётчиком.

    `records` — строки «скважина × месяц» с колонками REQUIRED_COLUMNS.
    `start_date` — начало экономического расчёта; год этой даты становится первым
    годом дисконтирования. Значение по умолчанию берёт эталон (минимальная дата
    данных), поэтому для сдачи его всегда задаём явно из конфига.

    Нормативы и таблица ЭЦН по умолчанию читаются из файла организаторов.
    """
    if assumptions is None or pumps is None:
        file_assumptions, file_pumps = load_norms()
        assumptions = file_assumptions if assumptions is None else assumptions
        pumps = file_pumps if pumps is None else pumps

    raw = chdd_model.compute_calculation(
        list(records),
        headers=headers,
        assumptions=assumptions,
        pumps=pumps,
        start_date=start_date,
        name=name,
        source_file=source_file,
    )
    return _to_result(raw)


def compute_npv_from_file(
    source_file: Path,
    *,
    start_date: Any = None,
    norms_file: Path = NORMS_FILE,
) -> NpvResult:
    """Тот же расчёт, но исходные данные берутся из .xlsx/.csv."""
    headers, records = load_records(source_file)
    assumptions, pumps = load_norms(norms_file)
    return compute_npv(
        records,
        headers=headers,
        start_date=start_date,
        assumptions=assumptions,
        pumps=pumps,
        source_file=str(source_file),
    )


def _to_result(raw: dict[str, Any]) -> NpvResult:
    summary = raw["summary"]
    diagnostics = raw["diagnostics"]
    return NpvResult(
        chdd_m=summary["totalChddM"],
        revenue_m=summary["totalRevenueM"],
        opex_m=summary["totalOpexM"],
        capex_m=summary["totalCapexM"],
        fcf_m=summary["totalFcfM"],
        oil_kt=summary["totalOilKt"],
        liquid_kt=summary["totalLiquidKt"],
        injection_km3=summary["totalInjectionK"],
        pump_changes=summary["pumpChanges"],
        start_stop_count=summary["startStopCount"],
        conversion_count=summary["conversionCount"],
        years=summary["years"],
        excluded_negative_rows=diagnostics["excludedNegativeRows"],
        warnings=tuple(diagnostics["warnings"]),
        raw=raw,
    )
