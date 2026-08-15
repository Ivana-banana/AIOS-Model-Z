"""Общие помощники тестов."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VENDOR_DIR = ROOT / "vendor" / "CHDD_PYTHON"
EXAMPLE_INPUT = VENDOR_DIR / "input" / "Пример_исходных_данных.xlsx"
MODEL_Z_SCHEDULE = ROOT / "vendor" / "Model_Z" / "Model_Z_sch.inc"


def make_row(
    data: str,
    well: str,
    *,
    wlpr: float = 0.0,
    womr: float = 0.0,
    wwir: float = 0.0,
    liquid: float = 0.0,
    oil: float = 0.0,
    injection: float = 0.0,
    thp: float = 100.0,
    bhp: float = 80.0,
) -> dict[str, object]:
    """Строка «скважина × месяц» в контракте эталонного расчётчика.

    `liquid`/`oil` — тонны за месяц, `injection` — м³ за месяц,
    `wlpr`/`wwir` — м³/сут.
    """
    return {
        "DATA": data,
        "well": well,
        "WLPT": 0.0,
        "WLPR": wlpr,
        "WOMT": 0.0,
        "WOMR": womr,
        "WWIR": wwir,
        "WWIT": 0.0,
        "THP": thp,
        "BHP": bhp,
        "WEFF": 1.0,
        "WLPT_Diff": liquid,
        "WOMT_Diff": oil,
        "WWIT_Diff": injection,
    }


def months(year: int, count: int, first: int = 1) -> list[str]:
    return [f"{year}-{month:02d}-01" for month in range(first, first + count)]
