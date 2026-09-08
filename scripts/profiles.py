"""Профили по скважинам: то, ради чего человек строит график.

    python scripts/profiles.py runs/base
    python scripts/profiles.py runs/base --months 12

Принцип 1 `CLAUDE.md` требует смотреть на данные глазами, а принцип 5 — чтобы
любой результат повторялся одной командой. Разбор профилей 18.08 первое
выполнил, второе нет: числа попали в отчёт, а скрипта, который их считает, не
осталось. Здесь он.

**Обводнённость печатается в двух базах сразу, и это не педантизм.** Выгрузка
даёт массы (`WLPT_Diff`, `WOMT_Diff` — тонны), поэтому естественная для неё
обводнённость — массовая; а `aios.balance` показывает объёмную, потому что
баланс воды считается в кубометрах. Разница доходит до шести процентных пунктов:
объёмные 0,80 — это 0,84 по массе, а порог «выше 0,9 по массе» соответствует
объёмным 0,875. Значит состав группы «безнадёжно обводнённых» зависит от базы, и
называть её надо явно.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aios import config as config_module  # noqa: E402
from aios import summary  # noqa: E402
from aios.summary import Fluids  # noqa: E402

BUCKETS = ((0.0, 0.1), (0.1, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01))


def mass_watercut(oil_t: float, liquid_t: float) -> float:
    """Обводнённость по массе — та, что прямо следует из контракта выгрузки."""
    return 1.0 - oil_t / liquid_t if liquid_t > 0 else 0.0


def volume_watercut(oil_t: float, liquid_t: float, fluids: Fluids) -> float:
    """Обводнённость по объёму — та, что показывает баланс воды."""
    oil_m3 = oil_t * 1000.0 / fluids.oil_density
    water_m3 = max(liquid_t - oil_t, 0.0) * 1000.0 / fluids.water_density
    total = oil_m3 + water_m3
    return water_m3 / total if total > 0 else 0.0


def by_well(rows: list[dict[str, Any]], months: int,
            fluids: Fluids) -> dict[str, dict[str, float]]:
    """Признаки по каждой скважине: за весь горизонт и за хвост в `months`."""
    dates = sorted({str(row["DATA"]) for row in rows})
    tail = set(dates[-months:]) if months else set(dates)

    wells: dict[str, dict[str, float]] = {}
    rates: dict[str, list[float]] = {}
    for row in rows:
        well = str(row["well"])
        state = wells.setdefault(well, dict.fromkeys(
            ("oil_all", "liquid_all", "oil_tail", "liquid_tail", "injection"), 0.0))
        oil, liquid = float(row["WOMT_Diff"]), float(row["WLPT_Diff"])
        if oil < 0 or liquid < 0:
            continue  # методика исключает такие строки целиком
        state["oil_all"] += oil
        state["liquid_all"] += liquid
        state["injection"] += max(float(row["WWIT_Diff"]), 0.0)
        if str(row["DATA"]) in tail:
            state["oil_tail"] += oil
            state["liquid_tail"] += liquid
            if float(row["WLPR"]) > 0:
                rates.setdefault(well, []).append(float(row["WLPR"]))

    for well, state in wells.items():
        state["wc_mass_all"] = mass_watercut(state["oil_all"], state["liquid_all"])
        state["wc_mass_tail"] = mass_watercut(state["oil_tail"], state["liquid_tail"])
        state["wc_volume_tail"] = volume_watercut(
            state["oil_tail"], state["liquid_tail"], fluids)
        state["rate_median"] = statistics.median(rates[well]) if rates.get(well) else 0.0
    return wells


def buckets(values: list[float]) -> list[tuple[str, int]]:
    return [
        (f"{low:.1f}–{min(high, 1.0):.1f}",
         sum(1 for value in values if low <= value < high))
        for low, high in BUCKETS
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Профили по скважинам из выгрузки прогона.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--months", type=int, default=12,
                        help="длина хвоста горизонта, месяцев (0 — весь горизонт)")
    parser.add_argument("--config", type=Path, default=ROOT / "config.toml")
    arguments = parser.parse_args(argv)

    refusal = config_module.refusal(arguments.config)
    if refusal:
        print(refusal, file=sys.stderr)
        return 1
    config = config_module.load(arguments.config)
    rows_csv = arguments.run_dir / "rows.csv"
    if rows_csv.is_file():
        with rows_csv.open(encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
        for row in rows:
            for column in ("WLPT_Diff", "WOMT_Diff", "WWIT_Diff", "WLPR", "WWIR"):
                row[column] = float(row[column])
    else:
        rows = summary.monthly_rows(arguments.run_dir / config.case.case_name,
                                    fluids=config.fluids)
    if not rows:
        print(f"нет выгрузки в {arguments.run_dir}", file=sys.stderr)
        return 1

    wells = by_well(rows, arguments.months, config.fluids)
    producing = {well: state for well, state in wells.items() if state["liquid_all"] > 0}
    active = {well: state for well, state in producing.items() if state["liquid_tail"] > 0}

    print(f"строк {len(rows)}, скважин {len(wells)}, добывающих {len(producing)}, "
          f"действующих в последние {arguments.months} мес. {len(active)}")
    print()
    print("Обводнённость: за весь горизонт и за хвост, в обеих базах")
    print(f"{'интервал':>10} {'горизонт(м)':>12} {'хвост(масса)':>13} {'хвост(объём)':>13}")
    for (label, whole), (_, tail_mass), (_, tail_volume) in zip(
        buckets([s["wc_mass_all"] for s in producing.values()]),
        buckets([s["wc_mass_tail"] for s in active.values()]),
        buckets([s["wc_volume_tail"] for s in active.values()]),
        strict=True,
    ):
        print(f"{label:>10} {whole:>12} {tail_mass:>13} {tail_volume:>13}")

    print()
    for basis in ("wc_mass_tail", "wc_volume_tail"):
        watered = {w: s for w, s in active.items() if s[basis] > 0.9}
        oil = sum(s["oil_tail"] for s in watered.values())
        share = oil / sum(s["oil_tail"] for s in active.values()) if active else 0.0
        name = "по массе" if basis == "wc_mass_tail" else "по объёму"
        print(f"обводнены выше 0,9 {name}: {len(watered)} скв., "
              f"нефти за хвост {oil:,.0f} т — {share:.1%} добычи действующего фонда")

    rates = [s["rate_median"] for s in active.values() if s["rate_median"] > 0]
    if rates:
        print(f"медиана дебита жидкости у действующих: {statistics.median(rates):.1f} м3/сут, "
              f"максимум {max(rates):.1f}")
    return 0


if __name__ == "__main__":
    # Вывод только в UTF-8, как во всех остальных скриптах пакета: умолчание
    # берётся из локали, и на узкой (`LANG=C`, консоль cp437) печать кириллицы
    # роняет скрипт — причём уже после того, как всё посчитано. Скрытая
    # зависимость от локали запрещена п. 3.5 ТЗ.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
