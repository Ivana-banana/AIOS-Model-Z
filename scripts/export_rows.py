"""Проверить прогон, выгрузить `rows.csv` и посчитать ЧДД эталонным расчётчиком.

    python scripts/export_rows.py runs/base

Печатает всё, что нужно проверить прежде, чем радоваться числу, и пишет
`rows.csv` — им и обмениваемся между машинами: мегабайты вместо сотен мегабайт
`.UNSMRY`, и его читает веб-платформа.

ЧДД считает только вендоренный расчётчик организаторов: своей формулы в проекте
нет и быть не должно (`CLAUDE.md`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aios import config as config_module  # noqa: E402
from aios import economics, summary  # noqa: E402

#: Ориентир организаторов по накопленной нефти за историю, тыс. т. Их число из
#: разбора задачи; расхождение больше пары процентов означает, что мы посчитали
#: не ту модель, а не что мы её улучшили.
REFERENCE_OIL_KT = 4771.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", nargs="?", default=None,
                        help="каталог прогона (по умолчанию [case].run_dir из конфигурации)")
    parser.add_argument("--config", default=config_module.DEFAULT_PATH)
    parser.add_argument("--case-name", default=None,
                        help="имя кейса без расширения (по умолчанию [case].case_name)")
    arguments = parser.parse_args(argv)

    config = config_module.load(arguments.config)
    run_dir = Path(arguments.run_dir) if arguments.run_dir else config.case.run_dir
    case_name = arguments.case_name or config.case.case_name
    case = run_dir / case_name

    if not case.with_suffix(".UNSMRY").is_file():
        print(f"нет результата расчёта: {case}.UNSMRY", file=sys.stderr)
        return 1

    rows = summary.monthly_rows(case, fluids=config.fluids)
    if not rows:
        print("выгрузка пустая: SUMMARY-секция не попала в кейс или прогон оборвался",
              file=sys.stderr)
        return 1

    wells = sorted({row["well"] for row in rows})
    dates = sorted({row["DATA"] for row in rows})
    oil_kt = sum(row["WOMT_Diff"] for row in rows) / 1000.0
    liquid_kt = sum(row["WLPT_Diff"] for row in rows) / 1000.0
    injection_km3 = sum(row["WWIT_Diff"] for row in rows) / 1000.0

    print(f"кейс:      {case}")
    print(f"строк:     {len(rows)} ({len(wells)} скважин × {len(dates)} месяцев)")
    print(f"период:    {dates[0]} — {dates[-1]}")
    print()
    print(f"нефть:     {oil_kt:11.1f} тыс. т   ориентир организаторов {REFERENCE_OIL_KT:.0f} тыс. т "
          f"({100.0 * (oil_kt - REFERENCE_OIL_KT) / REFERENCE_OIL_KT:+.1f} %)")
    print(f"жидкость:  {liquid_kt:11.1f} тыс. т")
    print(f"закачка:   {injection_km3:11.1f} тыс. м3")

    if oil_kt <= 0.0:
        print("\nнефти ноль: скважины остались без соединений с пластом (COMPDAT), "
              "прогон недействителен", file=sys.stderr)
        return 1

    problems = summary.sanity_problems(rows)
    print()
    print(f"--- санити-чеки: {'нарушений нет' if not problems else f'нарушений {len(problems)}'} ---")
    for line in problems[:15]:
        print(f"  {line}")
    if len(problems) > 15:
        print(f"  … ещё {len(problems) - 15}")
    if problems:
        # Отрицательные приращения — не наша ошибка и не повод «чистить» выгрузку:
        # методика исключает такие строки из экономики целиком, так же посчитают
        # и организаторы. Печатаем, считаем дальше.
        print("  (строки с отрицательными приращениями расчётчик исключает сам — "
              "правки выгрузки запрещены)")

    csv_path = summary.write_csv(rows, run_dir / "rows.csv")
    print()
    print(f"выгрузка:  {csv_path} ({csv_path.stat().st_size / 1e6:.1f} МБ)")

    result = economics.compute_npv(
        rows,
        headers=economics.REQUIRED_COLUMNS,
        start_date=config.economics.start_date.isoformat(),
        name=f"ЧДД {run_dir.name}",
        source_file=str(csv_path),
    )

    print()
    print(f"--- ЧДД (эталонный расчётчик, дисконт от {config.economics.start_date}) ---")
    print(f"ЧДД:              {result.chdd_m:12.1f} млн руб.")
    print(f"выручка:          {result.revenue_m:12.1f} млн руб.")
    print(f"OPEX:             {result.opex_m:12.1f} млн руб.")
    print(f"CAPEX:            {result.capex_m:12.1f} млн руб.")
    print(f"нефть:            {result.oil_kt:12.1f} тыс. т")
    print(f"лет:              {result.years:12d}")
    print(f"смен ЭЦН:         {result.pump_changes:12.0f}")
    print(f"пусков/остановок: {result.start_stop_count:12d}")
    print(f"переводов под закачку: {result.conversion_count:7d}")
    print(f"исключено отрицательных строк: {result.excluded_negative_rows:d}")
    for warning in result.warnings:
        print(f"  предупреждение: {warning}")

    print()
    print("записать строку в docs/experiments.md: дата, коммит, конфиг, seed, "
          "время счёта, число процессов, машина, ЧДД")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
