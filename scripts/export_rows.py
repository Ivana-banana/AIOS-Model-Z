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

from aios import (  # noqa: E402
    balance,
    economics,
    pvt,
    summary,
)
from aios import config as config_module  # noqa: E402

#: Ориентир организаторов по накопленной нефти за историю, тыс. т. Их число из
#: разбора задачи; расхождение больше пары процентов означает, что мы посчитали
#: не ту модель, а не что мы её улучшили.
REFERENCE_OIL_KT = 4771.0


def water_notes(rows: list[dict[str, object]], config) -> list[str]:
    """Что говорит о прогоне наше же ограничение по воде.

    Печатается здесь, а не только в цикле, потому что сдаваемый сценарий цикл не
    проходил: он собран `build_scaled_case.py` — тупым бейзлайном, — и до 23.08
    его никто не сверял с правилом, которое мы объявили своим (Р-61). Проверка
    ничего не режет и прогон не роняет: она называет годы, где правило нарушено,
    и оставляет решение человеку.
    """
    limits = config.limits
    if not limits.enforce_water_balance:
        return ["проверка выключена (`enforce_water_balance = false`)"]
    rule = limits.water_balance_rule
    if rule == "compensation":
        found = balance.compensation_problems(rows, fluids=config.fluids,
                                              ceiling=limits.water_tolerance)
    elif rule == "produced_water":
        found = balance.problems(rows, fluids=config.fluids,
                                 tolerance=limits.water_tolerance)
    elif rule == "external_water":
        found = balance.external_water_problems(
            rows, fluids=config.fluids,
            limit_m3_per_day=limits.max_external_water_m3_per_day)
    else:
        return [f"неизвестное правило {rule!r} — проверка не делалась"]
    return found or ["нарушений нет"]


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

    # плотность берётся по PVT-региону скважины; пометка печатается всегда —
    # молча посчитать чужой пласт по нашей плотности нельзя (ТЗ п. 4.3)
    deck_fluids = pvt.fluids_by_well(config.case.model_dir)
    print(f"плотности:     {deck_fluids.note}")
    rows = summary.monthly_rows(case, fluids=config.fluids,
                                well_fluids=deck_fluids.by_well)
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
    # Подпись периода обязательна: ниже идёт второй блок итогов — от эталонного
    # расчётчика, — и он считает те же величины за другой период (с даты
    # дисконтирования) и без строк с отрицательными приращениями. Пока оба блока
    # назывались просто «нефть», числа за разные периоды уже дважды попадали в
    # одну строку журнала.
    print(f"--- итоги по всей выгрузке ({dates[0]} — {dates[-1]}) ---")
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

    water = water_notes(rows, config)
    print()
    print(f"--- баланс воды: правило {config.limits.water_balance_rule} ---")
    for line in water[:5]:
        print(f"  {line}")
    if len(water) > 5:
        print(f"  … ещё {len(water) - 5} лет")

    csv_path = summary.write_csv(rows, run_dir / "rows.csv")
    # Пометка о происхождении плотностей ложится рядом с выгрузкой: `rows.csv`
    # уезжает на другую машину, а прокрутка терминала — нет.
    note_path = deck_fluids.write_note(csv_path, fluids=config.fluids,
                                       model_dir=config.case.model_dir)
    print()
    print(f"выгрузка:  {csv_path} ({csv_path.stat().st_size / 1e6:.1f} МБ)")
    print(f"источник:  {note_path}")

    result = economics.compute_npv(
        rows,
        headers=economics.REQUIRED_COLUMNS,
        start_date=config.economics.start_date.isoformat(),
        norms_file=config.economics.norms_file,
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
    print(f"жидкость:         {result.liquid_kt:12.1f} тыс. т")
    print(f"закачка:          {result.injection_km3:12.1f} тыс. м3")
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
    # Вывод только в UTF-8: под Windows умолчание — cp1251, и печать «→»
    # или части кириллицы роняет скрипт уже после расчёта, на выдаче
    # результата. Скрытая зависимость от локали запрещена п. 3.5 ТЗ.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
