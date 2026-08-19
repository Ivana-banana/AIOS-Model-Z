"""ЧДД по каждому сценарию пакета DoE — одной командой.

    python scripts/pack_npv.py runs/doe-injection
    python scripts/pack_npv.py runs/doe-injection --base 11851.8

Таблица «сценарий → ЧДД» в журнале уже была (16.08, пакеты `doe` и `doe-levels`),
но команды, которая её считает, не осталось — числа получены разовым разбором.
Принцип 5 `CLAUDE.md` требует обратного, а принцип 7 — чтобы такие числа вообще
попадали в журнал. Здесь эта команда.

Считается тем же путём, что и сдаваемое число: выгрузка `aios.summary` в 14
колонок контракта и **эталонный расчётчик организаторов**. Своей формулы ЧДД тут
нет и быть не может.

Сценарии без отметки `.doe-done` пропускаются с явной строкой: у оборванного
прогона `.UNSMRY` читается, но несёт обрезок истории, и его ЧДД — не ЧДД
сценария, а ЧДД его начала.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aios import config as config_module  # noqa: E402
from aios import economics, pvt, summary  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_root", type=Path, help="каталог пакета с doe.json")
    parser.add_argument("--config", type=Path, default=config_module.DEFAULT_PATH)
    parser.add_argument("--base", type=float, default=None,
                        help="ЧДД базового сценария, млн руб., для колонки «к базовому»")
    arguments = parser.parse_args(argv)

    config = config_module.load(arguments.config)
    manifest_path = arguments.run_root / "doe.json"
    if not manifest_path.is_file():
        print(f"нет манифеста {manifest_path}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    deck_fluids = pvt.fluids_by_well(config.case.model_dir)
    print(f"плотности: {deck_fluids.note}")
    print(f"пакет:     {arguments.run_root}")
    print()

    rows_out: list[tuple[str, float, float, float, float, float, float]] = []
    skipped: list[str] = []
    for scenario in manifest["scenarios"]:
        name = scenario["name"]
        case = arguments.run_root / name / config.case.case_name
        if not (case.parent / ".doe-done").is_file():
            skipped.append(f"{name}: нет отметки о завершении")
            continue
        rows = summary.monthly_rows(case, fluids=config.fluids,
                                    well_fluids=deck_fluids.by_well)
        if not rows:
            skipped.append(f"{name}: выгрузка пустая")
            continue
        result = economics.compute_npv(
            rows, headers=economics.REQUIRED_COLUMNS,
            start_date=config.economics.start_date.isoformat(), name=name,
        )
        rows_out.append((
            name, result.chdd_m,
            _mean(scenario.get("multipliers", {})),
            _mean(scenario.get("injector_multipliers", {})),
            result.oil_kt, result.liquid_kt, result.injection_km3,
        ))

    if not rows_out:
        print("ни одного досчитанного сценария", file=sys.stderr)
        for line in skipped:
            print(f"  {line}", file=sys.stderr)
        return 1

    header = f"{'сценарий':<10}{'ЧДД':>11}"
    if arguments.base is not None:
        header += f"{'к базовому':>12}"
    header += f"{'ср.отбор':>10}{'ср.закачка':>12}{'нефть':>10}{'жидкость':>11}{'закачка':>11}"
    print(header)
    for name, chdd, produce, inject, oil, liquid, injection in sorted(
        rows_out, key=lambda item: -item[1]
    ):
        line = f"{name:<10}{chdd:>11.1f}"
        if arguments.base is not None:
            line += f"{chdd - arguments.base:>+12.1f}"
        line += (f"{_cell(produce):>10}{_cell(inject):>12}"
                 f"{oil:>10.1f}{liquid:>11.1f}{injection:>11.1f}")
        print(line)

    print()
    print(f"досчитано {len(rows_out)} из {len(manifest['scenarios'])}, "
          f"нефть и жидкость — тыс. т, закачка — тыс. м3")
    for line in skipped:
        print(f"  пропущен {line}")
    return 0


def _mean(values: dict[str, float]) -> float | None:
    """Средний множитель сценария; `None`, если этот вид управления не менялся."""
    return statistics.fmean(values.values()) if values else None


def _cell(value: float | None) -> str:
    """Прочерк вместо нуля: пакет по нагнетателям не трогает отбор вовсе, и
    «0.00» в этой клетке читается как «отбор остановлен» — прямо противоположно
    тому, что было в сценарии."""
    return "—" if value is None else f"{value:.2f}"


if __name__ == "__main__":
    # Вывод только в UTF-8: умолчание берётся из локали, и на узкой печать
    # кириллицы роняет скрипт уже после счёта (п. 3.5 ТЗ).
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
