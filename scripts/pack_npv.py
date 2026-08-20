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

    rows_out: list[tuple[str, float, str, str, float, float, float]] = []
    skipped: list[str] = []
    for scenario in manifest["scenarios"]:
        name = scenario["name"]
        case = arguments.run_root / name / config.case.case_name
        if not (case.parent / ".doe-done").is_file():
            skipped.append(f"{name}: нет отметки о завершении")
            continue
        try:
            rows = summary.monthly_rows(case, fluids=config.fluids,
                                        well_fluids=deck_fluids.by_well)
        except ValueError as error:
            # один негодный сценарий не должен уносить таблицу по остальным
            # сорока семи, но и молча стать нулём в ней он тоже не должен
            skipped.append(f"{name}: {error}")
            continue
        if not rows:
            skipped.append(f"{name}: выгрузка пустая")
            continue
        result = economics.compute_npv(
            rows, headers=economics.REQUIRED_COLUMNS,
            start_date=config.economics.start_date.isoformat(), name=name,
        )
        rows_out.append((
            name, result.chdd_m,
            _multiplier_cell(scenario, "multipliers"),
            _multiplier_cell(scenario, "injector_multipliers"),
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
        line += (f"{produce:>10}{inject:>12}"
                 f"{oil:>10.1f}{liquid:>11.1f}{injection:>11.1f}")
        print(line)

    print()
    print(f"досчитано {len(rows_out)} из {len(manifest['scenarios'])}, "
          f"нефть и жидкость — тыс. т, закачка — тыс. м3")
    periods = max((len(item.get("periods") or ()) for item in manifest["scenarios"]),
                  default=0)
    if periods > 1:
        print(f"режим меняется внутри сценария: периодов {periods}, в колонках "
              "множителей показан размах средних по периодам, а не одно число")
    for line in skipped:
        print(f"  пропущен {line}")
    return 0


def _multiplier_cell(scenario: dict, key: str) -> str:
    """Множитель сценария: одно число при одном периоде, размах — при нескольких.

    Брать `scenario["multipliers"]` нельзя: это множители **первого периода**,
    поле оставлено ради совместимости манифеста и так и описано в `aios/doe.py`.
    На 4-периодном сценарии генератора оно давало 0,384 при среднем по сценарию
    0,453 и размахе 0,326…0,743 — то есть колонка описывала начало прогона, а
    читалась как весь прогон (замер 20.08).

    Размах, а не среднее: сценарий, идущий 0,7 → 1,3, в среднем равен единице,
    то есть «режим не менялся», — прямо противоположно тому, что в нём было.

    Прочерк вместо нуля: пакет по нагнетателям не трогает отбор вовсе, и «0.00»
    в этой клетке читается как «отбор остановлен».
    """
    periods = scenario.get("periods") or [scenario]
    means = [statistics.fmean(values.values())
             for period in periods
             if (values := period.get(key) or {})]
    if not means:
        return "—"
    if min(means) == max(means):
        return f"{means[0]:.2f}"
    return f"{min(means):.2f}…{max(means):.2f}"


if __name__ == "__main__":
    # Вывод только в UTF-8: умолчание берётся из локали, и на узкой печать
    # кириллицы роняет скрипт уже после счёта (п. 3.5 ТЗ).
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
